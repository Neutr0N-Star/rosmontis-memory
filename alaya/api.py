"""Alaya 稳定公共 API（v1）。

宿主只需要认识这一层：

    import alaya
    h = alaya.open("alaya.db", {"llm_callable": my_llm})
    h.wake("u1", "c1")
    ctx = h.context("u1", "c1", "他最近怎么样", budget=600)   # 纯函数：不写库、不超预算
    r = h.turn("u1", "c1", "我今天有点累", history=上一轮 messages)
    h.observe("u1", "c1", user_text, assistant_text)          # 四钩子里我们只负责记
    job = h.settle("u1", "c1")                                # 立即返回 job_id
    h.close()

契约来源：`docs/15-交付蓝图与模块契约.md` §3.3 / §3.4、`docs/_wip/A3-产品形态与接入.md` §3。

设计要点
--------
- `context` 是**纯函数**：内部走 harness.build_context(..., pure=True)，
  屏蔽检索日志落库；返回的 `memory_tokens` 被硬闸压到 `budget` 以内。
- `settle` **不阻塞**：后台线程跑结算/整理/做梦，立即返回 job_id；
  `wait(job_id)` 只是想同步看的调用方用的可选便利。
- 所有方法幂等：`wake` 靠种子 INSERT OR IGNORE；`observe` 对同一
  (space, conversation, user_text, assistant_text) 在窗口内去重；
  `settle` 对同一 session 的进行中任务去重；`close` 可重复调用。

v1 的 space / conversation 取值
-------------------------------
底层 `Memory` 仍然只有一层 `session_id`（WS-A 的 space/conversation 两层
隔离还没落盘）。v1 用 **space 作为 session_id**：同一 memory space 的不同
conversation 共享记忆（这正是「换会话不失忆」的产品要求），`conversation`
作为元数据原样带回、并参与 observe 幂等键。conversation 级隔离要等
WS-A 的 schema/gate 就位；这一点在 `docs/_wip/A7-接口实现报告.md` 里标为依赖。

错误形态
--------
方法出错抛 `AlayaError`（别名 `ApiError`）：`.code` / `.message` / `.retryable`，
`.as_dict()` 给出 `{"ok": False, "data": None, "error": {...}}` 的统一信封。
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
import uuid
from typing import Any, Callable

from .config import Config
from .harness import Harness, build_context as harness_build_context
from .tools import dispatch, item_state, transition_item

log = logging.getLogger("alaya.api")

__version__ = "1"
__all__ = ["open", "Handle", "AlayaError", "ApiError", "__version__"]


class AlayaError(Exception):
    """公共 API 的统一异常：带 code / message / retryable。"""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def as_dict(self) -> dict:
        return {
            "ok": False,
            "data": None,
            "error": {"code": self.code, "message": self.message, "retryable": self.retryable},
        }


ApiError = AlayaError

_CONFIG_FIELDS = {f.name for f in dataclasses.fields(Config)}
_HANDLE_OPTIONS = (
    "llm_callable",
    "llm",
    "system_prompt",
    "auto_tick",
    "memory_budget",
    "max_tool_rounds",
    "observe_window",
)


class Handle:
    """一个打开的 Alaya 记忆空间句柄。用 `alaya.open()` 创建。"""

    def __init__(
        self,
        cfg: Config,
        *,
        llm_callable: Callable | None = None,
        system_prompt: str | None = None,
        auto_tick: bool = False,
        memory_budget: int | None = None,
        max_tool_rounds: int | None = None,
        observe_window: float = 300.0,
    ):
        self._cfg = cfg
        self._closed = False
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._settle_by_sid: dict[str, dict] = {}
        self._observed: dict[tuple, float] = {}
        self._woken: set[str] = set()
        self.observe_window = float(observe_window)
        # auto_tick 默认关：服务/宿主形态由外部调度器管节拍，避免每轮后台抢线程。
        self._harness = Harness(
            cfg,
            llm_callable,
            system_prompt=system_prompt,
            auto_tick=auto_tick,
            memory_budget=memory_budget,
            max_tool_rounds=max_tool_rounds,
        )

    # ── 只读属性（给高级调用方/测试用，不参与契约）──────────
    @property
    def cfg(self) -> Config:
        return self._cfg

    @property
    def mem(self):
        return self._harness.mem

    @property
    def harness(self) -> Harness:
        return self._harness

    @property
    def closed(self) -> bool:
        return self._closed

    # ── 生命周期 ──────────────────────────────────────────
    def wake(self, space: str, conversation: str) -> dict:
        """会话开始：播种人格、载入常驻、报告状态。幂等。"""
        self._ensure_open()
        sid = _session(space, conversation)
        w = self._harness.wake(sid)
        with self._lock:
            self._woken.add(sid)
        return {
            "ok": True,
            "space": space,
            "conversation": conversation,
            "session_id": sid,
            **w,
        }

    def context(self, space: str, conversation: str, query: str, budget: int | None = None) -> dict:
        """取一份可注入的上下文。**纯函数**：不写库，且 memory_tokens 不超过 budget。"""
        self._ensure_open()
        sid = _session(space, conversation)
        if budget is not None:
            try:
                budget = int(budget)
            except (TypeError, ValueError):
                raise AlayaError("invalid_argument", "budget 必须是整数") from None
            if budget <= 0:
                raise AlayaError("invalid_argument", "budget 必须大于 0")

        ctx = harness_build_context(self._harness.mem, query or "", sid, budget, pure=True)
        return {
            "ok": True,
            "space": space,
            "conversation": conversation,
            "session_id": sid,
            "query": query or "",
            "system_prefix": ctx["system_prefix"],
            "memory_text": ctx["memory_text"],
            "tokens": ctx["memory_tokens"],
            "memory_tokens": ctx["memory_tokens"],
            "persona_tokens": ctx["persona_tokens"],
            "abstained": ctx["abstained"],
            "budget": budget if budget is not None else ctx.get("budget"),
            "parts": ctx["parts"],
            "latency_ms": ctx["latency_ms"],
            "as_of": ctx["as_of"],
            "pure": True,
        }

    def turn(
        self,
        space: str,
        conversation: str,
        text: str,
        *,
        history: list | None = None,
        extra_system: str | None = None,
    ) -> dict:
        """一轮完整对话（记录 → 组上下文 → 调模型 → 工具循环 → 回写）。"""
        self._ensure_open()
        sid = _session(space, conversation)
        if not (text or "").strip():
            raise AlayaError("invalid_argument", "text 不能为空")
        self._ensure_woken(sid)
        try:
            r = self._harness.turn(text, sid, history=history, extra_system=extra_system)
        except ValueError as e:
            raise AlayaError("invalid_argument", str(e)) from None
        except RuntimeError as e:
            raise AlayaError("llm_unavailable", f"模型没给出回复：{e}", retryable=True) from None
        return {"ok": True, "space": space, "conversation": conversation, "session_id": sid, **r}

    def observe(self, space: str, conversation: str, user_text: str, assistant_text: str) -> dict:
        """post_turn 钩子：只记录一来一回，不生成回答。幂等（窗口内同文本去重）。"""
        self._ensure_open()
        sid = _session(space, conversation)
        u = (user_text or "").strip()
        a = (assistant_text or "").strip()
        if not u and not a:
            raise AlayaError("invalid_argument", "user_text 与 assistant_text 至少给一个")

        key = (sid, u, a)
        now = time.time()
        with self._lock:
            prev = self._observed.get(key)
            if prev is not None and now - prev <= self.observe_window:
                return {
                    "ok": True,
                    "space": space,
                    "conversation": conversation,
                    "session_id": sid,
                    "queued": True,
                    "idempotent": True,
                    "written": [],
                }
            self._observed[key] = now

        out: dict[str, Any] = {
            "ok": True,
            "space": space,
            "conversation": conversation,
            "session_id": sid,
            "queued": True,
            "idempotent": False,
            "written": [],
        }
        if u:
            out["user"] = self._harness.mem.ingest_async(u, session_id=sid, role="user")
            out["written"].append("user")
        if a:
            out["assistant"] = self._harness.mem.ingest_async(a, session_id=sid, role="assistant")
            out["written"].append("assistant")
        return out

    def settle(self, space: str, conversation: str) -> dict:
        """session_end 钩子：立即返回 job_id，结算在后台跑（不阻塞退出流程）。"""
        self._ensure_open()
        sid = _session(space, conversation)
        now = time.time()
        with self._lock:
            job = self._settle_by_sid.get(sid)
            if job is not None and job.get("state") == "running":
                return {
                    "ok": True,
                    "space": space,
                    "conversation": conversation,
                    "session_id": sid,
                    "job_id": job["job_id"],
                    "queued": True,
                    "deduplicated": True,
                }
            job_id = "job_" + uuid.uuid4().hex[:16]
            job = {
                "job_id": job_id,
                "kind": "settle",
                "state": "running",
                "space": space,
                "conversation": conversation,
                "session_id": sid,
                "created_at": now,
                "finished_at": None,
                "result": None,
                "error": None,
            }
            self._jobs[job_id] = job
            self._settle_by_sid[sid] = job

        t = threading.Thread(target=self._settle_worker, args=(job,), daemon=True, name=f"alaya-settle-{job_id[:8]}")
        job["thread"] = t
        t.start()
        return {
            "ok": True,
            "space": space,
            "conversation": conversation,
            "session_id": sid,
            "job_id": job_id,
            "queued": True,
            "deduplicated": False,
        }

    def _settle_worker(self, job: dict) -> None:
        sid = job["session_id"]
        try:
            job["result"] = self._harness.sleep(sid)
            job["state"] = "done"
        except Exception as e:  # 后台任务失败不能影响调用方，但要留下可见原因
            job["state"] = "failed"
            job["error"] = f"{type(e).__name__}: {e}"
            log.warning("settle job %s failed: %s", job["job_id"], e)
        finally:
            job["finished_at"] = time.time()
            with self._lock:
                if self._settle_by_sid.get(sid) is job:
                    self._settle_by_sid.pop(sid, None)

    # ── 任务观察（settle 的可选同步查看法）────────────────
    def job(self, job_id: str) -> dict:
        """查后台任务状态。返回 {job_id, state, result, error, ...}。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise AlayaError("not_found", f"任务不存在：{job_id}")
            return _public_job(job)

    def wait(self, job_id: str, timeout: float = 60.0) -> dict:
        """等一个后台任务结束（可选便利；settle 本身不阻塞）。超时抛 retryable 错误。"""
        deadline = time.time() + max(0.0, float(timeout))
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    raise AlayaError("not_found", f"任务不存在：{job_id}")
                if job["state"] != "running":
                    return _public_job(job)
            if time.time() >= deadline:
                raise AlayaError("timeout", f"任务 {job_id} 在 {timeout}s 内未结束", retryable=True)
            time.sleep(0.02)

    # ── 工具（高级入口：宿主自己管循环时直接用）──────────
    def tool(self, space: str, conversation: str, name: str, args: dict) -> dict:
        """执行一次工具调用，返回统一结构。"""
        self._ensure_open()
        sid = _session(space, conversation)
        return dispatch(self._harness.mem, name, args or {}, sid)

    def item_state(self, item_id: str) -> dict:
        """查任意 item 的当前状态；不存在抛 not_found。"""
        self._ensure_open()
        d = item_state(self._harness.mem, item_id)
        if d is None:
            raise AlayaError("not_found", f"条目不存在：{item_id}")
        return {"ok": True, **d}

    def transition(self, item_id: str, action: str, *, target: str = "", reason: str = "", actor: str = "system") -> dict:
        """显式状态迁移，返回 {ok,id,state,from,to,reason}。"""
        self._ensure_open()
        return transition_item(self._harness.mem, item_id, action, target=target, reason=reason, actor=actor)

    def settle_now(self, space: str, conversation: str, *, timeout: float = 60.0) -> dict:
        """同步做一次完整 settle（峰终结算 + 整理 + 做梦）。

        `settle()` 是“立即返回 job_id”的异步形态；这个方法是守护/测试用的
        同步变体：等它返回时结算已经完成。两步都保持“不阻塞接入方逻辑”的默认选择。
        """
        self._ensure_open()
        sid = _session(space, conversation)
        self._ensure_woken(sid)
        try:
            result = self._harness.sleep(sid)
        except Exception as e:
            raise AlayaError("settle_failed", f"{type(e).__name__}: {e}", retryable=True) from None
        return {"ok": True, "space": space, "conversation": conversation,
                "session_id": sid, "result": result}

    # ── 关闭 ──────────────────────────────────────────────
    def close(self) -> None:
        """关闭句柄：等后台任务收尾。可重复调用。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._observed.clear()
            running = [j for j in self._jobs.values() if j.get("state") == "running"]
        try:
            self._harness.join(timeout=5.0)
        except Exception:
            pass
        for j in running:
            try:
                self.wait(j["job_id"], timeout=30.0)
            except Exception:
                pass
        try:
            self._harness.mem.wait_enrich(timeout=10.0)
        except Exception:
            pass
        try:
            self._harness.mem.close()  # Memory 目前没有 close；有就调，没有就忽略
        except AttributeError:
            pass
        except Exception as e:  # pragma: no cover
            log.debug("mem.close failed: %s", e)
        try:
            from .core import db as DB

            DB.close_thread_conns()
        except Exception:
            pass

    # ── 上下文管理 ────────────────────────────────────────
    def __enter__(self) -> "Handle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── 内部 ──────────────────────────────────────────────
    def _ensure_open(self) -> None:
        if self._closed:
            raise AlayaError("closed", "handle 已关闭", retryable=False)

    def _ensure_woken(self, sid: str) -> None:
        if sid in self._woken:
            return
        self._harness.wake(sid)
        with self._lock:
            self._woken.add(sid)


def _session(space: str, conversation: str) -> str:
    """space → 底层 session_id。v1 用 space 作为记忆空间（会话间共享）。"""
    s = (space or "").strip()
    if not s:
        raise AlayaError("invalid_argument", "space 不能为空")
    if s == "*":
        raise AlayaError("invalid_argument", "space 不能是 *（内部保留）")
    return s


def _public_job(job: dict) -> dict:
    return {
        "ok": job.get("state") != "failed",
        "job_id": job["job_id"],
        "kind": job.get("kind", "settle"),
        "state": job["state"],
        "space": job.get("space"),
        "conversation": job.get("conversation"),
        "session_id": job.get("session_id"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


def open(db_path: str, config: dict | None = None) -> Handle:
    """打开一个记忆空间，返回 Handle。

    `config` 可覆盖 `Config` 的任意字段（如 embed_provider / llm_hot / budget_tokens），
    另接受 Handle 级选项：

        llm_callable / llm   宿主模型回调
        system_prompt        本体 system 提示
        auto_tick            是否每轮后台节拍（默认 False，交给调度器）
        memory_budget        每轮检索 token 预算
        max_tool_rounds      工具循环上限（默认 3）
        observe_window       observe 幂等窗口秒数（默认 300）

    例：`alaya.open("alaya.db", {"embed_provider": "hash", "llm_hot": "rule", "llm_callable": fake})`
    """
    if isinstance(db_path, Config):
        cfg = db_path
        db_path = cfg.db_path
        opts: dict = {}
    else:
        if not isinstance(db_path, (str, bytes)) or not str(db_path).strip():
            raise AlayaError("invalid_argument", "db_path 不能为空")
        opts = dict(config or {}) if isinstance(config, dict) else {}
        if config is not None and not isinstance(config, dict):
            raise AlayaError("invalid_argument", "config 必须是 dict 或 None")
        cfg_kwargs = {k: v for k, v in opts.items() if k in _CONFIG_FIELDS and v is not None}
        cfg_kwargs["db_path"] = str(db_path)
        try:
            cfg = Config(**cfg_kwargs)
        except Exception as e:
            raise AlayaError("invalid_argument", f"配置不合法：{type(e).__name__}: {e}") from None

    handle_opts: dict = {k: opts.get(k) for k in _HANDLE_OPTIONS if k in opts}
    llm = handle_opts.pop("llm_callable", None) or handle_opts.pop("llm", None)
    try:
        return Handle(
            cfg,
            llm_callable=llm,
            system_prompt=handle_opts.get("system_prompt"),
            auto_tick=bool(handle_opts.get("auto_tick", False)),
            memory_budget=handle_opts.get("memory_budget"),
            max_tool_rounds=handle_opts.get("max_tool_rounds"),
            observe_window=handle_opts.get("observe_window", 300.0),
        )
    except AlayaError:
        raise
    except Exception as e:
        raise AlayaError("open_failed", f"打开记忆空间失败：{type(e).__name__}: {e}", retryable=True) from None
