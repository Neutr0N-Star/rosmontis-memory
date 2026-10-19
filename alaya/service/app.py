"""FastAPI 服务层。薄适配器：所有逻辑都在 core 里。

启动：python run.py serve
文档：http://127.0.0.1:8770/docs
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import Config
from ..core.memory import Memory
from ..tools import item_state as _item_state
from ..tools import transition_item

log = logging.getLogger("alaya.service")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

_mem: Memory | None = None
_harness = None


_META_PREFIX = __import__("re").compile(r"^[（(][^）)]{0,24}[）)]\s*")


def _clean_reply(text: str) -> str:
    """去掉模型开头的元叙述。

    「（保持自然回应）」这种是模型在自言自语，不是给用户看的。
    只清开头一层，不动正文里的括号。
    """
    t = (text or "").strip()
    for _ in range(2):
        t2 = _META_PREFIX.sub("", t).strip()
        if t2 == t:
            break
        t = t2
    return t


def _no_wildcard(session_id: str = "default") -> str:
    # `*` 是内部审计用的全库通配，不该由调用方传。
    # 之前它一直暴露在接口上：任何人传 session_id=* 就能读到所有会话的数据。
    if session_id == "*":
        raise HTTPException(status_code=400, detail="session_id 不能是 *（内部保留）")
    return session_id or "default"


_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _read_env_file() -> dict:
    d = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def _write_env_file(patch: dict) -> None:
    d = _read_env_file()
    for k, v in patch.items():
        if v is None or v == "":
            d.pop(k, None)
        else:
            d[k] = str(v)
    lines = ["# 藏识 Alaya 本地配置（自动生成，密钥不要提交）"]
    for k in sorted(d):
        lines.append(f"{k}={d[k]}")
    _ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        _ENV_FILE.chmod(0o600)
    except Exception:
        pass


def _reload_runtime():
    # 改完配置要重建 Memory 与 Harness，否则还在用旧模型
    global _mem, _harness
    _mem = None
    _harness = None
    for k in (
        "ALAYA_LLM_HOT", "ALAYA_LLM_DEEP", "ALAYA_LLM_HOT_MODEL", "ALAYA_LLM_DEEP_MODEL",
        "ALAYA_OPENAI_BASE", "ALAYA_OPENAI_KEY", "ALAYA_OPENAI_MODEL", "ALAYA_LLM_TEMP",
    ):
        os.environ.pop(k, None)
    for k, v in _read_env_file().items():
        os.environ.setdefault(k, v)
    try:
        get_mem()
        get_harness()
        return True
    except Exception as e:
        log.warning("reload after settings failed: %s", e)
        return False


class SettingsIn(BaseModel):
    llm_hot: str | None = None
    llm_deep: str | None = None
    llm_hot_model: str | None = None
    llm_deep_model: str | None = None
    openai_base: str | None = None
    openai_key: str | None = None
    openai_model: str | None = None
    temperature: float | None = None


class OnboardIn(BaseModel):
    session_id: str = "default"
    name: str = ""
    desc: str = ""
    seed: str = ""


def get_harness():
    """Harness 单例。llm_callable 从 cfg 的模型配置构造。

    旧实现把 messages 压成 system+对话交给 LLM.chat，模型根本没有工具可调，
    tool_calls 恒为 []——这是主动记忆在真实产品里从不发生的根因。
    现在统一用 `make_llm_callable`：原生工具协议优先（OpenAI 兼容 / Ollama，
    tool_calls 原样透传，含 id / arguments / usage / finish_reason），
    上游不支持时安静降级为纯文本通道，绝不抛。
    """
    global _harness
    if _harness is None:
        from ..harness import Harness, make_llm_callable

        mem = get_mem()
        llm_call = make_llm_callable(mem.llm_hot, clean=_clean_reply, max_tokens=400)

        _harness = Harness(mem.cfg, llm_call, auto_tick=False)
        _harness.mem = mem
    return _harness


# 模型必须定义在模块顶层：本文件启用了 `from __future__ import annotations`，
# 注解会被延迟成字符串，FastAPI 需要能在模块命名空间里解析到这些类，
# 否则会把 Pydantic 模型错误地当成 query 参数。
class EpisodeIn(BaseModel):
    text: str = Field(..., description="原文")
    session_id: str = "default"
    role: str = "user"
    t_event: float | None = None


class RecallIn(BaseModel):
    query: str
    session_id: str = "default"
    as_of: float | None = None
    budget: int | None = Field(None, description="token 预算；不传用配置默认值")
    exclude_ids: list[str] = Field(default_factory=list)


class PersonaIn(BaseModel):
    text: str
    session_id: str = "default"
    section: str = "free"
    author: str = "self"
    protected: bool = False
    priority: int = 100


class NoteIn(BaseModel):
    what: str
    session_id: str = "default"
    noticed: str = ""
    means: str = ""
    next_step: str = ""
    topic: str = ""


class PromoteIn(BaseModel):
    note_id: str
    target: str = "persona"
    section: str = "free"
    session_id: str = ""


class ReviseIn(BaseModel):
    note_id: str
    new_text: str
    reason: str = ""


class ContextIn(BaseModel):
    query: str
    session_id: str = "default"
    budget: int | None = None
    as_of: float | None = None


class TurnIn(BaseModel):
    text: str
    session_id: str = "default"


class TransitionIn(BaseModel):
    action: str = Field(..., description="promote / skip / mute / archive / restore / unmute")
    target: str = ""
    reason: str = ""
    actor: str = "system"


class CorrectIn(BaseModel):
    old_id: str
    new_text: str
    author: str = "self"


def get_mem() -> Memory:
    global _mem
    if _mem is None:
        _mem = Memory(Config())
    return _mem


def create_app() -> FastAPI:
    app = FastAPI(
        title="藏识 Alaya",
        version="1.0.0",
        description="长期陪伴 AI 的记忆器官：五层记忆 + 三轴 + 预算约束检索",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.exception_handler(RequestValidationError)
    async def _validation(request, exc):
        detail = exc.errors()[0] if exc.errors() else {}
        msg = f"{'.'.join(str(x) for x in detail.get('loc', []))}: {detail.get('msg', '参数不合法')}"
        return JSONResponse(
            {"ok": False, "data": None,
             "error": {"code": "invalid_argument", "retryable": False, "message": msg}},
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(request, exc: StarletteHTTPException):
        return JSONResponse(
            {"ok": False, "data": None,
             "error": {"code": f"http_{exc.status_code}", "retryable": exc.status_code >= 500,
                       "message": str(exc.detail)}},
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(
            {"ok": False, "data": None,
             "error": {"code": "internal", "retryable": True, "message": f"{type(exc).__name__}: {exc}"[:300]}},
            status_code=500,
        )

    # ── 工具 ──────────────────────────────────────────────
    def ok(data: Any) -> JSONResponse:
        """统一的成功包装。

        如果业务层自己返回了 {"ok": false, "reason": ...}，这里要把它翻成 4xx，
        不能包一层 ok:true 送出去——前端判 `if (!j.ok)` 会以为成功了。
        """
        if isinstance(data, dict) and data.get("ok") is False:
            return JSONResponse(
                {
                    "ok": False,
                    "data": None,
                    "error": {"code": "rejected", "retryable": False,
                              "message": str(data.get("reason") or "操作被拒绝")},
                },
                status_code=409,
            )
        return JSONResponse({"ok": True, "data": data, "error": None})

    def fail(code: str, msg: str, status: int = 400) -> JSONResponse:
        return JSONResponse({"ok": False, "data": None, "error": {"code": code, "message": msg}}, status_code=status)

    # ── 写 ────────────────────────────────────────────────
    @app.post("/v1/episodes", tags=["写入"])
    def post_episode(body: EpisodeIn):
        if not body.text.strip():
            return fail("empty_text", "text 不能为空")
        try:
            return ok(get_mem().ingest_async(body.text, body.session_id, body.role, body.t_event))
        except Exception as e:  # pragma: no cover
            log.exception("ingest failed")
            return fail("ingest_failed", str(e), 500)

    @app.post("/v1/episodes/batch", tags=["写入"])
    def post_batch(items: list[EpisodeIn]):
        try:
            return ok(get_mem().ingest_batch([i.model_dump() for i in items]))
        except Exception as e:  # pragma: no cover
            return fail("batch_failed", str(e), 500)

    # ── 读 ────────────────────────────────────────────────
    @app.post("/v1/recall", tags=["检索"])
    def post_recall(body: RecallIn):
        try:
            return ok(
                get_mem().recall(
                    body.query,
                    session_id=body.session_id,
                    as_of=body.as_of,
                    budget=body.budget,
                    exclude_ids=body.exclude_ids,
                )
            )
        except Exception as e:  # pragma: no cover
            log.exception("recall failed")
            return fail("recall_failed", str(e), 500)

    @app.get("/v1/episodes", tags=["读"])
    def get_episodes(
        session_id: str = Depends(_no_wildcard),
        limit: int = Query(50, le=500),
        offset: int = Query(0, ge=0),
        include_archived: bool = False,
    ):
        return ok(get_mem().episodes(session_id, limit, offset, include_archived))

    @app.get("/v1/entities", tags=["读"])
    def get_entities(
        session_id: str = Depends(_no_wildcard),
        type: str | None = None,
        limit: int = Query(100, le=500),
    ):
        return ok(get_mem().entities(session_id, type, limit))

    @app.get("/v1/entities/{entity_id}", tags=["读"])
    def get_entity(entity_id: str, as_of: float | None = None):
        d = get_mem().entity_detail(entity_id, as_of)
        if d is None:
            return fail("not_found", "实体不存在", 404)
        return ok(d)

    @app.get("/v1/facts", tags=["读"])
    def get_facts(
        session_id: str = Depends(_no_wildcard),
        status: str = "active",
        limit: int = Query(200, le=1000),
    ):
        return ok(get_mem().facts(session_id, status, limit))

    @app.get("/v1/state/{scope}", tags=["读"])
    def get_state(scope: str, session_id: str = "default", as_of: float | None = None):
        return ok(get_mem().state_snapshot(scope, as_of, _no_wildcard(session_id)))

    @app.get("/v1/timeline/{entity_id}", tags=["读"])
    def get_timeline(entity_id: str):
        return ok(get_mem().timeline(entity_id))

    @app.get("/v1/themes", tags=["读"])
    def get_themes(session_id: str | None = None, limit: int = 50):
        return ok(get_mem().themes(session_id, limit))

    @app.post("/v1/sessions/{session_id}/close", tags=["沉淀"])
    def post_close_session(session_id: str):
        """结束一次会话：峰终结算 + 沉淀 + 遗忘。"""
        return ok(get_mem().close_session(session_id))

    @app.get("/v1/sessions", tags=["读"])
    def get_sessions():
        return ok(get_mem().sessions())

    # ── item 状态机（运维接口；公共 API 与工具共用同一套 state）──
    @app.get("/v1/items/{item_id}/state", tags=["运维"])
    def get_item_state(item_id: str):
        """查任意 item 的当前状态（inbox/note/resident/important/archived/muted/dream）。"""
        d = _item_state(get_mem(), item_id)
        if d is None:
            return fail("not_found", "条目不存在", 404)
        return ok(d)

    @app.post("/v1/items/{item_id}/transition", tags=["运维"])
    def post_item_transition(item_id: str, body: TransitionIn):
        """显式状态迁移。gate 决定能不能迁；拒绝时返回原因而不是静默成功。"""
        try:
            r = transition_item(
                get_mem(), item_id, body.action,
                target=body.target, reason=body.reason, actor=body.actor,
            )
        except Exception as e:  # pragma: no cover
            log.exception("transition failed")
            return fail("transition_failed", f"{type(e).__name__}: {e}", 500)
        if not r.get("ok") and str(r.get("reason") or "").startswith("条目不存在"):
            return fail("not_found", str(r["reason"]), 404)
        return ok(r)

    # ── 主动记忆 ──────────────────────────────────────────
    @app.get("/v1/notebook", tags=["主动记忆"])
    def get_notebook(session_id: str = Depends(_no_wildcard), tier: str | None = None, limit: int = Query(50, le=200)):
        return ok(get_mem().notebook_list(session_id, tier, limit))

    @app.post("/v1/notebook", tags=["主动记忆"])
    def post_notebook(body: NoteIn):
        return ok(
            get_mem().notebook_write(
                body.what, body.session_id, body.noticed, body.means, body.next_step, body.topic
            )
        )

    @app.post("/v1/notebook/promote", tags=["主动记忆"])
    def post_notebook_promote(body: PromoteIn):
        return ok(get_mem().notebook_promote(body.note_id, body.target, body.section, body.session_id))

    @app.post("/v1/notebook/revise", tags=["主动记忆"])
    def post_notebook_revise(body: ReviseIn):
        return ok(get_mem().notebook_revise(body.note_id, body.new_text, body.reason))

    @app.post("/v1/notebook/revert", tags=["主动记忆"])
    def post_notebook_revert(promotion_id: str):
        return ok(get_mem().notebook_revert(promotion_id))

    @app.post("/v1/notebook/review", tags=["主动记忆"])
    def post_notebook_review(session_id: str = "default"):
        return ok(get_mem().notebook_review(session_id))

    # ── Harness：一轮对话 ─────────────────────────────────
    @app.post("/v1/turn", tags=["Harness"])
    def post_turn(body: TurnIn):
        """一轮对话：记录 → 组上下文 → 调模型 → 写回。"""
        if not (body.text or "").strip():
            return fail("empty_text", "说的是空的，她没法接话", 422)
        if not body.session_id:
            return fail("missing_session", "session_id 不能为空", 422)
        try:
            return ok(get_harness().turn(body.text, body.session_id))
        except ValueError as e:
            return fail("invalid_argument", str(e), 422)
        except RuntimeError as e:
            # 模型侧问题：明确告诉前端可以重试，不要伪装成成功
            return fail("llm_unavailable", f"模型没给出回复：{e}", 503)
        except Exception as e:
            log.exception("turn failed")
            return fail("turn_failed", f"{type(e).__name__}: {e}", 500)

    @app.post("/v1/session/wake", tags=["Harness"])
    def post_wake(session_id: str = "default"):
        return ok(get_harness().wake(session_id))

    @app.post("/v1/session/sleep", tags=["Harness"])
    def post_sleep(session_id: str = "default"):
        return ok(get_harness().sleep(session_id))

    @app.get("/v1/settings", tags=["设置"])
    def get_settings():
        # 密钥不回显，只说有没有
        cfg = get_mem().cfg
        return ok(
            {
                "llm_hot": cfg.llm_hot,
                "llm_deep": cfg.llm_deep,
                "llm_hot_model": cfg.llm_hot_model,
                "llm_deep_model": cfg.llm_deep_model,
                "openai_base": cfg.openai_base,
                "openai_model": cfg.openai_model,
                "openai_key_set": bool(cfg.openai_key),
                "temperature": cfg.llm_temperature,
                "embed_provider": cfg.embed_provider,
                "embed_model": cfg.embed_model,
            }
        )

    @app.post("/v1/settings", tags=["设置"])
    def post_settings(body: SettingsIn):
        patch = {}
        for env_key, val in (
            ("ALAYA_LLM_HOT", body.llm_hot),
            ("ALAYA_LLM_DEEP", body.llm_deep),
            ("ALAYA_LLM_HOT_MODEL", body.llm_hot_model),
            ("ALAYA_LLM_DEEP_MODEL", body.llm_deep_model),
            ("ALAYA_OPENAI_BASE", body.openai_base),
            ("ALAYA_OPENAI_KEY", body.openai_key),
            ("ALAYA_OPENAI_MODEL", body.openai_model),
            ("ALAYA_LLM_TEMP", body.temperature),
        ):
            if val is not None:
                patch[env_key] = val
        if not patch:
            return fail("empty", "没有要改的", 422)
        _write_env_file(patch)
        reloaded = _reload_runtime()
        return ok({"written": sorted(patch), "reloaded": reloaded})

    @app.post("/v1/onboard", tags=["Harness"])
    def post_onboard(body: OnboardIn):
        # 引导：新角色第一次见面由她先开口。
        # 玩家面对空屏不知道该说什么，这是最好用的一处改善——
        # 而且她开口的同时就把"自己是谁"写进了记忆，后面才有得记。
        h = get_harness()
        mem = h.mem
        sid = body.session_id
        try:
            mem.seed(sid)
            if body.seed:
                mem.persona_add(
                    f"你是{body.name or '她'}。{body.seed}", sid, "identity",
                    author="self", protected=True, priority=100,
                )
            prompt = (
                f"你要第一次见一个陌生人。\n\n"
                f"你的身份：{body.seed or ('你是' + (body.name or '她') + '。你记得来过的人，但不多问。')}\n"
                f"{'关于你：' + body.desc if body.desc else ''}\n\n"
                "说两到三句话：先自然地打个招呼，然后问对方一个具体问题"
                "（比如怎么称呼、今天过得怎样、来这儿做什么）。\n"
                "要求：口语，不超过 60 字，不要自我介绍式的长篇，不要说自己是 AI。"
            )
            reply = mem.llm_hot.chat("", prompt, max_tokens=180)
            reply = _clean_reply(reply or "").strip()
            if not reply:
                return fail("llm_unavailable", "她没开口（模型没响应）", 503)

            # 她说的第一句也要进记忆，否则下次她不知道自己说过什么
            mem.ingest_async(reply, sid, role="assistant")
            return ok({"reply": reply})
        except Exception as e:
            log.exception("onboard failed")
            return fail("onboard_failed", str(e), 500)

    @app.post("/v1/wipe", tags=["维护"])
    def post_wipe(session_id: str):
        """删掉一个角色的全部数据。不可逆。"""
        sid = _no_wildcard(session_id)
        return ok(get_mem().wipe(sid))

    # ── 做梦与节拍 ────────────────────────────────────────
    @app.post("/v1/dream", tags=["做梦"])
    def post_dream(session_id: str = "default", force: bool = False):
        """做一次梦：重放 → 关联 → 抽象 → 脱敏 → 成梦。"""
        return ok(get_mem().dream(session_id, force))

    @app.get("/v1/dreams", tags=["做梦"])
    def get_dreams(session_id: str = "default", limit: int = Query(20, le=100)):
        return ok(get_mem().dreams(session_id, limit))

    @app.get("/v1/dreams/candidates", tags=["做梦"])
    def get_dream_candidates(session_id: str = "default"):
        return ok(get_mem().dream_candidates(session_id))

    @app.get("/v1/inbox", tags=["做梦"])
    def get_inbox(session_id: str = "default"):
        """锦囊状态：积压多少、该不该整理。"""
        return ok(get_mem().inbox(session_id))

    @app.post("/v1/tick", tags=["做梦"])
    def post_tick(session_id: str = "default"):
        """维护节拍：该整理就整理，该做梦就做梦。接入方每轮调一次即可。"""
        return ok(get_mem().tick(session_id))

    # ── 上下文编译 ────────────────────────────────────────
    @app.post("/v1/context", tags=["上下文"])
    def post_context(body: ContextIn):
        """一次调用拿全：人格 + 常驻 + 检索。每轮对话前调一次即完成热重载。"""
        return ok(get_mem().build_context(body.query, body.session_id, body.budget, body.as_of))

    # ── 维护 ──────────────────────────────────────────────
    @app.post("/v1/forget/run", tags=["维护"])
    def post_forget(session_id: str | None = None):
        return ok(get_mem().forget(session_id))

    # ── 沉淀层 ────────────────────────────────────────────
    @app.get("/v1/pages", tags=["沉淀"])
    def get_pages(session_id: str = "default"):
        return ok(get_mem().pages(session_id))

    @app.post("/v1/pages/rebuild", tags=["沉淀"])
    def post_pages_rebuild(session_id: str = "default"):
        return ok(get_mem().rebuild_pages(session_id))

    @app.get("/v1/resident", tags=["沉淀"])
    def get_resident(session_id: str = "default"):
        return ok(get_mem().resident_block(session_id))

    @app.get("/v1/observations", tags=["沉淀"])
    def get_obs(session_id: str = "default"):
        return ok(get_mem().observations(session_id))

    @app.post("/v1/observations/derive", tags=["沉淀"])
    def post_obs(session_id: str = "default"):
        return ok(get_mem().derive_observations(session_id))

    # ── 人格与治理 ────────────────────────────────────────


    @app.get("/v1/persona", tags=["人格"])
    def get_persona(session_id: str = "default"):
        return ok(get_mem().persona_list(session_id))

    @app.post("/v1/persona", tags=["人格"])
    def post_persona(body: PersonaIn):
        return ok(get_mem().persona_add(body.text, body.session_id, body.section, body.author, body.protected, body.priority))

    @app.get("/v1/persona/compile", tags=["人格"])
    def get_persona_compile(session_id: str = "default"):
        return ok(get_mem().persona_compile(session_id))

    @app.post("/v1/promote", tags=["人格"])
    def post_promote(item_id: str, target_tier: str = "resident"):
        return ok(get_mem().promote(item_id, target_tier))

    @app.post("/v1/correct", tags=["人格"])
    def post_correct(body: CorrectIn):
        return ok(get_mem().correct(body.old_id, body.new_text, body.author))

    @app.post("/v1/remove", tags=["人格"])
    def post_remove(item_id: str, author: str = "teacher", reason: str = "manual"):
        return ok(get_mem().remove(item_id, author, reason))

    @app.get("/v1/audit", tags=["维护"])
    def get_audit(session_id: str | None = None):
        return ok(get_mem().audit(session_id))

    @app.get("/v1/stats", tags=["维护"])
    def get_stats():
        d = get_mem().stats()
        d["config"] = get_mem().cfg.as_dict()
        return ok(d)

    @app.get("/v1/config", tags=["维护"])
    def get_config():
        return ok(get_mem().cfg.as_dict())

    # ── 前端 ──────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    def index():
        for name in ("app.html", "index.html"):
            f = WEB_DIR / name
            if f.exists():
                return FileResponse(str(f))
        return JSONResponse({"ok": True, "data": "藏识 Alaya 服务运行中，web/ 下没有找到页面"})

    @app.get("/debug", include_in_schema=False)
    def debug_panel():
        f = WEB_DIR / "index.html"
        return FileResponse(str(f)) if f.exists() else JSONResponse({"ok": False}, status_code=404)

    @app.get("/health", include_in_schema=False)
    def health():
        return {"ok": True, "ts": time.time()}

    return app


app = create_app()
