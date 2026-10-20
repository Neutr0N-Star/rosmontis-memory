"""FastAPI 服务层。薄适配器：所有逻辑都在 core 里。

启动：python run.py serve
文档：http://127.0.0.1:8770/docs
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import __version__ as _ALAYA_VERSION
from ..config import Config
from ..core import jobs
from ..core.memory import Memory
from ..tools import item_state as _item_state
from ..tools import transition_item

log = logging.getLogger("alaya.service")

_mem: Memory | None = None
_harness = None

# ── 本地 token 鉴权（P0-4）────────────────────────────────────
# 路径先按 ALAYA_HOME / ~/.alaya 计算；等 D1 的 Config.home_dir() 落地后替换这里。
TOKEN_FILENAME = "token"
SECURITY_EXEMPT_PATHS = frozenset({"/healthz", "/readyz", "/openapi.json", "/docs", "/redoc"})
_CONFIRM_VALUES = frozenset({"yes", "1", "true", "y"})


def alaya_home_dir() -> Path:
    """本地数据目录：显式 ALAYA_HOME 优先，否则 ~/.alaya。"""
    raw = (os.environ.get("ALAYA_HOME") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".alaya"


def token_path() -> Path:
    return alaya_home_dir() / TOKEN_FILENAME


def _chmod_private(path: Path) -> None:
    try:
        os.chmod(path, 0o600)  # Windows 上可能不生效；本地默认权限可接受
    except OSError:
        pass


def ensure_local_token(path: Path | None = None) -> str:
    """读取本地 token；不存在则生成 secrets.token_urlsafe(32) 并写入。"""
    p = Path(path) if path is not None else token_path()
    try:
        if p.exists():
            token = p.read_text(encoding="utf-8").strip()
            if token:
                _chmod_private(p)
                return token
    except OSError:
        pass
    p.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    try:
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = p.read_text(encoding="utf-8").strip()
        except OSError:
            existing = ""
        if existing:
            _chmod_private(p)
            return existing
        p.write_text(token + "\n", encoding="utf-8")
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token + "\n")
    _chmod_private(p)
    return token


def configured_token() -> str | None:
    """显式 ALAYA_TOKEN 优先；否则读本地 token 文件（不存在返回 None）。"""
    raw = (os.environ.get("ALAYA_TOKEN") or "").strip()
    if raw:
        return raw
    try:
        token = token_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def _valid_bearer(header: str | None, token: str) -> bool:
    if not header or not token:
        return False
    scheme, _sep, value = header.partition(" ")
    return scheme.strip().lower() == "bearer" and secrets.compare_digest(value.strip(), token)


def unauthorized_response(message: str = "缺少或无效的 Authorization: Bearer token") -> JSONResponse:
    return JSONResponse(
        {
            "ok": False,
            "data": None,
            "error": {"code": "unauthorized", "retryable": False, "message": message},
        },
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


def is_confirmed(x_confirm: str | None, confirm: str | None) -> bool:
    """危险操作确认：X-Confirm: yes 头，或 ?confirm=yes。"""
    for value in (x_confirm, confirm):
        if isinstance(value, str) and value.strip().lower() in _CONFIRM_VALUES:
            return True
    return False


# HTTP /v1/settle 的后台 job 线程：lifespan 关停时要 join，不能硬杀。
_job_threads: set[threading.Thread] = set()
_job_threads_lock = threading.Lock()


def _write_shutdown_marker(payload: dict) -> None:
    """测试/运维用：设置 ALAYA_SHUTDOWN_MARKER 后，优雅关停时落标记文件。"""
    path = (os.environ.get("ALAYA_SHUTDOWN_MARKER") or "").strip()
    if not path:
        return
    try:
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as e:  # pragma: no cover - 关停路径不抛
        log.warning("shutdown marker write failed: %s", e)


def _run_job_once(job_id: str, mem: Memory) -> None:
    """抢到并执行一个指定 job（HTTP /v1/settle 的立即线程用）。"""
    cfg = mem.cfg
    claimed = jobs.claim_job(cfg, job_id, owner=f"service:{os.getpid()}", now=time.time())
    if not claimed:
        return
    try:
        jobs.execute_job(mem, claimed)
    except Exception as e:
        jobs.fail(cfg, job_id, e)
        log.warning("job %s failed: %s: %s", job_id, type(e).__name__, e)
    else:
        jobs.finish(cfg, job_id)


def _start_job_thread(job_id: str, mem: Memory) -> threading.Thread:
    """后台跑一个 job；线程登记在 lifespan 的 join 表里。"""

    def _worker() -> None:
        try:
            _run_job_once(job_id, mem)
        finally:
            with _job_threads_lock:
                _job_threads.discard(threading.current_thread())

    t = threading.Thread(target=_worker, daemon=True, name=f"alaya-job-{job_id[:8]}")
    with _job_threads_lock:
        _job_threads.add(t)
    t.start()
    return t


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
    idempotency_key: str | None = Field(None, description="等价于 Idempotency-Key 头")


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
    idempotency_key: str | None = Field(None, description="等价于 Idempotency-Key 头")


class SettleIn(BaseModel):
    """POST /v1/settle：打烊（立即返回 job_id，后台执行）。"""

    session_id: str = "default"
    idempotency_key: str | None = Field(None, description="等价于 Idempotency-Key 头")


class ObserveIn(BaseModel):
    """POST /v1/observe：宿主自管循环时回写一来一回（异步落库，立即返回）。"""

    session_id: str
    user_text: str = ""
    assistant_text: str = ""
    idempotency_key: str | None = Field(None, description="等价于 Idempotency-Key 头")


class StreamIn(BaseModel):
    """流式对话入参（SSE）：前端最常用的一条。"""
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
    @asynccontextmanager
    async def _lifespan(_app):
        """生命周期：关停时把在途后台任务收尾，别让 enrich/settle/job 线程被硬杀。"""
        yield
        joined_jobs = 0
        try:
            h = _harness
            if h is not None and hasattr(h, "join"):
                h.join(timeout=5.0)
                log.info("shutdown: 后台任务已收尾")
        except Exception as e:  # 关停路径不抛错，只记日志
            log.warning("shutdown join failed: %s", e)
        try:
            with _job_threads_lock:
                threads = list(_job_threads)
            for t in threads:
                t.join(timeout=5.0)
                joined_jobs += 1
            if joined_jobs:
                log.info("shutdown: %d 个 job 线程已收尾", joined_jobs)
        except Exception as e:
            log.warning("shutdown job join failed: %s", e)
        _write_shutdown_marker(
            {
                "shutdown": True,
                "harness_joined": _harness is not None,
                "job_threads_joined": joined_jobs,
                "at": time.time(),
            }
        )

    app = FastAPI(
        title="藏识 Alaya",
        version=_ALAYA_VERSION,
        description="长期陪伴 AI 的记忆器官：五层记忆 + 三轴 + 预算约束检索",
        lifespan=_lifespan,
    )
    # CORS：默认只放行本机常见前端端口；file:// 的 `Origin: null` 不再默认放行，
    # 需要它的用户显式配置 ALAYA_CORS_ORIGINS=null（P0-4）。
    _cors_env = (os.environ.get("ALAYA_CORS_ORIGINS") or "").strip()
    if _cors_env:
        _cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
    else:
        _cors_origins = [
            "http://127.0.0.1:8770", "http://localhost:8770",
            "http://127.0.0.1:3000", "http://localhost:3000",
            "http://127.0.0.1:5173", "http://localhost:5173",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        """除 /healthz /readyz 外的全部 HTTP 面要求 Bearer token（P0-4）。

        OPTIONS 预检放行（预检不带 Authorization；真正的请求照样要 token）。
        """
        if request.method == "OPTIONS" or request.url.path in SECURITY_EXEMPT_PATHS:
            return await call_next(request)
        try:
            token = configured_token() or ensure_local_token()
        except Exception as e:  # token 文件不可写等：明确 401，不裸奔
            log.warning("token resolve failed: %s", e)
            token = None
        if not token or not _valid_bearer(request.headers.get("authorization"), token):
            return unauthorized_response()
        return await call_next(request)

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

    def fail(code: str, msg: str, status: int = 400, retryable: bool | None = None) -> JSONResponse:
        """统一失败信封。5xx 默认标 retryable，客户端可据它决定是否重试。"""
        if retryable is None:
            retryable = status >= 500
        return JSONResponse(
            {"ok": False, "data": None,
             "error": {"code": code, "retryable": retryable, "message": msg}},
            status_code=status,
        )

    # ── 幂等键（Idempotency-Key）：24h 内同键返回首次响应 ──────
    def _idempotent(scope: str, key: Any, handler):
        """包一层显式幂等。key 非 str（直调测试时的 Header 默认对象）视为未传。"""
        if not isinstance(key, str) or not key.strip():
            return handler()
        cfg = get_mem().cfg
        state, cached = jobs.idem_begin(cfg, scope, key)
        if state == "invalid":
            return fail("invalid_idempotency_key", "Idempotency-Key 需为 1-200 字符", 400)
        if state == "hit" and cached is not None:
            return JSONResponse(
                cached["body"], status_code=int(cached.get("status_code") or 200),
                headers={"X-Idempotent-Replay": "1"},
            )
        if state == "busy":
            return fail(
                "idempotency_in_progress",
                "同一 Idempotency-Key 正在处理，请稍后重试", 409, retryable=True,
            )
        try:
            resp = handler()
        except Exception:
            jobs.idem_release(cfg, scope, key)
            raise
        try:
            body = json.loads(resp.body.decode("utf-8"))
            jobs.idem_store(cfg, scope, key, resp.status_code, body)
        except Exception as e:  # 缓存失败不能影响首次响应
            log.warning("idempotency store failed: %s", e)
        return resp

    # ── 写 ────────────────────────────────────────────────
    @app.post("/v1/episodes", tags=["写入"])
    def post_episode(
        body: EpisodeIn,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        def _do():
            if not body.text.strip():
                return fail("empty_text", "text 不能为空")
            try:
                return ok(get_mem().ingest_async(body.text, body.session_id, body.role, body.t_event))
            except Exception as e:  # pragma: no cover
                log.exception("ingest failed")
                return fail("ingest_failed", str(e), 500)

        key = idempotency_key if isinstance(idempotency_key, str) else None
        return _idempotent("POST:/v1/episodes", key or body.idempotency_key, _do)

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
    def post_turn(
        body: TurnIn,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        """一轮对话：记录 → 组上下文 → 调模型 → 写回。支持 Idempotency-Key。"""

        def _do():
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

        return _idempotent(
            "POST:/v1/turn",
            idempotency_key if isinstance(idempotency_key, str) else body.idempotency_key,
            _do,
        )

    @app.post("/v1/observe", tags=["Harness"])
    def post_observe(
        body: ObserveIn,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        """回写一来一回（宿主自管循环的 post_turn 钩子）：异步落库、立即返回。

        两个字段至少给一个；空字符串忽略；都给则各写一条 user / assistant episode。
        返回 `{queued: true, episode_ids: [...]}`（按 user、assistant 顺序）。
        """

        def _do():
            sid = _no_wildcard((body.session_id or "").strip())
            if not (body.session_id or "").strip():
                return fail("missing_session", "session_id 必填", 422)
            user_text = (body.user_text or "").strip()
            assistant_text = (body.assistant_text or "").strip()
            if not user_text and not assistant_text:
                return fail("empty_text", "user_text 与 assistant_text 至少给一个", 422)
            mem = get_mem()
            episode_ids: list[str] = []
            if user_text:
                r = mem.ingest_async(user_text, session_id=sid, role="user")
                if r.get("episode_id"):
                    episode_ids.append(r["episode_id"])
            if assistant_text:
                r = mem.ingest_async(assistant_text, session_id=sid, role="assistant")
                if r.get("episode_id"):
                    episode_ids.append(r["episode_id"])
            return ok({"queued": True, "episode_ids": episode_ids})

        key = idempotency_key if isinstance(idempotency_key, str) else None
        return _idempotent("POST:/v1/observe", key or body.idempotency_key, _do)

    @app.post("/v1/turn/stream", tags=["Harness"])
    def post_turn_stream(body: StreamIn):
        """一轮对话的 SSE 流：`meta → (tool)* → delta* → done`。

        前端用 `fetch` 读流即可（不必用 EventSource，因为要 POST body）。
        上游模型不支持 token 流时，服务端按段切分，**接口形态不退化**。
        """
        from fastapi.responses import StreamingResponse

        if not (body.text or "").strip():
            return fail("empty_text", "说的是空的，她没法接话", 422)
        harness = get_harness()

        def _sse(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        def _gen():
            try:
                for ev in harness.turn_stream(body.text, body.session_id):
                    yield _sse(ev["event"], ev["data"])
            except Exception as e:  # 流已经开了，错误只能作为事件送出
                log.exception("turn_stream failed")
                yield _sse(
                    "error",
                    {"code": "turn_failed", "retryable": True,
                     "message": f"{type(e).__name__}: {e}"[:300]},
                )

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/session/wake", tags=["Harness"])
    def post_wake(session_id: str = "default"):
        return ok(get_harness().wake(session_id))

    @app.post("/v1/session/sleep", tags=["Harness"])
    def post_sleep(session_id: str = "default"):
        return ok(get_harness().sleep(session_id))

    # ── 节拍 / 任务运维（C2）──────────────────────────────
    @app.post("/v1/settle", tags=["Harness"])
    def post_settle(
        body: SettleIn | None = None,
        session_id: str = "default",
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        """打烊：立即返回 job_id，峰终结算 + 整理 + 做梦在后台跑（不阻塞调用方）。

        兼容三种传 session 的写法：body `{"session_id"}`、query `?session_id=`、缺省 default。
        """

        def _do():
            raw_sid = (
                body.session_id
                if body is not None and body.session_id
                else (session_id or "default")
            )
            sid = _no_wildcard(raw_sid or "default")
            mem = get_mem()
            cfg = mem.cfg
            now = time.time()
            job = jobs.enqueue(
                cfg, jobs.KIND_SETTLE, sid, next_run=now, bucket=int(now), now=now
            )
            queued = False
            if job.get("status") in (jobs.STATUS_PENDING, jobs.STATUS_FAILED):
                _start_job_thread(job["job_id"], mem)
                queued = True
            return ok(
                {
                    "job_id": job["job_id"],
                    "session_id": sid,
                    "status": job.get("status"),
                    "queued": queued,
                }
            )

        key = idempotency_key if isinstance(idempotency_key, str) else None
        return _idempotent(
            "POST:/v1/settle", key or (body.idempotency_key if body is not None else None), _do
        )

    @app.get("/v1/jobs", tags=["运维"])
    def get_jobs(status: str | None = None, limit: int = Query(50, le=500)):
        """任务积压：各状态计数 + 最近 job 列表（运维/守护进程观测用）。"""
        cfg = get_mem().cfg
        now = time.time()
        return ok(
            {
                "summary": jobs.summary(cfg, now=now),
                "jobs": jobs.list_jobs(cfg, status=status, limit=limit),
            }
        )

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
    def post_settings(
        body: SettingsIn,
        x_confirm: str | None = Header(default=None, alias="X-Confirm"),
        confirm: str | None = None,
    ):
        """改本地模型/API 配置（会写 .env）：危险操作，必须显式确认。"""
        if not is_confirmed(x_confirm, confirm):
            return fail(
                "confirmation_required",
                "危险操作需要 X-Confirm: yes 头或 ?confirm=yes",
                403,
            )
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
    def post_wipe(
        session_id: str,
        x_confirm: str | None = Header(default=None, alias="X-Confirm"),
        confirm: str | None = None,
    ):
        """删掉一个角色的全部数据（不可逆）：必须显式确认。"""
        if not is_confirmed(x_confirm, confirm):
            return fail(
                "confirmation_required",
                "危险操作需要 X-Confirm: yes 头或 ?confirm=yes",
                403,
            )
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
        """一次调用拿全：人格 + 常驻 + 检索。

        与库层 `Handle.context` 走**同一条纯函数路径**：不写库、绝不超预算。
        （此前直接调 `Memory.build_context`，实测会写 retrieval_log 且 72 > budget 60。）
        """
        from ..harness import build_context as _build_context_pure

        ctx = _build_context_pure(
            get_mem(), body.query, body.session_id, body.budget, body.as_of, pure=True
        )
        return ok(ctx)

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

    # ── 服务信息（前端不在产品内：web/ 挂载与 /debug 已移除）────
    @app.get("/", include_in_schema=False)
    def index():
        return JSONResponse({"ok": True, "data": {"service": "alaya", "docs": "/docs"}})

    @app.get("/health", include_in_schema=False)
    def health():
        """兼容旧探针：只表示进程活着。"""
        return {"ok": True, "ts": time.time()}

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        """存活探针（liveness）：进程在、事件循环能响应即算活。"""
        return {"ok": True, "ts": time.time()}

    @app.get("/readyz", include_in_schema=False)
    def readyz():
        """就绪探针（readiness）：库可读 + schema 版本已知 + 降级状态可见。

        503 表示不该把流量打进来（库打不开 / schema 未知）；降级（无模型、哈希向量）
        不影响就绪，但会出现在 degraded 里，方便运维一眼看到。
        """
        checks: dict[str, Any] = {}
        ok = True
        try:
            mem = get_mem()
            import sqlite3

            conn = sqlite3.connect(f"file:{mem.cfg.db_path}?mode=ro", uri=True)
            try:
                uv = conn.execute("PRAGMA user_version").fetchone()[0]
                ep = conn.execute("SELECT COUNT(*) FROM episode").fetchone()[0]
            finally:
                conn.close()
            checks["db"] = {"ok": True, "user_version": uv, "episode": ep}
        except Exception as e:
            ok = False
            checks["db"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            st = get_mem().stats()
            checks["providers"] = {
                "ok": True,
                "degraded": bool(st.get("degraded")),
                "stats": st,
            }
        except Exception as e:
            checks["providers"] = {"ok": False, "error": type(e).__name__}
        body = {"ok": ok, "data": {"version": _ALAYA_VERSION, "checks": checks}}
        return JSONResponse(body, status_code=200 if ok else 503)

    return app


app = create_app()
