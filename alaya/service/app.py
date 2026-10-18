"""FastAPI 服务层。薄适配器：所有逻辑都在 core 里。

启动：python run.py serve
文档：http://127.0.0.1:8770/docs
"""

from __future__ import annotations

import logging
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


def get_harness():
    """Harness 单例。llm_callable 从 cfg 的模型配置构造。"""
    global _harness
    if _harness is None:
        from ..harness import Harness

        mem = get_mem()

        def llm_call(messages, tools=None):
            """把 messages 列表压成 (system, 对话) 交给底层 LLM。

            本地小模型对 function calling 支持参差，所以这里不用原生 tools，
            模型只负责说话，记忆写入靠 harness 的自动管线。
            """
            system_parts = []
            convo = []
            for m in messages or []:
                role = m.get("role", "user")
                content = m.get("content") or ""
                if role == "system":
                    system_parts.append(content)
                elif role == "tool":
                    convo.append(f"（工具结果）{content}")
                else:
                    convo.append(f"{'他' if role == 'user' else '你'}：{content}")
            out = mem.llm_hot.chat("\n".join(system_parts), "\n".join(convo), max_tokens=400)
            return {"content": _clean_reply(out or ""), "tool_calls": []}

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
    def get_state(scope: str, as_of: float | None = None):
        return ok(get_mem().state_snapshot(scope, as_of))

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
