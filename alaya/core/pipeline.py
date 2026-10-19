"""统一写入管线：全系统唯一的 `INSERT INTO episode` 写入点。

契约：`docs/15` §3.1
    submit(item, *, project=True) -> SubmitResult
    enrich(item_id) -> EnrichResult
    transition(item_id, action, ...) -> TransitionResult

纪律（docs/14 §1）：
1. 单一写入点：episode 的 INSERT 只在这里；状态与 `item_state_log` 同一事务落库。
2. 单一生命周期：状态由 `gate.decide()` 判、由 `transition()` 改；`tier/archived/muted`
   只是 `position()` 投影。
3. 主动条目也进异步 enrich：`submit(project=True)` 默认排后台抽取，结构投影与被动同构。

上下文：契约签名不带 cfg，所以用 `bind(cfg, embedder, llm)` 绑定当前库；显式传参优先。
`Memory.__init__` 会绑定，测试与脚本也可手动 `pipeline.bind(cfg)`。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import db as DB
from . import gate
from . import ids as IDS
from . import item as IT

log = logging.getLogger("alaya.pipeline")

# ── 进程内上下文（契约签名不带 cfg，靠 bind）──────────────────
_ctx_lock = threading.Lock()
_ctx: dict[str, Any] = {"cfg": None, "embedder": None, "llm": None}


def bind(cfg, embedder=None, llm=None) -> None:
    """绑定当前库的配置与依赖。Memory 初始化时调用；显式传参仍优先。"""
    with _ctx_lock:
        _ctx["cfg"] = cfg
        if embedder is not None:
            _ctx["embedder"] = embedder
        if llm is not None:
            _ctx["llm"] = llm


def active_config():
    return _ctx.get("cfg")


def _resolve(cfg=None, embedder=None, llm=None):
    with _ctx_lock:
        cfg = cfg or _ctx.get("cfg")
        embedder = embedder if embedder is not None else _ctx.get("embedder")
        llm = llm if llm is not None else _ctx.get("llm")
    if cfg is None:
        raise RuntimeError("pipeline 未绑定配置：先调用 pipeline.bind(cfg) 或显式传 cfg=")
    return cfg, embedder, llm


# ── 结果对象 ──────────────────────────────────────────────────
@dataclass
class SubmitResult:
    ok: bool
    item_id: str = ""
    state: str = ""
    tier: str = ""
    origin: str = ""
    duplicate: bool = False
    created: bool = False
    salience: float = 0.0
    emotion: float = 0.0
    enrich_queued: bool = False
    reason: str = ""
    log_id: int | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class EnrichResult:
    ok: bool
    item_id: str = ""
    skipped: bool = False
    reason: str = ""
    added: int = 0
    updated: int = 0
    invalidated: int = 0
    noop: int = 0
    states_changed: int = 0
    edges_touched: int = 0
    source: str = "rule"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class TransitionResult:
    ok: bool
    item_id: str = ""
    action: str = ""
    from_state: str = ""
    to_state: str = ""
    target: str = ""
    reason: str = ""
    gate_version: str = ""
    gates: dict = field(default_factory=dict)
    log_id: int | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["gate"] = d.pop("gates")
        return d


def _coerce_item(item: Any) -> IT.MemoryItem:
    if isinstance(item, IT.MemoryItem):
        return item
    if isinstance(item, dict):
        allowed = {
            "id", "session_id", "text", "origin", "actor", "memory_type", "state", "segments",
            "salience", "emotion", "t_event", "t_ingest", "role", "source", "author",
            "confirmed", "embedding", "meta",
        }
        return IT.MemoryItem(**{k: v for k, v in item.items() if k in allowed})
    return IT.MemoryItem.from_row(item)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _ensure_session(conn, session_id: str, now: float) -> None:
    conn.execute(
        """INSERT INTO session(id,agent_id,user_id,created_at,last_active) VALUES(?,'default','default',?,?)
           ON CONFLICT(id) DO UPDATE SET last_active=excluded.last_active""",
        (session_id, now, now),
    )


# ── 异步 enrich 池 ────────────────────────────────────────────
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()
_futures: set[Future] = set()


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alaya-enrich")
        return _pool


def enqueue_enrich(item_id: str, *, cfg=None, embedder=None, llm=None) -> bool:
    """把结构投影排进后台。返回是否真的排队。"""
    try:
        cfg, embedder, llm = _resolve(cfg, embedder, llm)
    except RuntimeError as e:
        log.debug("enqueue skipped: %s", e)
        return False

    def _job() -> None:
        try:
            enrich(item_id, cfg=cfg, embedder=embedder, llm=llm)
        except Exception as e:  # 后台失败不能影响写入
            log.debug("async enrich failed for %s: %s", item_id, e)

    fut = _get_pool().submit(_job)
    with _pool_lock:
        _futures.add(fut)
    fut.add_done_callback(_discard_future)
    return True


def _discard_future(fut: Future) -> None:
    with _pool_lock:
        _futures.discard(fut)


def pending_enrich() -> int:
    with _pool_lock:
        return len(_futures)


def wait_enrich(timeout: float | None = None) -> None:
    """测试/批处理用：等后台 enrich 全部跑完并关池（下次写入会重建）。"""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=False)
    with _pool_lock:
        _futures.clear()


# ── 唯一写入点 ────────────────────────────────────────────────
def submit(item: IT.MemoryItem | dict | Any, *, project: bool = True, cfg=None, embedder=None, llm=None) -> SubmitResult:
    """提交一个 item：落 episode + FTS + item_state_log，一步一事务。

    幂等：id 已存在时返回 duplicate=True，不再写任何行。
    `project=True`（契约默认）会把主动条目的结构投影排进后台 enrich。
    """
    cfg, embedder, llm = _resolve(cfg, embedder, llm)
    item = _coerce_item(item).normalized()
    text = (item.text or "").strip()
    if not text:
        return SubmitResult(ok=False, reason="text 不能为空")
    session_id = item.session_id or "default"

    from . import write as WR  # 延迟导入：write 依赖本模块的 submit

    now = time.time()
    t_event = float(item.t_event or now)
    t_ingest = float(item.t_ingest or now)
    decision = gate.decide(item, "create", cfg=cfg)
    state = decision["to"]
    tier, archived, muted = IT.projection(state)

    emotion = float(item.emotion or 0.0)
    if emotion <= 0.0:
        emotion = WR.emotion_of(text)
    salience = float(item.salience or 0.0)
    if salience <= 0.0:
        salience = WR.salience_of(
            text,
            emotion,
            actor=item.actor,
            memory_type=item.memory_type,
            segments=item.segments,
        )
    salience = _clamp(salience)
    emotion = _clamp(emotion)

    role = item.effective_role()
    item_id = item.id or IDS.episode_id(session_id, role, text, t_event)
    source = item.effective_source()
    author = item.effective_author()

    meta = dict(item.meta or {})
    if item.segments:
        meta.setdefault("segments", dict(item.segments))
    meta.setdefault("origin", item.origin)

    with DB.tx(cfg.db_path) as conn:
        row = conn.execute("SELECT id, state, tier, origin FROM episode WHERE id=?", (item_id,)).fetchone()
        if row is not None:
            return SubmitResult(
                ok=True,
                item_id=item_id,
                state=row["state"] or IT.compat_state(row["tier"]),
                tier=row["tier"],
                origin=row["origin"] or IT.ORIGIN_INGEST,
                duplicate=True,
                reason="已存在，跳过",
            )

        _ensure_session(conn, session_id, now)
        seq_row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS m FROM episode WHERE session_id=?", (session_id,)
        ).fetchone()
        seq = int(seq_row["m"] or 0) + 1

        conn.execute(
            """INSERT INTO episode
               (id,session_id,seq,role,text,t_event,t_ingest,source,tier,memory_type,author,
                confirmed,salience,emotion,embedding,meta_json,origin,state,state_at,gate_version,
                archived,muted)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item_id, session_id, seq, role, text, t_event, t_ingest, source, tier,
                item.memory_type, author, 1 if item.confirmed else 0, salience, emotion,
                item.embedding, json.dumps(meta, ensure_ascii=False), item.origin, state,
                now, gate.GATE_VERSION, archived, muted,
            ),
        )
        conn.execute(
            "INSERT INTO episode_fts(episode_id,tokens) VALUES(?,?)",
            (item_id, " ".join(_tokenize(text))),
        )
        cur = conn.execute(
            """INSERT INTO item_state_log
               (item_id,from_state,to_state,action,target,reason,gate_json,at,actor)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                item_id, None, state, "create", None, decision.get("reason", ""),
                gate.gate_json(decision), now, item.actor,
            ),
        )
        log_id = cur.lastrowid

    queued = False
    if project and item.origin in (IT.ORIGIN_INGEST, IT.ORIGIN_TOOL, IT.ORIGIN_SYSTEM):
        queued = enqueue_enrich(item_id, cfg=cfg, embedder=embedder, llm=llm)

    return SubmitResult(
        ok=True,
        item_id=item_id,
        state=state,
        tier=tier,
        origin=item.origin,
        created=True,
        salience=round(salience, 4),
        emotion=round(emotion, 4),
        enrich_queued=queued,
        reason=decision.get("reason", ""),
        log_id=log_id,
    )


def _tokenize(text: str) -> list[str]:
    from . import text as TX

    return TX.tokenize(text)


# ── 结构投影 ──────────────────────────────────────────────────
def enrich(item_id: str, *, cfg=None, embedder=None, llm=None) -> EnrichResult:
    """对已存在的 item 做结构投影（entity/fact/state_slot/hyperedge）。

    与被动写入共用 `write.enrich`，口径唯一；已投影过/已静音/已归档时跳过。
    """
    cfg, embedder, llm = _resolve(cfg, embedder, llm)
    from . import write as WR

    with DB.reader(cfg.db_path) as conn:
        row = conn.execute("SELECT state, tier, archived, muted FROM episode WHERE id=?", (item_id,)).fetchone()
    if row is None:
        return EnrichResult(ok=False, item_id=item_id, reason="条目不存在")
    state = row["state"] or IT.compat_state(row["tier"], row["archived"], row["muted"])
    if state in (IT.STATE_ARCHIVED, IT.STATE_MUTED):
        return EnrichResult(ok=True, item_id=item_id, skipped=True, reason=f"状态 {state} 不再投影")

    res = WR.enrich(cfg, llm, item_id, embedder=embedder)
    if not res.get("ok"):
        return EnrichResult(ok=False, item_id=item_id, reason=res.get("reason", "enrich 失败"), skipped=bool(res.get("skipped")))
    return EnrichResult(
        ok=True,
        item_id=item_id,
        skipped=bool(res.get("skipped")),
        reason=str(res.get("skipped") or ""),
        added=int(res.get("added", 0)),
        updated=int(res.get("updated", 0)),
        invalidated=int(res.get("invalidated", 0)),
        noop=int(res.get("noop", 0)),
        states_changed=int(res.get("states_changed", 0)),
        edges_touched=int(res.get("edges_touched", 0)),
        source=str(res.get("source", "rule")),
    )


# ── 状态转移 ──────────────────────────────────────────────────
def transition(
    item_id: str,
    action: str,
    *,
    target: str = "",
    reason: str = "",
    actor: str = "system",
    cfg=None,
    gate_decision: dict | None = None,
) -> TransitionResult:
    """唯一的状态转移入口：判定 → 更新 episode.state 与兼容投影 → 写 item_state_log。

    `promote_resident / demote / mark_important / skip / archive / review_* /
    promote_persona / promote_page / revise / revert_promotion` 全部收口到这里。
    人设/页面的实体写入由调用方（notebook）完成，本函数只负责状态与留痕。
    """
    cfg, _, _ = _resolve(cfg)
    with DB.reader(cfg.db_path) as conn:
        row = conn.execute("SELECT * FROM episode WHERE id=?", (item_id,)).fetchone()
        if row is None:
            fact = conn.execute("SELECT id FROM fact WHERE id=?", (item_id,)).fetchone()
            if fact is not None:
                return TransitionResult(
                    ok=False, item_id=item_id, action=action,
                    reason="事实条目不走人格升格，请用状态槽",
                )
            return TransitionResult(ok=False, item_id=item_id, action=action, reason="条目不存在")

    item = IT.MemoryItem.from_row(row)
    dec = gate_decision if gate_decision is not None else gate.decide(
        item, action, target=target, cfg=cfg, session_id=item.session_id
    )
    if not dec.get("ok"):
        return TransitionResult(
            ok=False,
            item_id=item_id,
            action=gate.normalize_action(action),
            from_state=dec.get("from", ""),
            to_state=dec.get("from", ""),
            target=target,
            reason=dec.get("reason", "闸门拒绝"),
            gate_version=dec.get("gate_version", gate.GATE_VERSION),
            gates=dec.get("gate", {}),
        )

    act = gate.normalize_action(action)
    from_state = dec.get("from", "") or item.state
    to_state = dec.get("to", "") or from_state
    target_value = target or dec.get("target", "") or ""
    now = time.time()
    gate_version = dec.get("gate_version", gate.GATE_VERSION)

    with DB.tx(cfg.db_path) as conn:
        if act in ("revise", "revert_promotion"):
            # 只留痕，不改状态（旧条留在原态 + superseded_by / 撤回指针）
            pass
        else:
            tier, archived, muted = IT.projection(to_state)
            conn.execute(
                """UPDATE episode
                   SET state=?, state_at=?, tier=?, archived=?, muted=?, gate_version=?
                   WHERE id=?""",
                (to_state, now, tier, archived, muted, gate_version, item_id),
            )
        payload_reason = reason or dec.get("reason", "")
        cur = conn.execute(
            """INSERT INTO item_state_log
               (item_id,from_state,to_state,action,target,reason,gate_json,at,actor)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                item_id, from_state or None, to_state, act, target_value or None,
                payload_reason, gate.gate_json(dec, reason=reason), now, actor,
            ),
        )
        log_id = cur.lastrowid

    return TransitionResult(
        ok=True,
        item_id=item_id,
        action=act,
        from_state=from_state,
        to_state=to_state,
        target=target_value,
        reason=payload_reason,
        gate_version=gate_version,
        gates=dec.get("gate", {}),
        log_id=log_id,
    )


def state_of(item_id: str, *, cfg=None) -> str:
    """读当前状态（含 v2 兼容投影回退）。供 API/测试使用。"""
    cfg, _, _ = _resolve(cfg)
    with DB.reader(cfg.db_path) as conn:
        row = conn.execute("SELECT state, tier, archived, muted FROM episode WHERE id=?", (item_id,)).fetchone()
    if row is None:
        return ""
    return row["state"] or IT.compat_state(row["tier"], row["archived"], row["muted"])


def state_log(item_id: str, *, cfg=None, limit: int = 50) -> list[dict]:
    cfg, _, _ = _resolve(cfg)
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT id,item_id,from_state,to_state,action,target,reason,gate_json,at,actor
               FROM item_state_log WHERE item_id=? ORDER BY at ASC, id ASC LIMIT ?""",
            (item_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_log(*, cfg=None, limit: int = 50) -> list[dict]:
    cfg, _, _ = _resolve(cfg)
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT id,item_id,from_state,to_state,action,target,reason,at,actor
               FROM item_state_log ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


__all__ = [
    "bind",
    "active_config",
    "submit",
    "enrich",
    "transition",
    "enqueue_enrich",
    "wait_enrich",
    "pending_enrich",
    "state_of",
    "state_log",
    "recent_log",
    "SubmitResult",
    "EnrichResult",
    "TransitionResult",
]
