"""给模型的工具集。

五个，少而够用。工具太多她会变成一台到处乱记的机器。

工具语义（WS-C 契约）：**提交 item，由 gate 定状态**。
所有写类工具统一返回：

    {"ok": bool, "id": str | None, "state": str, "from": str, "to": str, "reason": str}

`state` 取值与 docs/15 §3.1 的 item 状态机一致：
inbox / note / resident / important / archived / muted / dream。

每个 dispatch 的返回都必须是**模型能读懂的结果**——包括失败原因。
静默失败会让模型重复犯同一个错。

依赖点：`core/pipeline.py`、`core/gate.py`、`core/item.py` 由 WS-A 提供。
在它们就位之前，本模块做**惰性导入 + 降级**：写入口回落到现有
`Memory.notebook_*` / `Memory.promote`，状态由 notebook 的四段结构 gate 决定。
"""

from __future__ import annotations

import logging
import time
import uuid
from functools import lru_cache

log = logging.getLogger("rosmontis.tools")

# ── item 状态机（docs/15 §3.1）────────────────────────────────
STATE_INBOX = "inbox"
STATE_NOTE = "note"
STATE_RESIDENT = "resident"
STATE_IMPORTANT = "important"
STATE_ARCHIVED = "archived"
STATE_MUTED = "muted"
STATE_DREAM = "dream"

STATES = (STATE_INBOX, STATE_NOTE, STATE_RESIDENT, STATE_IMPORTANT, STATE_ARCHIVED, STATE_MUTED, STATE_DREAM)

_TIER_TO_STATE = {
    "inbox": STATE_INBOX,
    "retrieval": STATE_NOTE,
    "note": STATE_NOTE,
    "resident": STATE_RESIDENT,
    "important": STATE_IMPORTANT,
    "archived": STATE_ARCHIVED,
    "muted": STATE_MUTED,
    "dream": STATE_DREAM,
}

# ── JSON Schema（OpenAI function calling 格式，别的协议也能直接映射）──
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rosmontis_remember",
            "description": (
                "把一件事留给以后的自己。不是记流水账——只记你觉得将来还用得上的。"
                "如果只有『发生了什么』，它会进速记层，之后可能被丢弃；"
                "如果你还写了『我注意到什么』或『下次该怎么做』，它才会成为真正的笔记。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "what": {"type": "string", "description": "发生了什么，客观陈述"},
                    "noticed": {
                        "type": "string",
                        "description": "你注意到什么（别人可能没注意的细节）。可空。",
                    },
                    "means": {"type": "string", "description": "这对你来说意味着什么。可空。"},
                    "next": {
                        "type": "string",
                        "description": "下次遇到类似情况你该怎么做。可空，但这一项最有价值。",
                    },
                    "topic": {"type": "string", "description": "主题标签，用来归组"},
                },
                "required": ["what"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rosmontis_recall",
            "description": "主动查一件事。你已经拿到的上下文里没有再查，别重复查同一个问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "你要找什么"},
                    "budget": {"type": "integer", "description": "最多花多少 token，默认 600"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rosmontis_promote",
            "description": (
                "把一条笔记升格为「我就是这样」。只有当一条判断你反复想到、"
                "并且觉得它已经是你的一部分时才用。这会被系统限制频率。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string", "description": "哪条笔记"},
                    "section": {
                        "type": "string",
                        "enum": ["identity", "rule", "speech", "trait", "taste", "free"],
                        "description": "放进人格的哪个分区",
                    },
                },
                "required": ["note_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rosmontis_revise",
            "description": "订正一条你以前记错的笔记。原文会保留，不是覆盖。",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string", "description": "要改哪条"},
                    "new_text": {"type": "string", "description": "新的说法"},
                    "reason": {"type": "string", "description": "为什么改，必填"},
                },
                "required": ["note_id", "new_text", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rosmontis_skip",
            "description": "某件事你不想再提了。它不会被删除，但会沉下去，不再主动浮现。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "哪条记忆（episode / 事实 / 笔记都行）"},
                    "reason": {"type": "string", "description": "为什么不想再提"},
                },
                "required": ["item_id"],
            },
        },
    },
]


TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


# ── 统一结果 ──────────────────────────────────────────────────
def _result(
    ok: bool,
    *,
    item_id: str | None = None,
    state: str = "",
    from_state: str = "",
    to_state: str = "",
    reason: str = "",
) -> dict:
    """写类工具的唯一返回形态（六键，顺序稳定）。"""
    return {
        "ok": bool(ok),
        "id": item_id,
        "state": state,
        "from": from_state,
        "to": to_state,
        "reason": reason,
    }


def _field(obj, *names, default=None):
    """从 dict / dataclass / 对象里按候选名取值，取不到给 default。"""
    if obj is None:
        return default
    for n in names:
        try:
            if isinstance(obj, dict):
                if obj.get(n) is not None:
                    return obj[n]
            else:
                v = getattr(obj, n, None)
                if v is not None:
                    return v
        except Exception:
            continue
    return default


def _state_of_tier(tier: str | None) -> str:
    return _TIER_TO_STATE.get(str(tier or ""), STATE_NOTE)


def _episode_state(row) -> str:
    """episode 行 → 状态机里的 state。

    v3 起 `episode.state` 是唯一真值（WS-A 的状态机）；旧库/旧行没有 state 列，
    回落到 tier/archived/muted 三列兼容投影。
    """
    try:
        if hasattr(row, "keys") and "state" in row.keys():
            st = (row["state"] or "").strip().lower()
            if st in STATES:
                return st
    except Exception:
        pass
    if row["archived"]:
        return STATE_ARCHIVED
    if row["muted"]:
        return STATE_MUTED
    if row["source"] == "dream":
        return STATE_DREAM
    if row["superseded_by"]:
        return STATE_ARCHIVED
    return _state_of_tier(row["tier"])


@lru_cache(maxsize=64)
def _columns(db_path: str, table: str) -> frozenset:
    """表的列名（带缓存）。兼容迁移前后的旧库。"""
    from .core import db as DB

    try:
        with DB.reader(db_path) as conn:
            return frozenset(r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall())
    except Exception:
        return frozenset()


def _gate_action(action: str, target: str = "", current: str = "") -> str:
    """工具/HTTP 的通用动作 → core.gate 的标准动作名。"""
    a = (action or "").strip().lower()
    if a in ("promote", "upgrade"):
        if target in ("persona", "page"):
            return f"promote_{target}"
        if target in ("important", "mark_important"):
            return "mark_important"
        return "promote_resident"
    if a in ("skip", "mute", "muted"):
        return "skip"
    if a in ("archive", "remove", "delete", "forget", "drop"):
        return "archive"
    if a in ("unmute",):
        return "unmute"
    if a in ("unarchive", "restore", "unarchive_restore"):
        return "unmute" if current == STATE_MUTED else "unarchive"
    return a



def _fact_state(row) -> str:
    """fact 行 → 状态机里的 state。

    事实没有独立状态机：`invalidated` 一律停止检索，再用 t_valid_to 区分
    「静音（世界时间仍有效，只是不再浮现）」与「作废/归档（世界时间也结束）」。
    """
    st = str(row["status"] or "")
    if st == "active":
        return STATE_NOTE
    if st == "muted":
        return STATE_MUTED
    if st in ("invalidated", "archived"):
        try:
            if hasattr(row, "keys") and "t_valid_to" in row.keys() and row["t_valid_to"] is None:
                return STATE_MUTED
        except Exception:
            pass
        return STATE_ARCHIVED
    return _state_of_tier(st)


# ── WS-A pipeline 的惰性导入 ──────────────────────────────────
def _pipeline():
    """惰性拿 core.pipeline；还没就位就返回 None（降级到 Memory 方法）。"""
    try:
        from .core import pipeline as P  # noqa: WPS433

        if hasattr(P, "submit") or hasattr(P, "transition") or hasattr(P, "skip"):
            return P
    except Exception as e:  # 模块不存在 / 依赖没装好
        log.debug("core.pipeline not available: %s", e)
    return None


def _map_pipeline_result(r, fallback_id: str) -> dict:
    """把 SubmitResult / TransitionResult（dict 或 dataclass）映射成六键结果。"""
    ok = bool(_field(r, "ok", "success", default=True))
    state = _field(r, "state", "to_state", "to", default=STATE_NOTE)
    to_state = _field(r, "to_state", "to", default=state) or state
    from_state = _field(r, "from_state", "from", default="new")
    rid = _field(r, "id", "item_id", "note_id", default=fallback_id)
    reason = _field(r, "reason", "message", "detail", default="")
    if not ok:
        # 被 gate 拒绝：状态不发生迁移，state 停在 from
        state = from_state or state
    return _result(ok, item_id=rid, state=str(state), from_state=str(from_state), to_state=str(to_state), reason=str(reason))


# ── 写工具：提交 item，由 gate 定状态 ─────────────────────────
def _remember(mem, args: dict, session_id: str) -> dict:
    what = (args.get("what") or "").strip()
    if not what:
        return _result(False, state=STATE_INBOX, to_state=STATE_INBOX, reason="what 不能为空")
    noticed = (args.get("noticed") or "").strip()
    means = (args.get("means") or "").strip()
    next_step = (args.get("next") or "").strip()
    topic = (args.get("topic") or "").strip()
    text = "　".join(p for p in (what, f"注意到：{noticed}" if noticed else "", f"我想：{means}" if means else "", f"下次：{next_step}" if next_step else "") if p)
    fallback_state = STATE_NOTE if (noticed or means or next_step) else STATE_INBOX

    P = _pipeline()
    if P is not None and hasattr(P, "submit"):
        item = None
        try:
            from .core.item import MemoryItem  # type: ignore

            from .core.score import emotion_intensity

            now = time.time()
            item = MemoryItem(
                id="ep_tool_" + uuid.uuid4().hex[:12],
                session_id=session_id,
                text=text,
                origin="tool",
                actor="model",
                memory_type="fact",
                # state 留空：由 gate 按「有没有第二层」判定 inbox / note
                state="",
                segments={"what": what, "noticed": noticed, "means": means, "next": next_step, "topic": topic},
                salience=0.6,
                emotion=emotion_intensity(text),
                t_event=now,
                t_ingest=now,
            )
            r = P.submit(item, project=True, cfg=mem.cfg)
        except Exception as e:
            log.debug("pipeline.submit unavailable, fallback to notebook: %s", e)
            item = None
        else:
            return _map_pipeline_result(r, item.id)

    # 降级：现有 notebook gate（四段结构决定进笔记层还是速记层）
    r = mem.notebook_write(
        what=what,
        session_id=session_id,
        noticed=noticed,
        means=means,
        next_step=next_step,
        topic=topic,
    )
    if not r.get("ok"):
        return _result(False, state=fallback_state, to_state=fallback_state, reason=str(r.get("reason") or "写入被拒绝"))
    tier = r.get("tier") or ("retrieval" if r.get("has_second_layer") else "inbox")
    state = _state_of_tier(tier)
    note = r.get("note") or ("进了笔记层" if state == STATE_NOTE else "只有 what，留在速记层")
    return _result(True, item_id=r.get("id"), state=state, from_state="new", to_state=state, reason=str(note))


def _promote(mem, args: dict, session_id: str) -> dict:
    note_id = (args.get("note_id") or "").strip()
    if not note_id:
        return _result(False, state=STATE_NOTE, to_state=STATE_RESIDENT, reason="note_id 不能为空")
    section = (args.get("section") or "free") or "free"

    # WS-A 的 notebook.promote 已收口 gate + pipeline.transition；
    # 这里只负责调用与结果映射，状态一律以 gate 的 to 为准（不自行假设 resident）。
    r = mem.notebook_promote(note_id, target="persona", section=section, session_id=session_id)
    if not r.get("ok"):
        reason = str(r.get("reason") or "升格被拒绝")
        return _result(False, item_id=note_id, state=STATE_NOTE, from_state=STATE_NOTE,
                       to_state=STATE_RESIDENT, reason=reason)

    prev = str(r.get("from") or STATE_NOTE)
    state = str(r.get("state") or r.get("to") or STATE_NOTE)

    # 兼容旧版 notebook（只写 persona / notebook_promotion，不写状态机）：
    # 结果里没有 state 时才补一次状态留痕，避免重复过闸。
    P = _pipeline()
    if "state" not in r and "to" not in r and P is not None and hasattr(P, "transition"):
        try:
            tr = P.transition(note_id, "promote_persona", target=section,
                              reason="tool:rosmontis_promote", actor="model", cfg=mem.cfg)
            mapped = _map_pipeline_result(tr, note_id)
            if mapped["ok"]:
                prev, state = mapped["from"] or prev, mapped["to"] or state
        except Exception as e:
            log.debug("pipeline.transition unavailable, state stays tier projection: %s", e)

    return _result(
        True,
        item_id=r.get("id"),
        state=state,
        from_state=prev,
        to_state=state,
        reason=str(r.get("note") or "已写入人格，下一次 build_context 立刻生效"),
    )


def _revise(mem, args: dict, session_id: str) -> dict:
    note_id = (args.get("note_id") or "").strip()
    new_text = (args.get("new_text") or "").strip()
    if not (note_id and new_text):
        return _result(False, state=STATE_NOTE, to_state=STATE_NOTE, reason="note_id 与 new_text 都必须给")
    r = mem.notebook_revise(note_id, new_text, args.get("reason", "") or "")
    if not r.get("ok"):
        return _result(False, item_id=note_id, state=STATE_NOTE, from_state=STATE_NOTE,
                       to_state=STATE_NOTE, reason=str(r.get("reason") or "订正失败"))
    return _result(
        True,
        item_id=r.get("new_id"),
        state=STATE_NOTE,
        from_state=STATE_NOTE,
        to_state=STATE_NOTE,
        reason=str(r.get("note") or "旧说法保留但不再浮现"),
    )


def _skip(mem, args: dict, session_id: str) -> dict:
    item_id = (args.get("item_id") or "").strip()
    if not item_id:
        return _result(False, state=STATE_MUTED, to_state=STATE_MUTED, reason="item_id 不能为空")

    # 先看是不是 fact：状态机（episode 行）不覆盖事实层，事实走直接落库。
    cur = item_state(mem, item_id)
    if cur is not None and cur.get("kind") == "fact":
        return _skip_direct(mem, item_id)

    P = _pipeline()
    if P is not None and hasattr(P, "transition"):
        try:
            tr = P.transition(item_id, "skip", reason="tool:rosmontis_skip", actor="model", cfg=mem.cfg)
            mapped = _map_pipeline_result(tr, item_id)
            if mapped["ok"]:
                return mapped
            log.debug("gate rejected skip: %s", mapped["reason"])
        except Exception as e:
            log.debug("pipeline.transition unavailable, fallback to db: %s", e)

    return _skip_direct(mem, item_id)


def _skip_direct(mem, item_id: str) -> dict:
    """直接落库：episode 写 muted+长冷却（有 state 列则同步状态机）；fact 用 invalidated。"""
    from .core import db as DB

    now = time.time()
    ep_cols = _columns(mem.cfg.db_path, "episode")
    with DB.tx(mem.cfg.db_path) as conn:
        ep = conn.execute("SELECT * FROM episode WHERE id=?", (item_id,)).fetchone()
        if ep is not None:
            prev = _episode_state(ep)
            if "state" in ep_cols:
                conn.execute(
                    "UPDATE episode SET state='muted', state_at=?, tier='retrieval', archived=0, muted=1, "
                    "cooldown_until=? WHERE id=?",
                    (now, now + 86400 * 365 * 10, item_id),
                )
            else:
                conn.execute(
                    "UPDATE episode SET muted=1, cooldown_until=? WHERE id=?",
                    (now + 86400 * 365 * 10, item_id),
                )
            return _result(True, item_id=item_id, state=STATE_MUTED, from_state=prev,
                           to_state=STATE_MUTED, reason="不再主动提起了，条目仍保有原文")

        fact_cols = _columns(mem.cfg.db_path, "fact")
        fa = conn.execute("SELECT * FROM fact WHERE id=?", (item_id,)).fetchone()
        if fa is not None:
            prev = _fact_state(fa)
            # 静音 ≠ 作废：世界时间 t_valid_to 保持 NULL，只置 invalidated/archived
            # 让检索不再命中；这样 item_state 能把它与「归档」区分开。
            if "archived" in fact_cols:
                conn.execute(
                    "UPDATE fact SET status='invalidated', archived=1, t_invalid=?, last_access=? WHERE id=?",
                    (now, now, item_id),
                )
            else:
                conn.execute(
                    "UPDATE fact SET status='invalidated', t_invalid=?, last_access=? WHERE id=?",
                    (now, now, item_id),
                )
            return _result(True, item_id=item_id, state=STATE_MUTED, from_state=prev,
                           to_state=STATE_MUTED, reason="该事实不再参与检索，行仍保有权重与来源")

    return _result(False, item_id=item_id, state="", to_state=STATE_MUTED,
                   reason=f"条目不存在：{item_id}")


# ── 运维接口：查状态 / 显式迁移 ───────────────────────────────
def item_state(mem, item_id: str) -> dict | None:
    """查任意 item 的当前状态。找不到返回 None。"""
    from .core import db as DB

    with DB.reader(mem.cfg.db_path) as conn:
        ep = conn.execute("SELECT * FROM episode WHERE id=?", (item_id,)).fetchone()
        if ep is not None:
            return {
                "id": item_id,
                "kind": "episode",
                "session_id": ep["session_id"],
                "state": _episode_state(ep),
                "tier": ep["tier"],
                "source": ep["source"],
                "archived": bool(ep["archived"]),
                "muted": bool(ep["muted"]),
                "superseded": bool(ep["superseded_by"]),
            }
        fa = conn.execute("SELECT * FROM fact WHERE id=?", (item_id,)).fetchone()
        if fa is not None:
            cols = _columns(mem.cfg.db_path, "fact")
            return {
                "id": item_id,
                "kind": "fact",
                "session_id": fa["session_id"],
                "state": _fact_state(fa),
                "status": fa["status"],
                "archived": bool(fa["archived"]) if "archived" in cols else False,
                "superseded": bool(fa["superseded_by"]),
            }
    return None


def transition_item(
    mem,
    item_id: str,
    action: str,
    *,
    target: str = "",
    reason: str = "",
    actor: str = "system",
) -> dict:
    """显式状态迁移（供 HTTP 运维接口与宿主用）。返回六键统一结果。

    动作：promote（target=persona/page/resident/important）、skip/mute、archive、
    restore/unmute。episode 优先走 WS-A 的 gate+pipeline（唯一状态真值）；
    事实层（fact）不进状态机，走兼容投影的直接落库。
    """
    action = (action or "").strip().lower()
    cur = item_state(mem, item_id)
    if cur is None:
        return _result(False, item_id=item_id, state="", to_state="", reason=f"条目不存在：{item_id}")
    prev = cur["state"]
    target_state = _state_of_tier(target) if target else ""
    if target in ("persona", "resident", ""):
        target_state = STATE_RESIDENT if target != "important" else STATE_IMPORTANT

    P = _pipeline()
    if cur["kind"] == "episode" and P is not None and hasattr(P, "transition"):
        act = _gate_action(action, target, prev)
        try:
            tr = P.transition(item_id, act, target=target, reason=reason, actor=actor, cfg=mem.cfg)
            return _map_pipeline_result(tr, item_id)
        except Exception as e:
            log.debug("pipeline.transition unavailable, fallback: %s", e)

    if action in ("promote", "upgrade"):
        if cur["kind"] == "fact":
            return _result(False, item_id=item_id, state=prev, from_state=prev,
                           to_state=target_state or STATE_RESIDENT, reason="事实条目不走人格升格，请改写状态槽")
        tier = "important" if target_state == STATE_IMPORTANT else "resident"
        r = mem.promote(item_id, tier)
        if not r.get("ok"):
            return _result(False, item_id=item_id, state=prev, from_state=prev,
                           to_state=target_state or STATE_RESIDENT, reason=str(r.get("reason") or "升格被拒绝"))
        return _result(True, item_id=item_id, state=_state_of_tier(r.get("to")),
                       from_state=_state_of_tier(r.get("from")) or prev,
                       to_state=_state_of_tier(r.get("to")), reason=str(r.get("reason") or "已升格"))

    if action in ("skip", "mute"):
        return _skip_direct(mem, item_id)

    if action in ("archive", "remove"):
        from .core import db as DB

        now = time.time()
        ep_cols = _columns(mem.cfg.db_path, "episode")
        with DB.tx(mem.cfg.db_path) as conn:
            if cur["kind"] == "episode":
                if "state" in ep_cols:
                    conn.execute(
                        "UPDATE episode SET state='archived', state_at=?, tier='retrieval', archived=1, muted=0 "
                        "WHERE id=?",
                        (now, item_id),
                    )
                else:
                    conn.execute("UPDATE episode SET archived=1 WHERE id=?", (item_id,))
            else:
                fact_cols = _columns(mem.cfg.db_path, "fact")
                if "archived" in fact_cols:
                    conn.execute(
                        "UPDATE fact SET status='invalidated', archived=1, t_invalid=?, t_valid_to=? WHERE id=?",
                        (now, now, item_id),
                    )
                else:
                    conn.execute(
                        "UPDATE fact SET status='invalidated', t_invalid=?, t_valid_to=? WHERE id=?",
                        (now, now, item_id),
                    )
        return _result(True, item_id=item_id, state=STATE_ARCHIVED, from_state=prev,
                       to_state=STATE_ARCHIVED, reason="已归档，原文仍保留可查")

    if action in ("restore", "unmute", "unarchive"):
        from .core import db as DB

        ep_cols = _columns(mem.cfg.db_path, "episode")
        with DB.tx(mem.cfg.db_path) as conn:
            if cur["kind"] == "episode":
                if "state" in ep_cols:
                    conn.execute(
                        "UPDATE episode SET state='note', tier='retrieval', archived=0, muted=0, "
                        "cooldown_until=NULL WHERE id=?",
                        (item_id,),
                    )
                else:
                    conn.execute(
                        "UPDATE episode SET archived=0, muted=0, cooldown_until=NULL WHERE id=?",
                        (item_id,),
                    )
            else:
                fact_cols = _columns(mem.cfg.db_path, "fact")
                if "archived" in fact_cols:
                    conn.execute(
                        "UPDATE fact SET status='active', archived=0, t_invalid=NULL, t_valid_to=NULL WHERE id=?",
                        (item_id,),
                    )
                else:
                    conn.execute(
                        "UPDATE fact SET status='active', t_invalid=NULL, t_valid_to=NULL WHERE id=?",
                        (item_id,),
                    )
        return _result(True, item_id=item_id, state=STATE_NOTE, from_state=prev,
                       to_state=STATE_NOTE, reason="已恢复")

    return _result(False, item_id=item_id, state=prev, from_state=prev,
                   to_state=target_state or prev, reason=f"未知动作：{action or '(空)'}")


# ── dispatcher ────────────────────────────────────────────────
def dispatch(mem, name: str, args: dict, session_id: str) -> dict:
    """执行一次工具调用。

    `mem` 是 Memory 实例。所有失败都返回 {ok: false, reason}，
    让模型看到原因并自己调整——不静默吞掉。
    """
    args = args or {}
    try:
        if name == "rosmontis_remember":
            return _remember(mem, args, session_id)

        if name == "rosmontis_recall":
            q = (args.get("query") or "").strip()
            if not q:
                return {"ok": False, "reason": "query 不能为空"}
            r = mem.recall(q, session_id=session_id, budget=int(args.get("budget") or 600))
            if r.get("abstained"):
                return {"ok": True, "found": False, "reason": r.get("reason")}
            return {
                "ok": True,
                "found": True,
                "prompt": r.get("prompt", ""),
                "count": len(r.get("items", [])),
            }

        if name == "rosmontis_promote":
            return _promote(mem, args, session_id)

        if name == "rosmontis_revise":
            return _revise(mem, args, session_id)

        if name == "rosmontis_skip":
            return _skip(mem, args, session_id)

        return _result(False, state="", to_state="", reason=f"没有这个工具：{name}")

    except Exception as e:
        log.warning("tool %s failed: %s", name, e)
        # 失败原因必须带异常消息，只给类名模型没法自我修正
        return {"ok": False, "reason": f"执行出错：{type(e).__name__}: {e}"}
