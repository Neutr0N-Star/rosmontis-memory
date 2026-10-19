"""MemoryItem：主被动共用的唯一记忆对象。

契约来源：`docs/15-交付蓝图与模块契约.md` §3.1、`docs/_wip/A2-整合方案.md` §3.1–§3.3。

设计约束（硬）：
1. 对象同一：一条记忆就是一个 item；「被动/主动」只是 `origin`，不改变行为。
2. 生命周期唯一：`state` 是唯一真值；`tier / archived / muted` 是对外兼容投影，
   由本模块的 `projection()` 统一映射，任何入口不得自己写一套。
3. 不引入新表：item 就是 episode 行（A2 方案②）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

# ── 来源（谁发起）─────────────────────────────────────────────
ORIGIN_INGEST = "ingest"      # 被动：用户输入经抽取写入
ORIGIN_TOOL = "tool"          # 主动：模型调工具写
ORIGIN_SYSTEM = "system"      # 主动：系统整流（整理/结算）产候选
ORIGIN_DREAM = "dream"        # 做梦产物
ORIGIN_SUMMARY = "summary"    # 折叠摘要
ORIGINS = (ORIGIN_INGEST, ORIGIN_TOOL, ORIGIN_SYSTEM, ORIGIN_DREAM, ORIGIN_SUMMARY)

# ── 发起者（契约字段名是 actor，episode.author 是历史投影）──────
ACTOR_USER = "user"
ACTOR_MODEL = "model"
ACTOR_SYSTEM = "system"
ACTOR_TEACHER = "teacher"
ACTORS = (ACTOR_USER, ACTOR_MODEL, ACTOR_SYSTEM, ACTOR_TEACHER)

# ── 生命周期状态（7 态，A2 §3.2）──────────────────────────────
STATE_INBOX = "inbox"
STATE_NOTE = "note"
STATE_RESIDENT = "resident"
STATE_IMPORTANT = "important"
STATE_ARCHIVED = "archived"
STATE_MUTED = "muted"
STATE_DREAM = "dream"
STATES = (STATE_INBOX, STATE_NOTE, STATE_RESIDENT, STATE_IMPORTANT, STATE_ARCHIVED, STATE_MUTED, STATE_DREAM)

# state → (tier 兼容投影, archived, muted)
_STATE_PROJECTION: dict[str, tuple[str, int, int]] = {
    STATE_INBOX: ("inbox", 0, 0),
    STATE_NOTE: ("retrieval", 0, 0),
    STATE_RESIDENT: ("resident", 0, 0),
    STATE_IMPORTANT: ("important", 0, 0),
    STATE_ARCHIVED: ("retrieval", 1, 0),
    STATE_MUTED: ("retrieval", 0, 1),
    STATE_DREAM: ("dream", 0, 0),
}


def tier_of(state: str) -> str:
    """状态 → tier 兼容投影。未知状态退回 retrieval（绝不冒泡异常）。"""
    return _STATE_PROJECTION.get(state or "", _STATE_PROJECTION[STATE_NOTE])[0]


def archived_of(state: str) -> int:
    return _STATE_PROJECTION.get(state or "", _STATE_PROJECTION[STATE_NOTE])[1]


def muted_of(state: str) -> int:
    return _STATE_PROJECTION.get(state or "", _STATE_PROJECTION[STATE_NOTE])[2]


def projection(state: str) -> tuple[str, int, int]:
    """一次拿到 (tier, archived, muted)。写路径唯一的投影出口。"""
    return _STATE_PROJECTION.get(state or "", _STATE_PROJECTION[STATE_NOTE])


def compat_state(tier: str = "", archived: int | bool = 0, muted: int | bool = 0) -> str:
    """从旧库三列反推状态。迁移回填与 v2 回滚窗口都用它。

    优先级与 A2 §3.3 一致：muted → archived → tier 映射 → note。
    """
    if muted:
        return STATE_MUTED
    if archived:
        return STATE_ARCHIVED
    t = (tier or "").strip().lower()
    if t == "resident":
        return STATE_RESIDENT
    if t == "important":
        return STATE_IMPORTANT
    if t == "inbox":
        return STATE_INBOX
    if t == "dream":
        return STATE_DREAM
    return STATE_NOTE


def recall_visible(state: str) -> bool:
    """该状态的条目是否允许出现在召回里（C2：inbox 默认不出声）。读侧用它。"""
    return (state or "") not in (STATE_INBOX, STATE_ARCHIVED, STATE_MUTED)


def normalize_origin(value: str | None) -> str:
    v = (value or "").strip().lower()
    return v if v in ORIGINS else ORIGIN_INGEST


def normalize_actor(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in ACTORS:
        return v
    # episode 历史值兼容：self → model，主人写的 → user
    return {"self": ACTOR_MODEL, "teacher": ACTOR_TEACHER}.get(v, ACTOR_SYSTEM)


def author_for(actor: str) -> str:
    """actor → episode.author 兼容投影（self / teacher / system）。"""
    return {
        ACTOR_USER: "teacher",
        ACTOR_TEACHER: "teacher",
        ACTOR_MODEL: "self",
        ACTOR_SYSTEM: "system",
    }.get(normalize_actor(actor), "system")


def role_for(origin: str, actor: str) -> str:
    """origin/actor → episode.role。旧调用方显式传 role 时以它为优先。"""
    origin = normalize_origin(origin)
    actor = normalize_actor(actor)
    if origin == ORIGIN_DREAM:
        return "dream"
    if origin == ORIGIN_SUMMARY:
        return "summary"
    if origin == ORIGIN_TOOL:
        return "self"
    if origin == ORIGIN_SYSTEM:
        return "system"
    if actor in (ACTOR_USER, ACTOR_TEACHER):
        return "user"
    if actor in (ACTOR_MODEL,):
        return "assistant"
    return "system"


def source_for(origin: str) -> str:
    """origin → episode.source 兼容投影。"""
    return {
        ORIGIN_INGEST: "dialogue",
        ORIGIN_TOOL: "notebook",
        ORIGIN_SYSTEM: "manual",
        ORIGIN_DREAM: "dream",
        ORIGIN_SUMMARY: "summary",
    }.get(normalize_origin(origin), "dialogue")


def _rget(row: Any, key: str, default: Any = None) -> Any:
    """同时兼容 sqlite3.Row / dict / 普通对象，缺列时给默认值。"""
    try:
        if isinstance(row, Mapping) or hasattr(row, "keys"):
            if key in row.keys():
                return row[key]
            return default
        return getattr(row, key, default)
    except (IndexError, KeyError, TypeError):
        return default


def _segments_from_meta(meta: dict) -> dict:
    seg = meta.get("segments")
    if isinstance(seg, dict) and seg:
        return {str(k): "" if v is None else str(v) for k, v in seg.items()}
    return {k: meta.get(k, "") for k in ("what", "noticed", "means", "next") if meta.get(k)}


@dataclass
class MemoryItem:
    """一条记忆对象。§3.1 的字段顺序保持，后面是 episode 列的兼容补充。"""

    id: str = ""
    session_id: str = ""
    text: str = ""
    origin: str = ORIGIN_INGEST
    actor: str = ACTOR_SYSTEM
    memory_type: str = "fact"
    state: str = ""
    segments: dict = field(default_factory=dict)
    salience: float = 0.0
    emotion: float = 0.0
    t_event: float = 0.0
    t_ingest: float = 0.0

    # ── episode 兼容字段（§3.1 之外的实现补充，全部有默认值）──
    role: str = ""
    source: str = ""
    author: str = ""
    confirmed: bool = False
    embedding: bytes | None = field(default=None, repr=False)
    meta: dict = field(default_factory=dict)

    # ── 派生 ──────────────────────────────────────────────────
    def normalized(self) -> "MemoryItem":
        """规整枚举值并复制一份，避免 pipeline 原地改调用方对象。"""
        return replace(
            self,
            origin=normalize_origin(self.origin),
            actor=normalize_actor(self.actor),
            memory_type=(self.memory_type or "fact").strip().lower(),
            state=(self.state or "").strip().lower(),
            segments=dict(self.segments or {}),
            meta=dict(self.meta or {}),
        )

    def effective_role(self) -> str:
        return (self.role or role_for(self.origin, self.actor)).strip() or "system"

    def effective_author(self) -> str:
        base = (self.author or "").strip().lower()
        return base if base in ("self", "teacher", "system") else author_for(self.actor)

    def effective_source(self) -> str:
        base = (self.source or "").strip().lower()
        return base or source_for(self.origin)

    def has_second_layer(self) -> bool:
        """主动条目的第二层（noticed/means/next）。只有 what 的是速记。"""
        seg = self.segments or {}
        return any((seg.get(k) or "").strip() for k in ("noticed", "means", "next"))

    def promotion_text(self) -> str:
        """升格进人设/页面的那一句。与 notebook._essence 同口径。"""
        seg = self.segments or {}
        for key in ("next", "means", "noticed", "what"):
            v = (seg.get(key) or "").strip()
            if v:
                return v
        text = (self.text or "").strip()
        # meta 兼容：旧笔记把四段放在 meta_json 顶层
        for key in ("next", "means", "noticed", "what"):
            v = (self.meta.get(key) or "").strip()
            if v:
                return v
        return text

    def projection(self) -> tuple[str, int, int]:
        state = self.state or STATE_NOTE
        return projection(state)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "session_id": self.session_id,
            "text": self.text,
            "origin": self.origin,
            "actor": self.actor,
            "memory_type": self.memory_type,
            "state": self.state,
            "segments": dict(self.segments),
            "salience": round(float(self.salience or 0.0), 4),
            "emotion": round(float(self.emotion or 0.0), 4),
            "t_event": self.t_event,
            "t_ingest": self.t_ingest,
            "role": self.effective_role(),
            "source": self.effective_source(),
            "author": self.effective_author(),
            "confirmed": bool(self.confirmed),
        }
        return d

    @classmethod
    def from_row(cls, row: Any) -> "MemoryItem":
        """从 episode 行还原对象（召回、gate、transition 都用它）。"""
        raw_meta: dict = {}
        meta_text = _rget(row, "meta_json")
        if meta_text:
            try:
                loaded = json.loads(meta_text)
                if isinstance(loaded, dict):
                    raw_meta = loaded
            except (TypeError, ValueError):
                raw_meta = {}
        state = (_rget(row, "state") or "").strip().lower()
        # 兼容投影优先：forget/tools/dream 等旧入口仍可能直接改 archived/muted 或 tier，
        # 读到这种行时按旧列推导有效状态，避免状态机在「已归档」的行上继续升格。
        if _rget(row, "muted", 0):
            state = STATE_MUTED
        elif _rget(row, "archived", 0):
            state = STATE_ARCHIVED
        if state not in STATES:
            state = compat_state(_rget(row, "tier", ""), _rget(row, "archived", 0), _rget(row, "muted", 0))
        origin = _rget(row, "origin")
        segments = _segments_from_meta(raw_meta)
        t_event = _rget(row, "t_event", 0.0) or 0.0
        t_ingest = _rget(row, "t_ingest", 0.0) or 0.0
        return cls(
            id=str(_rget(row, "id", "") or ""),
            session_id=str(_rget(row, "session_id", "") or ""),
            text=str(_rget(row, "text", "") or ""),
            origin=normalize_origin(origin) if origin else _origin_from_legacy(row),
            actor=normalize_actor(_rget(row, "author", "")),
            memory_type=str(_rget(row, "memory_type", "fact") or "fact"),
            state=state,
            segments=segments,
            salience=float(_rget(row, "salience", 0.0) or 0.0),
            emotion=float(_rget(row, "emotion", 0.0) or 0.0),
            t_event=float(t_event),
            t_ingest=float(t_ingest),
            role=str(_rget(row, "role", "") or ""),
            source=str(_rget(row, "source", "") or ""),
            author=str(_rget(row, "author", "") or ""),
            confirmed=bool(_rget(row, "confirmed", 0)),
            embedding=_rget(row, "embedding"),
            meta=raw_meta,
        )


def _origin_from_legacy(row: Any) -> str:
    """v2 行没有 origin 时的推导（迁移回填用同一张表）。"""
    source = (_rget(row, "source", "") or "").strip().lower()
    role = (_rget(row, "role", "") or "").strip().lower()
    if source == "notebook":
        return ORIGIN_TOOL
    if source == "dream" or role == "dream":
        return ORIGIN_DREAM
    if source == "summary" or role == "summary":
        return ORIGIN_SUMMARY
    return ORIGIN_INGEST


def state_from_legacy(tier: str = "", archived: int | bool = 0, muted: int | bool = 0) -> str:
    """迁移回填：旧三列 → 状态。与 A2 §3.3 映射表一致。"""
    return compat_state(tier, archived, muted)


__all__ = [
    "MemoryItem",
    "ORIGINS",
    "ORIGIN_INGEST",
    "ORIGIN_TOOL",
    "ORIGIN_SYSTEM",
    "ORIGIN_DREAM",
    "ORIGIN_SUMMARY",
    "ACTORS",
    "ACTOR_USER",
    "ACTOR_MODEL",
    "ACTOR_SYSTEM",
    "ACTOR_TEACHER",
    "STATES",
    "STATE_INBOX",
    "STATE_NOTE",
    "STATE_RESIDENT",
    "STATE_IMPORTANT",
    "STATE_ARCHIVED",
    "STATE_MUTED",
    "STATE_DREAM",
    "tier_of",
    "archived_of",
    "muted_of",
    "projection",
    "compat_state",
    "state_from_legacy",
    "recall_visible",
    "normalize_origin",
    "normalize_actor",
    "author_for",
    "role_for",
    "source_for",
]
