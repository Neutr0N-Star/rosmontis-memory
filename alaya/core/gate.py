"""闸门：升格/降级/静音的唯一判定点。

A2 §3.4 / §5.2 判据 M5：所有生命周期动作经 `gate.decide()`；内部必须保留
`promotion.can_promote`（15 项基线里 `test_promotion_gate` 依赖它），再叠加
主动笔记原有的频率（≤2/会话）/幅度（≤200 字）/来源（必须来自笔记）三闸。

`decide()` 是纯判定：**不写库**。写库由 `pipeline.transition()` 负责，这样
判定表只有一张，状态与日志永远同时落。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from . import db as DB
from . import item as IT
from . import promotion as PR

log = logging.getLogger("alaya.gate")

# 契约版本：episode.gate_version 与 item_state_log.gate_json 都记它
GATE_VERSION = "3.0.0"

# 主动笔记升格人设的三道旧闸，原样保留
MAX_PROMOTIONS_PER_SESSION = 2
MAX_PERSONA_CHARS = 200

# 动作别名：对外 API 用旧词也能收口到同一张判定表
_ACTION_ALIASES = {
    "create": "create",
    "submit": "create",
    "promote": "promote_resident",
    "promote_resident": "promote_resident",
    "resident": "promote_resident",
    "demote": "demote",
    "retrieval": "demote",
    "mark_important": "mark_important",
    "important": "mark_important",
    "unmark_important": "unmark_important",
    "skip": "skip",
    "mute": "skip",
    "muted": "skip",
    "archive": "archive",
    "forget": "archive",
    "unmute": "unmute",
    "unarchive": "unarchive",
    "restore": "unarchive",
    "review": "review_note",
    "review_note": "review_note",
    "review_archive": "review_archive",
    "drop": "review_archive",
    "promote_persona": "promote_persona",
    "promote_page": "promote_page",
    "revise": "revise",
    "revert_promotion": "revert_promotion",
}

# 动作 → 允许的起始状态（空字符串 = 新建）
_FROM = {
    "promote_resident": {IT.STATE_NOTE, IT.STATE_IMPORTANT},
    "demote": {IT.STATE_RESIDENT, IT.STATE_IMPORTANT},
    "mark_important": {IT.STATE_NOTE},
    "unmark_important": {IT.STATE_IMPORTANT},
    "skip": {IT.STATE_INBOX, IT.STATE_NOTE, IT.STATE_RESIDENT, IT.STATE_IMPORTANT, IT.STATE_DREAM},
    "archive": {IT.STATE_INBOX, IT.STATE_NOTE, IT.STATE_RESIDENT, IT.STATE_IMPORTANT, IT.STATE_MUTED, IT.STATE_DREAM},
    "unmute": {IT.STATE_MUTED},
    "unarchive": {IT.STATE_ARCHIVED},
    "review_note": {IT.STATE_INBOX},
    "review_archive": {IT.STATE_INBOX},
    "promote_persona": {IT.STATE_NOTE, IT.STATE_RESIDENT, IT.STATE_IMPORTANT},
    "promote_page": {IT.STATE_NOTE, IT.STATE_RESIDENT, IT.STATE_IMPORTANT},
    # 订正/撤回只留痕，不改状态
    "revise": set(IT.STATES),
    "revert_promotion": set(IT.STATES),
}


def normalize_action(action: str) -> str:
    return _ACTION_ALIASES.get((action or "").strip().lower(), (action or "").strip().lower())


# ── 初始状态 ──────────────────────────────────────────────────
def initial_state(item: IT.MemoryItem) -> str:
    """新建条目的默认状态。与 A2 §3.2 状态表一致。

    - 被动 ingest：默认 note；种子/摘要/主人手记可 resident（沿用旧 default_tier）
    - 主动 remember：只有 what → inbox（不可检索）；有第二层 → note
    - 系统整流：只能产 note 候选（C7）
    - dream：dream 态；summary：note
    """
    origin = IT.normalize_origin(item.origin)
    if origin == IT.ORIGIN_DREAM:
        return IT.STATE_DREAM
    if origin == IT.ORIGIN_SUMMARY:
        return IT.STATE_NOTE
    if origin == IT.ORIGIN_SYSTEM:
        return IT.STATE_NOTE
    if origin == IT.ORIGIN_TOOL:
        return IT.STATE_NOTE if item.has_second_layer() else IT.STATE_INBOX

    # ingest：沿用 PR.default_tier，只做投影，不另立算法
    tier = PR.default_tier(item.effective_source(), item.memory_type, item.effective_author())
    ok, _ = PR.validate_combo(tier, item.memory_type)
    if not ok:
        return IT.STATE_NOTE
    return {
        PR.TIER_RESIDENT: IT.STATE_RESIDENT,
        PR.TIER_IMPORTANT: IT.STATE_IMPORTANT,
        PR.TIER_RETRIEVAL: IT.STATE_NOTE,
    }.get(tier, IT.STATE_NOTE)


# ── 频率闸：单会话最多改 2 次人设（原 notebook._promotions_in_session）──
def _promotions_in_session(cfg, session_id: str) -> int:
    if cfg is None or not session_id:
        return 0
    try:
        with DB.reader(cfg.db_path) as conn:
            made = conn.execute(
                """SELECT COUNT(*) c FROM item_state_log l
                   JOIN episode e ON e.id = l.item_id
                   WHERE e.session_id=? AND l.action IN ('promote_persona','promote_page')""",
                (session_id,),
            ).fetchone()
            reverted = conn.execute(
                """SELECT COUNT(*) c FROM item_state_log l
                   JOIN episode e ON e.id = l.item_id
                   WHERE e.session_id=? AND l.action='revert_promotion'""",
                (session_id,),
            ).fetchone()
        return max(0, int(made["c"] or 0) - int(reverted["c"] or 0))
    except Exception as e:  # 新库尚未迁移时不要让闸门崩掉
        log.debug("gate: count promotions failed: %s", e)
        return 0


def _pr_check(item: IT.MemoryItem, target_tier: str) -> tuple[bool, str]:
    """硬限制继续由 promotion.can_promote 说了算（15 项测试依赖它）。"""
    return PR.can_promote(
        target_tier,
        item.effective_author(),
        item.memory_type,
        bool(item.confirmed),
        item.effective_source(),
    )


def _result(ok: bool, action: str, src: str, dst: str, reason: str, checks: list[dict], target: str = "") -> dict:
    return {
        "ok": bool(ok),
        "action": action,
        "from": src,
        "to": dst,
        "target": target,
        "reason": reason,
        "gate_version": GATE_VERSION,
        "gate": {"version": GATE_VERSION, "checks": checks},
    }


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


# ── 唯一判定表 ────────────────────────────────────────────────
def decide(
    item: IT.MemoryItem | Any,
    action: str,
    *,
    target: str = "",
    cfg=None,
    session_id: str = "",
) -> dict:
    """判断一次动作能不能做。返回 {ok, from, to, target, reason, gate_version, gate}。

    `item` 可以是 MemoryItem、episode 行或 dict；`cfg` 用于频率闸；`session_id`
    缺省取 item.session_id。
    """
    if not isinstance(item, IT.MemoryItem):
        item = IT.MemoryItem.from_row(item)
    item = item.normalized()

    act = normalize_action(action)
    sid = session_id or item.session_id
    current = item.state if item.state in IT.STATES else ""
    checks: list[dict] = []

    if act == "create":
        requested = item.state if item.state in IT.STATES else ""
        if requested:
            # 订正继承旧状态等场景：只校验非法组合，不重新过升格闸
            ok, why = PR.validate_combo(IT.tier_of(requested), item.memory_type)
            checks.append(_check("combo", ok, why))
            to = requested if ok else IT.STATE_NOTE
            return _result(True, act, "", to, "继承状态" if ok else f"{why}，退回 note", checks)
        to = initial_state(item)
        if to == IT.STATE_RESIDENT:
            ok, why = _pr_check(item, PR.TIER_RESIDENT)
            checks.append(_check("can_promote", ok, why))
            if not ok:
                to = IT.STATE_NOTE
        if to == IT.STATE_INBOX:
            checks.append(_check("second_layer", False, "只有 what，进速记层（不可检索）"))
        return _result(True, act, "", to, "新建", checks)

    if act not in _FROM:
        return _result(False, act, current, current, f"未知动作 {action}", checks)

    src = current or IT.compat_state(item.meta.get("tier", ""))  # 兼容旧行
    allowed_from = _FROM[act]
    if act in ("revise", "revert_promotion"):
        checks.append(_check("exists", True, "只留痕"))
        return _result(True, act, src, src, "留痕", checks, target)

    if src not in allowed_from:
        # 幂等容错：已经在目标态就直接放行
        if act == "skip" and src == IT.STATE_MUTED:
            return _result(True, act, src, src, "已是静音", checks, target)
        if act == "archive" and src == IT.STATE_ARCHIVED:
            return _result(True, act, src, src, "已归档", checks, target)
        return _result(
            False,
            act,
            src,
            src,
            f"状态 {src or '空'} 不允许动作 {act}（允许：{'/'.join(sorted(allowed_from))}）",
            checks,
            target,
        )

    # ── 升常驻：PR 硬限制 ──
    if act == "promote_resident":
        ok, why = _pr_check(item, PR.TIER_RESIDENT)
        checks.append(_check("can_promote", ok, why))
        if not ok:
            return _result(False, act, src, src, why, checks, target)
        return _result(True, act, src, IT.STATE_RESIDENT, why, checks, target)

    if act == "demote":
        return _result(True, act, src, IT.STATE_NOTE, "降回笔记层", checks, target)

    if act == "mark_important":
        ok, why = _pr_check(item, PR.TIER_IMPORTANT)
        checks.append(_check("can_promote", ok, why))
        if not ok:
            return _result(False, act, src, src, why, checks, target)
        return _result(True, act, src, IT.STATE_IMPORTANT, why, checks, target)

    if act == "unmark_important":
        return _result(True, act, src, IT.STATE_NOTE, "取消重要标记", checks, target)

    if act == "skip":
        return _result(True, act, src, IT.STATE_MUTED, "静音，不再主动提起", checks, target)

    if act == "archive":
        return _result(True, act, src, IT.STATE_ARCHIVED, "归档，可回溯", checks, target)

    if act == "unmute":
        return _result(True, act, src, IT.STATE_NOTE, "解除静音", checks, target)

    if act == "unarchive":
        return _result(True, act, src, IT.STATE_NOTE, "解除归档", checks, target)

    if act == "review_note":
        return _result(True, act, src, IT.STATE_NOTE, "整理后进笔记层", checks, target)

    if act == "review_archive":
        return _result(True, act, src, IT.STATE_ARCHIVED, "整理不出，归档", checks, target)

    # ── 升人设 / 页面：PR + 频率 + 幅度 + 来源 + 系统不越权 ──
    if act in ("promote_persona", "promote_page"):
        ok, why = _pr_check(item, PR.TIER_RESIDENT)
        checks.append(_check("can_promote", ok, why))
        if not ok:
            return _result(False, act, src, src, why, checks, target)

        if item.origin == IT.ORIGIN_SYSTEM:
            checks.append(_check("origin", False, "系统产物只能进笔记层，人设需模型/用户发起"))
            return _result(False, act, src, src, "系统产物只能进笔记层，人设需模型/用户发起", checks, target)
        checks.append(_check("origin", True, item.origin))

        if item.effective_source() != "notebook":
            checks.append(_check("source", False, f"来源 {item.effective_source()} 不是笔记"))
            return _result(False, act, src, src, "只有自己写的笔记才能升格", checks, target)
        checks.append(_check("source", True, "notebook"))

        text = item.promotion_text()
        amp_ok = len(text) <= MAX_PERSONA_CHARS
        checks.append(_check("amplitude", amp_ok, f"{len(text)} 字 / 上限 {MAX_PERSONA_CHARS}"))
        if not amp_ok:
            return _result(
                False,
                act,
                src,
                src,
                f"单条升格内容超过 {MAX_PERSONA_CHARS} 字（当前 {len(text)}），先精简",
                checks,
                target,
            )

        n = _promotions_in_session(cfg, sid)
        freq_ok = n < MAX_PROMOTIONS_PER_SESSION
        checks.append(_check("frequency", freq_ok, f"{n}/{MAX_PROMOTIONS_PER_SESSION}"))
        if not freq_ok:
            return _result(
                False,
                act,
                src,
                src,
                f"单次会话最多改 {MAX_PROMOTIONS_PER_SESSION} 次人设（已用 {n} 次）。人格不该每天推翻",
                checks,
                target,
            )
        return _result(True, act, src, src, "允许升格", checks, target)

    return _result(False, act, src, src, f"动作 {act} 未实现", checks, target)


def gate_json(result: dict, *, reason: str = "") -> str:
    """把判定结果压成 item_state_log.gate_json。"""
    payload = {
        "version": result.get("gate_version", GATE_VERSION),
        "action": result.get("action", ""),
        "checks": (result.get("gate") or {}).get("checks", []),
    }
    if reason:
        payload["reason"] = reason
    return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "GATE_VERSION",
    "MAX_PROMOTIONS_PER_SESSION",
    "MAX_PERSONA_CHARS",
    "decide",
    "initial_state",
    "normalize_action",
    "gate_json",
]
