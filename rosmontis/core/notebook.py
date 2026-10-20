"""主动记忆：她自己写、自己改、自己升格。

整合后的写入纪律（docs/14 §1、A2 §3）：
- 笔记就是 episode（`origin='tool'`），但 **不再自己 INSERT**：一律走 `pipeline.submit`，
  与被动写入共用同一写入点、同一状态机、同一套 salience/enrich。
- 速记与笔记的区别是 `state='inbox'` vs `state='note'`（`tier` 只是兼容投影）。
- 升格到人设/页面走 `gate.decide()`；状态与审计由 `pipeline.transition()` 落。
- 只有 what 的条目留在速记层（inbox，不可检索），这是防流水账的硬闸。
- `notebook_promotion` 保留只读一版，不再写入；审计统一进 `item_state_log`。
"""

from __future__ import annotations

import json
import logging
import time

from ..config import Config
from . import db as DB
from . import gate
from . import ids as IDS
from . import item as IT
from . import persona as PER
from . import pipeline as TR
from . import promotion as PR

log = logging.getLogger("rosmontis.notebook")

# 人设改动的旧闸常量（判定已收口到 gate，这里保留常量名给外部引用）
MAX_PROMOTIONS_PER_SESSION = gate.MAX_PROMOTIONS_PER_SESSION
MAX_PERSONA_CHARS = gate.MAX_PERSONA_CHARS


def ensure_tables(cfg: Config) -> None:
    """兼容表：`notebook_promotion` 只读保留一版（A2 §3.3）。"""
    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS notebook_promotion (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                note_id     TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                promoted_at REAL NOT NULL,
                diff        TEXT,
                reverted    INTEGER NOT NULL DEFAULT 0
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_promo_note ON notebook_promotion(note_id)")


# ── 写 ────────────────────────────────────────────────────────
def remember(
    cfg: Config,
    embedder,
    session_id: str,
    what: str,
    noticed: str = "",
    means: str = "",
    next_step: str = "",
    topic: str = "",
    t_event: float | None = None,
) -> dict:
    """写一条笔记（统一管线）。

    四段：what / noticed / means / next。
    **只有 what 的条目进速记层（inbox），不进笔记层。** 这是防流水账的硬闸。
    写入后结构投影由 pipeline 排进异步 enrich——主动条目从此也有 entity/fact/超边。
    """
    ensure_tables(cfg)
    now = time.time()
    t_event = t_event or now
    what = (what or "").strip()
    if not what:
        return {"ok": False, "reason": "what 不能为空"}

    noticed = (noticed or "").strip()
    means = (means or "").strip()
    next_step = (next_step or "").strip()
    has_second_layer = bool(noticed or means or next_step)

    body = _compose(what, noticed, means, next_step)
    segments = {"what": what, "noticed": noticed, "means": means, "next": next_step}
    meta = dict(segments)
    meta["topic"] = (topic or "").strip()

    emb = None
    if embedder is not None:
        try:
            # 降级期不写哈希伪向量（I4：两入口一致）：先算，算完再按 degraded 判定
            blob = getattr(embedder, "embed_blob", None)
            blob = blob(body) if blob is not None else None
            emb = None if getattr(embedder, "degraded", False) else blob
        except Exception:
            emb = None

    item = IT.MemoryItem(
        id="",
        session_id=session_id,
        text=body,
        origin=IT.ORIGIN_TOOL,
        actor=IT.ACTOR_MODEL,
        memory_type=PR.T_FACT,
        segments=segments,
        salience=0.0,  # 0 = 交给管线的单一 salience 实现
        emotion=0.0,   # 0 = 交给管线的单一 emotion 实现
        t_event=t_event,
        t_ingest=now,
        role="self",
        source="notebook",
        author=PR.A_SELF,
        embedding=emb,
        meta=meta,
    )
    sub = TR.submit(item, project=True, cfg=cfg, embedder=embedder)
    if sub.duplicate:
        return {"ok": True, "id": sub.item_id, "duplicate": True, "tier": sub.tier, "state": sub.state}
    if not sub.ok:
        return {"ok": False, "reason": sub.reason}
    if emb is not None:
        from . import write as WR  # 延迟导入：write → pipeline，避免模块环

        WR.stamp_episode_embedding(cfg, sub.item_id, embedder)

    return {
        "ok": True,
        "id": sub.item_id,
        "state": sub.state,
        "tier": sub.tier,
        "has_second_layer": has_second_layer,
        "enrich_queued": sub.enrich_queued,
        "note": "进了笔记层" if has_second_layer else "只有 what，留在速记层，会话结束时会整理或丢弃",
    }


def _compose(what: str, noticed: str, means: str, next_step: str) -> str:
    """拼成可读、可检索的正文。顺序固定，方便后续解析。"""
    parts = [what]
    if noticed:
        parts.append(f"注意到：{noticed}")
    if means:
        parts.append(f"我想：{means}")
    if next_step:
        parts.append(f"下次：{next_step}")
    return "　".join(parts)


# ── 订正 ──────────────────────────────────────────────────────
def revise(cfg: Config, note_id: str, new_text: str, reason: str = "", author: str = PR.A_SELF) -> dict:
    """订正一条笔记。原文不删，新条指回旧条；状态继承旧条，旧条留 revise 审计。"""
    ensure_tables(cfg)
    with DB.reader(cfg.db_path) as conn:
        r = conn.execute("SELECT * FROM episode WHERE id=?", (note_id,)).fetchone()
    if r is None:
        return {"ok": False, "reason": "笔记不存在"}
    if r["source"] != "notebook":
        return {"ok": False, "reason": "只能订正自己写的笔记"}

    now = time.time()
    old = IT.MemoryItem.from_row(r)
    new_id = "ep_corr_" + IDS._digest(note_id, new_text, int(now))  # noqa: SLF001
    item = IT.MemoryItem(
        id=new_id,
        session_id=old.session_id,
        text=(new_text or "").strip(),
        origin=IT.ORIGIN_TOOL,
        actor=IT.ACTOR_MODEL,
        memory_type=old.memory_type,
        state=old.state,  # 继承旧条状态（inbox 仍是 inbox）
        salience=max(0.5, old.salience),
        emotion=old.emotion or 0.0,
        t_event=now,
        t_ingest=now,
        role="correction",
        source="notebook",
        author=author,
        meta={"revise_reason": reason, "revised_from": note_id, "is_correction": True},
    )
    sub = TR.submit(item, project=True, cfg=cfg)
    if not sub.ok:
        return {"ok": False, "reason": sub.reason}
    if sub.duplicate:
        return {"ok": True, "new_id": sub.item_id, "old_id": note_id, "duplicate": True}

    with DB.tx(cfg.db_path) as conn:
        conn.execute("UPDATE episode SET superseded_by=? WHERE id=?", (sub.item_id, note_id))

    TR.transition(
        note_id, "revise", target=sub.item_id, reason=reason or "订正",
        actor=IT.ACTOR_MODEL, cfg=cfg,
    )
    return {"ok": True, "new_id": sub.item_id, "old_id": note_id, "note": "旧说法保留但不再浮现"}


# ── 升格 ──────────────────────────────────────────────────────
def promote(cfg: Config, note_id: str, target: str = "persona", section: str = "free", session_id: str = "") -> dict:
    """把笔记升格到沉淀或人设。

    判定收口 `gate.decide()`：PR 硬限制 + 频率（≤2/会话）+ 幅度（≤200 字）+ 来源（必须笔记）
    + 系统不越权（system 产物不能直进 persona/page）。落库与审计由 pipeline.transition 完成。
    """
    ensure_tables(cfg)
    target = (target or "").lower()
    if target not in ("persona", "page"):
        return {"ok": False, "reason": "target 只能是 persona 或 page"}

    with DB.reader(cfg.db_path) as conn:
        r = conn.execute("SELECT * FROM episode WHERE id=?", (note_id,)).fetchone()
    if r is None:
        return {"ok": False, "reason": "笔记不存在"}
    if r["source"] != "notebook":
        return {"ok": False, "reason": "只有自己写的笔记才能升格"}
    if r["superseded_by"]:
        return {"ok": False, "reason": "这条已被订正，请升格新版"}

    item = IT.MemoryItem.from_row(r)
    sid = session_id or item.session_id
    action = "promote_persona" if target == "persona" else "promote_page"
    dec = gate.decide(item, action, target=section, cfg=cfg, session_id=sid)
    if not dec["ok"]:
        return {"ok": False, "reason": dec["reason"], "from": dec["from"], "to": dec["from"]}

    text = item.promotion_text()

    if target == "persona":
        pid = PER.add(cfg, item.session_id, text, section, author=PR.A_SELF, protected=False, priority=60)
        tr = TR.transition(
            note_id, action, target=pid, reason=dec["reason"], actor=IT.ACTOR_MODEL,
            cfg=cfg, gate_decision=dec,
        )
        if not tr.ok:
            return {"ok": False, "reason": tr.reason}
        return {
            "ok": True,
            "target": "persona",
            "id": pid,
            "text": text,
            "from": dec["from"],
            "to": tr.to_state,
            "state": tr.to_state,
            "note": "已写入人格，下一次 build_context 立刻生效",
        }

    # page：并入常驻活文档对应分区
    page_key = "relationship" if section == "relationship" else "user_profile"
    _append_to_page(cfg, item.session_id, page_key, text)
    tr = TR.transition(
        note_id, action, target=page_key, reason=dec["reason"], actor=IT.ACTOR_MODEL,
        cfg=cfg, gate_decision=dec,
    )
    if not tr.ok:
        return {"ok": False, "reason": tr.reason}
    return {
        "ok": True,
        "target": "page",
        "id": page_key,
        "text": text,
        "from": dec["from"],
        "to": tr.to_state,
        "state": tr.to_state,
        "note": "已并入常驻活文档",
    }


def _essence(r) -> str:
    """从笔记里抽出最该进人设的那一句（兼容旧接口，口径在 item.promotion_text）。"""
    return IT.MemoryItem.from_row(r).promotion_text()


def _append_to_page(cfg: Config, session_id: str, key: str, text: str) -> None:
    from . import pages as PG

    PG.ensure_tables(cfg)
    now = time.time()
    with DB.tx(cfg.db_path) as conn:
        row = conn.execute("SELECT body FROM page WHERE session_id=? AND key=?", (session_id, key)).fetchone()
        body = (row["body"] if row else "") or ""
        if text in body:
            return
        body = (body.rstrip("。") + "。" if body else "") + text
        title = PG.PAGE_DEFS.get(key, (key, ""))[0]
        conn.execute(
            """INSERT INTO page(id,session_id,key,title,body,sources,built_at,stale,version)
               VALUES(?,?,?,?,?,'[]',?,0,1)
               ON CONFLICT(session_id,key) DO UPDATE SET body=excluded.body, built_at=excluded.built_at,
                 version=page.version+1""",
            ("pg_" + IDS._digest(session_id, key), session_id, key, title, body, now),  # noqa: SLF001
        )


def revert(cfg: Config, promotion_id: str) -> dict:
    """回滚一次升格。人设条目置 active=0（不删除，撤回本身是历史）。

    `promotion_id` 兼容两种：item_state_log.id（新）或 notebook_promotion.id（旧只读记录）。
    """
    ensure_tables(cfg)
    log_row = None
    try:
        with DB.reader(cfg.db_path) as conn:
            log_row = conn.execute("SELECT * FROM item_state_log WHERE id=?", (int(promotion_id),)).fetchone()
    except (TypeError, ValueError):
        log_row = None

    if log_row is not None:
        item_id = log_row["item_id"]
        target_id = log_row["target"] or ""
        action = log_row["action"]
        if action == "promote_persona" and target_id:
            PER.revoke(cfg, target_id)
        tr = TR.transition(
            item_id, "revert_promotion", target=str(promotion_id),
            reason=f"撤回 {action}", actor=IT.ACTOR_SYSTEM, cfg=cfg,
        )
        return {"ok": tr.ok, "reverted": promotion_id, "target": target_id, "reason": tr.reason}

    # 旧记录：只读兼容，不更新旧表的 reverted 标记；撤回动作改记进 item_state_log
    with DB.reader(cfg.db_path) as conn:
        r = conn.execute("SELECT * FROM notebook_promotion WHERE id=?", (promotion_id,)).fetchone()
    if r is None:
        return {"ok": False, "reason": "记录不存在"}
    if r["reverted"]:
        return {"ok": False, "reason": "已回滚过"}
    if r["target_type"] == "persona":
        PER.revoke(cfg, r["target_id"])
    TR.transition(
        r["note_id"], "revert_promotion", target=str(promotion_id),
        reason=f"撤回旧升格记录 {promotion_id}", actor=IT.ACTOR_SYSTEM, cfg=cfg,
    )
    return {"ok": True, "reverted": promotion_id, "target": r["target_type"]}


# ── 整理 ──────────────────────────────────────────────────────
_INBOX_WHERE = "(state='inbox' OR (state IS NULL AND tier='inbox'))"


def review(cfg: Config, llm, session_id: str, now: float | None = None) -> dict:
    """整理速记层：能提炼的升级，提炼不出的归档。

    这是防"笔记层被流水账塞满"的兜底。状态转移全部走 pipeline.transition
    （旧实现直接 UPDATE archived=1，会绕过状态机与审计）。
    """
    ensure_tables(cfg)
    now = now or time.time()
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            f"""SELECT id, text, meta_json FROM episode
                WHERE session_id=? AND source='notebook' AND {_INBOX_WHERE}
                  AND superseded_by IS NULL AND archived=0
                ORDER BY t_event DESC LIMIT 100""",
            (session_id,),
        ).fetchall()

    if not rows:
        return {"scanned": 0, "promoted": 0, "dropped": 0}

    # 按主题聚一聚：同主题的速记合成一条带推断的笔记
    groups: dict[str, list] = {}
    for r in rows:
        try:
            meta = json.loads(r["meta_json"] or "{}")
        except Exception:
            meta = {}
        groups.setdefault(meta.get("topic") or "_", []).append(r)

    promoted = dropped = 0
    for topic, items in groups.items():
        if len(items) >= 3:
            body = "；".join((i["text"] or "")[:60] for i in items[:5])
            head = items[0]["id"]
            # 三条以上同主题：合成一条，升级进笔记层（origin=system：系统整流产候选）
            merged = f"关于「{topic or '这段时间'}」的几件小事：{body}"
            item = IT.MemoryItem(
                id="ep_merge_" + IDS._digest(head, int(now)),  # noqa: SLF001
                session_id=session_id,
                text=merged,
                origin=IT.ORIGIN_SYSTEM,
                actor=IT.ACTOR_SYSTEM,
                memory_type=PR.T_FACT,
                segments={},
                t_event=now,
                t_ingest=now,
                role="self",
                source="notebook",
                author=PR.A_SELF,
                meta={"merged_from": [i["id"] for i in items], "topic": topic, "is_merge": True},
            )
            sub = TR.submit(item, project=False, cfg=cfg)
            merged_id = sub.item_id if sub.ok else ""
            for i in items:
                TR.transition(
                    i["id"], "review_archive", target=merged_id,
                    reason="同主题合并进笔记层", actor=IT.ACTOR_SYSTEM, cfg=cfg,
                )
            promoted += len(items)
        else:
            # 凑不成组的：归档（可回溯），不是物理丢弃
            for i in items:
                TR.transition(
                    i["id"], "review_archive", reason="提炼不出，归档", actor=IT.ACTOR_SYSTEM, cfg=cfg,
                )
            dropped += len(items)

    return {"scanned": len(rows), "promoted": promoted, "dropped": dropped, "groups": len(groups)}


def list_notes(cfg: Config, session_id: str, tier: str | None = None, limit: int = 50) -> list[dict]:
    ensure_tables(cfg)
    sql = """SELECT id, text, tier, state, origin, salience, emotion, t_event, meta_json, superseded_by
             FROM episode WHERE session_id=? AND source='notebook' AND archived=0"""
    args: list = [session_id]
    if tier:
        sql += " AND tier=?"
        args.append(tier)
    sql += " ORDER BY t_event DESC LIMIT ?"
    args.append(limit)

    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(sql, args).fetchall()

    out = []
    for r in rows:
        try:
            meta = json.loads(r["meta_json"] or "{}")
        except Exception:
            meta = {}
        out.append(
            {
                "id": r["id"],
                "text": r["text"],
                "tier": r["tier"],
                "state": r["state"] or IT.compat_state(r["tier"]),
                "origin": r["origin"] or IT.ORIGIN_TOOL,
                "salience": round(r["salience"], 3),
                "emotion": round(r["emotion"] or 0, 3),
                "superseded": bool(r["superseded_by"]),
                "what": meta.get("what", ""),
                "noticed": meta.get("noticed", ""),
                "means": meta.get("means", ""),
                "next": meta.get("next", ""),
                "topic": meta.get("topic", ""),
            }
        )
    return out


# ── 锦囊：收集与整理分离 ──────────────────────────────────────
def inbox_status(cfg: Config, session_id: str, now: float | None = None) -> dict:
    """锦囊的状态：积压多少、最老一条多久了、该不该整理。

    李贺出门只往锦囊里投，回家才整理。这两件事必须分开——
    要是边走边整理，就没法走路了；要是回家也不整理，锦囊就成了垃圾袋。
    """
    ensure_tables(cfg)
    now = now or time.time()
    with DB.reader(cfg.db_path) as conn:
        row = conn.execute(
            f"""SELECT COUNT(*) c, MIN(t_event) oldest FROM episode
                WHERE session_id=? AND source='notebook' AND {_INBOX_WHERE}
                  AND superseded_by IS NULL AND archived=0""",
            (session_id,),
        ).fetchone()
        count = int(row["c"] or 0)
        oldest = row["oldest"]
        last = conn.execute(
            """SELECT MAX(t_event) t FROM episode
               WHERE session_id=? AND source='notebook'
                 AND (state IN ('note','resident','important') OR (state IS NULL AND tier!='inbox'))""",
            (session_id,),
        ).fetchone()
    age_h = (now - oldest) / 3600 if oldest else 0.0
    last_review_h = (now - (last["t"] or 0)) / 3600 if last and last["t"] else 999.0

    due = []
    if count >= cfg.policy.inbox_max_items:
        due.append(f"积压 {count} 条，超过上限 {cfg.policy.inbox_max_items}")
    if oldest and age_h >= cfg.policy.inbox_max_age_hours:
        due.append(f"最老一条已放 {age_h:.0f} 小时")
    if count and last_review_h >= cfg.policy.review_interval_hours and count >= 5:
        due.append(f"距上次整理 {last_review_h:.0f} 小时")

    return {
        "count": count,
        "oldest_age_hours": round(age_h, 1),
        "hours_since_review": round(min(last_review_h, 999), 1),
        "should_review": bool(due),
        "reasons": due,
    }


def scheduled_review(cfg: Config, llm, session_id: str, now: float | None = None) -> dict:
    """到点就整理。不等到会话结束，也不等她自己想起来。"""
    st = inbox_status(cfg, session_id, now)
    if not st["should_review"]:
        return {"skipped": True, "reason": "还没到整理的点", "status": st}
    rep = review(cfg, llm, session_id, now)
    rep["triggered_by"] = st["reasons"]
    return rep
