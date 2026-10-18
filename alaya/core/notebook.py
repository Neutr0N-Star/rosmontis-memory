"""主动记忆：她自己写、自己改、自己升格。

三重寄生，不新建平行系统：
- 笔记就是 episode（`source='notebook'`），自动继承三路召回、三空间衰减、ACT-R 激活
- 速记与笔记的区别就是 `tier='inbox'` vs `tier='retrieval'`
- 升格到沉淀/人设，走已有的 pages 与 persona

防流水账的硬闸：只有 `what` 的条目不许进笔记层。
"""

from __future__ import annotations

import json
import logging
import time

from ..config import Config
from . import db as DB
from . import ids as IDS
from . import persona as PER
from . import promotion as PR
from . import score as SC
from . import text as TX

log = logging.getLogger("alaya.notebook")

# 人设改动的闸门
MAX_PROMOTIONS_PER_SESSION = 2
MAX_PERSONA_CHARS = 200


def ensure_tables(cfg: Config) -> None:
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
    """写一条笔记。

    四段：what / noticed / means / next。
    **只有 what 的条目进速记层（inbox），不进笔记层。** 这是防流水账的硬闸。
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

    tier = PR.TIER_RETRIEVAL if has_second_layer else "inbox"
    body = _compose(what, noticed, means, next_step)
    emo = SC.emotion_intensity(body)
    salience = 0.55 + (0.12 if means else 0) + (0.10 if next_step else 0) + emo * 0.15

    ep_id = "ep_nb_" + IDS._digest(session_id, body, int(t_event))  # noqa: SLF001
    emb = embedder.embed_blob(body) if embedder is not None else None

    with DB.tx(cfg.db_path) as conn:
        exists = conn.execute("SELECT id FROM episode WHERE id=?", (ep_id,)).fetchone()
        if exists:
            return {"ok": True, "id": ep_id, "duplicate": True, "tier": tier}

        conn.execute(
            """INSERT INTO episode
               (id,session_id,seq,role,text,t_event,t_ingest,source,tier,memory_type,author,
                salience,emotion,embedding,meta_json)
               VALUES(?,?,0,'self',?,?,?,'notebook',?,'fact','self',?,?,?,?)""",
            (
                ep_id, session_id, body, t_event, now, tier,
                min(1.0, salience), emo, emb,
                json.dumps(
                    {"what": what, "noticed": noticed, "means": means, "next": next_step, "topic": topic},
                    ensure_ascii=False,
                ),
            ),
        )
        conn.execute(
            "INSERT INTO episode_fts(episode_id,tokens) VALUES(?,?)",
            (ep_id, " ".join(TX.tokenize(body + " " + topic))),
        )

    return {
        "ok": True,
        "id": ep_id,
        "tier": tier,
        "has_second_layer": has_second_layer,
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
    """订正一条笔记。复用 episode 的 correction 机制：原文不删，新条指回旧条。"""
    ensure_tables(cfg)
    now = time.time()
    with DB.reader(cfg.db_path) as conn:
        r = conn.execute("SELECT * FROM episode WHERE id=?", (note_id,)).fetchone()
    if r is None:
        return {"ok": False, "reason": "笔记不存在"}
    if r["source"] != "notebook":
        return {"ok": False, "reason": "只能订正自己写的笔记"}

    new_id = "ep_corr_" + IDS._digest(note_id, new_text, int(now))  # noqa: SLF001
    meta = {}
    try:
        meta = json.loads(r["meta_json"] or "{}")
    except Exception:
        pass
    meta["revise_reason"] = reason
    meta["revised_from"] = note_id

    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO episode
               (id,session_id,seq,role,text,t_event,t_ingest,source,tier,memory_type,author,
                salience,emotion,correction_of,meta_json)
               VALUES(?,?,?,'self',?,?,?,'notebook',?,'fact',?,?,?,?,?)""",
            (
                new_id, r["session_id"], r["seq"], new_text, now, now,
                r["tier"], author, max(0.5, r["salience"]), r["emotion"] or 0.0, note_id,
                json.dumps(meta, ensure_ascii=False),
            ),
        )
        conn.execute("UPDATE episode SET superseded_by=? WHERE id=?", (new_id, note_id))
        conn.execute(
            "INSERT INTO episode_fts(episode_id,tokens) VALUES(?,?)",
            (new_id, " ".join(TX.tokenize(new_text))),
        )
    return {"ok": True, "new_id": new_id, "old_id": note_id, "note": "旧说法保留但不再浮现"}


# ── 升格 ──────────────────────────────────────────────────────
def promote(cfg: Config, note_id: str, target: str = "persona", section: str = "free", session_id: str = "") -> dict:
    """把笔记升格到沉淀或人设。

    三道闸：频率（单次会话 ≤2）、幅度（≤200 字）、来源（必须是笔记）。
    """
    ensure_tables(cfg)
    now = time.time()
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
    sid = session_id or r["session_id"]

    text = _essence(r)
    if len(text) > MAX_PERSONA_CHARS:
        return {
            "ok": False,
            "reason": f"单条升格内容超过 {MAX_PERSONA_CHARS} 字（当前 {len(text)}），先精简",
        }

    if target == "persona":
        n = _promotions_in_session(cfg, sid, "persona")
        if n >= MAX_PROMOTIONS_PER_SESSION:
            return {
                "ok": False,
                "reason": f"单次会话最多改 {MAX_PROMOTIONS_PER_SESSION} 次人设（已用 {n} 次）。人格不该每天推翻",
            }
        pid = PER.add(cfg, r["session_id"], text, section, author=PR.A_SELF, protected=False, priority=60)
        _record(cfg, sid, note_id, "persona", pid, {"text": text, "section": section})
        return {"ok": True, "target": "persona", "id": pid, "text": text, "note": "已写入人格，下一次 build_context 立刻生效"}

    # page：并入常驻活文档对应分区
    page_key = "relationship" if section == "relationship" else "user_profile"
    _append_to_page(cfg, r["session_id"], page_key, text)
    _record(cfg, sid, note_id, "page", page_key, {"text": text})
    return {"ok": True, "target": "page", "id": page_key, "text": text, "note": "已并入常驻活文档"}


def _essence(r) -> str:
    """从笔记里抽出最该进人设的那一句。

    优先 next（给未来的指令），其次 means（推断），最后 what。
    人设里放的是"我该怎么做"，不是"发生过什么"。
    """
    try:
        meta = json.loads(r["meta_json"] or "{}")
    except Exception:
        meta = {}
    for key in ("next", "means", "noticed", "what"):
        v = (meta.get(key) or "").strip()
        if v:
            return v
    return (r["text"] or "").strip()


def _promotions_in_session(cfg: Config, session_id: str, target_type: str) -> int:
    with DB.reader(cfg.db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(*) c FROM notebook_promotion
               WHERE session_id=? AND target_type=? AND reverted=0""",
            (session_id, target_type),
        ).fetchone()
    return int(row["c"]) if row else 0


def _record(cfg: Config, session_id: str, note_id: str, target_type: str, target_id: str, diff: dict) -> None:
    now = time.time()
    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            """INSERT INTO notebook_promotion(id,session_id,note_id,target_type,target_id,promoted_at,diff)
               VALUES(?,?,?,?,?,?,?)""",
            (
                "np_" + IDS._digest(note_id, target_id, int(now)),  # noqa: SLF001
                session_id, note_id, target_type, target_id, now,
                json.dumps(diff, ensure_ascii=False),
            ),
        )


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
    """回滚一次升格。人设条目置 active=0（不删除，撤回本身是历史）。"""
    ensure_tables(cfg)
    with DB.reader(cfg.db_path) as conn:
        r = conn.execute("SELECT * FROM notebook_promotion WHERE id=?", (promotion_id,)).fetchone()
    if r is None:
        return {"ok": False, "reason": "记录不存在"}
    if r["reverted"]:
        return {"ok": False, "reason": "已回滚过"}

    if r["target_type"] == "persona":
        PER.revoke(cfg, r["target_id"])

    with DB.tx(cfg.db_path) as conn:
        conn.execute("UPDATE notebook_promotion SET reverted=1 WHERE id=?", (promotion_id,))
    return {"ok": True, "reverted": promotion_id, "target": r["target_type"]}


# ── 整理 ──────────────────────────────────────────────────────
def review(cfg: Config, llm, session_id: str, now: float | None = None) -> dict:
    """整理速记层：能提炼的升级，提炼不出的丢弃。

    这是防"笔记层被流水账塞满"的兜底。没有它，写入闸只是把问题延后。
    """
    ensure_tables(cfg)
    now = now or time.time()
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT id, text, meta_json FROM episode
               WHERE session_id=? AND source='notebook' AND tier='inbox'
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
    with DB.tx(cfg.db_path) as conn:
        for topic, items in groups.items():
            if len(items) >= 3:
                body = "；".join((i["text"] or "")[:60] for i in items[:5])
                head = items[0]["id"]
                # 三条以上同主题：合成一条，升级进笔记层
                merged = f"关于「{topic or '这段时间'}」的几件小事：{body}"
                conn.execute(
                    """INSERT OR IGNORE INTO episode
                       (id,session_id,seq,role,text,t_event,t_ingest,source,tier,memory_type,author,salience,meta_json)
                       VALUES(?,?,0,'self',?,?,?,'notebook','retrieval','fact','self',0.5,?)""",
                    (
                        "ep_merge_" + IDS._digest(head, now), session_id, merged, now, now,  # noqa: SLF001
                        json.dumps({"merged_from": [i["id"] for i in items], "topic": topic}, ensure_ascii=False),
                    ),
                )
                for i in items:
                    conn.execute("UPDATE episode SET archived=1 WHERE id=?", (i["id"],))
                promoted += len(items)
            else:
                # 凑不成组的：直接丢。速记是消耗品
                for i in items:
                    conn.execute("UPDATE episode SET archived=1 WHERE id=?", (i["id"],))
                dropped += len(items)

    return {"scanned": len(rows), "promoted": promoted, "dropped": dropped, "groups": len(groups)}


def list_notes(cfg: Config, session_id: str, tier: str | None = None, limit: int = 50) -> list[dict]:
    ensure_tables(cfg)
    sql = """SELECT id, text, tier, salience, emotion, t_event, meta_json, superseded_by
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
            """SELECT COUNT(*) c, MIN(t_event) oldest FROM episode
               WHERE session_id=? AND source='notebook' AND tier='inbox'
                 AND superseded_by IS NULL AND archived=0""",
            (session_id,),
        ).fetchone()
        count = int(row["c"] or 0)
        oldest = row["oldest"]
        last = conn.execute(
            "SELECT MAX(t_event) t FROM episode WHERE session_id=? AND source='notebook' AND tier='retrieval'",
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
