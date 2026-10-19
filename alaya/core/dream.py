"""做梦：记忆的加工，不是整理。

五步，对应睡眠对记忆做的四件事 + 一个副产品：

  重放 replay    把当天的记忆再取出来过一遍
  关联 bind      与已有主题/实体建立连接
  抽象 abstract  从多个事件里提炼判断（这是"内化"的原料）
  脱敏 decouple  情绪强度衰减，语义内容不变（这是"接受"的实现）
  成梦 dream     生成一段自我叙事，记录"我是怎么想通的"

关键前提：**梦不是日记**。
日记记"我此刻的感受"，梦产出"我加工后的理解"。前者是原料，后者是产品。
把原料当产品存，就会积累一堆需要反复表达的情绪碎片——那正是祥林嫂的素材库。

所以梦的存储规则：不进检索、不进常驻、只在显式查询时可见。
"""

from __future__ import annotations

import json
import logging
import time

from ..config import Config
from . import db as DB
from . import ids as IDS
from . import item as IT
from . import pipeline as TR
from . import score as SC
from . import text as TX
from .llm import LLM

log = logging.getLogger("alaya.dream")

MIN_INTERVAL_HOURS = 6      # 距上次做梦至少 6 小时
MIN_NEW_ITEMS = 5           # 当天至少 5 条新记忆才值得做
MAX_REPLAY = 30             # 一次最多重放 30 条
DECOUPLE_FACTOR = 0.6       # 情绪衰减系数
DECOUPLE_FLOOR = 0.15       # 情绪下限


def ensure_tables(cfg: Config) -> None:
    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS dream_log (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                dreamed_at  REAL NOT NULL,
                replayed    INTEGER NOT NULL DEFAULT 0,
                bound       INTEGER NOT NULL DEFAULT 0,
                abstracts   INTEGER NOT NULL DEFAULT 0,
                decoupled   INTEGER NOT NULL DEFAULT 0,
                text        TEXT,
                evidence    TEXT
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dream ON dream_log(session_id, dreamed_at)")


def should_dream(cfg: Config, session_id: str, now: float | None = None) -> tuple[bool, str]:
    """该不该做梦。不是每轮都做，也不是会话一结束就做。"""
    ensure_tables(cfg)
    now = now or time.time()

    with DB.reader(cfg.db_path) as conn:
        last = conn.execute(
            "SELECT dreamed_at FROM dream_log WHERE session_id=? ORDER BY dreamed_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if last:
            hours = (now - last["dreamed_at"]) / 3600.0
            if hours < MIN_INTERVAL_HOURS:
                return False, f"距上次做梦 {hours:.1f} 小时，还没到点"

        fresh = conn.execute(
            """SELECT COUNT(*) c FROM episode
               WHERE session_id=? AND t_event > ? AND archived=0
                 AND source IN ('dialogue','notebook')""",
            (session_id, now - 86400),
        ).fetchone()["c"]

    if fresh < MIN_NEW_ITEMS:
        return False, f"当天只有 {fresh} 条新记忆，不足 {MIN_NEW_ITEMS} 条"
    return True, f"可以做梦（{fresh} 条新记忆）"


def dream(cfg: Config, llm: LLM | None, embedder, session_id: str, now: float | None = None) -> dict:
    """做一次梦。返回加工报告。"""
    ensure_tables(cfg)
    now = now or time.time()
    out: dict = {"session_id": session_id, "replayed": 0, "bound": 0, "abstracts": 0, "decoupled": 0}

    # ── 1 重放 ──
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT id, text, emotion, salience, memory_type, meta_json
               FROM episode
               WHERE session_id=? AND t_event > ? AND archived=0
                 AND source IN ('dialogue','notebook')
                 AND salience >= 0.30
               ORDER BY salience DESC, t_event DESC LIMIT ?""",
            (session_id, now - 86400 * 2, MAX_REPLAY),
        ).fetchall()
    if len(rows) < 3:
        out["note"] = "可重放的记忆太少，这一夜没做梦"
        return out
    out["replayed"] = len(rows)

    # ── 2 关联：找出这批记忆共享的实体与主题 ──
    bound, themes = _bind(cfg, [r["id"] for r in rows], session_id)
    out["bound"] = bound
    out["themes"] = themes

    # ── 3 抽象：从这批记忆里提炼判断 ──
    abstracts = _abstract(llm, rows, themes)
    out["abstracts"] = len(abstracts)

    # ── 4 脱敏：情绪衰减，语义不动 ──
    if rows:
        with DB.tx(cfg.db_path) as conn:
            for r in rows:
                old = r["emotion"] or 0.0
                if old <= 0:
                    continue
                new = SC.decouple(old, DECOUPLE_FACTOR, DECOUPLE_FLOOR)
                if abs(new - old) > 1e-6:
                    conn.execute("UPDATE episode SET emotion=? WHERE id=?", (new, r["id"]))
                    out["decoupled"] += 1

    # ── 5 成梦：走统一管线（唯一写入点 + 状态机 + item_state_log）──
    body = _compose_dream(llm, rows, abstracts, themes)
    dream_id = "ep_dream_" + IDS._digest(session_id, int(now))  # noqa: SLF001
    evidence = [r["id"] for r in rows]
    embs = embedder.embed_blob(body) if embedder is not None else None

    dream_item = IT.MemoryItem(
        id=dream_id,
        session_id=session_id,
        text=body,
        origin=IT.ORIGIN_DREAM,
        actor=IT.ACTOR_MODEL,       # 她自己的加工；episode.author 兼容投影为 self
        memory_type="fact",
        salience=0.4,               # 旧口径保留：梦固定低显著度
        emotion=0.0,
        t_event=now,
        t_ingest=now,
        role="dream",
        source="dream",
        embedding=embs,
        meta={
            "evidence": evidence[:20],
            "abstracts": [a["text"] for a in abstracts],
            "themes": themes,
        },
    )
    sub = TR.submit(dream_item, project=False, cfg=cfg, embedder=embedder, llm=llm)
    if not sub.ok:
        out["note"] = f"成梦写入失败：{sub.reason}"
        return out
    if sub.duplicate:
        out["dream_id"] = sub.item_id
        out["note"] = "这一秒的梦已存在，跳过重复落库"
        return out
    dream_id = sub.item_id

    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            """INSERT INTO dream_log(id,session_id,dreamed_at,replayed,bound,abstracts,decoupled,text,evidence)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                "dr_" + IDS._digest(session_id, int(now)),  # noqa: SLF001
                session_id, now, out["replayed"], out["bound"], len(abstracts), out["decoupled"],
                body, json.dumps(evidence[:20]),
            ),
        )

    out["dream_id"] = dream_id
    out["body"] = body
    out["abstracts_detail"] = abstracts
    return out


def _bind(cfg: Config, ids: list[str], session_id: str) -> tuple[int, list[str]]:
    """关联：这批记忆共同出现的实体/主题。

    不做复杂图算法——共同实体就是最直接的关联证据。
    """
    if not ids:
        return 0, []
    marks = ",".join("?" * len(ids))
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            f"""SELECT object_entity_id AS eid, COUNT(*) c FROM fact
                WHERE source_episode IN ({marks}) AND object_entity_id IS NOT NULL
                GROUP BY object_entity_id HAVING c >= 2""",
            ids,
        ).fetchall()
        names = []
        for r in rows:
            n = conn.execute("SELECT name FROM entity WHERE id=?", (r["eid"],)).fetchone()
            if n:
                names.append(n["name"])

        # 超边也已经是一种关联，统计一下被覆盖到几条
        bound = conn.execute(
            f"""SELECT COUNT(DISTINCT edge_id) c FROM hyperedge_member
                WHERE member_type='episode' AND member_id IN ({marks})""",
            ids,
        ).fetchone()["c"]
    return int(bound), names[:6]


_ABSTRACT_SYS = """你在整理一段时间的经历，提炼出**关于对方或关于关系**的判断。

规则：
- 只能基于给的材料，不许补充材料里没有的事实、时间、地点、人物
- 判断要能站得住：材料里至少两处支持它，否则不要提
- 一条判断一句话，不超过 40 字
- 最多 3 条。**如果材料不足以支撑任何判断，返回空数组。**
- 不要写感受，写判断

格式：{"abstracts":[{"text":"判断内容","evidence":"哪几条材料支持"}]}"""


def _abstract(llm: LLM | None, rows, themes: list[str]) -> list[dict]:
    """抽象：从记忆里提炼判断。LLM 不可用时不提炼——宁可没有，不可编造。"""
    if llm is None or llm.provider == "rule":
        return []
    material = "\n".join(f"- {TX.snippets(r['text'], 70)}" for r in rows[:15])
    if themes:
        material += "\n\n共同提到的： " + "、".join(themes)
    data = llm.chat_json(_ABSTRACT_SYS, material, max_tokens=400)
    if not isinstance(data, dict):
        return []
    out = []
    for a in data.get("abstracts") or []:
        if not isinstance(a, dict):
            continue
        t = str(a.get("text", "")).strip()
        if 4 < len(t) <= 60:
            out.append({"text": t, "evidence": str(a.get("evidence", ""))[:80]})
    return out[:3]


def _compose_dream(llm: LLM | None, rows, abstracts: list[dict], themes: list[str]) -> str:
    """成梦：写成一段叙事，记录"我是怎么想通的"。

    这不是日记——它记的是加工过程，不是当时的情绪。
    """
    if llm is not None and llm.provider != "rule" and abstracts:
        material = "\n".join(f"- {a['text']}" for a in abstracts)
        out = llm.chat(
            "用第一人称写一小段（不超过 100 字）内心独白，像是夜里想通了什么。"
            "只写判断的变化过程，不要抒情，不要编造具体事件。",
            material,
            max_tokens=180,
        )
        if out and 10 < len(out.strip()) < 400:
            return out.strip()

    # 无 LLM：退化为陈述式，保住信息不造假
    if abstracts:
        return "想通了几件事：" + "；".join(a["text"] for a in abstracts) + "。"
    if themes:
        return f"今天反复出现的，是「{'、'.join(themes[:3])}」。"
    return "今天的事，先放着。"


def list_dreams(cfg: Config, session_id: str, limit: int = 20) -> list[dict]:
    ensure_tables(cfg)
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM dream_log WHERE session_id=? ORDER BY dreamed_at DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def candidates_for_promotion(cfg: Config, session_id: str, min_occurrences: int = 2) -> list[dict]:
    """升格候选：在至少 N 次梦里反复出现的判断。

    这条取代了 v1 的"提了 3 次就算消化"——那会把最反复、最侵入的记忆
    最先判定为已处理。这里数的是**梦里独立想到过几次**，不是被提及几次。
    """
    ensure_tables(cfg)
    # 抽象产物存在梦条目的 meta_json 里，不重复存一份
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT text, t_event, meta_json FROM episode
               WHERE session_id=? AND source='dream' AND archived=0
               ORDER BY t_event DESC LIMIT 60""",
            (session_id,),
        ).fetchall()

    seen: dict[str, list[float]] = {}
    for r in rows:
        try:
            meta = json.loads(r["meta_json"] or "{}")
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        for t in meta.get("abstracts") or []:
            t = str(t).strip()
            if 4 < len(t) <= 60:
                seen.setdefault(t, []).append(r["t_event"])

    out = []
    for text, times in seen.items():
        if len(times) >= min_occurrences:
            out.append(
                {
                    "text": text,
                    "occurrences": len(times),
                    "first_seen": min(times),
                    "last_seen": max(times),
                    "span_days": round((max(times) - min(times)) / 86400, 1),
                }
            )
    out.sort(key=lambda x: (-x["occurrences"], -x["last_seen"]))
    return out
