"""常驻活文档（Knowledge Pages）与后台归纳（Observations）。

这是把记忆从"被动检索库"变成"主动沉淀"的关键一层。

- **Pages**：预生成的常驻文档，注入时直接读、不检索、不调 LLM。
  解决的是"每轮都要跑一次语义检索才能知道用户是谁"这个成本问题。
- **Observations**：后台把多条零散事实归纳成认知，带证据链和置信度。
  解决的是"AI 只记得一堆碎片，说不出结论"这个质量问题。
"""

from __future__ import annotations

import json
import logging
import time

from ..config import Config
from . import db as DB
from . import ids as IDS
from . import text as TX
from .llm import LLM

log = logging.getLogger("rosmontis.pages")

# 预置页面：key → (标题, 数据来源说明)
PAGE_DEFS = {
    "user_profile": ("关于对方", "从事实层的长期属性归纳"),
    "relationship": ("我们的关系", "从关系类事实与承诺归纳"),
    "recent_state": ("最近的状态", "从近 14 天的事件与状态槽归纳"),
    "chronicle": ("共同经历", "从高重要度事件按时间归纳"),
}

PAGE_TOKEN_CAP = {"user_profile": 220, "relationship": 140, "recent_state": 160, "chronicle": 220}


def ensure_tables(cfg: Config) -> None:
    """页面表。放在这里而不是 schema.sql，方便已有库热升级。"""
    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS page (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                key         TEXT NOT NULL,
                title       TEXT NOT NULL,
                body        TEXT NOT NULL,
                sources     TEXT,
                built_at    REAL NOT NULL,
                stale       INTEGER NOT NULL DEFAULT 0,
                version     INTEGER NOT NULL DEFAULT 1
            )"""
        )
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_page ON page(session_id, key)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS observation (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                text        TEXT NOT NULL,
                kind        TEXT NOT NULL DEFAULT 'pattern',
                evidence    TEXT,
                confidence  REAL NOT NULL DEFAULT 0.6,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL,
                active      INTEGER NOT NULL DEFAULT 1
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_obs ON observation(session_id, active)")


def get_page(cfg: Config, session_id: str, key: str) -> dict | None:
    with DB.reader(cfg.db_path) as conn:
        r = conn.execute("SELECT * FROM page WHERE session_id=? AND key=?", (session_id, key)).fetchone()
    return dict(r) if r else None


def list_pages(cfg: Config, session_id: str) -> list[dict]:
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute("SELECT * FROM page WHERE session_id=? ORDER BY key", (session_id,)).fetchall()
    return [dict(r) for r in rows]


def compile_resident(cfg: Config, session_id: str, budget_tokens: int | None = None) -> dict:
    """编译常驻块：页面内容直接拼接，不检索、不调 LLM。

    这是 token 效率的关键——常驻内容在后台就建好了，每轮只是一次数据库读。
    """
    cap = budget_tokens or cfg.policy.resident_max_tokens
    pages = list_pages(cfg, session_id)
    if not pages:
        return {"text": "", "tokens": 0, "pages": 0, "truncated": False}

    order = ["user_profile", "relationship", "recent_state", "chronicle"]
    pages.sort(key=lambda p: order.index(p["key"]) if p["key"] in order else 9)

    lines = []
    used = 0
    truncated = False
    for p in pages:
        block = f"【{p['title']}】\n{p['body'].strip()}"
        cost = _tok(block)
        if used + cost > cap:
            truncated = True
            continue
        lines.append(block)
        used += cost

    return {"text": "\n".join(lines), "tokens": used, "pages": len(lines), "truncated": truncated}


def rebuild_pages(cfg: Config, llm: LLM | None, session_id: str, now: float | None = None) -> dict:
    """重建所有常驻页面。

    LLM 可用时由模型归纳；不可用时用模板拼接（信息不丢，只是不够凝练）。
    """
    ensure_tables(cfg)
    now = now or time.time()
    built = []
    for key, (title, _src) in PAGE_DEFS.items():
        body, sources = _build_one(cfg, llm, session_id, key, now)
        if not body:
            continue
        pid = "pg_" + IDS._digest(session_id, key)  # noqa: SLF001
        with DB.tx(cfg.db_path) as conn:
            conn.execute(
                """INSERT INTO page(id,session_id,key,title,body,sources,built_at,stale,version)
                   VALUES(?,?,?,?,?,?,?,0,1)
                   ON CONFLICT(session_id,key) DO UPDATE SET
                     body=excluded.body, sources=excluded.sources, built_at=excluded.built_at,
                     stale=0, version=page.version+1""",
                (pid, session_id, key, title, body, json.dumps(sources[:10]), now),
            )
        built.append({"key": key, "chars": len(body), "sources": len(sources)})
    return {"built": built, "count": len(built)}


def _build_one(cfg: Config, llm: LLM | None, session_id: str, key: str, now: float) -> tuple[str, list[str]]:
    facts, episodes, sources = _gather_material(cfg, session_id, key, now)
    if not facts and not episodes:
        return "", []

    if llm is not None and llm.provider != "rule":
        prompt_body = _material_text(facts, episodes)
        sys = _page_prompt(key)
        if sys:
            out = llm.chat(sys, prompt_body, max_tokens=320)
            if out and len(out.strip()) > 10:
                body = out.strip()[:1200]
                if _looks_fabricated(body, prompt_body):
                    # 编造就退回模板：宁可平淡，不可失真
                    return _template_page(key, facts, episodes), sources
                return body, sources

    return _template_page(key, facts, episodes), sources


def _page_prompt(key: str) -> str:
    base = (
        "你在整理一份关于某人/某段关系的长期档案。"
        "**只能用材料里出现过的信息。材料里没写的具体细节（时间、地点、物件、人物）一律不许出现。**"
        "如果材料不足以写出连贯段落，就直接复述材料里的短句，不要补充。"
        "用第三人称，不超过 120 字，写成连贯的短段落，不要用列表。"
    )
    extra = {
        "user_profile": "主题：这个人的身份、工作、住处、喜好与忌讳。",
        "relationship": "主题：我们之间的关系、约定、彼此在意的事。",
        "recent_state": "主题：最近两周他/她的状态与在做的事。",
        "chronicle": "主题：我们共同经历过的重要事件，按时间顺序。",
    }.get(key, "")
    return base + extra


def _gather_material(cfg: Config, session_id: str, key: str, now: float):
    """不同的页面取不同材料。"""
    facts: list[str] = []
    episodes: list[str] = []
    sources: list[str] = []
    with DB.reader(cfg.db_path) as conn:
        if key in ("user_profile", "relationship"):
            rows = conn.execute(
                """SELECT f.id, f.predicate, f.object_text, f.polarity, e.name AS subj
                   FROM fact f LEFT JOIN entity e ON e.id=f.subject_id
                   WHERE f.session_id=? AND f.status='active'
                   ORDER BY f.salience DESC LIMIT 40""",
                (session_id,),
            ).fetchall()
            for r in rows:
                # 画像页要全部属性；关系页只取关系类，避免两页内容一样
                if key == "relationship" and r["predicate"] not in ("关系是", "叫", "有"):
                    if "约" not in (r["object_text"] or "") and "承诺" not in (r["object_text"] or ""):
                        continue
                neg = "不" if r["polarity"] < 0 else ""
                facts.append(f"{r['subj']} {neg}{r['predicate']} {r['object_text']}")
                sources.append(r["id"])

        elif key == "recent_state":
            rows = conn.execute(
                """SELECT s.slot_key, s.value_text, e.name AS scope
                   FROM state_slot s LEFT JOIN entity e ON e.id=s.scope
                   WHERE s.session_id=? AND s.until IS NULL LIMIT 12""",
                (session_id,),
            ).fetchall()
            for r in rows:
                facts.append(f"{r['scope']} 的 {r['slot_key']} 是 {r['value_text']}")
            eps = conn.execute(
                """SELECT id, text FROM episode
                   WHERE session_id=? AND t_event > ? AND archived=0 AND folded_into IS NULL
                   ORDER BY t_event DESC LIMIT 8""",
                (session_id, now - 86400 * 14),
            ).fetchall()
            for e in eps:
                episodes.append(TX.snippets(e["text"], 60))
                sources.append(e["id"])

        elif key == "chronicle":
            rows = conn.execute(
                """SELECT id, text, t_event FROM episode
                   WHERE session_id=? AND salience >= 0.6 AND archived=0 AND folded_into IS NULL
                   ORDER BY t_event DESC LIMIT 12""",
                (session_id,),
            ).fetchall()
            for r in rows:
                episodes.append(TX.snippets(r["text"], 60))
                sources.append(r["id"])

    return facts, episodes, sources


def _material_text(facts: list[str], episodes: list[str]) -> str:
    parts = []
    if facts:
        parts.append("已知事实：\n" + "\n".join(f"- {f}" for f in facts[:20]))
    if episodes:
        parts.append("相关片段：\n" + "\n".join(f"- {e}" for e in episodes[:10]))
    return "\n\n".join(parts)


def _template_page(key: str, facts: list[str], episodes: list[str]) -> str:
    """无 LLM 时的模板。用自然句式而不是分号拼接——分号拼出来像报表，不像记忆。"""
    if key == "user_profile" and facts:
        head, rest = facts[0], facts[1:6]
        out = head + "。"
        if rest:
            out += "另外，" + "，".join(rest) + "。"
        return out
    if key == "relationship" and facts:
        return "，".join(facts[:4]) + "。"
    if key == "recent_state":
        bits = facts[:3] + episodes[:2]
        return "这段时间，" + "；".join(bits) + "。" if bits else ""
    if key == "chronicle" and episodes:
        return "记得" + "；也记得".join(episodes[:4]) + "。"
    return ""


def _looks_fabricated(body: str, material: str) -> bool:
    """粗检编造：把正文里的具体细节（数字、年份、常见场景词）回到材料里核对。

    这是个筛子，不是判据。宁可放弃 LLM 生成的段落退回模板，也不要让编造的
    细节每轮注入——评审里就撞见过把"他今天状态很差"扩写成"下午在办公室
    整理文件时感到疲惫"。
    """
    import re

    for n in re.findall(r"\d+", body):
        if n not in material:
            return True
    scene_words = ("办公室", "公司里", "同事", "会议室", "地铁", "公交车", "文件", "会议室")
    for w in scene_words:
        if w in body and w not in material:
            return True
    return False


# ── Observations：后台归纳 ────────────────────────────────────
def add_observation(cfg: Config, session_id: str, text: str, evidence: list[str], confidence: float = 0.6, kind: str = "pattern") -> str:
    ensure_tables(cfg)
    now = time.time()
    oid = "ob_" + IDS._digest(session_id, text)  # noqa: SLF001
    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            """INSERT INTO observation(id,session_id,text,kind,evidence,confidence,created_at,updated_at,active)
               VALUES(?,?,?,?,?,?,?,?,1)
               ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,
                 confidence=MAX(observation.confidence, excluded.confidence)""",
            (oid, session_id, text.strip(), kind, json.dumps(evidence[:12]), confidence, now, now),
        )
    return oid


def list_observations(cfg: Config, session_id: str, limit: int = 50) -> list[dict]:
    ensure_tables(cfg)
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM observation WHERE session_id=? AND active=1
               ORDER BY confidence DESC, updated_at DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def derive_observations(cfg: Config, llm: LLM | None, session_id: str, now: float | None = None) -> dict:
    """后台归纳：把零散事实凝练成带证据的认知。

    没有 LLM 时用规则：同一谓词出现 3 次以上就归成一条模式。
    """
    ensure_tables(cfg)
    now = now or time.time()
    made = 0

    with DB.reader(cfg.db_path) as conn:
        groups = conn.execute(
            """SELECT predicate, COUNT(*) c, GROUP_CONCAT(object_text, '|') objs,
                      GROUP_CONCAT(id, '|') ids
               FROM fact WHERE session_id=? AND status='active'
               GROUP BY predicate HAVING c >= 3""",
            (session_id,),
        ).fetchall()

    for g in groups:
        objs = (g["objs"] or "").split("|")[:8]
        ids = (g["ids"] or "").split("|")[:8]
        text = f"在「{g['predicate']}」上有多条相关记录：{('、'.join(objs))[:80]}"
        add_observation(cfg, session_id, text, ids, confidence=min(0.85, 0.5 + 0.05 * g["c"]), kind="pattern")
        made += 1

    if llm is not None and llm.provider != "rule" and made > 0:
        obs = list_observations(cfg, session_id, 20)
        material = "\n".join(f"- {o['text']}" for o in obs)
        out = llm.chat(
            "把下面这些零散观察归纳成 2-3 条更凝练的认知，每条一句话，不要编造材料里没有的内容。"
            "每行一条，不要编号。",
            material,
            max_tokens=200,
        )
        if out:
            for line in out.splitlines():
                line = line.strip("-· \t")
                if 6 < len(line) < 120:
                    add_observation(cfg, session_id, line, [], confidence=0.7, kind="insight")
                    made += 1

    return {"made": made, "total": len(list_observations(cfg, session_id))}


def _tok(s: str) -> int:
    from .score import estimate_tokens

    return estimate_tokens(s)
