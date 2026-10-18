"""写入管线：一次写入是一次状态转移，不是一次 append。

流程见 docs/03 §4.1。幂等、事务化、失败不留半条记忆。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field

from ..config import Config
from . import db as DB
from . import extract as EX
from . import ids as IDS
from . import promotion as PR
from . import score as SC
from . import text as TX
from . import timeparse
from . import vec as V
from .embed import Embedder
from .llm import LLM

log = logging.getLogger("alaya.write")


@dataclass
class IngestResult:
    episode_id: str
    duplicate: bool = False
    added: int = 0
    updated: int = 0
    invalidated: int = 0
    noop: int = 0
    entities_new: int = 0
    states_changed: int = 0
    edges_touched: int = 0
    salience: float = 0.0
    source: str = "rule"
    note: str | None = None
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def ensure_session(cfg: Config, conn, session_id: str, agent_id: str = "default", user_id: str = "default", now: float | None = None) -> None:
    now = now or time.time()
    conn.execute(
        """INSERT INTO session(id,agent_id,user_id,created_at,last_active) VALUES(?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET last_active=excluded.last_active""",
        (session_id, agent_id, user_id, now, now),
    )


def ingest(
    cfg: Config,
    embedder: Embedder,
    llm: LLM | None,
    text: str,
    session_id: str = "default",
    role: str = "user",
    t_event: float | None = None,
    now: float | None = None,
    auto_forget: bool | None = None,
    skip_extract: bool = False,
) -> IngestResult:
    now = now or time.time()
    t_event = t_event or now
    text = (text or "").strip()
    if not text:
        return IngestResult(episode_id="", note="empty")

    ep_id = IDS.episode_id(session_id, role, text, t_event)

    # ── 幂等检查 ──
    with DB.reader(cfg.db_path) as conn:
        row = conn.execute("SELECT id FROM episode WHERE id=?", (ep_id,)).fetchone()
        if row:
            return IngestResult(episode_id=ep_id, duplicate=True, note="已存在，跳过")

        # seq 不在这里算——并发时会读到同一个值。真正取号在事务里。

    # ── 抽取（事务外，因为要调 LLM，避免长事务锁库）──
    if skip_extract:
        ext = EX.Extraction(source="deferred")
    else:
        ext = EX.extract(text, llm, now, role=role, use_llm=cfg.use_llm_extract)

    # 降级时不要把哈希伪向量写进库。它不是语义向量，写进去之后
    # 服务恢复了这些行也永远检索不准，而 `degraded` 标记早就翻回 false 了。
    # 宁可这一轮没有向量（靠全文和图兜底），也不要留一条永久的坏向量。
    vec = embedder.embed(text)
    emb = None if embedder.degraded else V.pack(vec)
    emo = SC.emotion_intensity(text, is_session_tail=False)
    salience = _salience(text, emo, ext, role)

    # 特化维度：作者、来源、类型、层级
    author = PR.A_TEACHER if role == "user" else (PR.A_SELF if role == "assistant" else PR.A_SYSTEM)
    source = PR.S_DIALOGUE
    mtype = PR.classify_type(text, source)
    tier = PR.default_tier(source, mtype, author)
    okc, why = PR.validate_combo(tier, mtype)
    if not okc:
        tier = PR.TIER_RETRIEVAL
        ext.note = (ext.note or "") + f"|{why}"

    result = IngestResult(episode_id=ep_id, salience=salience, source=ext.source, note=ext.note)

    with DB.tx(cfg.db_path) as conn:
        ensure_session(cfg, conn, session_id, now=now)

        # 取号必须在事务内：并发写入时多个请求会读到同一个 MAX，
        # 结果 seq 全是 1。这一步之前放在 reader 里，是错的。
        seq_row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS m FROM episode WHERE session_id=?", (session_id,)
        ).fetchone()
        seq = int(seq_row["m"] or 0) + 1

        conn.execute(
            """INSERT INTO episode
               (id,session_id,seq,role,text,t_event,t_ingest,source,tier,memory_type,author,
                salience,emotion,embedding)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ep_id, session_id, seq, role, text, t_event, now,
                source, tier, mtype, author, salience, emo, emb,
            ),
        )
        conn.execute(
            "INSERT INTO episode_fts(episode_id,tokens) VALUES(?,?)",
            (ep_id, " ".join(TX.tokenize(text))),
        )

        ent_cache: dict[str, str] = {}
        for e in ext.entities:
            eid, created = _upsert_entity(conn, session_id, e.name, e.type, t_event, now)
            ent_cache[IDS.normalize(e.name)] = eid
            if created:
                result.entities_new += 1

        for f in ext.facts:
            act = _resolve_fact(cfg, conn, session_id, f, ep_id, ent_cache, t_event, now, embedder)
            result.details.append(act)
            kind = act.get("action")
            if kind == "ADD":
                result.added += 1
            elif kind == "UPDATE":
                result.updated += 1
            elif kind == "INVALIDATE":
                result.invalidated += 1
            else:
                result.noop += 1

        for s in ext.state_changes:
            if _apply_state(conn, session_id, s, ent_cache, t_event, now):
                result.states_changed += 1

        touched = _group_theme(cfg, conn, session_id, ext, ep_id, ent_cache, t_event, now)
        result.edges_touched = touched

        conn.execute(
            "UPDATE session SET last_active=? WHERE id=?", (now, session_id)
        )

    if auto_forget if auto_forget is not None else cfg.auto_forget:
        if result.added + result.updated > 0:
            _maybe_forget(cfg, session_id, now)

    return result


# ── 实体 ──────────────────────────────────────────────────────
def _upsert_entity(conn, session_id: str, name: str, etype: str, t_event: float, now: float) -> tuple[str, bool]:
    norm = IDS.normalize(name)
    if not norm:
        return "", False
    eid = IDS.entity_id(session_id, norm)
    row = conn.execute("SELECT id, mentions FROM entity WHERE id=?", (eid,)).fetchone()
    if row:
        conn.execute(
            "UPDATE entity SET mentions=mentions+1, last_seen=MAX(last_seen,?) WHERE id=?",
            (t_event, eid),
        )
        return eid, False
    conn.execute(
        """INSERT INTO entity(id,session_id,name,norm_name,type,aliases,attrs,salience,first_seen,last_seen,mentions)
           VALUES(?,?,?,?,?,'[]','{}',0.5,?,?,1)""",
        (eid, session_id, name, norm, etype, t_event, t_event),
    )
    return eid, True


# ── 事实消解 ──────────────────────────────────────────────────
def _resolve_fact(
    cfg: Config,
    conn,
    session_id: str,
    f: EX.FactCand,
    ep_id: str,
    ent_cache: dict[str, str],
    t_event: float,
    now: float,
    embedder=None,
) -> dict:
    # 谓词归一：LLM 会自由发挥（"搬到城东"→搬到），不归一的话
    # STATE_PREDICATES 查不到，状态槽就静默丢了。
    while f.predicate in EX._PREDICATE_ALIASES:
        f.predicate = EX._PREDICATE_ALIASES[f.predicate]
    # 事实也必须有向量，否则它在稠密召回里永远不存在——只能靠全文或图谱被捞到
    emb = None
    if embedder is not None:
        try:
            emb = embedder.embed_blob(f"{f.predicate} {f.obj}")
        except Exception:
            emb = None
    subj_norm = IDS.normalize(f.subject)
    subj_id = ent_cache.get(subj_norm)
    if not subj_id:
        subj_id, _ = _upsert_entity(conn, session_id, f.subject, "person", t_event, now)
        ent_cache[subj_norm] = subj_id

    obj_norm = IDS.normalize(f.obj)
    obj_ent = ent_cache.get(obj_norm)

    # 世界时间起点：能从原文解析出时间就用它，否则用事件时间
    vfrom = t_event
    if f.valid_from_text:
        span = timeparse.parse(str(f.valid_from_text), now)
        if span:
            vfrom = span.start

    rows = conn.execute(
        """SELECT * FROM fact
           WHERE session_id=? AND subject_id=? AND predicate=? AND status='active'
           ORDER BY t_valid_from DESC LIMIT 12""",
        (session_id, subj_id, f.predicate),
    ).fetchall()

    same = [r for r in rows if IDS.normalize(r["object_text"]) == obj_norm]
    same_pol = [r for r in same if int(r["polarity"]) == int(f.polarity)]

    if same_pol:
        best = same_pol[0]
        conn.execute(
            "UPDATE fact SET salience=MIN(1.0, salience+0.05), confidence=MAX(confidence,?) WHERE id=?",
            (f.confidence, best["id"]),
        )
        return {"action": "NOOP", "fact": best["id"], "reason": "同主语同谓词同客体已存在", "predicate": f.predicate}

    if f.predicate in EX.STATE_PREDICATES and rows:
        # 状态类谓词：不同客体视为状态变更 → 旧条作废
        for r in rows:
            _invalidate(conn, r["id"], t_event, now, valid_to=vfrom)
        new_id = _insert_fact(conn, session_id, subj_id, f, obj_ent, vfrom, ep_id, t_event, emb)
        return {
            "action": "UPDATE",
            "fact": new_id,
            "superseded": [r["id"] for r in rows],
            "reason": f"{f.predicate} 发生变更",
            "predicate": f.predicate,
        }

    opposite = [r for r in same if int(r["polarity"]) != int(f.polarity)]
    if opposite:
        # 极性反转：旧条作废（保留原极性，历史不可篡改），新条另立
        for r in opposite:
            _invalidate(conn, r["id"], t_event, now, valid_to=vfrom)
        new_id = _insert_fact(conn, session_id, subj_id, f, obj_ent, vfrom, ep_id, t_event, emb)
        return {
            "action": "INVALIDATE",
            "fact": new_id,
            "over": [r["id"] for r in opposite],
            "reason": "极性反转",
            "predicate": f.predicate,
        }

    new_id = _insert_fact(conn, session_id, subj_id, f, obj_ent, vfrom, ep_id, t_event, emb)

    # 确定性兜底：位置/工作/关系类事实直接驱动状态槽。
    # 不能指望 LLM 记得产出 state_changes——它经常忘，而槽是检索的主干，
    # 漏了比漏一条 fact 严重得多。「住在城东」本身就是 city=城东 的变更。
    if f.predicate in EX.STATE_PREDICATES and int(f.polarity) == 1:
        _apply_state(
            conn, session_id,
            EX.StateCand(scope=f.subject, slot=EX.STATE_PREDICATES[f.predicate], value=f.obj),
            ent_cache, t_event, now,
        )

    return {"action": "ADD", "fact": new_id, "predicate": f.predicate}


def _insert_fact(conn, session_id: str, subj_id: str, f: EX.FactCand, obj_ent, vfrom: float, ep_id: str, now: float, emb: bytes | None = None) -> str:
    fid = IDS.fact_id(session_id, IDS.normalize(f.subject), f.predicate, f.obj, vfrom, f.polarity)
    conn.execute(
        """INSERT OR IGNORE INTO fact
           (id,session_id,subject_id,predicate,object_text,object_entity_id,polarity,confidence,
            t_valid_from,t_valid_to,t_ingest,t_invalid,status,source_episode,salience,embedding)
           VALUES(?,?,?,?,?,?,?,?,?,NULL,?,NULL,'active',?,?,?)""",
        (
            fid, session_id, subj_id, f.predicate, f.obj, obj_ent,
            f.polarity, f.confidence, vfrom, now, ep_id,
            min(1.0, 0.45 + 0.35 * f.confidence), emb,
        ),
    )
    conn.execute(
        "INSERT INTO fact_fts(fact_id,tokens) VALUES(?,?)",
        (fid, " ".join(TX.tokenize(f"{f.predicate} {f.obj}"))),
    )
    return fid


def _invalidate(conn, fact_id: str, t_event: float, now: float, valid_to: float | None = None) -> None:
    """作废一条事实。

    `t_invalid` 是系统时间（我们何时判定它失效），`t_valid_to` 是世界时间
    （它在真实世界里何时不再为真）。两者分开：前者用 now，后者用**新事实的
    起始时刻**——用事件时间会产生"旧值和新值同时为真"的窗口。
    """
    conn.execute(
        """UPDATE fact SET status='invalidated', t_invalid=?,
                          t_valid_to=? WHERE id=?""",
        (now, valid_to if valid_to is not None else t_event, fact_id),
    )


# ── 状态槽 ────────────────────────────────────────────────────
def _apply_state(conn, session_id: str, s: EX.StateCand, ent_cache: dict[str, str], t_event: float, now: float) -> bool:
    scope_norm = IDS.normalize(s.scope)
    scope_id = ent_cache.get(scope_norm)
    if not scope_id:
        scope_id, _ = _upsert_entity(conn, session_id, s.scope, "person", t_event, now)
        ent_cache[scope_norm] = scope_id

    cur = conn.execute(
        """SELECT * FROM state_slot
           WHERE session_id=? AND scope=? AND slot_key=? AND until IS NULL
           ORDER BY since DESC LIMIT 1""",
        (session_id, scope_id, s.slot),
    ).fetchone()

    if cur and IDS.normalize(cur["value_text"]) == IDS.normalize(s.value):
        return False

    if cur:
        conn.execute("UPDATE state_slot SET until=? WHERE id=?", (t_event, cur["id"]))

    sid_slot = IDS.slot_id(session_id, scope_id, s.slot, t_event, s.value)
    sql = """INSERT INTO state_slot(id,session_id,scope,slot_key,value_text,since,until,confidence)
             VALUES(?,?,?,?,?,?,NULL,0.75)"""
    try:
        conn.execute(sql, (sid_slot, session_id, scope_id, s.slot, s.value, t_event))
    except sqlite3.IntegrityError:
        # 回填或同毫秒重入时 id 可能撞上已关闭的旧槽。
        # 不能 OR IGNORE（那会静默丢掉整条状态），换一个 id 再插。
        alt = "sl_" + IDS._digest(sid_slot, t_event, now)
        conn.execute(sql, (alt, session_id, scope_id, s.slot, s.value, t_event))
    return True


# ── 主题超边 ──────────────────────────────────────────────────
def _group_theme(
    cfg: Config,
    conn,
    session_id: str,
    ext: EX.Extraction,
    ep_id: str,
    ent_cache: dict[str, str],
    t_event: float,
    now: float,
) -> int:
    """把这条 episode 挂到一个已有的主题超边上，或新建一个。

    归组判据：实体重叠数 + 时间接近度。不额外调 LLM。
    """
    if not ext.entities and not ext.theme:
        return 0

    ent_ids = [ent_cache.get(IDS.normalize(e.name)) for e in ext.entities]
    ent_ids = [e for e in ent_ids if e]

    best_edge = None
    best_score = 0.0
    rows = conn.execute(
        """SELECT h.* FROM hyperedge h
           WHERE h.session_id=? AND (h.t_to IS NULL OR h.t_to > ?)
           ORDER BY h.created_at DESC LIMIT 40""",
        (session_id, t_event - 86400 * 60),
    ).fetchall()

    for r in rows:
        if ent_ids:
            marks = ",".join("?" * len(ent_ids))
            overlap = conn.execute(
                f"""SELECT COUNT(*) c FROM hyperedge_member
                    WHERE edge_id=? AND member_type='entity' AND member_id IN ({marks})""",
                (r["id"], *ent_ids),
            ).fetchone()["c"]
            if overlap == 0:
                continue
            score = overlap / max(1, len(ent_ids))
        else:
            score = 0.4
        if ext.theme and ext.theme in (r["label"] or ""):
            score += 0.5
        if score > best_score:
            best_score, best_edge = score, r

    if best_edge is not None and best_score >= 0.34:
        conn.execute(
            "INSERT OR IGNORE INTO hyperedge_member(edge_id,member_type,member_id,weight,added_at) VALUES(?,?,?,1.0,?)",
            (best_edge["id"], "episode", ep_id, now),
        )
        for e in ent_ids:
            conn.execute(
                "INSERT OR IGNORE INTO hyperedge_member(edge_id,member_type,member_id,weight,added_at) VALUES(?,?,?,0.6,?)",
                (best_edge["id"], "entity", e, now),
            )
        conn.execute(
            "UPDATE hyperedge SET member_count=(SELECT COUNT(*) FROM hyperedge_member WHERE edge_id=?), t_to=? WHERE id=?",
            (best_edge["id"], t_event, best_edge["id"]),
        )
        return 1

    label = (ext.theme or "").strip() or _label_from_entities(conn, ent_ids) or "未命名主题"
    hid = IDS.edge_id(session_id, "theme", label)
    conn.execute(
        """INSERT OR IGNORE INTO hyperedge(id,session_id,kind,label,summary,salience,t_from,t_to,created_at,member_count)
           VALUES(?,?,'theme',?,?,?,?,?,?,0)""",
        (hid, session_id, label, None, 0.5, t_event, t_event, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO hyperedge_member(edge_id,member_type,member_id,weight,added_at) VALUES(?,?,?,1.0,?)",
        (hid, "episode", ep_id, now),
    )
    for e in ent_ids:
        conn.execute(
            "INSERT OR IGNORE INTO hyperedge_member(edge_id,member_type,member_id,weight,added_at) VALUES(?,?,?,0.6,?)",
            (hid, "entity", e, now),
        )
    conn.execute(
        "UPDATE hyperedge SET member_count=(SELECT COUNT(*) FROM hyperedge_member WHERE edge_id=?) WHERE id=?",
        (hid, hid),
    )
    return 1


def _label_from_entities(conn, ent_ids: list[str]) -> str:
    names = []
    for e in ent_ids[:2]:
        row = conn.execute("SELECT name FROM entity WHERE id=?", (e,)).fetchone()
        if row:
            names.append(row["name"])
    return "·".join(names)


# ── 重要度 ────────────────────────────────────────────────────
def _salience(text: str, emo: float, ext: EX.Extraction, role: str) -> float:
    base = 0.42
    if role == "user":
        base += 0.06
    base += emo * 0.30
    if len(text) > 80:
        base += 0.04
    if ext.state_changes:
        base += 0.10
    if any(f.predicate in ("工作在", "住在", "关系是") for f in ext.facts):
        base += 0.08
    if not ext.facts and not ext.entities:
        base -= 0.10
    return float(max(0.05, min(1.0, base)))


def _maybe_forget(cfg: Config, session_id: str, now: float) -> None:
    try:
        from .forget import run_decay_only

        run_decay_only(cfg, session_id, now)
    except Exception as e:  # 遗忘失败不能影响写入
        log.debug("inline decay failed: %s", e)


# ── 后置抽取：给已经入库的 episode 补事实（异步路径用）────────
def enrich(cfg: Config, llm: LLM | None, episode_id: str, now: float | None = None, embedder=None) -> dict:
    """对已存在的 episode 做抽取并写入事实层。

    异步写入用：原文先落库保证不丢，抽取在后台跑，跑完再补。
    """
    now = now or time.time()
    with DB.reader(cfg.db_path) as conn:
        r = conn.execute("SELECT * FROM episode WHERE id=?", (episode_id,)).fetchone()
    if r is None:
        return {"ok": False, "reason": "episode 不存在"}
    if r["muted"] or r["archived"]:
        return {"ok": False, "reason": "条目已删除或归档，不再抽取"}
    with DB.reader(cfg.db_path) as conn:
        dead = conn.execute(
            "SELECT 1 FROM tombstone WHERE chunk_key=? OR line_ids LIKE ?",
            (episode_id, f"%{episode_id}%"),
        ).fetchone()
    if dead:
        return {"ok": False, "reason": "条目已进墓碑，不再抽取"}
    if r["meta_json"] and '"enriched": true' in r["meta_json"]:
        return {"ok": True, "skipped": "已抽取过"}

    ext = EX.extract(r["text"], llm, now, role=r["role"], use_llm=cfg.use_llm_extract)
    res = {"ok": True, "added": 0, "updated": 0, "invalidated": 0, "states_changed": 0, "source": ext.source}

    # 用抽取结果修正 episode 的类型与层级
    mtype = ext.facts[0].memory_type if ext.facts else PR.classify_type(r["text"], r["source"])
    if mtype and mtype != r["memory_type"]:
        with DB.tx(cfg.db_path) as conn:
            conn.execute("UPDATE episode SET memory_type=? WHERE id=?", (mtype, episode_id))

    if ext.empty():
        with DB.tx(cfg.db_path) as conn:
            conn.execute(
                "UPDATE episode SET meta_json=? WHERE id=?",
                ('{"enriched": true, "empty": true}', episode_id),
            )
        return res

    with DB.tx(cfg.db_path) as conn:
        ent_cache: dict[str, str] = {}
        for e in ext.entities:
            eid, _ = _upsert_entity(conn, r["session_id"], e.name, e.type, r["t_event"], now)
            ent_cache[IDS.normalize(e.name)] = eid

        for f in ext.facts:
            act = _resolve_fact(cfg, conn, r["session_id"], f, episode_id, ent_cache, r["t_event"], now, embedder)
            k = act.get("action")
            if k == "ADD":
                res["added"] += 1
            elif k == "UPDATE":
                res["updated"] += 1
            elif k == "INVALIDATE":
                res["invalidated"] += 1

        for s in ext.state_changes:
            if _apply_state(conn, r["session_id"], s, ent_cache, r["t_event"], now):
                res["states_changed"] += 1

        _group_theme(cfg, conn, r["session_id"], ext, episode_id, ent_cache, r["t_event"], now)

        conn.execute(
            "UPDATE episode SET meta_json=? WHERE id=?",
            ('{"enriched": true}', episode_id),
        )
    return res
