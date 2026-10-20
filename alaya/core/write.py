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
from . import item as IT
from . import pipeline as TR
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


def embedding_identity(embedder) -> tuple[str | None, int | None]:
    """当前 embedder 的向量身份（写 episode/fact.embed_model/embed_dim 用）。"""
    if embedder is None:
        return None, None
    try:
        model_id = str(
            getattr(embedder, "model_id", "") or getattr(embedder, "model", "") or ""
        ).strip()
        dim = int(getattr(embedder, "dim", 0) or 0)
    except (TypeError, ValueError):
        return None, None
    return (model_id or None), (dim or None)


def stamp_episode_embedding(cfg: Config, item_id: str, embedder) -> bool:
    """给刚写入的 episode 补向量身份。

    `pipeline.submit` 的契约没有 embed_model/embed_dim 字段（v4 只加列，不改
    pipeline 接口），所以写入成功后在这里用一次 UPDATE 补齐；旧数据保持 NULL。
    """
    if not item_id:
        return False
    model_id, dim = embedding_identity(embedder)
    if model_id is None:
        return False
    try:
        with DB.tx(cfg.db_path) as conn:
            conn.execute(
                "UPDATE episode SET embed_model=?, embed_dim=? WHERE id=?",
                (model_id, dim, item_id),
            )
        return True
    except Exception as e:  # 身份补写失败不影响记忆本身
        log.debug("stamp embedding identity failed: %s", e)
        return False


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
    """同步写入：抽取 → `pipeline.submit()` 落 item → 结构投影。

    episode 的 INSERT 已收口到 pipeline；这里只负责抽取与投影（与异步 enrich 同一实现）。
    """
    now = now or time.time()
    t_event = t_event or now
    text = (text or "").strip()
    if not text:
        return IngestResult(episode_id="", note="empty")

    ep_id = IDS.episode_id(session_id, role, text, t_event)

    # ── 幂等预检（避免白跑一次 LLM；真正的幂等在 pipeline.submit 事务内）──
    with DB.reader(cfg.db_path) as conn:
        row = conn.execute("SELECT id FROM episode WHERE id=?", (ep_id,)).fetchone()
        if row:
            return IngestResult(episode_id=ep_id, duplicate=True, note="已存在，跳过")

    # ── 抽取（事务外，因为要调 LLM，避免长事务锁库）──
    if skip_extract:
        ext = EX.Extraction(source="deferred")
    else:
        ext = EX.extract(text, llm, now, role=role, use_llm=cfg.use_llm_extract)

    # 降级时不要把哈希伪向量写进库（I4：两入口都不写伪向量）
    vec = embedder.embed(text)
    emb = None if embedder.degraded else V.pack(vec)
    emo = emotion_of(text)
    mtype = PR.classify_type(text, PR.S_DIALOGUE)
    author = "teacher" if role == "user" else ("self" if role == "assistant" else "system")
    actor = {
        "user": IT.ACTOR_USER,
        "assistant": IT.ACTOR_MODEL,
        "system": IT.ACTOR_SYSTEM,
    }.get(role, IT.ACTOR_SYSTEM)
    salience = salience_of(
        text,
        emo,
        actor=actor,
        memory_type=mtype,
        has_state=bool(ext.state_changes),
        has_facts=bool(ext.facts),
        has_entities=bool(ext.entities),
        fact_predicates=[f.predicate for f in ext.facts],
    )

    item = IT.MemoryItem(
        id=ep_id,
        session_id=session_id,
        text=text,
        origin=IT.ORIGIN_INGEST,
        actor=actor,
        memory_type=mtype,
        salience=salience,
        emotion=emo,
        t_event=t_event,
        t_ingest=now,
        role=role,
        source="dialogue",
        author=author,
        embedding=emb,
        meta={"extract_source": ext.source} if ext.source not in ("rule", "") else {},
    )
    sub = TR.submit(item, project=False, cfg=cfg, embedder=embedder, llm=llm)
    if sub.duplicate:
        return IngestResult(episode_id=sub.item_id, duplicate=True, note="已存在，跳过")
    if not sub.ok:
        return IngestResult(episode_id="", note=sub.reason or "写入失败")
    # 向量身份：pipeline 的写入契约没有 embed_model/embed_dim 字段，提交成功后就地补齐
    if emb is not None:
        stamp_episode_embedding(cfg, sub.item_id, embedder)

    result = IngestResult(episode_id=sub.item_id, salience=sub.salience, source=ext.source, note=ext.note)

    if not ext.empty():
        with DB.tx(cfg.db_path) as conn:
            counts = _project_extraction(
                cfg, conn, session_id, ext, sub.item_id, t_event, now, embedder
            )
            conn.execute("UPDATE session SET last_active=? WHERE id=?", (now, session_id))
        result.entities_new = counts["entities_new"]
        result.added = counts["added"]
        result.updated = counts["updated"]
        result.invalidated = counts["invalidated"]
        result.noop = counts["noop"]
        result.states_changed = counts["states_changed"]
        result.edges_touched = counts["edges_touched"]
        result.details = counts["details"]

    if auto_forget if auto_forget is not None else cfg.auto_forget:
        if result.added + result.updated > 0:
            _maybe_forget(cfg, session_id, now)

    return result


def _project_extraction(
    cfg: Config,
    conn,
    session_id: str,
    ext: EX.Extraction,
    ep_id: str,
    t_event: float,
    now: float,
    embedder=None,
) -> dict:
    """结构投影的唯一实现：被动同步写入与主动异步 enrich 共用。"""
    out = {
        "entities_new": 0,
        "added": 0,
        "updated": 0,
        "invalidated": 0,
        "noop": 0,
        "states_changed": 0,
        "edges_touched": 0,
        "details": [],
    }
    ent_cache: dict[str, str] = {}
    state_fact: dict[tuple[str, str], str] = {}
    for e in ext.entities:
        eid, created = _upsert_entity(conn, session_id, e.name, e.type, t_event, now)
        ent_cache[IDS.normalize(e.name)] = eid
        if created:
            out["entities_new"] += 1

    for f in ext.facts:
        act = _resolve_fact(cfg, conn, session_id, f, ep_id, ent_cache, t_event, now, embedder)
        out["details"].append(act)
        kind = act.get("action")
        if kind == "ADD":
            out["added"] += 1
        elif kind == "UPDATE":
            out["updated"] += 1
        elif kind == "INVALIDATE":
            out["invalidated"] += 1
        else:
            out["noop"] += 1
        # 状态类事实 → 它驱动的那个槽（scope 归一 + slot_key），供下面的
        # state_changes 认领归属；NOOP 也记，指向已有的那条事实。
        slot_key = EX.STATE_PREDICATES.get(f.predicate)
        if slot_key and act.get("fact"):
            state_fact[(IDS.normalize(f.subject), slot_key)] = act["fact"]

    for s in ext.state_changes:
        # 同一条 episode 抽出的 state_changes 往往就是上面那条事实的投影；
        # 有对应事实时把 fact id 带进槽，`correct()` 与 audit 才有线可查。
        src = state_fact.get((IDS.normalize(s.scope), s.slot))
        if _apply_state(conn, session_id, s, ent_cache, t_event, now, source_fact=src):
            out["states_changed"] += 1

    out["edges_touched"] = _group_theme(cfg, conn, session_id, ext, ep_id, ent_cache, t_event, now)
    return out


def _merge_meta(conn, item_id: str, patch: dict) -> None:
    """合并 meta_json，而不是覆盖——主动条目的四段 segments 不能被 enrich 抹掉。"""
    row = conn.execute("SELECT meta_json FROM episode WHERE id=?", (item_id,)).fetchone()
    meta: dict = {}
    if row is not None and row["meta_json"]:
        try:
            loaded = json.loads(row["meta_json"])
            if isinstance(loaded, dict):
                meta = loaded
        except (TypeError, ValueError):
            meta = {}
    meta.update(patch)
    conn.execute("UPDATE episode SET meta_json=? WHERE id=?", (json.dumps(meta, ensure_ascii=False), item_id))


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
    # 降级期不写哈希伪向量（与 episode 口径一致）：留 NULL 等 embed_backfill 回算
    emb = None
    if embedder is not None and not getattr(embedder, "degraded", False):
        try:
            emb = embedder.embed_blob(f"{f.predicate} {f.obj}")
            if getattr(embedder, "degraded", False):  # 本次调用把服务打挂了
                emb = None
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
        hist_end = _supersede(conn, rows, vfrom, t_event, now)
        new_id, _inserted = _insert_fact(
            conn, session_id, subj_id, f, obj_ent, vfrom, ep_id, t_event, emb,
            embedder=embedder, valid_to=hist_end,
        )
        # 抽取只吐 facts、不吐 state_changes 时，槽只能由这里驱动。漏了这一步，
        # 槽会停在旧值上、source_fact 还指着刚被作废的事实——检索里的「当前状态」
        # 永远是老地址（真实 LLM 也不总记得吐 state_changes）。
        if int(f.polarity) == 1:
            _apply_state(
                conn, session_id,
                EX.StateCand(scope=f.subject, slot=EX.STATE_PREDICATES[f.predicate], value=f.obj),
                ent_cache, t_event, now, source_fact=new_id,
            )
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
        hist_end = _supersede(conn, opposite, vfrom, t_event, now)
        new_id, _inserted = _insert_fact(
            conn, session_id, subj_id, f, obj_ent, vfrom, ep_id, t_event, emb,
            embedder=embedder, valid_to=hist_end,
        )
        return {
            "action": "INVALIDATE",
            "fact": new_id,
            "over": [r["id"] for r in opposite],
            "reason": "极性反转",
            "predicate": f.predicate,
        }

    new_id, inserted = _insert_fact(conn, session_id, subj_id, f, obj_ent, vfrom, ep_id, t_event, emb, embedder=embedder)

    # 确定性兜底：位置/工作/关系类事实直接驱动状态槽。
    # 不能指望 LLM 记得产出 state_changes——它经常忘，而槽是检索的主干，
    # 漏了比漏一条 fact 严重得多。「住在城东」本身就是 city=城东 的变更。
    if f.predicate in EX.STATE_PREDICATES and int(f.polarity) == 1:
        _apply_state(
            conn, session_id,
            EX.StateCand(scope=f.subject, slot=EX.STATE_PREDICATES[f.predicate], value=f.obj),
            ent_cache, t_event, now, source_fact=new_id,
        )

    if not inserted:
        # 事实已经在库里（同主语/谓词/客体/起点/极性）：报 EXISTS，不冒充 ADD。
        # 旧实现 INSERT OR IGNORE 之后照样返回 ADD，`added` 统计被灌水，
        # 上层看到"又记住了一条"其实什么都没变。
        return {"action": "EXISTS", "fact": new_id, "predicate": f.predicate}
    return {"action": "ADD", "fact": new_id, "predicate": f.predicate}


def _supersede(conn, rows, vfrom: float, t_event: float, now: float) -> float | None:
    """按时间轴收口被新事实顶掉的旧事实，返回新事实该闭合到哪个时刻。

    正常时序（`vfrom` 不早于旧条起点）：旧条在 `vfrom` 结束——它确实被取代了。
    补录（`vfrom` 早于旧条起点）：**旧条不动**，它仍然有效到下一次变更；新事实
    只占 `[vfrom, 旧条起点)` 这段历史，否则会写出
    `t_valid_to < t_valid_from` 的负区间（"这条事实在它成立之前就失效了"）。
    """
    later = [float(r["t_valid_from"]) for r in rows if float(r["t_valid_from"]) > vfrom]
    for r in rows:
        if float(r["t_valid_from"]) <= vfrom:
            _invalidate(conn, r["id"], t_event, now, valid_to=vfrom)
    return min(later) if later else None


def _insert_fact(conn, session_id: str, subj_id: str, f: EX.FactCand, obj_ent, vfrom: float, ep_id: str, now: float, emb: bytes | None = None, embedder=None, valid_to: float | None = None) -> tuple[str, bool]:
    """插一条事实，返回 (fact_id, 是否真的插入)。

    `INSERT OR IGNORE` 的 rowcount 是唯一的"到底插没插"证据：它被忽略时
    不能继续插 FTS 行（会攒重复行），也不能向上报 ADD。

    `valid_to` 非空表示这条是**补录的历史事实**：它当时为真、现在已经结束，
    所以直接以 invalidated + 闭合区间落库，不抢占"当前有效"的位置。
    """
    fid = IDS.fact_id(session_id, IDS.normalize(f.subject), f.predicate, f.obj, vfrom, f.polarity)
    model_id, dim = embedding_identity(embedder) if emb is not None else (None, None)
    status = "active" if valid_to is None else "invalidated"
    if valid_to is not None and valid_to < vfrom:  # 不变式：闭区间不得为负
        valid_to = vfrom
    cur = conn.execute(
        """INSERT OR IGNORE INTO fact
           (id,session_id,subject_id,predicate,object_text,object_entity_id,polarity,confidence,
            t_valid_from,t_valid_to,t_ingest,t_invalid,status,source_episode,salience,embedding,
            embed_model,embed_dim)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            fid, session_id, subj_id, f.predicate, f.obj, obj_ent,
            f.polarity, f.confidence, vfrom, valid_to, now,
            now if valid_to is not None else None, status, ep_id,
            min(1.0, 0.45 + 0.35 * f.confidence), emb, model_id, dim,
        ),
    )
    if cur.rowcount <= 0:
        return fid, False
    conn.execute(
        "INSERT INTO fact_fts(fact_id,tokens) VALUES(?,?)",
        (fid, " ".join(TX.tokenize(f"{f.predicate} {f.obj}"))),
    )
    return fid, True


def _invalidate(conn, fact_id: str, t_event: float, now: float, valid_to: float | None = None) -> None:
    """作废一条事实。

    `t_invalid` 是系统时间（我们何时判定它失效），`t_valid_to` 是世界时间
    （它在真实世界里何时不再为真）。两者分开：前者用 now，后者用**新事实的
    起始时刻**——用事件时间会产生"旧值和新值同时为真"的窗口。

    `MAX(?, t_valid_from)` 是区间不变式：`t_valid_to` 永远不早于 `t_valid_from`。
    正常时序下它是恒等操作；补录场景若还有人传进更早的时刻，宁可收紧到起点，
    也不留一条"在成立之前就失效"的负区间。
    """
    conn.execute(
        """UPDATE fact SET status='invalidated', t_invalid=?,
                          t_valid_to=MAX(COALESCE(?, t_valid_from), t_valid_from)
           WHERE id=?""",
        (now, valid_to if valid_to is not None else t_event, fact_id),
    )


# ── 状态槽 ────────────────────────────────────────────────────
def _insert_slot(
    conn,
    session_id: str,
    scope_id: str,
    slot_key: str,
    value: str,
    since: float,
    until: float | None,
    source_fact: str | None,
    now: float,
) -> bool:
    """写一条槽。区间合法性由调用方保证，这里只做最后一道拦截。"""
    if until is not None and until < since:
        # 不变式（A-3）：槽区间不许为负。走到这支说明上面的分支判断有漏，
        # 宁可丢一条历史槽，也不往库里写"结束早于开始"的状态。
        log.warning(
            "state_slot 负区间已拒绝：%s=%s since=%s until=%s", slot_key, value, since, until
        )
        return False
    sid_slot = IDS.slot_id(session_id, scope_id, slot_key, since, value)
    sql = """INSERT INTO state_slot(id,session_id,scope,slot_key,value_text,since,until,source_fact,confidence)
             VALUES(?,?,?,?,?,?,?,?,0.75)"""
    try:
        conn.execute(sql, (sid_slot, session_id, scope_id, slot_key, value, since, until, source_fact))
    except sqlite3.IntegrityError:
        # 回填或同毫秒重入时 id 可能撞上已关闭的旧槽。
        # 不能 OR IGNORE（那会静默丢掉整条状态），换一个 id 再插。
        alt = "sl_" + IDS._digest(sid_slot, since, now)
        conn.execute(sql, (alt, session_id, scope_id, slot_key, value, since, until, source_fact))
    return True


def _apply_state(
    conn,
    session_id: str,
    s: EX.StateCand,
    ent_cache: dict[str, str],
    t_event: float,
    now: float,
    source_fact: str | None = None,
) -> bool:
    """落一条状态槽，返回是否写入。

    `source_fact` 记「这个槽由哪条事实推出」：`correct()` 的槽失效与 audit 的
    stale_state 都靠它串联，之前该列恒 NULL，两条链路永久空跑。
    """
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

    if cur is None:
        return _insert_slot(conn, session_id, scope_id, s.slot, s.value, t_event, None, source_fact, now)

    if t_event >= cur["since"]:
        # 正常时序：旧槽关在事件时刻，新槽从事件时刻开始
        conn.execute("UPDATE state_slot SET until=? WHERE id=?", (t_event, cur["id"]))
        return _insert_slot(conn, session_id, scope_id, s.slot, s.value, t_event, None, source_fact, now)

    # 补录的历史时刻早于当前槽起点：**不许**拿它关掉当前槽。否则历史状态会
    # 顶替当前真理，同时制造 until < since 的负区间。只插一个已闭合的
    # 历史槽 [t_event, cur.since)，当前槽原样保留。
    covered = conn.execute(
        """SELECT value_text FROM state_slot
           WHERE session_id=? AND scope=? AND slot_key=?
             AND since <= ? AND (until IS NULL OR until > ?)""",
        (session_id, scope_id, s.slot, t_event, t_event),
    ).fetchall()
    if any(IDS.normalize(r["value_text"]) == IDS.normalize(s.value) for r in covered):
        return False
    return _insert_slot(
        conn, session_id, scope_id, s.slot, s.value, t_event, cur["since"], source_fact, now
    )


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


# ── 重要度与情感：全系统唯一的实现（M3 打分单点）────────────
# 状态类谓词：出现时说明这次写入改变了世界状态，重要度抬高
_STATE_PREDICATES = ("工作在", "住在", "关系是")


def emotion_of(text: str) -> float:
    """情绪强度的唯一实现。被动/主动/系统条目都走它。"""
    return SC.emotion_intensity(text or "", is_session_tail=False)


def salience_of(
    text: str,
    emotion: float = 0.0,
    *,
    actor: str = "system",
    memory_type: str = "fact",
    has_state: bool = False,
    has_facts: bool = False,
    has_entities: bool = False,
    fact_predicates: list[str] | tuple[str, ...] = (),
    segments: dict | None = None,
) -> float:
    """重要度的唯一实现（原 `write._salience` + `notebook` 内联公式合并）。

    口径只看内容证据，**不看 origin**——同内容经主动/被动两个入口写入必须同分（M3）。
    - 用户/主人说的话略高
    - 情绪、长文本、状态变更、位置/工作/关系事实各自加权
    - 抽取不出任何结构的段落降权
    - 主动条目的第二层（noticed/means/next）等价于结构化证据，逐段小幅加权
    """
    base = 0.42
    if (actor or "").strip().lower() in ("user", "teacher"):
        base += 0.06
    base += float(emotion or 0.0) * 0.30
    if len(text or "") > 80:
        base += 0.04
    if has_state:
        base += 0.10
    if any(p in _STATE_PREDICATES for p in (fact_predicates or ())):
        base += 0.08
    if not has_facts and not has_entities:
        base -= 0.10
    if segments:
        base += 0.04 * sum(1 for k in ("noticed", "means", "next") if (segments.get(k) or "").strip())
    return float(max(0.05, min(1.0, base)))


def _maybe_forget(cfg: Config, session_id: str, now: float) -> None:
    try:
        from .forget import run_decay_only

        run_decay_only(cfg, session_id, now)
    except Exception as e:  # 遗忘失败不能影响写入
        log.debug("inline decay failed: %s", e)


# ── 后置抽取：给已经入库的 episode 补结构投影（异步路径用）────
def enrich(cfg: Config, llm: LLM | None, episode_id: str, now: float | None = None, embedder=None) -> dict:
    """对已存在的 episode 做抽取并写入结构层。

    异步写入用：原文先落库保证不丢，抽取在后台跑，跑完再补。
    与被动同步写入共用 `_project_extraction`，口径唯一（M2）。
    """
    now = now or time.time()
    with DB.reader(cfg.db_path) as conn:
        r = conn.execute("SELECT * FROM episode WHERE id=?", (episode_id,)).fetchone()
    if r is None:
        return {"ok": False, "reason": "episode 不存在"}

    state = r["state"] if "state" in r.keys() and r["state"] else IT.compat_state(
        r["tier"], r["archived"], r["muted"]
    )
    if state in (IT.STATE_MUTED, IT.STATE_ARCHIVED) or r["muted"] or r["archived"]:
        return {"ok": False, "reason": "条目已删除或归档，不再抽取"}

    import json as _json

    meta: dict = {}
    if r["meta_json"]:
        try:
            loaded = _json.loads(r["meta_json"])
            if isinstance(loaded, dict):
                meta = loaded
        except (TypeError, ValueError):
            meta = {}
    if meta.get("enriched"):
        return {"ok": True, "skipped": "已抽取过"}

    with DB.reader(cfg.db_path) as conn:
        dead = conn.execute(
            "SELECT 1 FROM tombstone WHERE chunk_key=? OR line_ids LIKE ?",
            (episode_id, f"%{episode_id}%"),
        ).fetchone()
    if dead:
        return {"ok": False, "reason": "条目已进墓碑，不再抽取"}

    ext = EX.extract(r["text"], llm, now, role=r["role"], use_llm=cfg.use_llm_extract)
    res = {
        "ok": True,
        "added": 0,
        "updated": 0,
        "invalidated": 0,
        "noop": 0,
        "states_changed": 0,
        "edges_touched": 0,
        "source": ext.source,
    }

    # 用抽取结果修正 episode 的类型（不改状态，状态只由 gate/pipeline 改）
    mtype = ext.facts[0].memory_type if ext.facts else PR.classify_type(r["text"], r["source"])

    if ext.empty():
        with DB.tx(cfg.db_path) as conn:
            if mtype and mtype != r["memory_type"]:
                conn.execute("UPDATE episode SET memory_type=? WHERE id=?", (mtype, episode_id))
            # 评审 B4：enrich 结束在同一事务重算 salience，保证两入口同输入同分
            new_sal = _canonical_salience(r, ext, mtype)
            conn.execute("UPDATE episode SET salience=? WHERE id=?", (new_sal, episode_id))
            _merge_meta(conn, episode_id, {"enriched": True, "empty": True})
        res["salience"] = new_sal
        return res

    with DB.tx(cfg.db_path) as conn:
        if mtype and mtype != r["memory_type"]:
            conn.execute("UPDATE episode SET memory_type=? WHERE id=?", (mtype, episode_id))
        # 注意：必须在 _project_extraction 之前算——它会把 predicate 归一化，
        # 而同步入口的 salience 是按归一化前的原文谓词算的（顺序必须一致）。
        new_sal = _canonical_salience(r, ext, mtype)
        counts = _project_extraction(
            cfg, conn, r["session_id"], ext, episode_id, r["t_event"], now, embedder
        )
        conn.execute("UPDATE episode SET salience=? WHERE id=?", (new_sal, episode_id))
        res["added"] = counts["added"]
        res["updated"] = counts["updated"]
        res["invalidated"] = counts["invalidated"]
        res["noop"] = counts["noop"]
        res["states_changed"] = counts["states_changed"]
        res["edges_touched"] = counts["edges_touched"]
        res["salience"] = new_sal
        _merge_meta(conn, episode_id, {"enriched": True})
    return res


def _canonical_salience(row, ext: EX.Extraction, memory_type: str = "") -> float:
    """enrich 后的 salience 重算（评审 B4 / M3）。

    enrich 与同步 ingest 用同一份 `salience_of`，输入取库里的文本、情绪、
    actor 与刚抽出的结构证据，因此同输入两入口差 = 0。
    """
    try:
        item = IT.MemoryItem.from_row(row)
    except Exception:  # 极端脏行：退回最小输入，绝不让 enrich 崩在打分上
        item = IT.MemoryItem(
            id=str(row["id"]), session_id=str(row["session_id"]), text=str(row["text"] or ""),
            actor=IT.ACTOR_SYSTEM, memory_type=memory_type or "fact",
        )
    return salience_of(
        row["text"] or "",
        float(row["emotion"] or 0.0),
        actor=item.actor,
        memory_type=memory_type or item.memory_type,
        has_state=bool(ext.state_changes),
        has_facts=bool(ext.facts),
        has_entities=bool(ext.entities),
        fact_predicates=[f.predicate for f in ext.facts],
        segments=item.segments,
    )
