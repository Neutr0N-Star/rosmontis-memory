"""四阶段检索 + 预算约束打包。

阶段：理解问题 → 三路召回 → 融合重排 → 渐进式注入
输出的 prompt 片段可直接塞进对话上下文。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..config import Config
from . import db as DB
from . import ids as IDS
from . import memory_field as MF
from . import score as SC
from . import text as TX
from . import timeparse
from . import vec as V
from .embed import Embedder

log = logging.getLogger("alaya.recall")

# 四级反射深度。规模按查询复杂度伸缩——问"我住哪"不必扫全库。
# 参数参考 NeuralMemory 的 ReflexPipeline（INSTANT / CONTEXT / HABIT / DEEP）。
REFLEX_DEPTH = {
    "instant": {"dense": 6, "sparse": 6, "graph": 0, "hops": 0, "budget": 0.45},
    "context": {"dense": 30, "sparse": 30, "graph": 18, "hops": 1, "budget": 1.00},
    "habit":   {"dense": 55, "sparse": 55, "graph": 35, "hops": 2, "budget": 1.25},
    "deep":    {"dense": 90, "sparse": 90, "graph": 60, "hops": 3, "budget": 1.60},
}

_DEEP_HINTS = ("总结", "梳理", "所有", "全部", "历史上", "回顾", "都有哪些", "列出")
_HABIT_HINTS = ("为什么", "怎么", "以前", "变化", "后来", "原因", "经过", "什么时候开始")

# 意图关键词
_WANT_POLARITY = ("喜欢", "讨厌", "爱吃", "怕", "害怕", "愿意", "想不想", "会不会")
_WANT_PAST = ("以前", "之前", "当时", "那时候", "去年", "曾经", "原来", "过去")
_WANT_NOW = ("现在", "目前", "最近", "如今", "这阵子", "还在")
_WANT_CHANGE = ("变了", "变化", "后来", "改成", "不再是", "还住", "还喜欢")
_WANT_IDENTITY = ("我叫", "我是谁", "名字", "我的名字", "我是做什么", "我住哪", "我在哪")


@dataclass
class RecallItem:
    kind: str
    ref_id: str
    level: int
    text: str
    score: float
    tokens: int
    meta: dict = field(default_factory=dict)


@dataclass
class RecallResult:
    query: str
    abstained: bool = False
    reason: str | None = None
    items: list[RecallItem] = field(default_factory=list)
    prompt: str = ""
    tokens_used: int = 0
    budget: int = 0
    plan: dict = field(default_factory=dict)
    excluded: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "abstained": self.abstained,
            "reason": self.reason,
            "prompt": self.prompt,
            "tokens_used": self.tokens_used,
            "budget": self.budget,
            "plan": self.plan,
            "items": [
                {
                    "kind": i.kind,
                    "id": i.ref_id,
                    "level": i.level,
                    "text": i.text,
                    "score": round(i.score, 4),
                    "tokens": i.tokens,
                }
                for i in self.items
            ],
            "excluded": self.excluded,
            "latency_ms": round(self.latency_ms, 1),
        }


def recall(
    cfg: Config,
    embedder: Embedder,
    query: str,
    session_id: str | None = None,
    as_of: float | None = None,
    budget: int | None = None,
    exclude_ids: list[str] | None = None,
    now: float | None = None,
) -> RecallResult:
    t0 = time.perf_counter()
    now = now or time.time()
    budget = budget or cfg.budget_tokens
    as_of = as_of or now
    res = RecallResult(query=query, budget=budget)

    plan = _plan(query, now, as_of)
    res.plan = plan

    # 反近因开关：问"现在/我是谁"用常规新近性；其余（问过去、问变化、泛问）
    # 用陈旧度加权，让久未想起的事有机会浮现。
    stale_mode = plan["intent"] not in ("now", "identity", "polarity")

    depth_cfg = REFLEX_DEPTH.get(plan["depth"], REFLEX_DEPTH["context"])
    if budget is not None:
        budget = max(80, int(budget * depth_cfg["budget"]))
        res.budget = budget
    cands = _gather(cfg, embedder, query, session_id, plan, as_of, budget == 0)
    if not cands:
        res.abstained = True
        res.reason = "记忆库中没有相关内容"
        res.latency_ms = (time.perf_counter() - t0) * 1000
        _log(cfg, session_id, res)
        return res

    cands = SC.rank_and_densify(
        cands, now, cfg.weights, cfg.budget, stale_mode=stale_mode, policy=cfg.policy
    )

    # 拒答只看**检索层**的得分。状态快照与常驻条目是"我一直知道的"，
    # 它们不该把无关问题撑成"有相关记忆"——那会让拒答永远失效。
    retrievable = [c for c in cands if c.kind not in ("state", "resident")]
    top = max((c.score for c in retrievable), default=0.0)
    # 双门：分数要过线，原始相关度也要过线。
    # 只看分数的话，图扩散和全文匹配能把「今天天气怎么样」撑出 10 条
    # 高分记忆——分数是合成的，rel 才是「这段话跟我真有关系吗」。
    top_rel = max((c.rel for c in retrievable), default=0.0)
    # rel 门必须跟向量质量挂钩：
    #   真语义向量（bge-m3）→ 0.40，汉字短句间余弦普遍偏高，严门才能拦住天气类
    #   降级（真模型掉线、embedding 缺失）→ 0.25，全文匹配的 rel 天然偏低
    #   hash 伪向量（测试/无模型）→ 0.15，它的余弦没有语义意义，只留底线
    prov = getattr(embedder, "provider", "") if embedder is not None else ""
    if embedder is None or prov == "hash":
        # hash 伪向量的余弦没有语义意义，rel 门整个跳过——
        # 拒答只靠 score 门。真语义向量才配得上 0.40 的严门。
        rel_gate = None
    elif getattr(embedder, "degraded", False):
        rel_gate = 0.25
    else:
        rel_gate = 0.40
    if top < cfg.weights.theta_abstain or (rel_gate is not None and top_rel < rel_gate):
        res.abstained = True
        res.reason = (
            f"没有找到相关回忆（最高分 {top:.2f} < 阈值 {cfg.weights.theta_abstain}）"
            if retrievable
            else "记忆库里没有可检索的内容"
        )
        res.excluded = [{"id": c.ref_id, "kind": c.kind, "why": "低于拒答阈值"} for c in retrievable[:5]]
        # 仍然给出常驻部分：她"一直知道"的事与问题是否相关无关
        try:
            from . import pages as PG

            blk = PG.compile_resident(cfg, session_id or "default", cfg.policy.resident_max_tokens)
            if blk["text"]:
                res.items = [RecallItem("page", "resident_pages", 3, blk["text"], 1.0, blk["tokens"], {"pages": blk["pages"]})]
                res.tokens_used = blk["tokens"]
                res.prompt = _render(res.items)
        except Exception:
            pass
        res.latency_ms = (time.perf_counter() - t0) * 1000
        _log(cfg, session_id, res)
        return res

    skip = set(exclude_ids or [])
    items, used, excluded = _pack(cfg, cands, budget, skip, plan)

    # 常驻活文档：预生成，直接读，不检索不调 LLM。单独成块，不占检索预算。
    try:
        from . import pages as PG

        res_block = PG.compile_resident(cfg, session_id or "default", cfg.policy.resident_max_tokens)
        if res_block["text"]:
            items.insert(
                0,
                RecallItem("page", "resident_pages", 3, res_block["text"], 1.0, res_block["tokens"], {"pages": res_block["pages"]}),
            )
            used += res_block["tokens"]
    except Exception as e:  # 页面层失败不影响检索
        log.debug("resident pages failed: %s", e)

    res.items = items
    res.tokens_used = used
    res.excluded = excluded
    res.prompt = _render(items)
    res.latency_ms = (time.perf_counter() - t0) * 1000
    _log(cfg, session_id, res)
    return res


def _adjacency(cfg: Config, sid: str) -> dict[str, list[tuple[str, float]]]:
    """从超边成员构建邻接表（只连 fact 与 episode）。

    连续场的拉普拉斯算子在图上就是"邻居"。这里用超边做超边内两两连接，
    权重随超边大小衰减——一个 50 人的大主题里，任意两条的关联本来就不强。
    """
    adj: dict[str, list[tuple[str, float]]] = {}
    try:
        with DB.reader(cfg.db_path) as conn:
            edges = conn.execute(
                """SELECT edge_id FROM hyperedge
                   WHERE (?='*' OR session_id=?) AND member_count BETWEEN 2 AND 30
                   ORDER BY member_count DESC LIMIT 200""",
                (sid, sid),
            ).fetchall()
            for e in edges:
                members = conn.execute(
                    """SELECT member_type, member_id FROM hyperedge_member
                       WHERE edge_id=? AND member_type IN ('fact','episode') LIMIT 30""",
                    (e["edge_id"],),
                ).fetchall()
                if len(members) < 2:
                    continue
                w = round(0.85 / max(1, len(members) - 1), 3)
                keys = [f"{m['member_type']}:{m['member_id']}" for m in members]
                for i, a in enumerate(keys):
                    for b in keys[i + 1 :]:
                        adj.setdefault(a, []).append((b, w))
                        adj.setdefault(b, []).append((a, w))
    except Exception as e:
        log.debug("adjacency failed: %s", e)
    return adj


def _expand_by_field(
    cfg: Config,
    sid: str,
    cands: list[SC.Candidate],
    seen: dict,
    as_of: float,
    plan: dict,
) -> list[SC.Candidate]:
    """图扩散：命中节点把激活沿结构传给邻居。

    这一步补的是向量检索的结构性缺陷——它只在查询那一刻建立相似,
    而扩散走的是**已经存在的结构关联**。命中了"团子"的记忆，
    会顺带把同属"养狗"主题的其他片段也拉起来。
    """
    adj = _adjacency(cfg, sid)
    if not adj:
        return cands
    seeds = {f"{c.kind}:{c.ref_id}": max(0.15, min(1.0, c.rel)) for c in cands if c.rel > 0.2}
    if not seeds:
        return cands

    hops = int(plan.get("_hops", 2))
    act = MF.spread(seeds, adj, gamma=MF.DEFAULT_GAMMA, hops=hops, floor=0.06)
    extra = 0
    for nid, a in act.items():
        if nid in seeds:
            continue
        try:
            kind, real = nid.split(":", 1)
        except ValueError:
            continue
        if kind not in ("fact", "episode") or real in seen:
            continue
        if extra >= 12:  # 扩散补进来的条目上限，防止一次检索被撑爆
            break
        c = _load_candidate(cfg, kind, real, sid, as_of, plan)
        if c is None:
            continue
        c.rel = max(c.rel, min(0.9, a * 0.9))
        c.graph_prox = max(c.graph_prox, a)
        seen[real] = c
        cands.append(c)
        extra += 1
    if extra:
        log.debug("field expanded %d candidates", extra)
    return cands


# ── 阶段 1：理解问题 ──────────────────────────────────────────
def _plan(query: str, now: float, as_of: float) -> dict:
    q = query or ""
    span = timeparse.parse(q, now)
    intent = "general"
    if any(k in q for k in _WANT_CHANGE):
        intent = "change"
    elif any(k in q for k in _WANT_PAST):
        intent = "past"
    elif any(k in q for k in _WANT_NOW):
        intent = "now"
    elif any(k in q for k in _WANT_IDENTITY):
        intent = "identity"
    elif any(k in q for k in _WANT_POLARITY):
        intent = "polarity"

    # 反射深度：先看问题形态，再看长度
    if any(k in q for k in _DEEP_HINTS):
        depth = "deep"
    elif any(k in q for k in _HABIT_HINTS) or span is not None:
        depth = "habit"
    elif intent in ("identity", "now") and len(q) <= 12:
        depth = "instant"
    else:
        depth = "context"

    return {
        "intent": intent,
        "depth": depth,
        "time_span": None if not span else [span.start, span.end, span.granularity],
        "as_of": as_of,
        "tokens": TX.tokenize(q),
        "wants_negative": any(w in q for w in ("不喜欢", "讨厌", "怕", "不想要")),
    }


# ── 阶段 2：三路召回 ──────────────────────────────────────────
def _gather(
    cfg: Config,
    embedder: Embedder,
    query: str,
    session_id: str | None,
    plan: dict,
    as_of: float,
    state_only: bool = False,
) -> list[SC.Candidate]:
    sid = session_id or "default"
    depth = REFLEX_DEPTH.get(str(plan.get("depth") or "context"), REFLEX_DEPTH["context"])
    seen: dict[str, SC.Candidate] = {}

    _collect_states(cfg, sid, as_of, seen, plan)

    if state_only:
        return list(seen.values())

    plan["_k_dense"] = depth["dense"]
    dense_ids = _dense(cfg, embedder, query, sid, as_of, plan)
    sparse_ids = _sparse(cfg, query, sid, as_of, depth["sparse"])
    graph_ids = _graph(cfg, plan, sid, as_of, depth["graph"])

    fused = V.rrf_fuse([dense_ids, sparse_ids, graph_ids], k=60.0)
    # 直接用 RRF 分作为"被多路共同命中"的证据，稍后与向量余弦合成 rel
    dense_rank = {rid: i for i, rid in enumerate(dense_ids)}

    for rid, fscore in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        kind, real_id = rid.split(":", 1)
        if real_id in seen:
            seen[real_id].rel = max(seen[real_id].rel, min(1.0, fscore * 30))
            continue
        c = _load_candidate(cfg, kind, real_id, sid, as_of, plan)
        if c is None:
            continue
        c.rel = min(1.0, fscore * 30)
        if real_id in dense_rank:
            c.graph_prox = max(c.graph_prox, 0.05)
        seen[real_id] = c

    # 对稠密命中补一次精确余弦，避免 RRF 粗粒度丢掉真实相似度
    if dense_ids:
        _refine_cosine(cfg, embedder, query, seen, dense_ids)

    # 图扩散：把结构上相邻的记忆也拉进候选
    plan["_hops"] = depth["hops"]
    if depth["hops"] <= 0:
        return list(seen.values())
    return _expand_by_field(cfg, sid, list(seen.values()), seen, as_of, plan)


def _collect_states(cfg: Config, sid: str, as_of: float, seen: dict, plan: dict) -> None:
    """常驻层：状态快照 + tier=resident 的事件。每轮注入，不参与冷却，不计数。

    这是陪伴场景的地基——她不该靠语义相似才想起"老师是谁"。
    """
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT s.*, e.name AS scope_name FROM state_slot s
               LEFT JOIN entity e ON e.id = s.scope
               WHERE (? = '*' OR s.session_id = ?)
                 AND s.since <= ? AND (s.until IS NULL OR s.until > ?)
               ORDER BY s.since DESC LIMIT 60""",
            (sid, sid, as_of, as_of),
        ).fetchall()
    for r in rows:
        c = SC.Candidate(
            kind="state",
            ref_id=r["id"],
            subject=r["scope"],
            predicate=r["slot_key"],
            object_text=r["value_text"],
            t_ref=r["since"],
            salience=0.92,
            scope_ok=True,
            tier="resident",
            base_text=f"{r['scope_name'] or r['scope']}·{r['slot_key']} = {r['value_text']}（自 {timeparse.humanize(r['since'])}）",
            meta={"scope_name": r["scope_name"], "confidence": r["confidence"]},
        )
        c.rel = 0.55 if plan.get("intent") in ("identity", "now", "change") else 0.35
        c.emotion = 0.25
        c.time_fit = 1.0
        seen[c.ref_id] = c

    # tier=resident 的记忆条目（用户画像/承诺/编年史摘要）
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM episode
               WHERE (?='*' OR session_id=?) AND tier='resident' AND muted=0
                 AND archived=0 AND t_ingest <= ?
               ORDER BY salience DESC, t_event DESC LIMIT 40""",
            (sid, sid, as_of),
        ).fetchall()
    for r in rows:
        c = SC.Candidate(
            kind="resident",
            ref_id=r["id"],
            t_ref=r["t_event"],
            salience=r["salience"],
            emotion=r["emotion"] or 0.0,
            access_count=r["access_count"] or 0,
            last_recall=r["last_access"] or 0.0,
            author=r["author"],
            tier="resident",
            memory_type=r["memory_type"],
            base_text=(r["text"] or "")[:200],
            summary=(r["text"] or "")[:200],
            raw_text=r["text"] or "",
            meta={"source": r["source"], "author": r["author"], "type": r["memory_type"]},
        )
        c.rel = 0.6
        c.time_fit = 1.0
        seen[c.ref_id] = c


def _dense(cfg: Config, embedder: Embedder, query: str, sid: str, as_of: float, plan: dict) -> list[str]:
    qv = embedder.embed_blob(query)
    if not qv:
        return []
    k = int(plan.get("_k_dense") or cfg.recall_k_dense)
    out: list[tuple[float, str]] = []
    with DB.reader(cfg.db_path) as conn:
        for table, kind in (("fact", "fact"), ("episode", "episode")):
            if table == "fact":
                rows = conn.execute(
                    """SELECT id, embedding FROM fact
                       WHERE (?='*' OR session_id=?) AND embedding IS NOT NULL
                         AND status!='invalidated' AND t_ingest <= ?
                       ORDER BY t_ingest DESC LIMIT 20000""",
                    (sid, sid, as_of),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, embedding FROM episode
                       WHERE (?='*' OR session_id=?) AND embedding IS NOT NULL
                         AND archived=0 AND t_ingest <= ? AND source!='dream'
                       ORDER BY t_ingest DESC LIMIT 20000""",
                    (sid, sid, as_of),
                ).fetchall()
            if not rows:
                continue
            ids = [r["id"] for r in rows]
            blobs = [r["embedding"] for r in rows]
            # 维度校验：一条旧模型的 512 维向量混进来会让整批 reshape 失败。
            # 坏行剔除并计数，不让全库检索废掉。
            need = cfg.embed_dim * 4
            good_idx = [i for i, b in enumerate(blobs) if b and len(b) == need]
            if len(good_idx) != len(blobs):
                log.warning("dense: %d 条向量维度不符已跳过", len(blobs) - len(good_idx))
            ids = [ids[i] for i in good_idx]
            blobs = [blobs[i] for i in good_idx]
            if not ids:
                continue
            sims = V.cosine_batch(qv, blobs, cfg.embed_dim)
            for i, s in zip(ids, sims):
                if s > 0.05:
                    out.append((s, f"{kind}:{i}"))

    out.sort(key=lambda x: x[0], reverse=True)
    return [rid for _, rid in out[:k]]


def _sparse(cfg: Config, query: str, sid: str, as_of: float, k: int | None = None) -> list[str]:
    k = k or cfg.recall_k_sparse
    toks = TX.tokenize(query)
    fq = TX.fts_query(toks)
    if not fq:
        return []
    out: list[str] = []
    with DB.reader(cfg.db_path) as conn:
        for fts, kind, table in (("fact_fts", "fact", "fact"), ("episode_fts", "episode", "episode")):
            try:
                rows = conn.execute(
                    f"""SELECT f.rowid, t.id FROM {fts} f
                        JOIN {table} t ON t.id = f.{'fact_id' if kind=='fact' else 'episode_id'}
                        WHERE {fts} MATCH ? AND (?='*' OR t.session_id=?) AND t.t_ingest <= ?
                          AND t.source!='dream'
                        ORDER BY rank LIMIT ?""",
                    (fq, sid, sid, as_of, k),
                ).fetchall()
            except Exception as e:  # FTS 语法异常不该拖垮检索
                log.debug("fts failed %s: %s", fts, e)
                continue
            out.extend(f"{kind}:{r['id']}" for r in rows)
    return list(dict.fromkeys(out))


def _graph(cfg: Config, plan: dict, sid: str, as_of: float, k: int | None = None) -> list[str]:
    k = k or cfg.recall_k_graph
    if k <= 0:
        return []
    """图游走：query 命中的实体 → 一跳邻居的事实与超边成员。"""
    toks = plan.get("tokens") or []
    if not toks:
        return []
    out: list[str] = []
    with DB.reader(cfg.db_path) as conn:
        marks = ",".join("?" * len(toks))
        rows = conn.execute(
            f"""SELECT id FROM entity
                WHERE (?='*' OR session_id=?) AND (norm_name IN ({marks}))
                LIMIT 12""",
            (sid, sid, *[t.lower() for t in toks]),
        ).fetchall()
        ent_ids = [r["id"] for r in rows]
        if not ent_ids:
            return []
        marks_e = ",".join("?" * len(ent_ids))
        frows = conn.execute(
            f"""SELECT id FROM fact
                WHERE subject_id IN ({marks_e}) AND status!='invalidated' AND t_ingest <= ?
                ORDER BY salience DESC LIMIT ?""",
            (*ent_ids, as_of, k),
        ).fetchall()
        out.extend(f"fact:{r['id']}" for r in frows)

        orows = conn.execute(
            f"""SELECT id FROM fact
                WHERE object_entity_id IN ({marks_e}) AND status!='invalidated' AND t_ingest <= ?
                ORDER BY salience DESC LIMIT ?""",
            (*ent_ids, as_of, max(2, k // 2)),
        ).fetchall()
        out.extend(f"fact:{r['id']}" for r in orows)

        erows = conn.execute(
            f"""SELECT DISTINCT m.member_id, m.member_type FROM hyperedge_member m
                WHERE m.member_type='entity' AND m.member_id IN ({marks_e}) LIMIT 20""",
            tuple(ent_ids),
        ).fetchall()
        for r in erows:
            mrows = conn.execute(
                """SELECT member_type, member_id FROM hyperedge_member
                   WHERE edge_id IN (SELECT edge_id FROM hyperedge_member
                                     WHERE member_type='entity' AND member_id=?)
                     AND member_type IN ('fact','episode') LIMIT 10""",
                (r["member_id"],),
            ).fetchall()
            for m in mrows:
                out.append(f"{m['member_type']}:{m['member_id']}")
    return list(dict.fromkeys(out))


def _load_candidate(cfg: Config, kind: str, real_id: str, sid: str, as_of: float, plan: dict) -> SC.Candidate | None:
    with DB.reader(cfg.db_path) as conn:
        if kind == "fact":
            r = conn.execute(
                """SELECT f.*, e.name AS subject_name FROM fact f
                   LEFT JOIN entity e ON e.id=f.subject_id WHERE f.id=?""",
                (real_id,),
            ).fetchone()
            if r is None or r["status"] == "invalidated":
                return None
            if r["t_ingest"] > as_of:
                return None
            if r["t_valid_from"] > as_of:
                return None
            # 世界时间有效性
            if r["t_valid_to"] is not None and r["t_valid_to"] < as_of and r["t_invalid"] is not None:
                return None
            neg = r["polarity"] < 0
            conflict = bool(plan.get("wants_negative")) != neg and plan.get("intent") == "polarity"
            subj = r["subject_name"] or r["subject_id"]
            neg_mark = "不" if neg else ""
            base = f"{subj}：{neg_mark}{r['predicate']} {r['object_text']}"
            summary = f"{subj}{neg_mark}{r['predicate']} {r['object_text']}"
            return SC.Candidate(
                kind="fact",
                ref_id=real_id,
                subject=r["subject_id"],
                predicate=r["predicate"],
                object_text=r["object_text"],
                polarity=r["polarity"],
                t_ref=r["t_valid_from"],
                salience=r["salience"],
                access_count=r["access_count"] or 0,
                polarity_conflict=conflict,
                base_text=base,
                summary=summary,
                raw_text=r["object_text"],
                meta={"source_episode": r["source_episode"], "confidence": r["confidence"]},
            )
        if kind == "episode":
            r = conn.execute("SELECT * FROM episode WHERE id=?", (real_id,)).fetchone()
            if r is None or r["archived"]:
                return None
            if r["t_ingest"] > as_of:
                return None
            # 冷却：刚被想起过的不再重复浮现，否则会反复说同一句话
            if r["muted"]:
                return None
            if r["cooldown_until"] and r["cooldown_until"] > as_of:
                return None
            # 已被订正的旧说法：默认不再浮现，但保留可查（时间旅行能看到）
            if r["superseded_by"]:
                return None
            emo = r["emotion"] or SC.emotion_intensity(r["text"] or "")
            return SC.Candidate(
                kind="episode",
                ref_id=real_id,
                t_ref=r["t_event"],
                salience=r["salience"],
                emotion=emo,
                access_count=r["access_count"] or 0,
                last_recall=r["last_access"] or 0.0,
                author=r["author"],
                tier=r["tier"],
                memory_type=r["memory_type"],
                cooldown_until=r["cooldown_until"] or 0.0,
                correction_of=r["correction_of"],
                base_text=(r["text"] or "")[:60],
                summary=(r["text"] or "")[:90],
                raw_text=r["text"] or "",
                meta={
                    "role": r["role"],
                    "t_event": r["t_event"],
                    "source": r["source"],
                    "author": r["author"],
                    "type": r["memory_type"],
                },
            )
        if kind == "theme":
            r = conn.execute("SELECT * FROM hyperedge WHERE id=?", (real_id,)).fetchone()
            if r is None:
                return None
            return SC.Candidate(
                kind="theme",
                ref_id=real_id,
                t_ref=r["t_from"],
                salience=r["salience"],
                base_text=r["label"],
                summary=r["summary"] or "",
                raw_text=r["summary"] or "",
                meta={"kind": r["kind"]},
            )
    return None


def _refine_cosine(cfg: Config, embedder: Embedder, query: str, seen: dict, dense_ids: list[str]) -> None:
    qv = embedder.embed_blob(query)
    want = {rid.split(":", 1)[1]: rid.split(":", 1)[0] for rid in dense_ids}
    ids = [i for i in want if i in seen]
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(f"SELECT id, embedding FROM fact WHERE id IN ({placeholders})", ids).fetchall()
        rows += conn.execute(f"SELECT id, embedding FROM episode WHERE id IN ({placeholders})", ids).fetchall()
    for r in rows:
        if r["embedding"] and r["id"] in seen:
            sim = V.cosine(qv, r["embedding"], cfg.embed_dim)
            seen[r["id"]].rel = max(0.0, sim)


# ── 阶段 4：预算打包（质量优先，三阶段）────────────────────────
def _pack(
    cfg: Config,
    cands: list[SC.Candidate],
    budget: int,
    skip: set[str],
    plan: dict,
) -> tuple[list[RecallItem], int, list[dict]]:
    """预算约束下的记忆组装。

    优化顺序是硬的：**A 按质量选 → B 压表达 → C 才砍内容**。
    不允许出现"为了省 token 换成一条更便宜但更差的记忆"。
    """
    bp = cfg.budget
    avail = [c for c in cands if c.ref_id not in skip]
    if not avail:
        return [], 0, []

    states = [c for c in avail if c.kind in ("state", "resident")]
    others = [c for c in avail if c.kind not in ("state", "resident")]
    # 「重要」档插队到普通记忆之前（不是替代常驻）
    imp = [c for c in others if c.tier == "important"]
    plain = [c for c in others if c.tier != "important"]
    others = imp + plain
    excluded: list[dict] = []

    # ── 阶段 A：质量筛选 ──────────────────────────────────
    # 预算大约能装多少条（按一条 L1 约 22 token 估），再乘候选池倍数
    approx_slots = max(bp.floor_facts, int(budget / 22))
    pool_size = int(approx_slots * bp.pool_multiplier)
    pool = others[:pool_size]

    # 冗余过滤（MMR）：同一事实/高度相似的情节只留一条
    kept: list[SC.Candidate] = []
    for c in pool:
        f = SC.mmr_factor(c, kept, bp.redundancy_lambda)
        if f <= 0.05:
            excluded.append({"id": c.ref_id, "kind": c.kind, "why": "与已选条目完全重复"})
            continue
        c.score *= f
        kept.append(c)

    # ── 阶段 B：全保真起步，超预算再压表达 ────────────────
    entries: list[list] = []  # [cand, level, text, tokens]
    for c in states:
        t = c.text_at(1)
        entries.append([c, 1, t, SC.estimate_tokens(t)])
    for c in kept:
        lv = 3 if c.kind == "episode" else 2
        t = c.text_at(lv)
        entries.append([c, lv, t, SC.estimate_tokens(t)])

    total = sum(e[3] for e in entries)

    if total > budget:
        # 从得分最低的开始降级表达——先降"低分的说法"，不降"高分的内容"
        # 常驻条目永不降级：它们代表"我一直知道的事"，压掉就是失忆
        degradable = sorted(
            [e for e in entries if e[0].kind not in ("state", "resident")],
            key=lambda e: e[0].score,
        )
        for e in degradable:
            if total <= budget:
                break
            c, lv = e[0], e[1]
            while lv > 1 and total > budget:
                lv -= 1
                nt = c.text_at(lv)
                ntok = SC.estimate_tokens(nt)
                if ntok < e[3]:
                    total -= e[3] - ntok
                    e[1], e[2], e[3] = lv, nt, ntok

    # ── 阶段 C：仍超预算才砍内容（保性能地板）────────────
    if total > budget:
        floor_ids = {e[0].ref_id for e in entries if e[0].kind in ("state", "resident")}
        non_state = [e for e in entries if e[0].kind not in ("state", "resident")]
        for e in non_state[: bp.floor_facts]:
            floor_ids.add(e[0].ref_id)

        drop = sorted([e for e in entries if e[0].ref_id not in floor_ids], key=lambda e: e[0].score)
        for e in drop:
            if total <= budget:
                break
            total -= e[3]
            excluded.append({"id": e[0].ref_id, "kind": e[0].kind, "why": "降级后仍超预算，按低分裁剪"})
            entries = [x for x in entries if x is not e]

    # 预算有余量时，把高分项升回完整形态
    left = budget - total
    if left > 0:
        upgradable = sorted(
            [e for e in entries if e[0].kind == "episode" and e[1] < 3],
            key=lambda e: e[0].score,
            reverse=True,
        )[:4]
        for e in upgradable:
            c = e[0]
            for lv in range(3, e[1], -1):
                nt = c.text_at(lv)
                ntok = SC.estimate_tokens(nt)
                delta = ntok - e[3]
                if 0 < delta <= budget - total:
                    total += delta
                    e[1], e[2], e[3] = lv, nt, ntok
                    break

    items = [
        RecallItem(e[0].kind, e[0].ref_id, e[1], e[2], e[0].score, e[3], e[0].meta or {})
        for e in entries
    ]
    items.sort(key=lambda i: (0 if i.kind == "state" else 1, -i.score))
    return items, total, excluded


def _render(items: list[RecallItem]) -> str:
    """把选中的记忆渲染成可注入的上下文块。"""
    if not items:
        return ""
    groups = {"page": [], "resident": [], "state": [], "fact": [], "episode": [], "theme": []}
    for i in items:
        groups.setdefault(i.kind, []).append(i)

    lines: list[str] = ["<memory>"]
    if groups["page"]:
        lines.append(groups["page"][0].text)
    if groups["state"]:
        lines.append("【当前状态】")
        lines.extend(f"- {i.text}" for i in groups["state"])
    if groups["resident"]:
        lines.append("【一直知道的】")
        lines.extend(f"- {i.text}" for i in groups["resident"])
    if groups["fact"]:
        lines.append("【已知事实】")
        lines.extend(f"- {i.text}" for i in groups["fact"])
    if groups["theme"]:
        lines.append("【长期主题】")
        lines.extend(f"- {i.text}" for i in groups["theme"])
    if groups["episode"]:
        lines.append("【想起来的片段】")
        lines.extend(f"- {i.text}" for i in groups["episode"])
    lines.append("</memory>")
    return "\n".join(lines)


def _log(cfg: Config, sid: str | None, res: RecallResult) -> None:
    try:
        import json

        with DB.tx(cfg.db_path) as conn:
            conn.execute(
                """INSERT INTO retrieval_log(ts,session_id,query,plan_json,selected_json,excluded_json,latency_ms,abstained)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    time.time(),
                    sid,
                    res.query,
                    json.dumps(res.plan, ensure_ascii=False),
                    json.dumps([i.ref_id for i in res.items]),
                    json.dumps(res.excluded[:10], ensure_ascii=False),
                    res.latency_ms,
                    1 if res.abstained else 0,
                ),
            )
    except Exception as e:  # 审计失败不影响主流程
        log.debug("recall log failed: %s", e)
