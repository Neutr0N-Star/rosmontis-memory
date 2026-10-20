"""语义场：点火 → 稀疏扩散 → 激活 → 空间归属 → 场指标。

docs/15 §3.2 的 WS-B 接口实现。数据访问全在这里，动力学在 memory_field.py；
本模块对数据库**只读**，唯一副作用是 logging。

四步检索里的前两步（docs/15 §1.2）：

1. 点火 `seed_nodes()`：向量 kNN（top k_vec）+ FTS5 关键词（top k_fts）+
   实体命中（query 解析出的实体及其 1 跳邻域）。三路各自带点火强度，
   同名节点取最大强度并合并来源。
2. 扩散 `propagate()`：从种子出发做稀疏拉普拉斯迭代（≤3 次），
   每次只需当前前沿的邻居，复杂度 O(E × iter)，**没有稠密矩阵**。
   冷启动（无结构边）时直接返回种子，`iterations=0`，不报错。

另外三个契约函数：

- `activation(item_id, t)`：从库里取痕迹时间，按 ACT-R 幂律算 `A_i(t)`；
- `space_of(item)`：identity / knowledge / operational 三空间归属；
- `field_metrics(session_id)`：场规模、边数、度分布熵，供 WS-D 采集。

保护
- 任何一路失败都不拖垮其它路（向量服务挂了仍有 FTS+实体点火）；
- 空种子返回空激活图，由 recall 走现有拒答；
- 迭代次数硬上限 `MAX_HOPS=3`，邻接按前沿增量加载并设边数上限，防环防爆；
- `ALAYA_FIELD=0` 可整路关闭语义场（A/B 与回退用）。
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

from . import db as DB
from . import memory_field as MF
from . import text as TX
from . import vec as V

log = logging.getLogger("alaya.field")

MAX_ITERATIONS = MF.MAX_HOPS          # 迭代硬上限 3
MAX_SEEDS = 96                         # 种子集上限，防止一次点火打爆
MIN_PHI = 1e-3                         # 低于此激活视为噪声，剪掉
MAX_PHI_NODES = 4000                   # 单次扩散保留节点数上限
MAX_EDGES_PER_STEP = 4000              # 单步加载边数上限（保护 10 万级库）
MAX_PAIRS_PER_EDGE = 700               # 单条超边内部最多连多少对
SEMANTIC_MIN_SIM = 0.72                # docs/15 §1.1 语义边阈值（点火用绝对量）

# 边类型：结构边（超边成员）/ 共现边（entity↔fact/episode）/ 溯源边（fact↔episode）
EDGE_KINDS = ("structural", "cooccur", "provenance")

_NODE_KINDS = ("fact", "episode", "entity")

# 线程本地默认上下文：让契约签名 `seed_nodes(query, session_id)` 也能工作。
_tls = threading.local()


# ── 数据结构 ──────────────────────────────────────────────────
@dataclass(frozen=True)
class Seed:
    """一个点火种子。

    key    "fact:fa_xxx" / "episode:ep_xxx" / "entity:en_xxx"
    score  点火强度 0-1（向量用余弦校准，FTS 用排名，实体用 salience）
    source 来源，多路命中用 "+" 连接，例如 "vec+fts"
    """

    key: str
    score: float = 0.5
    source: str = "vec"

    @property
    def kind(self) -> str:
        return self.key.split(":", 1)[0]

    @property
    def ref_id(self) -> str:
        return self.key.split(":", 1)[1] if ":" in self.key else self.key


class PhiMap(dict):
    """`propagate()` 的返回：激活字典 + 本轮场的元信息。

    继承 dict，所以完全满足契约 `-> dict[str, float]`；额外带
    `iterations / edge_types / dist / seed_count`，供 recall 写
    `retrieval_log.plan_json`，不必再查一遍库。
    """

    def __init__(self, data=None, *, iterations: int = 0, edge_types=(), dist=None, seed_count: int = 0):
        super().__init__(data or {})
        self.iterations = int(iterations)
        self.edge_types = tuple(sorted(edge_types or ()))
        self.dist: dict[str, int] = dict(dist or {})
        self.seed_count = int(seed_count)


class _Budget:
    """单步边数预算，防止大库上一条查询把内存打满。"""

    __slots__ = ("left",)

    def __init__(self, limit: int = MAX_EDGES_PER_STEP):
        self.left = max(0, int(limit))

    def take(self) -> bool:
        if self.left <= 0:
            return False
        self.left -= 1
        return True


# ── 上下文（cfg / embedder）──────────────────────────────────
def bind(cfg, embedder=None) -> None:
    """绑定当前线程的默认配置。显式传 cfg 的调用不受影响。"""
    _tls.cfg = cfg
    _tls.embedder = embedder


def _resolve(cfg, embedder=None):
    real_cfg = cfg if cfg is not None else getattr(_tls, "cfg", None)
    real_emb = embedder if embedder is not None else getattr(_tls, "embedder", None)
    return real_cfg, real_emb


def enabled() -> bool:
    """语义场总开关。`ALAYA_FIELD=0/false/off/no` 关闭（其余一律开启）。"""
    v = os.environ.get("ALAYA_FIELD", "1")
    return str(v).strip().lower() not in ("0", "false", "off", "no")


def _as_of(as_of: float | None) -> float:
    return float(as_of) if as_of else time.time()


def _marks(n: int) -> str:
    return ",".join("?" * n)


def _join_key(kind: str, real_id: str) -> str:
    return f"{kind}:{real_id}"


def _split_keys(keys) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        if not isinstance(k, str) or ":" not in k:
            continue
        kind, real = k.split(":", 1)
        if kind in _NODE_KINDS and real:
            buckets[kind].append(real)
    return buckets


# ── 点火：向量 / FTS / 实体 ──────────────────────────────────
def _offer(bucket: dict[str, list], key: str, score: float, source: str) -> None:
    cur = bucket.get(key)
    if cur is None:
        bucket[key] = [float(score), {source}]
        return
    if float(score) > cur[0]:
        cur[0] = float(score)
    cur[1].add(source)


def _calibrate(cos: float) -> float:
    """余弦 → 点火强度。绝对量软映射，不做排名归一（保留"低相似度就是低强度"）。"""
    x = max(0.0, min(1.0, float(cos)))
    return max(0.05, min(1.0, (x - 0.20) / 0.55))


def _fact_live(alias: str = "") -> str:
    """事实的时间轴判据（与 recall.py 的三路召回同口径），占两个 `as_of` 参数。

    - 有效性是区间：`as_of` 落在 `[t_valid_from, t_valid_to)` 内就是当时为真，
      后来被订正不该让它在历史查询里消失；
    - 但 `invalidated` 且区间没有终点（mute / forget 直接改 status）的"死事实"
      永远排除——无法证明它当时真到什么时候，放进来等于让死事实复活。
    """
    a = f"{alias}." if alias else ""
    return (
        f"{a}t_valid_from <= ? AND ({a}t_valid_to IS NULL OR {a}t_valid_to > ?) "
        f"AND ({a}status='active' OR {a}t_valid_to IS NOT NULL)"
    )


def _vec_seeds(cfg, embedder, query: str, sid: str, as_of: float, k: int) -> list[tuple[str, float]]:
    """向量 kNN 点火。embedder 缺失、降级、维度不符都只是少一路，不抛出。"""
    if embedder is None or k <= 0:
        return []
    try:
        blob = embedder.embed_blob(query)
    except Exception as e:  # 向量服务抖动不该拖垮检索
        log.debug("field vec embed failed: %s", e)
        return []
    if not blob:
        return []
    # 能不能比看查询向量自己的长度：cfg.embed_dim 是配置猜测，与模型维度不一致
    # 时按它判会把整库向量判成坏数据（与 recall._dense 同口径）。
    need = len(blob)
    out: list[tuple[float, str]] = []
    try:
        with DB.reader(cfg.db_path) as conn:
            for table, kind, extra, extra_args in (
                # 事实的有效性见 _fact_live：区间判定 + 死事实排除
                ("fact", "fact", _fact_live(), (as_of, as_of)),
                ("episode", "episode", "(archived=0 AND source!='dream')", ()),
            ):
                rows = conn.execute(
                    f"""SELECT id, embedding FROM {table}
                        WHERE (?='*' OR session_id=?) AND embedding IS NOT NULL
                          AND t_ingest<=? AND {extra}
                        ORDER BY t_ingest DESC LIMIT 20000""",
                    (sid, sid, as_of, *extra_args),
                ).fetchall()
                if not rows:
                    continue
                ids = []
                blobs = []
                for r in rows:
                    b = r["embedding"]
                    if b and len(b) == need:
                        ids.append(r["id"])
                        blobs.append(b)
                if not ids:
                    continue
                sims = V.cosine_batch(blob, blobs, int(cfg.embed_dim))
                for i, s in zip(ids, sims):
                    if s > 0.05:
                        out.append((float(s), _join_key(kind, i)))
    except Exception as e:
        log.debug("field vec knn failed: %s", e)
        return []
    out.sort(key=lambda x: x[0], reverse=True)
    return [(key, _calibrate(sim)) for sim, key in out[: max(1, int(k))]]


def _fts_seeds(cfg, query: str, sid: str, as_of: float, k: int) -> list[tuple[str, float]]:
    """FTS5 关键词点火。中文没装 jieba 时用字符二元组，FTS 侧同样按二元组建索引。"""
    if k <= 0:
        return []
    toks = TX.tokenize(query)
    fq = TX.fts_query(toks)
    if not fq:
        return []
    out: list[tuple[str, float]] = []
    try:
        with DB.reader(cfg.db_path) as conn:
            for fts, kind, table, idcol, extra, extra_args in (
                ("fact_fts", "fact", "fact", "fact_id", _fact_live("t"), (as_of, as_of)),
                (
                    "episode_fts", "episode", "episode", "episode_id",
                    "(t.archived=0 AND t.source!='dream')", (),
                ),
            ):
                rows = conn.execute(
                    f"""SELECT t.id AS rid FROM {fts} f
                        JOIN {table} t ON t.id = f.{idcol}
                        WHERE {fts} MATCH ? AND (?='*' OR t.session_id=?)
                          AND t.t_ingest<=? AND {extra}
                        ORDER BY rank LIMIT ?""",
                    (fq, sid, sid, as_of, *extra_args, int(k)),
                ).fetchall()
                for pos, r in enumerate(rows):
                    # 排名越靠前点火越强；最低保留 0.30，免得 FTS 长尾没有扩散价值。
                    out.append((_join_key(kind, r["rid"]), max(0.30, 0.92 - 0.02 * pos)))
    except Exception as e:  # FTS 语法/缺表都不拖垮检索
        log.debug("field fts failed: %s", e)
        return []
    return out


def _entity_seeds(cfg, query: str, sid: str, as_of: float) -> list[tuple[str, float]]:
    """实体命中点火：query 解析出的实体 + 它的 1 跳邻域（fact/episode）。

    实体本身作为图上的中间节点保留（后面扩散要经过它），1 跳邻域直接
    作为种子给较低强度——这样冷启动没有结构边时也还有候选。
    """
    toks = TX.tokenize(query)
    if not toks:
        return []
    ltok = [t.lower() for t in toks][:8]
    out: list[tuple[str, float]] = []
    try:
        with DB.reader(cfg.db_path) as conn:
            # 规范化名精确命中 + 每个 token 的前缀兜底。原先对全表逐行做子串
            # 比较（20k entity 实测 2.06ms），现在两个条件都能走 norm_name 索引
            # （0.01ms）。前缀这一支负责"token 是实体名的一部分"的情形，
            # 例如 token=字节 命中实体「字节跳动」——去掉它召回会明显掉。
            likes = " OR ".join(["norm_name LIKE ?"] * len(ltok))
            rows = conn.execute(
                f"""SELECT id, name, norm_name, salience FROM entity
                    WHERE (?='*' OR session_id=?)
                      AND (norm_name IN ({_marks(len(ltok))})
                           OR (length(norm_name)>=2 AND ({likes})))
                    ORDER BY salience DESC, mentions DESC LIMIT 16""",
                (sid, sid, *ltok, *[t + "%" for t in ltok]),
            ).fetchall()
            ent_ids: list[str] = []
            for r in rows:
                ent_ids.append(r["id"])
                out.append((_join_key("entity", r["id"]), 0.55 + 0.35 * max(0.0, min(1.0, float(r["salience"] or 0.5)))))
            if not ent_ids:
                return out

            frows = conn.execute(
                f"""SELECT id, source_episode FROM fact
                    WHERE (subject_id IN ({_marks(len(ent_ids))})
                           OR object_entity_id IN ({_marks(len(ent_ids))}))
                      AND {_fact_live()}
                      AND t_ingest<=?
                    ORDER BY salience DESC LIMIT 24""",
                (*ent_ids, *ent_ids, as_of, as_of, as_of),
            ).fetchall()
            # fact 的溯源 episode 直接在同一批行里带回，少一次往返。
            seen_eps: list[str] = []
            for r in frows:
                out.append((_join_key("fact", r["id"]), 0.52))
                ep = r["source_episode"] or ""
                if ep and ep not in seen_eps:
                    seen_eps.append(ep)
            for ep in seen_eps[:24]:
                out.append((_join_key("episode", ep), 0.45))
    except Exception as e:
        log.debug("field entity ignition failed: %s", e)
        return out
    return out


def seed_nodes(
    query: str,
    session_id: str,
    *,
    k_vec: int = 24,
    k_fts: int = 24,
    cfg=None,
    embedder=None,
    as_of: float | None = None,
    routes: dict | None = None,
) -> list[Seed]:
    """三路点火，返回按点火强度降序的种子列表。

    参数
        query/session_id: 查询与库分区
        k_vec/k_fts: 向量路、FTS 路各取多少
        cfg/embedder: 不传则用 `bind()` 的线程默认值
        as_of: 时间旅行截止时刻（epoch 秒），默认现在
        routes: 可选 `{"vec": [node_key...], "fts": [node_key...]}`。
                recall 已经算过 dense/sparse 时直接复用，避免同一查询重复扫库。
    返回
        list[Seed]，空列表表示"没点着"（调用方走拒答）。
    """
    cfg, embedder = _resolve(cfg, embedder)
    if cfg is None or not enabled():
        return []
    sid = session_id or "default"
    at = _as_of(as_of)
    bucket: dict[str, list] = {}

    has_routes = bool(routes)
    if has_routes:
        for pos, key in enumerate(routes.get("vec") or []):
            _offer(bucket, str(key), max(0.10, 0.95 - 0.02 * pos), "vec")
        for pos, key in enumerate(routes.get("fts") or []):
            _offer(bucket, str(key), max(0.10, 0.90 - 0.02 * pos), "fts")
    else:
        for key, score in _vec_seeds(cfg, embedder, query, sid, at, int(k_vec)):
            _offer(bucket, key, score, "vec")
        for key, score in _fts_seeds(cfg, query, sid, at, int(k_fts)):
            _offer(bucket, key, score, "fts")

    for key, score in _entity_seeds(cfg, query, sid, at):
        _offer(bucket, key, score, "entity")

    seeds = [Seed(k, round(v[0], 4), "+".join(sorted(v[1]))) for k, v in bucket.items()]
    seeds.sort(key=lambda s: (-s.score, s.key))
    return seeds[:MAX_SEEDS]


# ── 扩散：从前沿增量加载稀疏邻接 ──────────────────────────────
def _add_pair(adj: dict[str, dict[str, float]], a: str, b: str, w: float, budget: _Budget) -> None:
    """加一条无向边（对称存两份），同名边保留最大权重。"""
    if not a or not b or a == b:
        return
    ww = float(w)
    if ww <= 0.0:
        return
    ww = min(1.0, ww)
    if adj.get(a, {}).get(b, 0.0) >= ww:
        return
    if not budget.take():
        return
    adj.setdefault(a, {})[b] = ww
    adj.setdefault(b, {})[a] = ww


def _structural_edges(conn, sid: str, buckets: dict[str, list[str]], adj, kinds: set) -> None:
    """结构边：同一条 hyperedge 内的成员两两相连（docs/15 §1.1）。

    一次 SQL 查完三类前沿成员所在的超边，避免按 kind 往返三次。
    """
    clauses: list[str] = []
    params: list = []
    for mtype in ("fact", "episode", "entity"):
        ids = buckets.get(mtype) or []
        if not ids:
            continue
        clauses.append(f"(m.member_type=? AND m.member_id IN ({_marks(len(ids))}))")
        params.extend([mtype, *ids])
    if not clauses:
        return
    rows = conn.execute(
        f"""SELECT m.edge_id AS eid FROM hyperedge_member m
            JOIN hyperedge h ON h.id = m.edge_id
            WHERE (?='*' OR h.session_id=?) AND ({" OR ".join(clauses)})
            LIMIT 240""",
        (sid, sid, *params),
    ).fetchall()
    edge_ids = list(dict.fromkeys(r["eid"] for r in rows))[:200]
    if not edge_ids:
        return
    rows = conn.execute(
        f"""SELECT edge_id, member_type, member_id, weight FROM hyperedge_member
            WHERE edge_id IN ({_marks(len(edge_ids))}) LIMIT 3000""",
        tuple(edge_ids),
    ).fetchall()
    groups: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for r in rows:
        mt = r["member_type"]
        if mt not in _NODE_KINDS:
            continue
        groups[r["edge_id"]].append((mt, r["member_id"], max(0.01, float(r["weight"] or 1.0))))
    if not groups:
        return
    kinds.add("structural")
    budget = _Budget()
    for _eid, members in groups.items():
        n = len(members)
        if n < 2:
            continue
        if n <= 12:
            pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
        else:
            # 大主题：只把每个成员接到权重最高的 6 个枢纽上，避免 O(n²)
            order = sorted(range(n), key=lambda i: -members[i][2])
            hubs = set(order[:6])
            pairs = ((i, h) for i in range(n) for h in hubs if i != h)
        for i, j in pairs:
            if budget.left <= 0:
                return
            mi, mj = members[i], members[j]
            w = 0.85 * min(mi[2], mj[2]) / max(1, n - 1)
            _add_pair(adj, _join_key(mi[0], mi[1]), _join_key(mj[0], mj[1]), w, budget)


def _fact_edges(conn, buckets: dict[str, list[str]], as_of: float, adj, kinds: set) -> None:
    """共现/溯源边：fact ↔ 它的 entity（主语/客体）与 source_episode。

    三类前沿（fact / entity / episode）合成一条 SQL，减少往返。
    """
    fact_ids = buckets.get("fact") or []
    ent_ids = buckets.get("entity") or []
    ep_ids = buckets.get("episode") or []
    conds: list[str] = []
    params: list = []
    if fact_ids:
        conds.append(f"id IN ({_marks(len(fact_ids))})")
        params.extend(fact_ids)
    if ent_ids:
        conds.append(
            f"(subject_id IN ({_marks(len(ent_ids))}) OR object_entity_id IN ({_marks(len(ent_ids))}))"
        )
        params.extend(ent_ids)
        params.extend(ent_ids)
    if ep_ids:
        conds.append(f"source_episode IN ({_marks(len(ep_ids))})")
        params.extend(ep_ids)
    if not conds:
        return
    try:
        rows = conn.execute(
            f"""SELECT id, subject_id, object_entity_id, source_episode FROM fact
                WHERE ({' OR '.join(conds)})
                  AND {_fact_live()}
                  AND t_ingest<=?
                ORDER BY salience DESC LIMIT 600""",
            (*params, as_of, as_of, as_of),
        ).fetchall()
    except Exception as e:  # 脏数据/缺列只少一类边
        log.debug("field fact edge query failed: %s", e)
        return
    if not rows:
        return
    budget = _Budget()
    ep_keys = {_join_key("episode", e) for e in ep_ids}
    for r in rows:
        fid = r["id"]
        subject = r["subject_id"] or ""
        obj = r["object_entity_id"] or ""
        source_ep = r["source_episode"] or ""
        fkey = _join_key("fact", fid)
        linked = False
        for ent in (subject, obj):
            if ent:
                _add_pair(adj, fkey, _join_key("entity", ent), 0.55 if ent == subject else 0.48, budget)
                linked = True
        if linked:
            kinds.add("cooccur")
        if source_ep:
            _add_pair(adj, fkey, _join_key("episode", source_ep), 0.50, budget)
            kinds.add("provenance")
            if _join_key("episode", source_ep) in ep_keys:
                for ent in (subject, obj):
                    if ent:
                        _add_pair(adj, _join_key("episode", source_ep), _join_key("entity", ent), 0.35, budget)
                        kinds.add("cooccur")


def _neighbors(cfg, sid: str, frontier: list[str], as_of: float) -> tuple[dict[str, dict[str, float]], set[str]]:
    """加载前沿节点的一跳稀疏邻接。返回 (adj, 边类型集合)。"""
    adj: dict[str, dict[str, float]] = {}
    kinds: set[str] = set()
    buckets = _split_keys(frontier)
    if not buckets:
        return adj, kinds
    try:
        with DB.reader(cfg.db_path) as conn:
            _structural_edges(conn, sid, buckets, adj, kinds)
            _fact_edges(conn, buckets, as_of, adj, kinds)
    except Exception as e:  # 读库失败退化为不扩散
        log.debug("field neighbors failed: %s", e)
        return {}, set()
    return adj, kinds


def propagate(
    seeds: list[Seed] | list[str],
    session_id: str,
    *,
    iterations: int = 3,
    decay: float | None = None,
    cfg=None,
    as_of: float | None = None,
) -> PhiMap:
    """从种子出发做稀疏拉普拉斯扩散，返回候选激活 φ。

    参数
        seeds: `seed_nodes()` 的结果（也接受 "kind:id" 字符串，强度按 0.5）
        session_id: 库分区
        iterations: 迭代轮数，clamp 到 [0, MAX_ITERATIONS=3]；0 表示只点火不扩散
        decay: 每轮混合系数 α（默认 DEFAULT_GAMMA=0.45），clamp 到 [0,1]
        cfg/as_of: 同 `seed_nodes`
    返回
        PhiMap（dict 子类）：{node_key: 激活值}，附带 iterations / edge_types /
        dist（节点到种子的跳数）/ seed_count。

    保护
        - 空种子 → 空图，iterations=0；
        - 无结构边（冷启动）→ 原样返回种子，iterations=0，不报错；
        - 迭代上限 + 每步边数预算，环路不会死循环。
    """
    cfg, _ = _resolve(cfg, None)
    sid = session_id or "default"
    at = _as_of(as_of)
    raw_iters = MAX_ITERATIONS if iterations is None else int(iterations)
    iters = max(0, min(raw_iters, MAX_ITERATIONS))
    alpha = MF.DEFAULT_GAMMA if decay is None else max(0.0, min(1.0, float(decay)))

    phi: dict[str, float] = {}
    dist: dict[str, int] = {}
    for s in seeds or []:
        if isinstance(s, Seed):
            key, score = s.key, float(s.score)
        else:
            key, score = str(s), 0.5
        if not key:
            continue
        phi[key] = max(phi.get(key, 0.0), max(0.0, min(1.0, score)))
        dist.setdefault(key, 0)
    if not phi or cfg is None or iters <= 0:
        return PhiMap(phi, iterations=0, edge_types=(), dist=dist, seed_count=len(dist))

    adj: dict[str, dict[str, float]] = {}
    expanded: set[str] = set()
    edge_types: set[str] = set()
    used = 0
    for step in range(iters):
        frontier = [k for k in phi if k not in expanded]
        if not frontier:
            break
        neighbors, kinds = _neighbors(cfg, sid, frontier, at)
        if not neighbors:
            break  # 冷启动 / 没有结构边：退化为纯点火
        for a, nbrs in neighbors.items():
            for b, w in nbrs.items():
                ww = min(1.0, float(w))
                if ww > adj.get(a, {}).get(b, 0.0):
                    adj.setdefault(a, {})[b] = ww
                    adj.setdefault(b, {})[a] = ww
        expanded.update(frontier)
        edge_types |= kinds
        phi = MF.laplacian_step(phi, adj, alpha)
        for k in phi:
            if k not in dist:
                dist[k] = step + 1
        phi = {k: v for k, v in phi.items() if v >= MIN_PHI}
        used = step + 1
        if len(phi) > MAX_PHI_NODES:
            keep = sorted(phi.items(), key=lambda kv: -kv[1])[:MAX_PHI_NODES]
            phi = dict(keep)
            dist = {k: v for k, v in dist.items() if k in phi}

    return PhiMap(phi, iterations=used, edge_types=edge_types, dist=dist, seed_count=len(dist))


# ── ACT-R 激活 ────────────────────────────────────────────────
def activation_from_parts(
    encode_t: float | None,
    last_access: float | None,
    access_count: int,
    salience: float,
    t: float,
) -> float:
    """由库里已有字段重建痕迹时间并算 ACT-R 激活（纯函数，不查库）。

    库里没有逐次访问时间线，只有 `last_access + access_count`：
    - 编码时刻（t_event / t_valid_from）永远算一条痕迹；
    - 最近访问时刻按 `min(access_count, 16)` 次重复计入（次数越多痕迹越密）；
    - 没有 last_access 时退化为只有编码痕迹。
    """
    traces: list[float] = []
    if encode_t:
        try:
            traces.append(float(encode_t))
        except (TypeError, ValueError):
            pass
    n = max(0, int(access_count or 0))
    try:
        last = float(last_access or 0.0)
    except (TypeError, ValueError):
        last = 0.0
    if last > 0:
        traces.extend([last] * min(max(n, 1), 16))
    elif n > 1 and encode_t:
        traces.extend([float(encode_t)] * min(n - 1, 8))
    return MF.activation_value(traces, salience, t)


def activation(item_id: str, t: float, *, cfg=None) -> float:
    """单条记忆在时刻 t 的 ACT-R 激活 `A_i(t)`。

    参数
        item_id: "fact:fa_xxx" / "episode:ep_xxx"，也接受裸 id（自动探测表）
        t: 当前时间（epoch 秒）
        cfg: 不传则用 `bind()` 的线程默认值
    返回
        float；查不到该条目时返回基础激活（不抛错）
    """
    cfg, _ = _resolve(cfg, None)
    if cfg is None or not item_id:
        return MF.BASE_ACTIVATION
    kind, real = (item_id.split(":", 1) + [""])[:2] if ":" in item_id else ("", item_id)
    tables = [(kind, real)] if kind in _NODE_KINDS else [("fact", item_id), ("episode", item_id)]
    for table, rid in tables:
        if not rid:
            continue
        try:
            with DB.reader(cfg.db_path) as conn:
                if table == "fact":
                    r = conn.execute(
                        "SELECT t_valid_from, t_ingest, last_access, access_count, salience FROM fact WHERE id=?",
                        (rid,),
                    ).fetchone()
                    encode = (r["t_valid_from"] or r["t_ingest"]) if r else None
                else:
                    r = conn.execute(
                        "SELECT t_event, last_access, access_count, salience FROM episode WHERE id=?",
                        (rid,),
                    ).fetchone()
                    encode = r["t_event"] if r else None
        except Exception as e:
            log.debug("field activation lookup failed: %s", e)
            continue
        if r is None:
            continue
        return activation_from_parts(encode, r["last_access"], r["access_count"], r["salience"], t)
    return MF.BASE_ACTIVATION


# ── 空间归属 ──────────────────────────────────────────────────
def _field_of(item, name: str, default: str = "") -> str:
    if item is None:
        return default
    if isinstance(item, dict):
        v = item.get(name)
    else:
        v = getattr(item, name, None)
    return str(v) if v is not None else default


def space_of(item: dict) -> str:
    """把条目映射到 identity / knowledge / operational。

    参数
        item: dict（或带同名属性的对象），可含 kind / tier / source / memory_type
    返回
        "identity" | "knowledge" | "operational"
    """
    return MF.classify_space(
        _field_of(item, "kind"),
        _field_of(item, "tier"),
        _field_of(item, "source"),
        _field_of(item, "memory_type"),
    )


# ── 场指标 ──────────────────────────────────────────────────
def _entropy(counts: list[int]) -> float:
    """度分布香农熵（bits）；全零返回 0。"""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return round(ent, 4)


def field_metrics(session_id: str, *, cfg=None) -> dict:
    """场的规模 / 边数 / 度分布熵，供 WS-D 采集与健康检查。

    参数
        session_id: 库分区；"*" 表示全库
        cfg: 不传则用 `bind()` 的线程默认值
    返回
        dict：nodes / node_total / edges / avg_degree / max_degree /
        degree_entropy / structural_nodes / spaces / session_id
    """
    cfg, _ = _resolve(cfg, None)
    sid = session_id or "default"
    empty = {
        "session_id": sid,
        "nodes": {},
        "node_total": 0,
        "edges": {"structural": 0, "cooccur": 0, "provenance": 0, "total": 0},
        "avg_degree": 0.0,
        "max_degree": 0,
        "degree_entropy": 0.0,
        "structural_nodes": 0,
        "spaces": {"identity": 0, "knowledge": 0, "operational": 0},
    }
    if cfg is None:
        return empty
    who = "(?='*' OR session_id=?)"
    args = (sid, sid)
    try:
        with DB.reader(cfg.db_path) as conn:
            def one(sql: str, *a) -> int:
                try:
                    row = conn.execute(sql, a).fetchone()
                    return int(row[0]) if row else 0
                except Exception:
                    return 0

            nodes = {
                "episode": one(f"SELECT COUNT(*) FROM episode WHERE {who} AND archived=0", *args),
                "fact": one(f"SELECT COUNT(*) FROM fact WHERE {who} AND status!='invalidated'", *args),
                "entity": one(f"SELECT COUNT(*) FROM entity WHERE {who}", *args),
                "hyperedge": one(f"SELECT COUNT(*) FROM hyperedge WHERE {who}", *args),
                "state": one(f"SELECT COUNT(*) FROM state_slot WHERE {who}", *args),
            }
            structural = one(
                f"""SELECT COUNT(*) FROM hyperedge_member m
                    JOIN hyperedge h ON h.id=m.edge_id WHERE {who.replace('session_id','h.session_id')}""",
                *args,
            )
            cooccur = one(
                f"""SELECT COALESCE(SUM(1 + (object_entity_id IS NOT NULL)), 0) FROM fact
                    WHERE {who} AND status!='invalidated'""",
                *args,
            )
            provenance = one(
                f"""SELECT COUNT(*) FROM fact WHERE {who} AND status!='invalidated'
                      AND source_episode IS NOT NULL""",
                *args,
            )
            deg_rows = conn.execute(
                f"""SELECT m.member_id AS nid, COUNT(*) AS c FROM hyperedge_member m
                    JOIN hyperedge h ON h.id=m.edge_id
                    WHERE {who.replace('session_id','h.session_id')}
                    GROUP BY m.member_id LIMIT 20000""",
                args,
            ).fetchall()
            degrees = [int(r["c"]) for r in deg_rows]
            spaces = {"identity": 0, "knowledge": 0, "operational": 0}
            srows = conn.execute(
                f"""SELECT tier, source, memory_type, COUNT(*) AS c FROM episode
                    WHERE {who} GROUP BY tier, source, memory_type""",
                args,
            ).fetchall()
            for r in srows:
                sp = MF.classify_space("episode", r["tier"], r["source"], r["memory_type"])
                spaces[sp] += int(r["c"])
    except Exception as e:  # 指标失败不该影响主流程
        log.debug("field_metrics failed: %s", e)
        return empty

    total_edges = structural + cooccur + provenance
    return {
        "session_id": sid,
        "nodes": nodes,
        "node_total": sum(nodes.values()),
        "edges": {
            "structural": structural,
            "cooccur": cooccur,
            "provenance": provenance,
            "total": total_edges,
        },
        "avg_degree": round(sum(degrees) / len(degrees), 3) if degrees else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "degree_entropy": _entropy(degrees),
        "structural_nodes": len(degrees),
        "spaces": spaces,
    }
