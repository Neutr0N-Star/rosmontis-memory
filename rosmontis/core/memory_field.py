"""记忆场：激活、扩散、衰减（纯数学层，不碰数据库）。

本模块只做场动力学的计算，数据访问在 `core/field.py`。分成两层是为了让
扩散/激活可以被单测直接喂数组，不必建库。

三个来源，都是**离散图上的实现**，不引入 PDE 求解器：

1. **ACT-R 基础激活**（docs/15 §1.1）
   `A_i(t) = ln( Σ_j (t - t_j)^-d ) + B_i`，`d = 0.5`，`B_i = salience × 2`。
   比单纯的 `access_count` 精细：同样被访问 5 次，昨天访问 5 次和一年前
   访问 5 次，激活完全不同。`activation_odds()` 再把 A 映射到 0-1，供打分用。

2. **图扩散激活**（离散版 `D∇²φ`）
   连续场上的拉普拉斯算子，在图上就是「邻居均值减自身」。本模块用
   **稀疏邻接表的行归一化随机游走**实现稳定离散热传导：

       φ' = (1 - α)·φ + α·Âφ,     Â = D⁻¹A

   固定 α∈[0,1] 时 φ' 是 φ 与邻居均值的凸组合，因此**有界、不发散**；
   `α` 取空间衰减倍率，identity 几乎不扩散、operational 扩散最快。
   复杂度 O(E) / 次，迭代次数上限 `MAX_HOPS = 3`，全程没有稠密矩阵。

3. **三空间衰减率**（docs/15 §1.1）
   `identity 0.1 / knowledge 1.0 / operational 3.0`，数字越大衰减越快。
   - 打分：乘在激活项上（人格不靠"最近想起"排序，操作状态靠它）；
   - 遗忘：乘在 Δt 上（identity 的 S 除以 0.1，等于半衰期 ×10）。

安全阀：结构关键节点（连接度高）永远不归档。删掉枢纽会让图断成孤岛。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# 空间 → 衰减倍率（数字越大衰减越快）
SPACE_DECAY = {
    "identity": 0.10,     # 人格、种子、常驻状态：几乎不衰减
    "knowledge": 1.00,    # 事实、状态、实体、主题：按相关度生死
    "operational": 3.00,  # 会话情节、冷却、工具行为：烧得快，清得也快
}
SPACES = ("identity", "knowledge", "operational")

# ACT-R 参数
ACT_R_DECAY = 0.5      # 时间衰减指数 d
BASE_ACTIVATION = 0.0  # 基础激活
# ACT-R 用「秒」作时间单位时，单条 1 秒前 / 1 天前的迹分别约 0 / -5.68。
# 取两者在对数时间上的中点做 logistic 中心：1 天前 ≈ 0.5，1 小时前 ≈ 0.8。
ACT_R_CENTER = -0.25 * math.log(86400.0)  # ≈ -2.84

# 扩散参数
DEFAULT_GAMMA = 0.45   # 每轮扩散的混合系数 α
MAX_HOPS = 3


def classify_space(kind: str = "", tier: str = "", source: str = "", memory_type: str = "") -> str:
    """把一条记忆映射到三种空间之一（docs/15 §1.1 的归属口径）。

    输入输出
        kind: persona / episode / fact / state / entity / theme / page / resident
        tier: resident / retrieval / important
        source: dialogue / summary / self_reflection / seed / dream ...
        memory_type: fact / preference / principle / lesson / tool_behavior ...
        返回 "identity" | "knowledge" | "operational"
    """
    k = (kind or "").lower()
    tr = (tier or "").lower()
    s = (source or "").lower()
    mt = (memory_type or "").lower()

    if k == "persona" or mt == "identity" or s == "seed":
        return "identity"
    if k in ("entity", "fact", "state", "theme", "page"):
        return "knowledge"
    if k in ("episode", "resident"):
        if tr == "resident":
            return "identity"  # 常驻：一直知道的事，不按情节衰减
        if s in ("summary", "self_reflection"):
            return "knowledge"
        if mt in ("principle", "preference", "lesson"):
            return "knowledge"
        return "operational"
    if mt == "tool_behavior":
        return "operational"
    return "knowledge"


def space_of(kind: str, tier: str = "", source: str = "", memory_type: str = "") -> str:
    """`classify_space` 的旧签名包装（保留给已有调用方）。"""
    return classify_space(kind, tier, source, memory_type)


def decay_factor(space: str) -> float:
    """空间衰减倍率。未知空间按 knowledge 处理。"""
    return SPACE_DECAY.get(space, 1.0)


def space_weight(space: str) -> float:
    """空间倍率相对 knowledge 的比值：identity 0.1 / knowledge 1.0 / operational 3.0。

    打分时乘在激活项上：人格几乎不因"最近被想起"而加权，操作状态加权最猛。
    """
    return decay_factor(space) / SPACE_DECAY["knowledge"]


# ── ACT-R 激活 ────────────────────────────────────────────────
def activation_value(traces: list[float] | tuple[float, ...], salience: float, now: float) -> float:
    """ACT-R 幂律基础激活 `A = ln(Σ_j (t-t_j)^-0.5) + B`，`B = salience × 2`。

    参数
        traces: 每次痕迹的时间戳（epoch 秒）；只看最近 32 条，避免长历史压死
        salience: 0-1 静态重要度，作为 B_i
        now: 当前时间（epoch 秒）
    返回
        float，最近/越多痕迹越高；同样次数越久远越低。可为负。
    """
    base = BASE_ACTIVATION + 2.0 * max(0.0, min(1.0, float(salience or 0.0)))
    total = 0.0
    for tj in list(traces)[-32:]:
        try:
            dt = max(1.0, float(now) - float(tj))
        except (TypeError, ValueError):
            continue
        total += dt ** (-ACT_R_DECAY)
    if total <= 0.0:
        return base
    return math.log(total) + base


def activation_odds(a: float) -> float:
    """把 ACT-R 激活值映射为 0-1 的召回强度（logistic）。

    A 是对数量纲，直接当线性系数会在大负数区把分数压死。以 1 天前单条
    痕迹的激活为中心做 logistic 后，时间衰减仍然单调，但落在可用的 (0,1)。
    """
    x = max(-60.0, min(60.0, float(a) - ACT_R_CENTER))
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass
class Activation:
    """一条记忆的激活状态（内存对象，供离线演算与单测使用）。"""

    base: float = 0.5            # 静态重要度（= salience）
    traces: list[float] = field(default_factory=list)  # 每次访问的时间戳
    space: str = "knowledge"

    def act_r(self, now: float) -> float:
        """ACT-R 基础激活。没有访问记录时退化为静态重要度。"""
        return activation_value(self.traces, self.base, now)

    def retention(self, now: float, half_life_days: float) -> float:
        """保留率。空间倍率作用在有效半衰期上：identity 更慢，operational 更快。"""
        last = self.traces[-1] if self.traces else now
        days = max(0.0, now - last) / 86400.0
        eff_half = max(1.0, half_life_days / max(1e-6, decay_factor(self.space)))
        return 0.5 ** (days / eff_half)


# ── 稀疏扩散 ──────────────────────────────────────────────────
def laplacian_step(
    phi: dict[str, float],
    adjacency: dict[str, dict[str, float]],
    alpha: float = DEFAULT_GAMMA,
) -> dict[str, float]:
    """图上的离散热传导一步：φ' = (1-α)φ + α·D⁻¹Aφ。

    参数
        phi: {node_id: 激活值}
        adjacency: 稀疏邻接表 {node_id: {neighbor_id: weight}}，必须对称
        alpha: 混合系数，clamp 到 [0,1]
    返回
        新的激活字典（含 φ 中原有但无边的孤立节点，值不变）

    复杂度 O(E)；α≤1 时结果是"自身与邻居加权均值"的凸组合，**有界不发散**。
    没有使用任何稠密矩阵。
    """
    if not phi:
        return {}
    a = max(0.0, min(1.0, float(alpha)))
    degree: dict[str, float] = {}
    for node, nbrs in adjacency.items():
        if nbrs:
            degree[node] = sum(max(0.0, float(w)) for w in nbrs.values())

    out: dict[str, float] = dict(phi)
    for node, nbrs in adjacency.items():
        deg = degree.get(node) or 0.0
        if deg <= 0.0:
            continue
        acc = 0.0
        for nb, w in nbrs.items():
            ww = max(0.0, float(w))
            if ww > 0.0:
                acc += phi.get(nb, 0.0) * ww
        out[node] = (1.0 - a) * phi.get(node, 0.0) + a * (acc / deg)
    return out


def spread(
    seeds: dict[str, float],
    adjacency: dict[str, list[tuple[str, float]]],
    gamma: float = DEFAULT_GAMMA,
    hops: int = MAX_HOPS,
    floor: float = 0.05,
) -> dict[str, float]:
    """多轮扩散的便捷包装（列表式邻接表，兼容旧调用方）。

    参数
        seeds: {node_id: 激活值}
        adjacency: {node_id: [(neighbor_id, weight)]}
        gamma: 每轮混合系数
        hops: 迭代轮数，上限 MAX_HOPS
        floor: 低于此值的节点不再保留，防止无意义的长尾
    返回
        {node_id: 激活值}，含仍高于 floor 的种子
    """
    adj = {k: {nb: w for nb, w in v} for k, v in (adjacency or {}).items()}
    phi = dict(seeds or {})
    iters = max(0, min(int(hops if hops is not None else MAX_HOPS), MAX_HOPS))
    for _ in range(iters):
        phi = laplacian_step(phi, adj, gamma)
        phi = {k: v for k, v in phi.items() if v >= floor}
        if not phi:
            break
    return phi


def critical_nodes(adjacency: dict[str, list[tuple[str, float]]] | dict[str, dict[str, float]], top_k: int = 20) -> set[str]:
    """结构关键节点：度中心性最高的那批。

    完整做法是找割点（Tarjan），但在这个规模下度高 ≈ 关键——删掉高频实体，
    它连着的那些记忆就永久孤岛化了。用度做近似，成本低一个数量级。
    """
    degree: dict[str, float] = {}
    for node, nbrs in (adjacency or {}).items():
        items = nbrs.items() if isinstance(nbrs, dict) else nbrs
        for nb, w in items:
            degree[node] = degree.get(node, 0.0) + float(w)
            degree[nb] = degree.get(nb, 0.0) + float(w)
    if not degree:
        return set()
    ranked = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)
    threshold = ranked[min(len(ranked) - 1, max(0, top_k - 1))][1]
    if threshold <= 0:
        return set()
    return {n for n, d in degree.items() if d >= threshold}


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """把一组激活值线性归一到 0-1（用于展示/指标，不用于相对排序）。

    注意：这是 min-max 归一；等价于排序时保留单调性，仅在需要固定量纲时用。
    """
    if not scores:
        return {}
    vals = sorted(scores.values())
    lo, hi = vals[0], vals[-1]
    if hi - lo < 1e-9:
        return {k: 0.5 for k in scores}
    span = hi - lo
    return {k: max(0.0, min(1.0, (v - lo) / span)) for k, v in scores.items()}
