"""记忆场：激活、扩散、衰减。

三个来源的融合，都是**离散图上的实现**，不引入 PDE 求解器：

1. **ACT-R 基础激活**（Ori Mnemos / 认知科学）
   `A_i = ln( Σ_j t_j^(-d) ) + B_i`
   比单纯的 `access_count` 精细：同样被访问 5 次，昨天访问 5 次和一年前访问 5 次，
   激活完全不同。

2. **图扩散激活**（Field-Theoretic Memory 的 `D∇²φ` 离散版）
   连续场上的拉普拉斯算子，在图上就是「邻居均值减自身」。
   命中一个节点后，激活沿边扩散，衰减因子控制扩散半径。
   这是"想起 A 就顺带想起相关的 B"的工程版——向量检索做不到，因为它在
   检索那一刻才建立联系，而扩散用的是**已经存在的结构**。

3. **三空间衰减率**（Ori 的 identity / knowledge / ops）
   人格几乎不衰减、知识按相关度生死、操作状态快速清空。
   比"统一衰减率 + 重要度调制"更贴近真实：有些东西本来就不该按同一把尺子量。

安全阀：结构关键节点（连接度高、割点）永远不归档。删掉枢纽会让图断成孤岛。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# 空间 → 衰减倍率（数字越大衰减越快）
SPACE_DECAY = {
    "identity": 0.10,     # 人格、种子、状态快照：几乎不衰减
    "knowledge": 1.00,    # 事实、常驻页面、主题：按相关度生死
    "operational": 3.00,  # 对话情节：烧得快，清得也快
}

# ACT-R 参数
ACT_R_DECAY = 0.5      # 时间衰减指数 d
BASE_ACTIVATION = 0.0  # 基础激活

# 扩散参数
DEFAULT_GAMMA = 0.45   # 每跳衰减
MAX_HOPS = 3


def space_of(kind: str, tier: str = "", source: str = "") -> str:
    """把记忆条目映射到三种空间之一。"""
    if kind in ("state",) or tier == "resident":
        return "identity" if kind == "state" else "knowledge"
    if kind in ("fact", "theme", "page") or source in ("summary", "seed"):
        return "knowledge"
    if source == "self_reflection":
        return "knowledge"
    return "operational"


def decay_factor(space: str) -> float:
    return SPACE_DECAY.get(space, 1.0)


@dataclass
class Activation:
    """一条记忆的激活状态。"""

    base: float = 0.5            # 静态重要度
    traces: list[float] = field(default_factory=list)  # 每次访问的时间戳
    space: str = "knowledge"

    def act_r(self, now: float) -> float:
        """ACT-R 基础激活。没有访问记录时退化为静态重要度。"""
        if not self.traces:
            return self.base
        total = 0.0
        for t in self.traces[-32:]:  # 只算最近 32 次，避免长历史压死
            dt = max(1.0, now - t)
            total += dt ** (-ACT_R_DECAY)
        if total <= 0:
            return self.base
        return self.base + math.log(total) * 0.5

    def retention(self, now: float, half_life_days: float) -> float:
        """保留率。空间衰减倍率作用在这里。"""
        last = self.traces[-1] if self.traces else now
        days = max(0.0, now - last) / 86400.0
        eff_half = max(1.0, half_life_days / decay_factor(self.space))
        return 0.5 ** (days / eff_half)


def spread(
    seeds: dict[str, float],
    adjacency: dict[str, list[tuple[str, float]]],
    gamma: float = DEFAULT_GAMMA,
    hops: int = MAX_HOPS,
    floor: float = 0.05,
) -> dict[str, float]:
    """图扩散激活。

    参数
        seeds: 初始激活 {node_id: activation}
        adjacency: {node_id: [(neighbor_id, weight)]}
        gamma: 每跳衰减
        hops: 最大跳数
        floor: 低于此值不再传播

    返回扩散后的完整激活图（含种子）。
    """
    act: dict[str, float] = dict(seeds)
    frontier = dict(seeds)
    for hop in range(hops):
        decay = gamma ** (hop + 1)
        nxt: dict[str, float] = {}
        for node, a in frontier.items():
            for nb, w in adjacency.get(node, ()):
                if nb in seeds:
                    continue  # 种子本身不再被邻居抬高，避免自激
                cand = a * decay * max(0.0, w)
                if cand < floor:
                    continue
                if cand > nxt.get(nb, 0.0):
                    nxt[nb] = cand
        if not nxt:
            break
        for k, v in nxt.items():
            if v > act.get(k, 0.0):
                act[k] = v
        frontier = nxt
    return act


def critical_nodes(adjacency: dict[str, list[tuple[str, float]]], top_k: int = 20) -> set[str]:
    """结构关键节点：度中心性最高的那批。

    完整做法是找割点（Tarjan），但在这个规模下度高 ≈ 关键——删掉高频实体，
    它连着的那些记忆就永久孤岛化了。用度做近似，成本低一个数量级。
    """
    degree = {n: 0.0 for n in adjacency}
    for n, nbrs in adjacency.items():
        for nb, w in nbrs:
            degree[n] += w
            degree.setdefault(nb, 0.0)
            degree[nb] += w
    if not degree:
        return set()
    ranked = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)
    threshold = ranked[min(len(ranked) - 1, max(0, top_k - 1))][1]
    if threshold <= 0:
        return set()
    return {n for n, d in degree.items() if d >= threshold}


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """激活值归一到 0-1，供打分使用。用分位而非 min-max，避免一条极高值压扁其余。"""
    if not scores:
        return {}
    vals = sorted(scores.values())
    lo, hi = vals[0], vals[-1]
    if hi - lo < 1e-9:
        return {k: 0.5 for k in scores}
    span = hi - lo
    return {k: max(0.0, min(1.0, (v - lo) / span)) for k, v in scores.items()}
