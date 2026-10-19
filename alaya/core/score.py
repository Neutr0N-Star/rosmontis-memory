"""打分、情感强度、token 估算、MMR 去冗余。

这一层决定"哪条记忆值得占预算"，是整个帕累托优化的核心。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..config import BudgetPlan, ScoringWeights
from . import field as FD
from . import memory_field as MF
from . import vec as V

# ── 情感词表：分强度，带否定翻转 ──────────────────────────────
_EMO_STRONG = {
    "崩溃", "绝望", "恨", "恶心", "痛不欲生", "狂喜", "爱死", "最讨厌", "最怕",
    "分手", "去世", "死了", "离婚", "癌症", "自杀", "霸凌", "家暴", "失业",
    "考上", "录取", "求婚", "结婚了", "出生", "怀孕", "手术",
    # 补：之前漏掉的一批日常高频表达。"我今天真的很难受"曾被评为 0.0
    "撑不住", "扛不住", "受不了", "熬不住", "喘不过气", "窒息", "麻木",
    "不想活", "没意思", "活着没意义", "空落落", "心慌", "绞痛", "发抖",
    "发抖", "哭了一整夜", "睡不着", "噩梦", "崩溃了", "炸了",
}
_EMO_MED = {
    "开心", "难过", "害怕", "焦虑", "生气", "委屈", "孤独", "感动", "兴奋", "疲惫",
    "失望", "担心", "喜欢", "讨厌", "想念", "后悔", "害羞", "紧张", "安心", "温暖",
    "痛苦", "幸福", "压力", "烦", "累", "哭", "笑", "怕", "爱",
    "难受", "不舒服", "低落", "沮丧", "无力", "心累", "心酸", "失落", "烦闷",
    "郁闷", "憋着", "冷战", "吵架", "崩溃边缘", "硬撑", "强撑",
}
_EMO_WEAK = {
    "还行", "一般", "有点", "稍微", "还算", "稍微有点", "不太", "挺", "蛮",
}
_NEGATORS = ("不", "没", "别", "无", "非", "未")
# 程度副词：出现时把情感强度抬一档。"特别难受"和"有点难受"不该同分。
_INTENSIFIERS = ("特别", "非常", "超级", "极其", "太", "巨", "好想", "根本", "完全", "真的", "一点也")

_EMO_RE = re.compile("|".join(sorted(map(re.escape, _EMO_STRONG | _EMO_MED), key=len, reverse=True)))
_EXCLAM = re.compile(r"[!！]{2,}")
_QMARK = re.compile(r"[?？]{2,}")
_ELLIPSIS = re.compile(r"(\.{3,}|。{3,}|…{2,})")


def emotion_intensity(text: str, is_session_tail: bool = False, is_peak: bool = False) -> float:
    """0-1 的情感强度。用于打分与重要度。

    三项加权：词汇强度 0.6 + 标点强度 0.2 + 峰终加成 0.2。
    """
    if not text:
        return 0.0
    t = text

    lexical = 0.0
    hits = 0
    for m in _EMO_RE.finditer(t):
        word = m.group(0)
        base = 1.0 if word in _EMO_STRONG else 0.6
        # 否定翻转：前两个字里有否定词则降级而非取反
        pre = t[max(0, m.start() - 2) : m.start()]
        if any(n in pre for n in _NEGATORS):
            base *= 0.45
        lexical = max(lexical, base)
        hits += 1
    if hits >= 3:
        lexical = min(1.0, lexical + 0.15)
    if lexical > 0 and any(w in t for w in _INTENSIFIERS):
        lexical = min(1.0, lexical + 0.22)

    punct = 0.0
    if _EXCLAM.search(t):
        punct = 0.8
    elif _QMARK.search(t):
        punct = 0.55
    elif _ELLIPSIS.search(t):
        punct = 0.35
    elif t.rstrip().endswith(("！", "!")):
        punct = 0.25

    peak = (0.35 if is_peak else 0.0) + (0.2 if is_session_tail else 0.0)
    return float(max(0.0, min(1.0, lexical * 0.6 + punct * 0.2 + peak * 0.2)))


def decouple(emotion: float, factor: float = 0.6, floor: float = 0.15) -> float:
    """做梦时的情绪脱敏。强度降，内容不变；降到下限就停。

    下限是必要的：脱十次就到零，那叫遗忘不叫消化。
    """
    if emotion <= 0:
        return 0.0
    return max(floor, emotion * factor) if emotion > floor else emotion


# ── token 估算 ────────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文约 1.6 字/token，英文约 4 字符/token。"""
    if not text:
        return 0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.6 + other / 4.0) + 2)


# ── 候选与打分 ────────────────────────────────────────────────
@dataclass
class Candidate:
    """一条候选记忆，携带四个呈现级别。"""

    kind: str          # state / fact / episode / theme
    ref_id: str
    subject: str = ""
    predicate: str = ""
    object_text: str = ""
    polarity: int = 1
    t_ref: float = 0.0
    salience: float = 0.5
    emotion: float = 0.0
    access_count: int = 0
    last_recall: float = 0.0      # 最近一次被想起（对话或独白都算）
    # 特化维度
    author: str = "system"        # self / teacher / system
    tier: str = "retrieval"       # resident / retrieval / important
    memory_type: str = "fact"
    cooldown_until: float = 0.0
    muted: bool = False
    correction_of: str | None = None
    # 语义场：空间归属与 ACT-R 激活（docs/15 §1.2 第 3 步）
    space: str = ""                # identity / knowledge / operational，空则按 kind/tier/source 推断
    activation: float | None = None  # A_i(t)，None 时由 t_ref/last_recall/access_count 现算
    # 分项
    rel: float = 0.0
    graph_prox: float = 0.0
    time_fit: float = 0.5
    scope_ok: bool = True
    polarity_conflict: bool = False
    base_text: str = ""       # 用于生成各级文案
    summary: str = ""
    raw_text: str = ""
    meta: dict | None = None

    # 结果
    score: float = 0.0
    level: int = 1
    value_density: float = 0.0
    token_cost: int = 0

    def text_at(self, level: int) -> str:
        if level <= 0:
            return f"[{self.predicate}:{self.object_text}]" if self.predicate else (self.summary or self.raw_text)[:24]
        if level == 1:
            if self.kind == "state":
                return self.base_text or f"{self.predicate} = {self.object_text}"
            return self.base_text or f"{self.predicate} {self.object_text}".strip()
        if level == 2:
            return self.summary or (self.raw_text[:90] + "…" if len(self.raw_text) > 90 else self.raw_text)
        return self.raw_text or self.summary or self.base_text

    def tokens_at(self, level: int) -> int:
        return estimate_tokens(self.text_at(level))


_FIDELITY_DEFAULT = {3: 1.00, 2: 0.86, 1: 0.72, 0: 0.46}


def fidelity(level: int, bp: BudgetPlan | None = None) -> float:
    """某个呈现级别保留了多少信息。用于衡量降级表达造成的质量损失。"""
    if bp is None:
        return _FIDELITY_DEFAULT.get(level, 0.72)
    return {3: bp.fidelity_l3, 2: bp.fidelity_l2, 1: bp.fidelity_l1, 0: bp.fidelity_l0}.get(level, 0.72)


def score_one(
    c: Candidate,
    now: float,
    w: ScoringWeights,
    stale_mode: bool = False,
    tau_days: float = 14.0,
    w_fresh: float = 0.55,
    w_stale: float = 1.00,
) -> float:
    """单条打分。

    stale_mode 是特化开关：
    - False（问"现在"）：新的事更相关，用 recency
    - True（问"以前"或无明确时间）：**久没被想起的事权重更高**，用 stale

    后者是长期陪伴场景的核心差异点。通用系统一律用 recency，结果是 AI
    只记得最近三天，像一个上下文窗口而不是一个人。
    """
    tau = 86400.0 * 30 * (0.5 + c.salience)  # 重要的事衰减更慢
    age = max(0.0, now - c.t_ref) if c.t_ref else 0.0
    recency = math.exp(-age / tau) if tau > 0 else 0.0

    if stale_mode:
        last = c.last_recall or c.t_ref
        days = max(0.0, now - last) / 86400.0 if last else 0.0
        stale = 1.0 - math.exp(-days / tau_days) if tau_days > 0 else 0.0
        time_factor = w_fresh + (w_stale - w_fresh) * stale
    else:
        time_factor = 0.6 * c.time_fit + 0.4 * recency

    raw = (
        w.rel * c.rel
        + w.time * time_factor
        + w.salience * c.salience
        + w.graph * c.graph_prox
        + w.emotion * c.emotion
        - (w.polarity_penalty if c.polarity_conflict else 0.0)
        - (w.scope_penalty if not c.scope_ok else 0.0)
    )
    # docs/15 §1.2：把裸 access_count 加成项换成 ACT-R 激活 A_i(t)，并乘空间倍率。
    # 公式骨架仍是 docs/03 §4.3 的 `final = raw × (1 + α·x)`，α=w.access_boost 不变；
    # 只是 x 从 log(1+access_count) 换成 activation_odds(A)·space_weight：
    #   - 同样 5 次访问，昨天访问和一年前访问的 A 差很多（幂律时间衰减）；
    #   - identity 几乎不吃"最近想起"的加成，operational 吃满（×3）。
    act = c.activation
    if act is None:
        act = FD.activation_from_parts(c.t_ref, c.last_recall, c.access_count, c.salience, now)
    space = c.space or MF.classify_space(c.kind, c.tier, "", c.memory_type)
    boost = 1.0 + w.access_boost * MF.activation_odds(act) * MF.space_weight(space)
    return max(0.0, min(1.5, raw * min(1.5, max(1.0, boost))))


def calibrate_rel(cosines: list[float]) -> list[float]:
    """把余弦映射到 [0,1] 的相关度。

    **不按排名归一化。** 排名归一化有个致命副作用：只要候选中有一条，
    它就一定拿到高分——余弦 0.06 的无关内容会被归一成 1.0，拒答机制直接失效。

    这里用绝对量软映射：余弦 0.20 以下视为无关（映射为 0），0.75 以上视为强相关。
    """
    out = []
    for c in cosines:
        x = max(0.0, min(1.0, float(c)))
        mapped = (x - 0.20) / 0.55
        out.append(max(0.0, min(1.0, mapped)))
    return out


def rank_and_densify(
    cands: list[Candidate],
    now: float,
    w: ScoringWeights,
    bp: BudgetPlan,
    stale_mode: bool = True,
    policy=None,
) -> list[Candidate]:
    """打分 + 校准 + 按**质量**排序。

    排序键是分数，不是「分数 / token 成本」。

    token 只在三个地方起作用：阶段 B 的降级顺序、阶段 C 的裁剪顺序、以及同分决胜。
    """
    if not cands:
        return []
    calibrated = calibrate_rel([c.rel for c in cands])
    for c, n in zip(cands, calibrated):
        c.rel = n

    p = policy
    for c in cands:
        # 语义场：先定空间归属并算一次 ACT-R 激活，供 score_one 使用。
        # 空间按 docs/15 §1.1 的归属口径，激活用库里已有的痕迹字段重建。
        if not c.space:
            c.space = MF.classify_space(
                c.kind,
                c.tier,
                str((c.meta or {}).get("source") or ""),
                c.memory_type,
            )
        if c.activation is None:
            c.activation = FD.activation_from_parts(c.t_ref, c.last_recall, c.access_count, c.salience, now)
        c.score = score_one(
            c,
            now,
            w,
            stale_mode=stale_mode,
            tau_days=getattr(p, "stale_tau_days", 14.0),
            w_fresh=getattr(p, "w_fresh", 0.55),
            w_stale=getattr(p, "w_stale", 1.0),
        )
        c.level = 3 if c.kind == "episode" else 2
        c.token_cost = max(1, c.tokens_at(c.level))
        if c.tier == "important":
            c.score = min(1.5, c.score * 1.18)
        c.value_density = (c.score * fidelity(c.level, bp)) / c.token_cost

    # 同分时按 id 排序，保证结果可复现——不能由输入顺序决定
    cands.sort(key=lambda x: (-x.score, -x.value_density, x.ref_id))
    return cands


def mmr_factor(c: Candidate, selected: list[Candidate], lam: float) -> float:
    """冗余惩罚。

    判据分两类：
    - 结构化条目（fact/state）：看 (subject, predicate, object) 三元组。
      同主语同谓词**不同客体不罚**（喜欢猫和喜欢狗是两件事）。
      极性相反不罚反奖——那是"变了"的证据，正是该一起给模型看的。
    - 原文条目（episode）：看字符二元组 jaccard，判不准就算了，宁可多给一条。
    """
    if not selected:
        return 1.0

    for s in selected:
        if c.kind in ("fact", "state") and s.kind == c.kind:
            if c.subject and s.subject and c.subject == s.subject and c.predicate == s.predicate:
                if (c.object_text or "").strip() == (s.object_text or "").strip():
                    return 0.0  # 完全同一条，硬剔除
                if c.polarity != s.polarity:
                    return 1.0 + lam * 0.4  # 极性相反：加分
            continue
        if c.raw_text and s.raw_text:
            sim = _quick_jaccard(c.raw_text, s.raw_text)
            if sim > 0.92:
                return 0.0
            if sim > 0.6:
                return max(0.0, 1.0 - lam * sim)

    return 1.0


def _quick_jaccard(a: str, b: str) -> float:
    from .text import jaccard

    return jaccard(a, b)


def degrade_options(c: Candidate, current: int) -> list[tuple[int, int, str]]:
    """列出从 current 往下的降级选项：[(级别, 省下 token, 文本)]。"""
    out: list[tuple[int, int, str]] = []
    cur_tok = c.tokens_at(current)
    for lv in range(current - 1, -1, -1):
        t = c.text_at(lv)
        tok = estimate_tokens(t)
        if tok < cur_tok:
            out.append((lv, cur_tok - tok, t))
    return out
