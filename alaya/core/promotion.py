"""升格闸门：一条记忆能不能进常驻、能不能变成原则。

这是整套设计里唯一一处"分数说了不算"的地方——由 (作者, 类型, 确认状态)
三者共同决定。通用记忆系统没有这道闸门，所以它们会在几百轮之后变成
"知道自己所有设定但没有自我"的东西。

规则来源：`docs/00-痛点调研.md` P3、P9。
"""

from __future__ import annotations

# 记忆类型
T_FACT = "fact"
T_EMOTION = "emotion"
T_LESSON = "lesson"
T_PRINCIPLE = "principle"
T_PREFERENCE = "preference"
T_TOOL = "tool_behavior"

MEMORY_TYPES = (T_FACT, T_EMOTION, T_LESSON, T_PRINCIPLE, T_PREFERENCE, T_TOOL)

# 层级
TIER_RESIDENT = "resident"
TIER_RETRIEVAL = "retrieval"
TIER_IMPORTANT = "important"

TIERS = (TIER_RESIDENT, TIER_RETRIEVAL, TIER_IMPORTANT)

# 作者
A_SELF = "self"
A_TEACHER = "teacher"
A_SYSTEM = "system"

# 来源
S_DIALOGUE = "dialogue"
S_REFLECTION = "self_reflection"
S_SEED = "seed"
S_SUMMARY = "summary"
S_MANUAL = "manual"

# 人格分区及其保留优先级（数字小的先保留）
SECTION_ORDER = {
    "identity": 0,
    "rule": 1,
    "speech": 2,
    "trait": 3,
    "taste": 4,
    "free": 5,
}

SECTION_LABEL = {
    "identity": "我是谁",
    "rule": "铁律",
    "speech": "说话方式",
    "trait": "性格",
    "taste": "品味",
    "free": "自由区",
}

# 类型判定的关键词线索（规则兜底用；LLM 可用时优先让模型判）
_TOOL_HINTS = ("工具", "命令", "报错", "grep", "search_files", "list_files", "接口", "参数", "返回空")
_LESSON_HINTS = ("教训", "规矩", "下次", "以后不", "记住了", "错案", "认错", "先查")
_PRINCIPLE_HINTS = ("原则", "铁律", "绝不", "永远不", "必须", "不许", "底线")
_EMOTION_HINTS = ("难过", "开心", "委屈", "生气", "怕", "心疼", "高兴", "烦", "难受", "感动")
_PREF_HINTS = ("喜欢", "讨厌", "顺眼", "别扭", "偏好", "不爱", "爱吃")


def classify_type(text: str, source: str = S_DIALOGUE) -> str:
    """粗判记忆类型。LLM 不可用时的兜底，宁可判 fact 也不乱升格。"""
    t = text or ""
    if source == S_REFLECTION:
        if any(k in t for k in _PRINCIPLE_HINTS):
            return T_PRINCIPLE
        if any(k in t for k in _LESSON_HINTS):
            return T_LESSON
        if any(k in t for k in _TOOL_HINTS):
            return T_TOOL
        if any(k in t for k in _EMOTION_HINTS):
            return T_EMOTION
        return T_FACT
    # 对话与手记：工具行为优先识别，避免技术细节污染人设
    if any(k in t for k in _TOOL_HINTS) and len(t) < 200:
        return T_TOOL
    if any(k in t for k in _PRINCIPLE_HINTS):
        return T_PRINCIPLE
    if any(k in t for k in _LESSON_HINTS):
        return T_LESSON
    if any(k in t for k in _PREF_HINTS):
        return T_PREFERENCE
    if any(k in t for k in _EMOTION_HINTS) and len(t) < 80:
        return T_EMOTION
    return T_FACT


def default_tier(source: str, memory_type: str, author: str) -> str:
    """新条目的默认层级。"""
    if memory_type == T_TOOL:
        return TIER_RETRIEVAL
    if source == S_SEED:
        return TIER_RESIDENT
    if source == S_SUMMARY:
        return TIER_RESIDENT
    if source == S_MANUAL and author == A_TEACHER:
        return TIER_RESIDENT
    if memory_type == T_PRINCIPLE and author == A_SELF:
        return TIER_RETRIEVAL  # 自己写的原则要先被确认才升常驻
    return TIER_RETRIEVAL


def can_promote(
    target_tier: str,
    author: str,
    memory_type: str,
    confirmed: bool,
    source: str = S_DIALOGUE,
) -> tuple[bool, str]:
    """能不能升到目标层级。返回 (是否允许, 理由)。

    默认从宽——大多数组合直接放行。只有一条是真的硬限制：
    **工具行为不能进常驻/人设**，因为技术细节会污染人格表达，
    让 AI 在需要"像个人"的时候突然说起参数和报错。

    其余的组合（作者、是否确认）只作为提示，不阻塞。
    """
    if target_tier not in TIERS:
        return False, f"未知层级 {target_tier}"

    if memory_type == T_TOOL:
        return False, "工具行为不升常驻：技术细节会污染人格表达"

    if target_tier == TIER_RESIDENT:
        if source in (S_SEED, S_SUMMARY):
            return True, "种子与摘要按定义常驻"
        if memory_type == T_EMOTION:
            if not confirmed:
                return False, "单次情绪不进常驻：一次情绪波动会被当成固定的自我认知"
            return True, "情绪已确认"
        return True, "允许"

    if target_tier in (TIER_RETRIEVAL, TIER_IMPORTANT):
        return True, "检索层无限制"

    return True, "允许"


def validate_combo(tier: str, memory_type: str) -> tuple[bool, str]:
    """写入期校验。只挡真正有害的组合。"""
    if tier == TIER_RESIDENT and memory_type == T_TOOL:
        return False, "resident + tool_behavior 非法"
    return True, ""


def describe(memory_type: str) -> str:
    return {
        T_FACT: "事实",
        T_EMOTION: "情绪",
        T_LESSON: "教训",
        T_PRINCIPLE: "原则",
        T_PREFERENCE: "偏好",
        T_TOOL: "工具行为",
    }.get(memory_type, memory_type)
