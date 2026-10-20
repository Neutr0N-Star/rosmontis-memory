"""记忆抽取：LLM 结构化输出为主，规则只做兜底。

设计取舍（依据 2026 年主流实践）：
1. 抽取是 LLM 任务，不是正则任务。正则只覆盖"我叫X"这类极小高频模式，
   覆盖率低是故意的——宁可漏抽，也不要抽出错的东西污染长期记忆。
2. 输出用 JSON schema 约束，且明确允许空列表。模型有强烈的"必须产出"倾向，
   不写"可以为空"就会把"用户打了个招呼"也记成记忆。
3. 要求每条事实 self-contained：脱离对话也能看懂。
   "用户同意了第二个选项"这种条目三周后毫无用处，还会误导。
4. 置信度自评，供写入层分流。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from . import timeparse
from .llm import LLM

log = logging.getLogger("rosmontis.extract")

PREDICATES = [
    "工作在", "住在", "喜欢", "讨厌", "害怕", "有", "叫", "是", "关系是",
    "计划", "正在做", "去过", "认识", "属于", "其他",
]

STATE_PREDICATES = {
    "工作在": "job",
    "住在": "city",
    "关系是": "relationship",
}

# LLM 会自由发挥谓词（"搬到城东"→搬到），归一到标准谓词再驱动状态槽。
# 不归一的话 STATE_PREDICATES 查不到，槽就静默丢了。
_PREDICATE_ALIASES = {
    "搬到": "住在",
    "搬回": "住在",
    "迁到": "住在",
    "搬去": "住在",
    "跳槽到": "工作在",
    "跳槽去": "工作在",
    "入职": "工作在",
}

ENTITY_TYPES = ("person", "place", "thing", "concept", "org")

_MEMORY_TYPES = ("fact", "preference", "emotion", "lesson", "principle", "plan", "constraint")


@dataclass
class FactCand:
    subject: str
    predicate: str
    obj: str
    polarity: int = 1
    confidence: float = 0.7
    valid_from_text: str | None = None
    memory_type: str = "fact"


@dataclass
class EntityCand:
    name: str
    type: str = "concept"


@dataclass
class StateCand:
    scope: str
    slot: str
    value: str


@dataclass
class Extraction:
    entities: list[EntityCand] = field(default_factory=list)
    facts: list[FactCand] = field(default_factory=list)
    state_changes: list[StateCand] = field(default_factory=list)
    theme: str | None = None
    source: str = "rule"
    note: str | None = None
    confidence: float = 0.0

    def empty(self) -> bool:
        return not (self.entities or self.facts or self.state_changes)


# ── Prompt：schema 约束 + 允许空 + self-contained + few-shot ──
_SYS = """你是一个记忆提炼器。从对话里提取值得长期记住的信息，输出 JSON。

提取标准（满足才提，不满足就不提）：
- 关于人的持久信息：身份、工作、住处、偏好、恐惧、关系、拥有的东西、长期计划、明确的规则
- 每条必须 self-contained：脱离这段对话也能看懂。写成"用户喜欢喝美式"而不是"用户同意了第二个"
- 情绪只在强烈且明确时提取，一次普通的心情波动不算
- 技术细节、工具报错、系统参数不提取（那不是关于人的信息）

铁律：
- 如果没有值得记住的东西，返回空数组。不要为了不返回空而编造。
- 每类最多 2 条。宁少勿滥。
- **位置、工作、关系这类会变的，必须同时产出 state_changes**（slot: job/city/relationship）——"搬到、搬回、跳槽、分手"都是变更，哪怕话语里没明说"我现在住在"
- **犹豫、打算、纠结（"要不要""在考虑""还没想好"）也是记忆**：predicate 用"计划"，memory_type 用 plan。这类最容易漏
- 否定要正确表达：polarity = -1 表示"不喜欢/不是/没有"

字段说明：
- predicate 必须是：工作在/住在/喜欢/讨厌/害怕/有/叫/是/关系是/计划/正在做/去过/认识/属于/其他
- polarity：1 肯定，-1 否定
- confidence：0-1，用户直接说的接近 0.9，推测的不超过 0.5
- valid_from：如果话语暗示了起始时间（如"三个月前"），填那个说法；没有就 null
- memory_type：fact/preference/emotion/lesson/principle/plan/constraint

输出格式（严格 JSON，不要任何解释）：
{"entities":[{"name":"...","type":"person|place|thing|concept|org"}],
 "facts":[{"subject":"我","predicate":"喜欢","object":"猫","polarity":1,"confidence":0.9,"valid_from":null,"memory_type":"preference"}],
 "state_changes":[{"scope":"我","slot":"job|city|relationship","value":"..."}],
 "theme":"一句话主题或null"}"""

_FEWSHOT = """<示例1>
对话：我在字节跳动做后端，三个月前刚跳过来的。
输出：{"entities":[{"name":"字节跳动","type":"org"}],"facts":[{"subject":"我","predicate":"工作在","object":"字节跳动","polarity":1,"confidence":0.92,"valid_from":"三个月前","memory_type":"fact"}],"state_changes":[{"scope":"我","slot":"job","value":"字节跳动"}],"theme":"工作变动"}

<示例2>
对话：说实话我不太喜欢猫，主要是怕毛。
输出：{"entities":[{"name":"猫","type":"thing"},{"name":"毛","type":"thing"}],"facts":[{"subject":"我","predicate":"喜欢","object":"猫","polarity":-1,"confidence":0.85,"valid_from":null,"memory_type":"preference"},{"subject":"我","predicate":"害怕","object":"毛","polarity":1,"confidence":0.8,"valid_from":null,"memory_type":"emotion"}],"state_changes":[],"theme":"偏好"}

<示例3>
对话：今天天气不错啊，你吃了吗？
输出：{"entities":[],"facts":[],"state_changes":[],"theme":null}

<示例4>
对话：帮我看看这个报错，grep 又返回空了。
输出：{"entities":[],"facts":[],"state_changes":[],"theme":null}

<示例5>
对话：我上个月搬到城东了，离公司近多了。
输出：{"entities":[{"name":"城东","type":"place"}],"facts":[{"subject":"我","predicate":"住在","object":"城东","polarity":1,"confidence":0.9,"valid_from":"上个月","memory_type":"fact"}],"state_changes":[{"scope":"我","slot":"city","value":"城东"}],"theme":"搬家"}

<示例6>
对话：其实我又搬回城西了，还是那边住着习惯。
输出：{"entities":[{"name":"城西","type":"place"}],"facts":[{"subject":"我","predicate":"住在","object":"城西","polarity":1,"confidence":0.9,"valid_from":null,"memory_type":"fact"}],"state_changes":[{"scope":"我","slot":"city","value":"城西"}],"theme":"搬家"}

<示例7>
对话：最近在想要不要辞职去做点自己的事，还没想好。
输出：{"entities":[],"facts":[{"subject":"我","predicate":"计划","object":"辞职去做自己的事","polarity":1,"confidence":0.6,"valid_from":null,"memory_type":"plan"}],"state_changes":[],"theme":"职业纠结"}
"""


def extract(
    text: str,
    llm: LLM | None,
    now: float,
    role: str = "user",
    use_llm: bool = True,
) -> Extraction:
    """主入口。LLM 优先；不可用时落极简规则。"""
    if not text or not text.strip():
        return Extraction(source="none")

    if use_llm and llm is not None and llm.provider != "rule":
        got = _extract_llm(text, llm, now, role)
        if got is not None:
            if got.empty():
                got.note = "llm_empty"
            return got
        fb = _extract_rule(text, now, role)
        fb.note = "llm_failed_fallback_rule"
        return fb

    return _extract_rule(text, now, role)


def _extract_llm(text: str, llm: LLM, now: float, role: str) -> Extraction | None:
    user = f"{_FEWSHOT}\n当前时间：{timeparse.humanize(now)}\n说话人：{role}\n\n<对话>\n{text}\n</对话>\n输出："
    data = llm.chat_json(_SYS, user, max_tokens=700)
    if data is None:
        return None
    if isinstance(data, list):
        data = {"facts": data}
    if not isinstance(data, dict):
        return None

    out = Extraction(source="llm")
    for e in _as_list(data.get("entities")):
        name = str(e.get("name", "")).strip()
        if not name or len(name) > 40:
            continue
        typ = str(e.get("type", "concept")).strip().lower()
        out.entities.append(EntityCand(name=name, type=typ if typ in ENTITY_TYPES else "concept"))

    for f in _as_list(data.get("facts")):
        subj = str(f.get("subject", "")).strip()
        pred = str(f.get("predicate", "")).strip()
        obj = str(f.get("object", "")).strip()
        if not (subj and pred and obj):
            continue
        if len(obj) > 80:
            obj = obj[:80]
        mtype = str(f.get("memory_type", "fact")).strip().lower()
        out.facts.append(
            FactCand(
                subject=subj,
                predicate=_canon_predicate(pred),
                obj=obj,
                polarity=-1 if _to_int(f.get("polarity"), 1) < 0 else 1,
                confidence=_clamp(f.get("confidence"), 0.75),
                valid_from_text=f.get("valid_from") or None,
                memory_type=mtype if mtype in _MEMORY_TYPES else "fact",
            )
        )

    for s in _as_list(data.get("state_changes")):
        scope = str(s.get("scope", "")).strip()
        slot = str(s.get("slot", "")).strip().lower()
        val = str(s.get("value", "")).strip()
        if scope and slot in ("job", "city", "relationship") and val:
            out.state_changes.append(StateCand(scope=scope, slot=slot, value=val[:60]))

    theme = data.get("theme")
    if isinstance(theme, str) and theme.strip() and theme.strip().lower() != "null":
        out.theme = theme.strip()[:30]

    if out.facts:
        out.confidence = sum(f.confidence for f in out.facts) / len(out.facts)
    return out


# ── 极简规则兜底：只处理最基础的几句，不追求覆盖 ────────────
# 刻意保持很少。规则覆盖越多，越容易把自然语言硬掰成错误结构。
_FALLBACK = [
    (re.compile(r"(?:我|俺)\s*叫\s*(?P<o>[^，。！？\s]{1,12})"), "叫", 1, "person"),
    (re.compile(r"(?:我|俺)\s*住(?:在)?\s*(?P<o>[^，。！？\s]{2,16})"), "住在", 1, "place"),
    (re.compile(r"(?:我|俺)\s*在\s*(?P<o>[^，。！？\s]{2,14})\s*(?:上班|工作|实习)"), "工作在", 1, "org"),
]


def _extract_rule(text: str, now: float, role: str) -> Extraction:
    """兜底：只认三个最稳的模式。抽不到就算了，错抽代价更大。"""
    out = Extraction(source="rule_min")
    if role == "assistant":
        return out

    time_hint = timeparse.parse(text, now)
    valid_text = time_hint.raw if time_hint else None

    for pat, pred, pol, etype in _FALLBACK:
        m = pat.search(text)
        if not m:
            continue
        obj = (m.group("o") or "").strip("的了呢吧啊 ")
        if not obj:
            continue
        out.facts.append(
            FactCand(
                subject="我",
                predicate=pred,
                obj=obj,
                polarity=pol,
                confidence=0.55,
                valid_from_text=valid_text,
            )
        )
        out.entities.append(EntityCand(name=obj, type=etype))
        if pred in STATE_PREDICATES:
            out.state_changes.append(StateCand(scope="我", slot=STATE_PREDICATES[pred], value=obj))
    out.confidence = 0.55 if out.facts else 0.0
    return out


# ── 工具 ──────────────────────────────────────────────────────
def _canon_predicate(p: str) -> str:
    p = (p or "").strip()
    if p in PREDICATES:
        return p
    table = {
        "上班": "工作在", "工作": "工作在", "就职于": "工作在", "任职于": "工作在",
        "居住": "住在", "定居": "住在", "搬家": "住在",
        "喜爱": "喜欢", "爱好": "喜欢", "偏爱": "喜欢",
        "不喜欢": "讨厌", "厌恶": "讨厌", "反感": "讨厌",
        "恐惧": "害怕", "怕": "害怕",
        "拥有": "有", "养": "有", "饲养": "有",
        "名字": "叫", "昵称": "叫",
        "计划要": "计划", "打算": "计划", "准备": "计划",
        "去过": "去过", "到过": "去过",
        "处于": "关系是", "关系": "关系是",
    }
    for k, v in table.items():
        if k in p:
            return v
    return "其他"


def _as_list(x) -> list:
    if isinstance(x, list):
        return [i for i in x if isinstance(i, dict)]
    if isinstance(x, dict):
        return [x]
    return []


def _to_int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _clamp(v, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default
