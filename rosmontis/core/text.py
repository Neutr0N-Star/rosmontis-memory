"""文本处理：中文分词、停用词、n-gram。

jieba 可选；缺失时退化为字符二元组（中文召回率略降但功能完整）。
"""

from __future__ import annotations

import re

try:  # 可选增强
    import jieba  # type: ignore

    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False

_PUNCT = re.compile(r"[\s\u3000-\u303f\uff00-\uff65!-/:-@\[-`{-~]+")
_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")

STOPWORDS = {
    # 中文高频虚词
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "们", "这", "那", "有", "和",
    "就", "都", "也", "还", "但", "而", "与", "或", "被", "把", "给", "让", "对", "从",
    "不", "很", "太", "更", "最", "会", "能", "要", "想", "说", "个", "些", "吧", "吗",
    "啊", "呢", "哦", "嗯", "呀", "哈", "嘛", "着", "过", "地", "得", "之", "其", "为",
    "以", "及", "等", "于", "如", "若", "则", "且", "并", "已", "又", "再", "才", "只",
    # 英文高频
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or",
    "in", "on", "at", "for", "with", "that", "this", "it", "i", "you", "he", "she", "we",
    "they", "do", "does", "did", "have", "has", "had", "not", "but", "so", "if", "as",
}


def has_cjk(text: str) -> bool:
    return bool(_HAS_CJK.search(text or ""))


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    """切词。返回小写 token 列表；中文用 jieba，英文按空白与标点切。"""
    if not text:
        return []
    clean = _PUNCT.sub(" ", text)
    if _HAS_JIEBA:
        toks = [t.strip().lower() for t in jieba.cut_for_search(clean)]
    else:
        toks = _fallback_split(clean)
    out = []
    for t in toks:
        if not t or t.isspace():
            continue
        if not keep_stop and t in STOPWORDS:
            continue
        if len(t) == 1 and not t.isdigit() and not _HAS_CJK.match(t):
            continue
        out.append(t)
    return out


def _fallback_split(clean: str) -> list[str]:
    """无 jieba：英文按空格，中文按二元组。"""
    out: list[str] = []
    for chunk in clean.split():
        if has_cjk(chunk):
            out.extend(chunk[i : i + 2] for i in range(max(1, len(chunk) - 1)))
            if len(chunk) == 1:
                out.append(chunk)
        else:
            out.append(chunk)
    return out


def fts_query(tokens: list[str]) -> str:
    """把 token 拼成 FTS5 查询串：OR 连接，转义双引号。"""
    safe = []
    for t in tokens:
        t = t.replace('"', "")
        if t and t not in STOPWORDS:
            safe.append(f'"{t}"')
    return " OR ".join(safe)


def snippets(text: str, limit: int = 120) -> str:
    """截断展示用。"""
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= limit else t[:limit] + "…"


def bigrams(text: str) -> set[str]:
    """字符二元组集合，用于近似重复检测。"""
    t = _PUNCT.sub("", text or "")
    if len(t) < 2:
        return {t} if t else set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def jaccard(a: str, b: str) -> float:
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
