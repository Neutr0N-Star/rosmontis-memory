"""上下文编译：一次调用拿全该注入的东西。

热重载的工程实现就在这个函数里——人格每次现编译，改完下一轮就是新的。
接入方只需要在每轮对话前调一次，不用监听任何东西。
"""

from __future__ import annotations

import time

from ..config import Config
from . import pages as PG
from . import persona as PER
from . import recall as RC
from . import score as SC
from .embed import Embedder


def build_context(
    cfg: Config,
    embedder: Embedder,
    query: str,
    session_id: str = "default",
    budget: int | None = None,
    as_of: float | None = None,
    include_persona: bool = True,
) -> dict:
    """编译一份可直接塞进 system prompt 的上下文。

    分三块，各有各的预算，互不挪用：
    - persona：我是谁（7600 字上限，整条取舍）
    - resident：我一直知道的（活文档，不检索）
    - recalled：这次想起来的（按问题浮现）

    返回的 `prompt` 已经是拼好的文本，`parts` 里是分块明细。
    """
    t0 = time.perf_counter()
    # `as_of` 用 is None 判：0.0 是"时间轴原点"，不是"没传"
    now = time.time() if as_of is None else float(as_of)

    persona_text = ""
    persona_meta: dict = {}
    if include_persona:
        p = PER.compile_persona(cfg, session_id)
        persona_text = p["text"]
        persona_meta = {k: v for k, v in p.items() if k != "text"}

    r = RC.recall(
        cfg,
        embedder,
        query=query,
        session_id=session_id,
        as_of=as_of,
        budget=budget,
    )

    blocks: list[str] = []
    if persona_text:
        blocks.append(f"<persona>\n{persona_text}\n</persona>")
    if r.prompt:
        blocks.append(r.prompt)

    prompt = "\n\n".join(blocks)
    persona_tokens = SC.estimate_tokens(persona_text)

    return {
        "prompt": prompt,
        "parts": {
            "persona": {
                "text": persona_text,
                "tokens": persona_tokens,
                "chars": len(persona_text),
                "budget": persona_meta.get("budget"),
                "dropped": persona_meta.get("dropped_count", 0),
            },
            "memory": {
                "text": r.prompt,
                "tokens": r.tokens_used,
                "items": len(r.items),
                "abstained": r.abstained,
                "reason": r.reason,
                "depth": r.plan.get("depth"),
                "budget": r.budget,
            },
        },
        "tokens": persona_tokens + r.tokens_used,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "session_id": session_id,
        "as_of": now,
    }


def build_prompt_only(
    cfg: Config,
    embedder: Embedder,
    query: str,
    session_id: str = "default",
    budget: int | None = None,
) -> str:
    """只要文本的简版。给不方便处理 dict 的调用方。"""
    return build_context(cfg, embedder, query, session_id, budget)["prompt"]
