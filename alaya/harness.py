"""Harness：把记忆接进对话循环的那根线。

    h = Harness(cfg, llm_callable)
    h.wake("u1")
    r = h.turn("我今天有点累", "u1")
    print(r["reply"])
    h.sleep("u1")

它做的是三件事的合称：一圈固定循环、一套工具、一个节拍。
详细设计见 docs/12-harness设计.md。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .config import Config
from .core.memory import Memory
from .tools import TOOLS, dispatch

log = logging.getLogger("alaya.harness")

MAX_TOOL_ROUNDS = 3


@dataclass
class TurnResult:
    reply: str
    memory_used: dict = field(default_factory=dict)
    tools_called: list[dict] = field(default_factory=list)
    abstained: bool = False
    latency_ms: float = 0.0
    rounds: int = 1

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Harness:
    """包在模型外面的一圈流程。

    `llm_callable(messages, tools) -> {"content": str, "tool_calls": [{"name","arguments"}]}`

    不绑定任何 SDK：OpenAI、Ollama、DeepSeek 或一个假模型都能塞进来。
    """

    def __init__(
        self,
        cfg: Config | None = None,
        llm_callable: Callable | None = None,
        system_prompt: str | None = None,
        auto_tick: bool = True,
        memory_budget: int | None = None,
    ):
        self.cfg = cfg or Config()
        self.mem = Memory(self.cfg)
        self.llm = llm_callable
        self.system_prompt = system_prompt or "你在和一个人长期相处，自然一点，不要每次都复述记忆。"
        self.auto_tick = auto_tick
        self.memory_budget = memory_budget
        self._lock = threading.Lock()
        self._bg: set[threading.Thread] = set()

    # ── 醒 ────────────────────────────────────────────────
    def wake(self, session_id: str = "default") -> dict:
        """会话开始：播种、载入人格、报告状态。

        不返回 prompt——prompt 每轮由 build_context 现编译，这样改人设能立刻生效。
        """
        self.mem.seed(session_id)
        p = self.mem.persona_compile(session_id)
        return {
            "session_id": session_id,
            "persona_chars": p["chars"],
            "persona_budget": p["budget"],
            "persona_items": p["included"],
            "inbox": self.mem.inbox(session_id),
        }

    # ── 一息 ──────────────────────────────────────────────
    def turn(self, user_text: str, session_id: str = "default") -> dict:
        """一轮对话。"""
        t0 = time.perf_counter()
        text = (user_text or "").strip()

        # 空输入必须挡住。之前不挡的后果是模型自己编一段小说，
        # 再被当成"她说过的话"写进长期记忆——最脏的一种污染。
        if not text:
            raise ValueError("text 不能为空")

        # ① 先落库再说话：抽取异步，不挡路
        if text:
            self.mem.ingest_async(text, session_id=session_id, role="user")

        # ② 节拍放后台，绝不阻塞对话
        if self.auto_tick:
            self._spawn(self._bg_tick, session_id)

        # ③ 组装上下文（人格 + 常驻 + 检索，一次拿全）
        ctx = self.mem.build_context(text or "……", session_id, self.memory_budget)

        # ④ 调模型，必要时跑工具循环
        messages = self._build_messages(ctx, text)
        reply, tools_called, rounds = self._run_loop(messages, session_id)

        # ⑤ 她说过的话也要记：不对称的记忆是残的
        if reply:
            self.mem.ingest_async(reply, session_id=session_id, role="assistant")

        return TurnResult(
            reply=reply,
            memory_used={
                "persona_chars": ctx["parts"]["persona"]["chars"],
                "memory_items": ctx["parts"]["memory"]["items"],
                "memory_tokens": ctx["parts"]["memory"]["tokens"],
                "depth": ctx["parts"]["memory"]["depth"],
                "abstained": ctx["parts"]["memory"]["abstained"],
            },
            tools_called=tools_called,
            abstained=bool(ctx["parts"]["memory"]["abstained"]),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            rounds=rounds,
        ).to_dict()

    def _build_messages(self, ctx: dict, user_text: str) -> list[dict]:
        system = self.system_prompt
        if ctx["parts"]["persona"]["text"]:
            system += "\n\n" + ctx["parts"]["persona"]["text"]
        if ctx["parts"]["memory"]["text"]:
            system += "\n\n" + ctx["parts"]["memory"]["text"]
        messages = [{"role": "system", "content": system}]
        if user_text:
            messages.append({"role": "user", "content": user_text})
        return messages

    def _run_loop(self, messages: list[dict], session_id: str) -> tuple[str, list[dict], int]:
        if self.llm is None:
            return "（没有接模型）", [], 1

        called: list[dict] = []
        reply = ""
        for round_no in range(MAX_TOOL_ROUNDS):
            try:
                resp = self.llm(messages, TOOLS)
            except Exception as e:
                log.warning("llm call failed: %s", e)
                return f"（模型调用失败：{type(e).__name__}）", called, round_no + 1

            resp = resp or {}
            content = (resp.get("content") or "").strip()
            calls = resp.get("tool_calls") or []

            if not content and not calls:
                # 模型返回空：可能是不可用、超时、或拒绝回答。
                # 不能当成正常回复往下走（会写进记忆），必须往上抛。
                raise RuntimeError("模型没有返回内容")

            if content:
                reply = content
            if not calls:
                return reply, called, round_no + 1

            # 把模型这一轮的意图回灌，再执行
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for c in calls:
                name = c.get("name", "")
                args = c.get("arguments") or {}
                result = dispatch(self.mem, name, args, session_id)
                called.append({"tool": name, "args": args, "ok": result.get("ok", False),
                               "reason": result.get("reason", "")})
                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": _stringify(result),
                    }
                )

        # 工具轮次用完，让她带着已有结果收尾
        try:
            resp = self.llm(messages, None)
            reply = (resp or {}).get("content") or reply
        except Exception:
            pass
        return reply or "（工具调用达到上限）", called, MAX_TOOL_ROUNDS

    # ── 眠 ────────────────────────────────────────────────
    def sleep(self, session_id: str = "default") -> dict:
        """会话结束：峰终结算、整理锦囊、够条件就做梦。"""
        out = {"session_id": session_id, "steps": []}

        try:
            out["close"] = self.mem.close_session(session_id)
            out["steps"].append("峰终结算")
        except Exception as e:
            out["close_error"] = str(e)

        try:
            st = self.mem.inbox(session_id)
            if st["should_review"]:
                out["review"] = self.mem.notebook_review(session_id)
                out["steps"].append("整理锦囊")
        except Exception as e:
            out["review_error"] = str(e)

        try:
            d = self.mem.dream(session_id)
            if not d.get("skipped"):
                out["dream"] = d
                out["steps"].append("做梦")
            else:
                out["dream_skipped"] = d.get("reason")
        except Exception as e:
            out["dream_error"] = str(e)

        return out

    # ── 节拍 ──────────────────────────────────────────────
    def tick(self, session_id: str = "default") -> dict:
        return self.mem.tick(session_id)

    def _bg_tick(self, session_id: str) -> None:
        try:
            self.mem.tick(session_id)
        except Exception as e:
            log.debug("bg tick failed: %s", e)

    def _spawn(self, fn, *args) -> None:
        """后台跑，但记住线程，退出时可等。"""
        with self._lock:
            self._bg = {t for t in self._bg if t.is_alive()}
            if len(self._bg) >= 2:
                return  # 最多两个后台任务，避免堆积
            t = threading.Thread(target=fn, args=args, daemon=True)
            self._bg.add(t)
        t.start()

    def join(self, timeout: float = 5.0) -> None:
        """等后台任务收尾。测试和进程退出前调。"""
        with self._lock:
            threads = list(self._bg)
        for t in threads:
            t.join(timeout=timeout)
        with self._lock:
            self._bg.clear()


def _stringify(d: dict) -> str:
    """工具结果转成模型能读的文本。失败原因必须原样带上。"""
    if d.get("ok"):
        parts = [f"{k}={v}" for k, v in d.items() if k != "ok" and v not in (None, "", [], {})]
        return "成功：" + "；".join(parts) if parts else "成功"
    return "失败：" + str(d.get("reason", "未知原因"))
