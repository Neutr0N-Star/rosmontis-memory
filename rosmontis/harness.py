"""Harness：把记忆接进对话循环的那根线。

    h = Harness(cfg, llm_callable)
    h.wake("u1")
    r = h.turn("我今天有点累", "u1")
    print(r["reply"])
    h.sleep("u1")

它做的是三件事的合称：一圈固定循环、一套工具、一个节拍。
详细设计见 docs/12-harness设计.md。

接入契约（WS-C）：
- `turn(user_text, session_id, *, history=None, extra_system=None)`：
  历史缺口已补，宿主可以把上一轮的 messages 直接递进来。
- `build_context(...)`：上下文组装是**公开的模块级函数**，不再埋在 turn 里；
  `pure=True` 时走纯函数路径（不写 retrieval_log），供 rosmontis/api.py 与四钩子宿主复用。
- `llm_callable(messages, tools) -> dict`：返回值可以带 `usage` 与 `finish_reason`，
  缺省也不会崩；`tool_calls` 既接受 OpenAI 原生格式，也接受
  `[{"name","arguments"}]` 的扁平格式，内部统一补 `id`/`tool_call_id`。
- `chat_with_tools(llm, messages, tools)`：OpenAI 兼容 / Ollama 的原生工具调用透传；
  上游没有工具调用能力时返回 None，由调用方安静降级（绝不抛）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .core.memory import Memory
from .core.score import estimate_tokens
from .tools import TOOLS, dispatch

log = logging.getLogger("rosmontis.harness")

MAX_TOOL_ROUNDS = 3
# history 只带最近这么多条，防止宿主把整个上下文塞进来把预算撑爆
MAX_HISTORY_MESSAGES = 40


class LLMUnavailableError(RuntimeError):
    """没有真实模型输出：`llm_callable` 未配置 / 调用失败 / 返回既无 content 也无 tool_calls。

    引擎层的明确信号：调用方（`rosmontis.api.Handle` / HTTP 层）据此回 `llm_unavailable`
    与 503，**绝不**把占位文本当回复写进长期记忆。
    """

    code = "llm_unavailable"
    retryable = True

    def __init__(self, message: str = "没有真实模型输出"):
        super().__init__(message)


@dataclass
class TurnResult:
    reply: str
    memory_used: dict = field(default_factory=dict)
    tools_called: list[dict] = field(default_factory=list)
    abstained: bool = False
    latency_ms: float = 0.0
    rounds: int = 1
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ── 纯函数上下文：不写库、不超预算 ────────────────────────────
_QUIET_LOCK = threading.Lock()
_QUIET_STATE: threading.local | None = None


def _quiet_recall_log() -> threading.local:
    """给 `core.recall._log` 套一层按线程生效的静音开关。

    纯函数契约要求 context 不写库。检索日志是 recall 路径上唯一的写操作，
    这里只在本线程设置静音标记，其它线程照常落日志（不全局关审计）。
    """
    global _QUIET_STATE
    from .core import recall as RC

    with _QUIET_LOCK:
        if _QUIET_STATE is None:
            orig = RC._log
            state = threading.local()

            def _guarded(cfg, sid, res):
                if getattr(state, "quiet", False):
                    return
                return orig(cfg, sid, res)

            RC._log = _guarded  # type: ignore[assignment]
            _QUIET_STATE = state
        return _QUIET_STATE


@contextmanager
def quiet_recall_log():
    """本线程内屏蔽 RetrievalLog 落库（纯函数 context 用）。"""
    state = _quiet_recall_log()
    prev = getattr(state, "quiet", False)
    state.quiet = True
    try:
        yield
    finally:
        state.quiet = prev


def fit_memory_text(text: str, budget: int | None) -> tuple[str, int]:
    """把已渲染的记忆块压到预算内：**整行丢弃，绝不截半句**。

    recall 内部会按反射深度放大预算（habit 1.25x / deep 1.6x），
    公共 API 的「绝不超预算」是硬契约，所以这里做最后一道闸。
    丢弃顺序从尾部开始：状态/常驻在前、情节片段在后，先丢低分片段。
    """
    if not text:
        return "", 0
    tok = estimate_tokens(text)
    if budget is None or budget <= 0 or tok <= budget:
        return text, tok

    lines = text.splitlines()
    wrapped = bool(lines) and lines[0].strip() == "<memory>"
    body = [ln for ln in lines if ln.strip() not in ("<memory>", "</memory>")]
    while body:
        candidate = "\n".join((["<memory>"] if wrapped else []) + body + (["</memory>"] if wrapped else []))
        if estimate_tokens(candidate) <= budget:
            return candidate, estimate_tokens(candidate)
        body.pop()
    # 一条都装不下：返回空而不是硬凑（宁可少注入，也不超预算）
    return "", 0


def build_context(
    mem: Memory,
    query: str,
    session_id: str = "default",
    budget: int | None = None,
    as_of: float | None = None,
    *,
    pure: bool = False,
) -> dict:
    """组装一份可注入的上下文（人格 + 常驻 + 检索）。

    这是 harness.turn 与公共 API 共用的唯一组装入口，不再把逻辑埋在 turn 里。
    `pure=True` 时保证不写库（episode / retrieval_log 计数不变），且
    返回的 `memory_tokens` 绝不超过 `budget`。
    """
    if pure:
        with quiet_recall_log():
            raw = mem.build_context(query or "……", session_id, budget, as_of)
    else:
        raw = mem.build_context(query or "……", session_id, budget, as_of)

    raw = dict(raw or {})
    parts = dict(raw.get("parts") or {})
    persona_part = dict(parts.get("persona") or {})
    memory_part = dict(parts.get("memory") or {})

    persona_text = persona_part.get("text") or ""
    memory_text = memory_part.get("text") or ""
    budget_eff = memory_part.get("budget")
    if budget is not None:
        memory_text, memory_tokens = fit_memory_text(memory_text, budget)
        budget_eff = budget
    else:
        memory_tokens = memory_part.get("tokens") or estimate_tokens(memory_text)

    memory_part["text"] = memory_text
    memory_part["tokens"] = memory_tokens
    memory_part["budget"] = budget_eff
    persona_tokens = persona_part.get("tokens") or estimate_tokens(persona_text)
    persona_part["tokens"] = persona_tokens
    parts["persona"] = persona_part
    parts["memory"] = memory_part

    blocks: list[str] = []
    if persona_text:
        blocks.append(f"<persona>\n{persona_text}\n</persona>")
    if memory_text:
        blocks.append(memory_text)

    return {
        "prompt": "\n\n".join(blocks),
        "parts": parts,
        "tokens": persona_tokens + memory_tokens,
        "persona_tokens": persona_tokens,
        "memory_tokens": memory_tokens,
        "system_prefix": persona_text,
        "memory_text": memory_text,
        "abstained": bool(memory_part.get("abstained")),
        "reason": memory_part.get("reason"),
        "depth": memory_part.get("depth"),
        "budget": budget_eff,
        "latency_ms": raw.get("latency_ms"),
        "session_id": session_id,
        "as_of": raw.get("as_of"),
        "pure": pure,
    }


def render_system(ctx: dict, base_prompt: str | None = None, extra_system: str | None = None) -> str:
    """把上下文渲染成 system prompt。宿主自己拼 messages 时也用它，保持一致。"""
    system = (base_prompt or "").strip()
    for block in (ctx.get("system_prefix") or "", ctx.get("memory_text") or "", extra_system or ""):
        if block:
            system = (system + "\n\n" + block).strip() if system else block.strip()
    return system


# ── 工具调用格式归一 ──────────────────────────────────────────
def _call_id(name: str, args: dict, index: int) -> str:
    raw = f"{name}|{json.dumps(args, ensure_ascii=False, sort_keys=True)}|{index}"
    return "call_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def normalize_tool_calls(raw_calls: Any) -> list[dict]:
    """把各种工具调用格式归一成内部结构：

        {"id", "name", "arguments": dict, "type": "function"}

    接受：OpenAI 原生 `{"id","type","function":{"name","arguments": "<json>"}}`、
    扁平 `{"name","arguments": {...}}`、以及缺 id 的响应（自动补 call_ 前缀）。
    永不抛异常：坏条目直接跳过。
    """
    out: list[dict] = []
    if not raw_calls:
        return out
    if not isinstance(raw_calls, (list, tuple)):
        raw_calls = [raw_calls]
    for i, c in enumerate(raw_calls):
        if not isinstance(c, dict):
            continue
        fn = c.get("function") if isinstance(c.get("function"), dict) else {}
        name = str(c.get("name") or fn.get("name") or "").strip()
        if not name:
            continue
        args: Any = c.get("arguments", fn.get("arguments"))
        if isinstance(args, str):
            s = args.strip()
            if not s:
                parsed: dict = {}
            else:
                try:
                    loaded = json.loads(s)
                    parsed = loaded if isinstance(loaded, dict) else {"_value": loaded}
                except Exception:
                    parsed = {"_raw": s}
        elif isinstance(args, dict):
            parsed = args
        elif args is None:
            parsed = {}
        else:
            parsed = {"_value": args}
        cid = str(c.get("id") or c.get("tool_call_id") or "").strip() or _call_id(name, parsed, i)
        out.append({"id": cid, "name": name, "arguments": parsed, "type": "function"})
    return out


def to_openai_tool_call(call: dict) -> dict:
    """内部结构 → OpenAI assistant.tool_calls 条目（只保留协议字段）。"""
    return {
        "id": call["id"],
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
        },
    }


def _merge_usage(total: dict, usage: Any) -> dict:
    if not isinstance(usage, dict):
        return total
    for k, v in usage.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            total[k] = total.get(k, 0) + v
        elif k not in total:
            total[k] = v
    return total


# ── OpenAI 兼容 / Ollama 原生工具调用透传（service 层使用）─────
def parse_openai_chat_response(data: dict) -> dict:
    """OpenAI chat/completions 响应 → 统一响应。补 id、保留 usage/finish_reason。"""
    data = data or {}
    choices = data.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    msg = choice.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    return {
        "content": content,
        "tool_calls": normalize_tool_calls(msg.get("tool_calls")),
        "usage": data.get("usage") or {},
        "finish_reason": choice.get("finish_reason"),
    }


def parse_ollama_chat_response(data: dict) -> dict:
    """Ollama /api/chat 响应 → 统一响应。"""
    data = data or {}
    msg = data.get("message") or {}
    usage = {}
    if data.get("prompt_eval_count") is not None:
        usage["prompt_tokens"] = data.get("prompt_eval_count") or 0
    if data.get("eval_count") is not None:
        usage["completion_tokens"] = data.get("eval_count") or 0
    if usage:
        usage["total_tokens"] = sum(usage.values())
    content = msg.get("content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    return {
        "content": content,
        "tool_calls": normalize_tool_calls(msg.get("tool_calls")),
        "usage": usage,
        "finish_reason": data.get("done_reason"),
    }


def chat_with_tools(llm: Any, messages: list[dict], tools: list[dict] | None = None, *, max_tokens: int = 400) -> dict | None:
    """把带 tool 轮的 messages 原样发给底层模型，带回 tool_calls。

    这是修「主动记忆在真实产品里从不发生」的根因：service 层原先只用
    `LLM.chat(system, user)` 拼一段文本，模型根本没有工具可调，tool_calls
    恒为空。这里直连 OpenAI 兼容端点（或 Ollama），把工具协议完整走通。

    任何失败（无密钥、无网络、上游不支持 tools、返回坏 JSON）都返回 None，
    调用方据此安静降级到纯文本通道——绝不抛异常。
    """
    if llm is None:
        return None
    provider = str(getattr(llm, "provider", "") or "").lower()
    timeout = float(getattr(llm, "timeout", 45.0) or 45.0)
    temp = float(getattr(llm, "temperature", 0.1) or 0.1)

    try:
        if provider == "openai":
            base = str(getattr(llm, "openai_base", "") or "").rstrip("/")
            key = str(getattr(llm, "openai_key", "") or "")
            if not (base and key):
                return None
            payload: dict = {
                "model": getattr(llm, "openai_model", "") or getattr(llm, "model", ""),
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            req = urllib.request.Request(
                base + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            try:
                llm.calls = int(getattr(llm, "calls", 0)) + 1
            except Exception:
                pass
            return parse_openai_chat_response(data)

        if provider == "ollama":
            base = str(getattr(llm, "ollama_url", "") or "").rstrip("/")
            if not base:
                return None
            payload = {
                "model": getattr(llm, "model", ""),
                "messages": messages,
                "stream": False,
                "options": {"temperature": temp, "num_predict": max_tokens},
            }
            if tools:
                payload["tools"] = tools
            req = urllib.request.Request(
                base + "/api/chat",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            try:
                llm.calls = int(getattr(llm, "calls", 0)) + 1
            except Exception:
                pass
            return parse_ollama_chat_response(data)
    except Exception as e:
        try:
            llm.failures = int(getattr(llm, "failures", 0)) + 1
            llm.last_error = f"{type(e).__name__}: {e}"
        except Exception:
            pass
        log.debug("chat_with_tools failed (%s): %s", provider or "unknown", e)
    return None


def make_llm_callable(llm: Any, *, clean: Callable[[str], str] | None = None, max_tokens: int = 400) -> Callable:
    """构造服务层使用的 `llm_callable(messages, tools)`。

    原生工具协议优先（OpenAI 兼容 / Ollama，tool_calls 原样透传）；
    上游不支持 / 没密钥 / 超时 → 安静降级到旧的纯文本通道（tool_calls=[]），
    绝不抛异常。服务层 `service/app.py` 与宿主适配器都可以直接复用它。
    """

    def _clean(text: str) -> str:
        return clean(text) if clean else text

    def llm_call(messages: list[dict], tools: list[dict] | None = None) -> dict:
        resp = chat_with_tools(llm, messages, tools, max_tokens=max_tokens)
        if resp is not None:
            resp["content"] = _clean(resp.get("content") or "")
            return resp

        # 降级：把 tool 轮压成可读对话，交给 LLM.chat 的纯文本接口
        system_parts: list[str] = []
        convo: list[str] = []
        for m in messages or []:
            role = m.get("role", "user")
            content = m.get("content") or ""
            if role == "system":
                system_parts.append(content)
            elif role == "tool":
                convo.append(f"（工具结果）{content}")
            elif role == "assistant" and m.get("tool_calls"):
                names = "、".join(
                    str((c.get("function") or {}).get("name") or c.get("name") or "")
                    for c in m.get("tool_calls") or []
                )
                convo.append(f"（我调用了工具：{names}）")
            elif role == "assistant":
                convo.append(f"你：{content}")
            else:
                convo.append(f"他：{content}")
        out = None
        try:
            if llm is not None and hasattr(llm, "chat"):
                out = llm.chat("\n".join(system_parts), "\n".join(convo), max_tokens=max_tokens)
        except Exception as e:  # pragma: no cover
            log.debug("fallback llm chat failed: %s", e)
        return {
            "content": _clean(out or ""),
            "tool_calls": [],
            "usage": {},
            "finish_reason": None,
        }

    return llm_call


class Harness:
    """包在模型外面的一圈流程。

    `llm_callable(messages, tools) -> {"content": str, "tool_calls": [...],
    "usage": {...}, "finish_reason": str}`

    不绑定任何 SDK：OpenAI、Ollama、DeepSeek 或一个假模型都能塞进来。
    `tool_calls` 支持 OpenAI 原生嵌套 `function.arguments`（JSON 串）与扁平
    `{"name","arguments": {...}}` 两种写法，缺 id 时自动补。
    """

    def __init__(
        self,
        cfg: Config | None = None,
        llm_callable: Callable | None = None,
        system_prompt: str | None = None,
        auto_tick: bool = True,
        memory_budget: int | None = None,
        max_tool_rounds: int | None = None,
    ):
        self.cfg = cfg or Config()
        self.mem = Memory(self.cfg)
        self.llm = llm_callable
        self.system_prompt = system_prompt or "你在和一个人长期相处，自然一点，不要每次都复述记忆。"
        self.auto_tick = auto_tick
        self.memory_budget = memory_budget
        self.max_tool_rounds = int(max_tool_rounds or MAX_TOOL_ROUNDS)
        self._lock = threading.Lock()
        self._bg: set[threading.Thread] = set()

    # ── 醒 ────────────────────────────────────────────────
    def wake(self, session_id: str = "default") -> dict:
        """会话开始：播种、载入人格、报告状态。

        不返回 prompt——prompt 每轮由 build_context 现编译，这样改人设能立刻生效。
        四钩子契约需要 persona_text / resident_text / inbox / degraded。
        """
        self.mem.seed(session_id)
        p = self.mem.persona_compile(session_id)
        try:
            resident = self.mem.resident_block(session_id)
        except Exception as e:  # 常驻层坏了不该挡会话开始
            log.debug("resident block failed: %s", e)
            resident = {"text": "", "tokens": 0, "pages": 0}
        return {
            "session_id": session_id,
            "persona_text": p["text"],
            "persona_chars": p["chars"],
            "persona_budget": p["budget"],
            "persona_items": p["included"],
            "resident_text": resident.get("text", ""),
            "resident_tokens": resident.get("tokens", 0),
            "resident_pages": resident.get("pages", 0),
            "inbox": self.mem.inbox(session_id),
            "degraded": bool(getattr(getattr(self.mem, "embedder", None), "degraded", False)),
        }

    # ── 上下文（公开、可复用、可纯）───────────────────────
    def build_context(
        self,
        query: str,
        session_id: str = "default",
        budget: int | None = None,
        as_of: float | None = None,
        *,
        pure: bool = False,
    ) -> dict:
        return build_context(self.mem, query, session_id, budget, as_of, pure=pure)

    # ── 一息 ──────────────────────────────────────────────
    def turn(
        self,
        user_text: str,
        session_id: str = "default",
        *,
        history: list[dict] | None = None,
        extra_system: str | None = None,
    ) -> dict:
        """一轮对话。

        `history`：宿主上一轮的 messages（`[{"role","content"}]`），只取 user/assistant。
        `extra_system`：宿主自己的额外 system 段，拼在本体提示之后。
        """
        t0 = time.perf_counter()
        text = (user_text or "").strip()

        # 空输入必须挡住。之前不挡的后果是模型自己编一段小说，
        # 再被当成"她说过的话"写进长期记忆——最脏的一种污染。
        if not text:
            raise ValueError("text 不能为空")

        # ① 先落库再说话：抽取异步，不挡路
        self.mem.ingest_async(text, session_id=session_id, role="user")

        # ② 节拍放后台，绝不阻塞对话
        if self.auto_tick:
            self._spawn(self._bg_tick, session_id)

        # ③ 组装上下文（人格 + 常驻 + 检索，一次拿全）
        ctx = build_context(self.mem, text, session_id, self.memory_budget)

        # ④ 调模型，必要时跑工具循环
        messages = self._build_messages(ctx, text, history=history, extra_system=extra_system)
        reply, tools_called, rounds, usage, finish_reason = self._run_loop(messages, session_id)

        # ⑤ 她说过的话也要记：不对称的记忆是残的
        if reply:
            self.mem.ingest_async(reply, session_id=session_id, role="assistant")

        return TurnResult(
            reply=reply,
            memory_used={
                "persona_chars": ctx["parts"]["persona"]["chars"],
                "memory_items": ctx["parts"]["memory"].get("items"),
                "memory_tokens": ctx["memory_tokens"],
                "depth": ctx["parts"]["memory"].get("depth"),
                "abstained": ctx["parts"]["memory"].get("abstained"),
                "tokens": ctx["tokens"],
            },
            tools_called=tools_called,
            abstained=bool(ctx["parts"]["memory"].get("abstained")),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            rounds=rounds,
            usage=usage,
            finish_reason=finish_reason,
        ).to_dict()

    def turn_stream(
        self,
        user_text: str,
        session_id: str = "default",
        *,
        chunk_chars: int = 24,
        history: list[dict] | None = None,
        extra_system: str | None = None,
    ):
        """流式版 `turn`：先 `meta`，再 `tool*`，再 `delta*`，最后 `done`。

        上游模型不支持 token 流时，这里按段切分回复文本——**接口形态不退化**，
        前端不需要区分"真流式"和"服务端切段"。

        助手文本只在全部 delta 发完之后才写回记忆：客户端中途断开时生成器被关闭，
        流程走不到写回那一步，避免把半截话当成长期记忆。
        """
        t0 = time.perf_counter()
        text = (user_text or "").strip()
        if not text:
            raise ValueError("text 不能为空")

        self.mem.ingest_async(text, session_id=session_id, role="user")
        if self.auto_tick:
            self._spawn(self._bg_tick, session_id)

        ctx = build_context(self.mem, text, session_id, self.memory_budget)
        yield {
            "event": "meta",
            "data": {
                "persona_chars": ctx["parts"]["persona"]["chars"],
                "memory_items": ctx["parts"]["memory"].get("items"),
                "memory_tokens": ctx.get("memory_tokens"),
                "depth": ctx["parts"]["memory"].get("depth"),
                "abstained": bool(ctx["parts"]["memory"].get("abstained")),
            },
        }

        messages = self._build_messages(ctx, text, history=history, extra_system=extra_system)
        try:
            reply, tools_called, rounds, usage, finish_reason = self._run_loop(messages, session_id)
        except LLMUnavailableError as e:
            # 已经发过 meta，只能把错误作为事件送出；不写 assistant 行、不发 done。
            yield {
                "event": "error",
                "data": {
                    "code": getattr(e, "code", "llm_unavailable"),
                    "retryable": bool(getattr(e, "retryable", True)),
                    "status": 503,
                    "message": str(e)[:300],
                },
            }
            return

        for call in tools_called:
            yield {"event": "tool", "data": call}

        for piece in _chunk_text(reply, chunk_chars):
            yield {"event": "delta", "data": {"text": piece}}

        if reply:
            self.mem.ingest_async(reply, session_id=session_id, role="assistant")

        yield {
            "event": "done",
            "data": {
                "reply": reply,
                "rounds": rounds,
                "tools_called": tools_called,
                "finish_reason": finish_reason,
                "usage": usage,
                "memory_tokens": ctx.get("memory_tokens"),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        }

    def _build_messages(
        self,
        ctx: dict,
        user_text: str,
        history: list[dict] | None = None,
        extra_system: str | None = None,
    ) -> list[dict]:
        """把上下文渲染成 messages。history 只取 user/assistant，超长部分截断。"""
        system = self.system_prompt
        if ctx.get("system_prefix"):
            system += "\n\n" + ctx["system_prefix"]
        if ctx.get("memory_text"):
            system += "\n\n" + ctx["memory_text"]
        if extra_system:
            system += "\n\n" + extra_system

        messages: list[dict] = [{"role": "system", "content": system}]
        for m in (history or [])[-MAX_HISTORY_MESSAGES:]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                system += "\n\n" + str(content or "")
                messages[0] = {"role": "system", "content": system}
                continue
            if role not in ("user", "assistant") or not content:
                continue
            messages.append({"role": role, "content": str(content)})
        if user_text:
            # 避免宿主把本轮输入也塞进 history 造成重复
            if not (
                messages
                and messages[-1].get("role") == "user"
                and messages[-1].get("content") == user_text
            ):
                messages.append({"role": "user", "content": user_text})
        return messages

    def _run_loop(
        self, messages: list[dict], session_id: str
    ) -> tuple[str, list[dict], int, dict, str | None]:
        if self.llm is None:
            # 没接模型就不该有"她说过的话"：抛明确错误，不写 assistant 行。
            raise LLMUnavailableError("llm_callable 未配置：没有真实模型输出")

        called: list[dict] = []
        reply = ""
        usage: dict = {}
        finish_reason: str | None = None
        for round_no in range(self.max_tool_rounds):
            try:
                resp = self.llm(messages, TOOLS)
            except LLMUnavailableError:
                raise
            except Exception as e:
                log.warning("llm call failed: %s", e)
                raise LLMUnavailableError(f"模型调用失败：{type(e).__name__}: {e}") from e

            resp = resp if isinstance(resp, dict) else {}
            content = (resp.get("content") or "").strip()
            calls = normalize_tool_calls(resp.get("tool_calls"))
            usage = _merge_usage(usage, resp.get("usage"))
            if resp.get("finish_reason"):
                finish_reason = resp.get("finish_reason")

            if not content and not calls:
                # 模型返回空：不可用、超时、或拒绝回答。不能落库成"她说过的话"。
                raise LLMUnavailableError("模型返回既无 content 也无 tool_calls")

            if content:
                reply = content
            if not calls:
                return reply, called, round_no + 1, usage, finish_reason

            # 把模型这一轮的意图回灌，再执行（OpenAI 兼容格式，带 id）
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [to_openai_tool_call(c) for c in calls],
                }
            )
            for c in calls:
                name = c["name"]
                args = c["arguments"]
                try:
                    result = dispatch(self.mem, name, args, session_id)
                except Exception as e:  # dispatch 内部已兜底，这里再兜一层
                    result = {"ok": False, "reason": f"工具派发失败：{type(e).__name__}: {e}"}
                called.append(
                    {
                        "tool": name,
                        "args": args,
                        "ok": bool(result.get("ok")),
                        "reason": result.get("reason", "") or "",
                        "id": result.get("id"),
                        "state": result.get("state"),
                        "tool_call_id": c["id"],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "name": name,
                        "content": _stringify(result),
                    }
                )

        # 工具轮次用完，让她带着已有结果收尾；仍没有内容 → 同样是"模型不可用"
        try:
            resp = self.llm(messages, None)
        except Exception as e:
            raise LLMUnavailableError(f"模型调用失败：{type(e).__name__}: {e}") from e
        resp = resp if isinstance(resp, dict) else {}
        final = (resp.get("content") or "").strip() or reply
        usage = _merge_usage(usage, resp.get("usage"))
        if resp.get("finish_reason"):
            finish_reason = resp.get("finish_reason")
        if not final:
            raise LLMUnavailableError("工具轮次用尽仍未收到内容")
        return final, called, self.max_tool_rounds, usage, finish_reason

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


def _chunk_text(text: str, size: int = 24):
    """把整段回复切成小块，供"上游不支持流式"时前端仍有逐段体验。"""
    text = text or ""
    size = max(1, int(size or 24))
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _stringify(d: dict) -> str:
    """工具结果转成模型能读的文本。失败原因必须原样带上。"""
    if not isinstance(d, dict):
        return f"成功：{d}"
    if d.get("ok"):
        parts = [f"{k}={v}" for k, v in d.items() if k != "ok" and v not in (None, "", [], {}, False)]
        return "成功：" + "；".join(parts) if parts else "成功"
    return "失败：" + str(d.get("reason") or "未知原因")
