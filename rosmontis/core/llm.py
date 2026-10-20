"""LLM Provider：热路径（每条消息）与深路径（关键判断）分开配置。

设计原则：
- 任何调用失败都不抛异常给业务层，返回 None，由调用方走规则兜底。
- 超时短、不重试（陪伴场景宁可少抽一条，也不能卡住对话）。
- JSON 解析带修复：本地小模型经常吐 ```json 包裹或不闭合的括号。
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

log = logging.getLogger("rosmontis.llm")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class LLM:
    def __init__(
        self,
        provider: str,
        model: str,
        ollama_url: str,
        timeout: float = 45.0,
        temperature: float = 0.1,
        openai_base: str = "",
        openai_key: str = "",
        openai_model: str = "",
    ):
        self.provider = provider
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.openai_base = openai_base.rstrip("/")
        self.openai_key = openai_key
        self.openai_model = openai_model or model
        self.last_error: str | None = None
        self.calls = 0
        self.failures = 0

    def available(self) -> bool:
        if self.provider == "ollama":
            try:
                with urllib.request.urlopen(self.ollama_url + "/api/tags", timeout=3):
                    return True
            except Exception:
                return False
        if self.provider == "openai":
            return bool(self.openai_base and self.openai_key)
        return False

    def chat(self, system: str, user: str, max_tokens: int = 700, json_mode: bool = False) -> str | None:
        self.last_error = None
        """返回原始文本；失败返回 None。"""
        if self.provider == "ollama":
            return self._ollama(system, user, max_tokens, json_mode)
        if self.provider == "openai":
            return self._openai(system, user, max_tokens, json_mode)
        return None

    def chat_json(self, system: str, user: str, max_tokens: int = 700) -> dict | list | None:
        raw = self.chat(system, user, max_tokens=max_tokens, json_mode=True)
        if not raw:
            return None
        return parse_json_loose(raw)

    # ── 后端 ──────────────────────────────────────────────
    def _ollama(self, system: str, user: str, max_tokens: int, json_mode: bool) -> str | None:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            req = urllib.request.Request(
                self.ollama_url + "/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            self.calls += 1
            return (data.get("message") or {}).get("content") or None
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            log.debug("ollama chat failed: %s", self.last_error)
            return None

    def _openai(self, system: str, user: str, max_tokens: int, json_mode: bool) -> str | None:
        # 失败原因要留在 last_error 里——401（密钥错）和超时对用户是两件事，
        # 全吞成 None 的话，前端只会显示「模型没响应」，用户不知道去查密钥。
        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            req = urllib.request.Request(
                self.openai_base + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            self.calls += 1
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            log.debug("openai chat failed: %s", self.last_error)
            return None


def parse_json_loose(raw: str) -> dict | list | None:
    """尽最大努力从模型输出里捞出 JSON。

    本地 2B 模型常见的坏格式：```json 包裹、前后带解释、单引号、尾逗号、
    对象未闭合。逐级尝试，全部失败返回 None。
    """
    if not raw:
        return None
    text = raw.strip()

    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()

    for candidate in (text, _slice_braces(text), _slice_brackets(text)):
        if not candidate:
            continue
        for fixer in (lambda s: s, _fix_common, _fix_common_aggressive):
            try:
                return json.loads(fixer(candidate))
            except Exception:
                continue
    return None


def _slice_braces(s: str) -> str | None:
    i, j = s.find("{"), s.rfind("}")
    return s[i : j + 1] if 0 <= i < j else None


def _slice_brackets(s: str) -> str | None:
    i, j = s.find("["), s.rfind("]")
    return s[i : j + 1] if 0 <= i < j else None


def _fix_common(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = s.replace("，", ",").replace("：", ":")
    return s


def _fix_common_aggressive(s: str) -> str:
    s = _fix_common(s)
    s = re.sub(r"'([^']*)'", r'"\1"', s)
    if s.count("{") > s.count("}"):
        s += "}" * (s.count("{") - s.count("}"))
    if s.count("[") > s.count("]"):
        s += "]" * (s.count("[") - s.count("]"))
    return s
