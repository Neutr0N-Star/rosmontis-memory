"""Embedding Provider。

支持 ollama（本地）与 hash（无依赖兜底）。降级状态显式可见，不静默。
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from collections import OrderedDict

from . import vec as V

log = logging.getLogger("alaya.embed")


class _LRU:
    def __init__(self, cap: int = 4096):
        self.cap = cap
        self.data: OrderedDict[str, list[float]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
            return None

    def put(self, key: str, val: list[float]) -> None:
        with self.lock:
            self.data[key] = val
            self.data.move_to_end(key)
            while len(self.data) > self.cap:
                self.data.popitem(last=False)


class Embedder:
    def __init__(self, provider: str, model: str, dim: int, ollama_url: str, timeout: float = 8.0):
        self.provider = provider if provider in ("ollama", "hash") else "hash"
        self.model = model
        self.dim = dim
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.cache = _LRU()
        self.degraded = False
        self.degraded_reason: str | None = None
        self._probe_done = False

    # ── 对外 ──────────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        t = (text or "").strip()
        if not t:
            return [0.0] * self.dim
        hit = self.cache.get(t)
        if hit is not None:
            return hit
        v = self._embed_remote(t) if self.provider == "ollama" else None
        if v is None:
            if not self.degraded:
                self.degraded = True
                self.degraded_reason = "embedding 服务不可用，已退化为确定性哈希向量"
                log.warning("embed degraded -> hash")
            v = V.deterministic_hash_vector(t, self.dim)
        else:
            self.degraded = False
            self.degraded_reason = None
        self.cache.put(t, v)
        return v

    def embed_blob(self, text: str) -> bytes:
        return V.pack(self.embed(text))

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def health(self) -> dict:
        ok = False
        detail = ""
        if self.provider == "ollama":
            try:
                req = urllib.request.Request(self.ollama_url + "/api/tags")
                with urllib.request.urlopen(req, timeout=3) as r:
                    data = json.loads(r.read().decode("utf-8"))
                names = [m.get("name", "") for m in data.get("models", [])]
                ok = any(self.model.split(":")[0] in n for n in names)
                detail = "models=" + ",".join(names[:5])
            except Exception as e:  # pragma: no cover
                detail = f"{type(e).__name__}: {e}"
        return {
            "provider": self.provider,
            "model": self.model,
            "dim": self.dim,
            "reachable": ok,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "detail": detail,
        }

    # ── 内部 ──────────────────────────────────────────────
    def _embed_remote(self, text: str) -> list[float] | None:
        payload = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url + "/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            embs = data.get("embeddings")
            if embs:
                return list(embs[0])
            emb = data.get("embedding")
            if emb:
                return list(emb)
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            log.debug("embed remote failed: %s", e)
            return None
