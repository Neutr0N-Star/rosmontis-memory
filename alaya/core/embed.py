"""Embedding Provider。

支持 ollama（本地语义向量）、hash（测试/降级兜底）与 none（无向量模式）。
降级状态显式可见，不静默。

**断路器（P0）**：探测一次失败即置 `_available=False`，此后在退避窗口内
**不再打远程**，直接返回兜底向量（或空）；指数退避 30s → 5min 封顶后重探，
成功后自动切回语义模式。目的不是"关掉向量"，而是 Ollama 半死时单轮对话
不再出现 8s 级等待。

**provider 扩展点**：`provider_registry` 目前只有内置 ollama/hash/none。
后续「模型安装器」线要加 local(ONNX) 等 provider 时，在这里注册并实现
`embed(text)->list[float] | None`；本文件不内置 ONNX 推理。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict

from ..errors import AlayaError
from . import vec as V

log = logging.getLogger("alaya.embed")

# provider 扩展点：仅登记内置实现；新 provider（如 local/onnx）由独立工作线注册。
# 契约：对象需提供 embed(text) -> list[float] | None；可选 embed_many。
provider_registry: dict[str, str] = {
    "ollama": "builtin-http",
    "hash": "builtin-test-fallback",
    "none": "builtin-disabled",
}


class EmbedProviderError(AlayaError):
    """配置了未注册的 provider：启动即失败，不静默降级成 hash。

    静默换 hash 的代价是"以为在跑语义检索、实际全库不落向量"，运维在
    `/readyz` 上也看不出来。hash / none 是真实模式，但必须**显式**写才生效。
    """

    def __init__(self, provider: str):
        self.provider = provider
        self.available = sorted(provider_registry)
        super().__init__(
            "embed_provider_unavailable",
            f"embedding provider 不可用：{provider!r}；"
            f"可用：{'、'.join(self.available)}（hash / none 需显式配置才生效）",
            retryable=False,
        )


# 断路器参数
BACKOFF_BASE = 30.0        # 首次失败后 30s 重探
BACKOFF_MAX = 300.0        # 封顶 5min
PROBE_TIMEOUT = 2.0        # 探测用短超时：不让"半死"状态拖满 8s
HEALTH_TIMEOUT = 2.0


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

    def clear(self) -> None:
        with self.lock:
            self.data.clear()


class Embedder:
    def __init__(
        self,
        provider: str,
        model: str,
        dim: int,
        ollama_url: str,
        timeout: float = 8.0,
        *,
        model_id: str | None = None,
        clock=None,
        probe_timeout: float = PROBE_TIMEOUT,
    ):
        prov = str(provider or "").strip().lower()
        if prov not in provider_registry:
            # 拼错的 provider 静默变 hash，是最难查的一类"库半失明"：写入端
            # 以为在算语义向量，实际上一个字都没算出来。启动就炸比静默好。
            raise EmbedProviderError(prov or provider)
        self.provider = prov
        self.model = model
        # 向量身份：写入 episode.embed_model / fact.embed_model 的稳定标识。
        # 默认取模型名；换模型/换档时由 config.embed_model_id 显式区分。
        self.model_id = (model_id or model or "").strip()
        self.dim = int(dim)
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = float(timeout)
        self.probe_timeout = float(probe_timeout)
        self.cache = _LRU()
        self.degraded = False
        self.degraded_reason: str | None = None
        # 配置维度与实际模型输出不一致（首次拿到远程向量时才能发现）。
        # 这类错误最容易被吞成"静默跳过所有向量"，所以要单独外露成 misconfigured。
        self.dim_mismatch: str | None = None

        # ── 断路器状态 ──────────────────────────────────────
        self._clock = clock or time.monotonic
        self._available: bool | None = None   # None=未探测 / True=可用 / False=断路
        self._fail_streak = 0
        self._open_until = 0.0
        self._probe_done = False              # 用的就是原来那枚死标记：是否已探测过
        # 可观测计数（测试/运维用）
        self.remote_attempts = 0
        self.probe_attempts = 0
        self.fallback_count = 0
        self.health_attempts = 0
        self._remote_timeout: float | None = None  # 本次远程调用的超时（探测用短超时）

        if self.provider == "none":
            self.degraded = True
            self.degraded_reason = "embedding provider=none（无向量模式）"
        elif self.provider == "hash":
            # hash 是测试/兜底通道，不是语义向量：对外的 vector_mode 记为 degraded，
            # 写路径据此不落伪向量（留 NULL 给 embed_backfill / 真实模型恢复后补）。
            self.degraded = True
            self.degraded_reason = "hash provider（确定性哈希向量，非语义；仅供测试/兜底）"

    # ── 身份 / 模式 ─────────────────────────────────────────
    def identity(self) -> dict:
        return {"model": self.model_id, "model_name": self.model, "dim": self.dim}

    def vector_mode(self) -> str:
        """semantic / degraded / none / misconfigured：给 /readyz 与 stats 的一眼判据。"""
        if self.provider == "none":
            return "none"
        if self.dim_mismatch:
            # 维度对不上的向量永远进不了余弦，等同于没有向量——但它不是
            # "服务挂了"，而是"配置写错了"，必须让运维看到区别。
            return "misconfigured"
        if self.provider != "ollama":
            return "degraded"       # hash：有向量但不是语义向量，仅供测试/兜底
        return "degraded" if self.degraded else "semantic"

    def circuit_state(self) -> dict:
        now = self._clock()
        open_ = self._available is False
        if not open_:
            state = "closed"
        elif now < self._open_until:
            state = "open"
        else:
            state = "half-open"
        return {
            "state": state,
            "fail_streak": self._fail_streak,
            "open_until": self._open_until,
            "next_probe_in": round(max(0.0, self._open_until - now), 3) if open_ else 0.0,
            "probe_done": self._probe_done,
            "remote_attempts": self.remote_attempts,
            "probe_attempts": self.probe_attempts,
            "fallback_count": self.fallback_count,
        }

    # ── 对外 ──────────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        t = (text or "").strip()
        if not t:
            return [0.0] * self.dim
        hit = self.cache.get(t)
        if hit is not None:
            return hit
        if self.provider == "ollama":
            v = self._try_remote(t)
            if v is not None and len(v) != self.dim and not self.dim_mismatch:
                # 真实模型的输出维度与配置不一致：不静默按配置截断/补零（那会
                # 造出一堆无意义向量），而是标记出来让 /readyz 与 context() 说话。
                self.dim_mismatch = (
                    f"embed_dim 配置为 {self.dim}，但模型返回 {len(v)} 维；"
                    f"请核对 ALAYA_EMBED_DIM 与 {self.model_id or self.model} 的实际维度"
                )
                self.degraded = True
                self.degraded_reason = self.dim_mismatch
                log.warning("embedding 维度不一致：%s", self.dim_mismatch)
        elif self.provider == "hash":
            v = None
        else:  # none
            v = None
        if v is None:
            if self.provider == "none":
                v = []
            else:
                if self.provider == "hash" and not self.degraded:
                    self.degraded = True
                    self.degraded_reason = "hash provider（确定性哈希向量，非语义）"
                v = V.deterministic_hash_vector(t, self.dim)
                self.fallback_count += 1
        # 降级期的哈希向量不缓存：否则恢复后同一句话会直接命中旧哈希向量，
        # 永远等不到下一次远程探测（缓存键无法区分向量身份）。
        cacheable = self.provider != "ollama" or not self.degraded
        if cacheable:
            self.cache.put(t, v)
        return v

    def embed_blob(self, text: str) -> bytes:
        vec = self.embed(text)
        return V.pack(vec) if vec else b""

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def health(self, probe: bool = True) -> dict:
        """健康快照。`probe=False` 绝不联网（/readyz、stats 用），只看断路器。"""
        ok = False
        detail = ""
        circuit = self.circuit_state()
        if self.provider == "ollama":
            if not probe:
                ok = self._available is True
                detail = "circuit open" if circuit["state"] == "open" else (
                    "unprobed" if self._available is None else "closed"
                )
            elif self._circuit_blocked():
                ok = False
                detail = f"circuit open，{circuit['next_probe_in']:.0f}s 后重探"
            else:
                ok, detail = self._probe_health()
        elif self.provider == "hash":
            detail = "hash provider（非语义向量）"
        elif self.provider == "none":
            detail = "无向量模式"
        return {
            "provider": self.provider,
            "model": self.model,
            "model_id": self.model_id,
            "dim": self.dim,
            "reachable": ok,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "vector_mode": self.vector_mode(),
            "circuit": circuit,
            "detail": detail,
        }

    # ── 断路器 ────────────────────────────────────────────
    def _circuit_blocked(self) -> bool:
        if self._available is not False:
            return False
        return self._clock() < self._open_until

    def _try_remote(self, text: str) -> list[float] | None:
        if self._circuit_blocked():
            return None
        now = self._clock()
        first_probe = self._available is None   # 首次探测用短超时，避免半死拖 8s
        probing = self._available is not True
        self.remote_attempts += 1
        if probing:
            self.probe_attempts += 1
        self._probe_done = True
        timeout = self.probe_timeout if first_probe else self.timeout
        self._remote_timeout = timeout
        try:
            v = self._embed_remote(text)
        except Exception as e:  # provider 实现抛异常也按失败处理，不能把超时冒给业务层
            log.debug("embed remote raised: %s", e)
            v = None
        if v is None:
            self._record_failure(now)
            return None
        self._record_success()
        return v

    def _record_failure(self, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        entered = self._available is not False
        self._fail_streak += 1
        self._available = False
        delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (self._fail_streak - 1)))
        self._open_until = now + delay
        self.degraded = True
        self.degraded_reason = f"embedding 服务不可用（连续 {self._fail_streak} 次），{delay:.0f}s 后重探"
        if entered:
            self.cache.clear()  # 不把降级期的哈希向量带进恢复后
        log.warning("embed circuit open: streak=%d, retry in %.0fs", self._fail_streak, delay)

    def _record_success(self) -> None:
        recovered = self._available is False or self.degraded
        if recovered:
            log.info("embed circuit recovered -> semantic")
            self.cache.clear()  # 丢弃降级期缓存的哈希向量
        self._available = True
        self._fail_streak = 0
        self._open_until = 0.0
        self.degraded = False
        self.degraded_reason = None

    def _probe_health(self) -> tuple[bool, str]:
        now = self._clock()
        self.health_attempts += 1
        try:
            req = urllib.request.Request(self.ollama_url + "/api/tags")
            with urllib.request.urlopen(req, timeout=self.probe_timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            names = [m.get("name", "") for m in data.get("models", [])]
            ok = any(self.model.split(":")[0] in n for n in names)
            if ok:
                self._record_success()
            return ok, "models=" + ",".join(names[:5])
        except Exception as e:  # pragma: no cover - 网络异常路径
            self._record_failure(now)
            return False, f"{type(e).__name__}: {e}"

    # ── 远程调用 ──────────────────────────────────────────
    def _embed_remote(self, text: str) -> list[float] | None:
        payload = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url + "/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._remote_timeout or self.timeout) as r:
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


__all__ = ["Embedder", "provider_registry", "BACKOFF_BASE", "BACKOFF_MAX"]
