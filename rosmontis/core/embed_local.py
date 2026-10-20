"""本地 ONNX embedding 推理（provider=local）。

把 `rosmontis models install` 下载的模型资产真正用起来：tokenizer 编码 →
onnx session 前向 → masked mean pooling → L2 归一化。

设计约束：
- **惰性加载**：session 与 tokenizer 在首次 embed 时才建（构造 Embedder 的
  路径上不起推理线程，serve 启动不被模型加载阻塞）。
- **维度以模型为准**：加载后读模型目录 `config.json` 的 `hidden_size` 作为
  真值——模型资产表（models.MODELS）里的 dim 只是下载期的缺省，表写错了
  也不该产出 misconfigured 的假向量。
- **不做网络重试/断路器**：本地推理没有网络语义；失败由 Embedder 统一
  降级为 hash 并点名原因。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("rosmontis.embed.local")


class LocalModelError(RuntimeError):
    """local 通道的模型/依赖问题。消息应当能直接给用户看。"""


class LocalOnnxEmbedder:
    """单模型 ONNX 推理器。线程安全（加载有锁，推理共享 session）。"""

    def __init__(self, onnx_path: str, tokenizer_path: str, model_dir: str, fallback_dim: int = 384):
        self.onnx_path = onnx_path
        self.tokenizer_path = tokenizer_path
        self.model_dir = model_dir
        self.fallback_dim = int(fallback_dim)
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None
        self._output_key: str | None = None   # sentence_embedding / last_hidden_state
        self._loaded = False
        self.dim: int | None = None           # 加载后由 config.json 的 hidden_size 回填

    # ── 加载 ──────────────────────────────────────────────
    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as e:
                raise LocalModelError(
                    f"local 通道缺少依赖：{e.name}。"
                    f"安装：pip install 'rosmontis-memory[embed]'"
                ) from e

            for p, what in ((self.onnx_path, "ONNX 模型"), (self.tokenizer_path, "tokenizer")):
                if not Path(p).exists():
                    raise LocalModelError(
                        f"{what}文件不存在：{p}。运行：rosmontis models install "
                        f"（目录：{self.model_dir}）"
                    )

            opts = ort.SessionOptions()
            # 桌面场景默认占一半核：留核给前端/LLM，别把机器吃满
            opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            try:
                self._session = ort.InferenceSession(
                    self.onnx_path, opts, providers=["CPUExecutionProvider"]
                )
                self._tokenizer = Tokenizer.from_file(self.tokenizer_path)
                self._tokenizer.no_padding()
            except Exception as e:
                raise LocalModelError(f"模型加载失败（{self.onnx_path}）：{e}") from e

            self._resolve_io_signature()
            self._loaded = True
            log.info("local ONNX 模型已加载：dim=%s, output=%s", self.dim, self._output_key)

    def _resolve_io_signature(self) -> None:
        """确定输入名集合与输出张量；dim 以 config.json 的 hidden_size 为真值。"""
        inputs = {i.name for i in self._session.get_inputs()}
        outputs = [o.name for o in self._session.get_outputs()]
        if "sentence_embedding" in outputs:
            self._output_key = "sentence_embedding"
        elif "last_hidden_state" in outputs:
            self._output_key = "last_hidden_state"
        else:
            self._output_key = outputs[0]   # 未知导出格式：取第一个输出，pooling 时按形状判断
        self._has_token_type = "token_type_ids" in inputs

        hidden = None
        cfg = Path(self.model_dir) / "config.json"
        try:
            meta = json.loads(cfg.read_text(encoding="utf-8"))
            hidden = int(meta.get("hidden_size") or 0) or None
        except Exception as e:
            log.debug("读取模型 config.json 失败（%s）：%s", cfg, e)
        if hidden is None:
            # config.json 缺失/损坏时退回资产表给的缺省；此时若与真实输出不符，
            # 仍会被 dim_mismatch 机制抓住，不会静默。
            hidden = self.fallback_dim
        self.dim = hidden

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ── 推理 ──────────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        """单条文本 → 归一化向量。失败抛异常，由调用方降级。"""
        self._load()
        enc = self._tokenizer.encode(text)
        if not enc.ids:
            return [0.0] * (self.dim or self.fallback_dim)
        feed: dict[str, list] = {"input_ids": [list(enc.ids)], "attention_mask": [list(enc.attention_mask)]}
        if self._has_token_type:
            feed["token_type_ids"] = [[0] * len(enc.ids)]
        out = self._session.run(None, feed)

        if self._output_key == "sentence_embedding":
            vec = [float(x) for x in out[0][0]]
        else:
            hidden = out[0]                     # [1, seq, h] 或 [1, h]
            if len(hidden.shape) == 2:          # 已池化
                vec = [float(x) for x in hidden[0]]
            else:                               # masked mean pooling
                import numpy as np
                mask = np.asarray(enc.attention_mask, dtype="float32")
                h = hidden[0]                   # [seq, h]
                summed = (h * mask[:, None]).sum(axis=0)
                counts = max(1.0, float(mask.sum()))
                vec = [float(x) for x in (summed / counts)]
        return _l2_normalize(vec)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm <= 0.0:
        return vec
    return [x / norm for x in vec]


__all__ = ["LocalOnnxEmbedder", "LocalModelError"]
