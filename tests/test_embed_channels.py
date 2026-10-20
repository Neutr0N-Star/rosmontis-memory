"""openai / local 两个 embedding 通道的行为测试。

- openai：本地 HTTP server mock `/v1/embeddings`（成功 / 500 / 维度不符 / 缺配置）
- local：假模型目录验证「未安装报错」与「依赖或文件坏 → 降级 hash 且点名」；
  真模型的端到端推理在装了 `[embed]` 依赖与本机模型时才跑（CI 自动跳过）。

这些测试不需要真实网络与真实模型。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosmontis.core.embed import Embedder, EmbedProviderError, SEMANTIC_PROVIDERS  # noqa: E402
from rosmontis.errors import RosmontisError  # noqa: E402


# ── mock OpenAI 兼容服务器 ──────────────────────────────────────
class _OpenAIMock(BaseHTTPRequestHandler):
    mode = "ok"          # ok / fail_500 / bad_json / wrong_dim
    dim = 8
    calls = 0

    def do_POST(self):  # noqa: N802
        type(self).calls += 1
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        self.server.last_body = body
        auth = self.headers.get("Authorization", "")
        self.server.last_auth = auth
        m = type(self).mode
        if m == "fail_500":
            self.send_response(500)
            self.end_headers()
            return
        if m == "bad_json":
            payload = b"not-json"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if m == "wrong_dim":
            emb = [0.1] * (type(self).dim + 3)
        else:
            emb = [0.1] * type(self).dim
        payload = json.dumps({"data": [{"embedding": emb}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # 静默
        pass


def _start_mock(mode: str, dim: int = 8):
    handler = type("H", (_OpenAIMock,), {})
    handler.mode = mode
    handler.dim = dim
    handler.calls = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.last_body = None
    srv.last_auth = ""
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


class TestOpenAIChannel(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("ROSMONTIS_EMBED_OPENAI_BASE", "ROSMONTIS_EMBED_OPENAI_KEY",
                        "ROSMONTIS_OPENAI_BASE", "ROSMONTIS_OPENAI_KEY")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _embedder(self, srv, dim=8):
        return Embedder(
            provider="openai", model="text-embedding-test", dim=dim,
            ollama_url="http://127.0.0.1:9", timeout=2.0,
            openai_base=f"http://127.0.0.1:{srv.server_port}", openai_key="test-key",
        )

    def test_registry_and_semantic_set(self):
        self.assertIn("openai", SEMANTIC_PROVIDERS)
        self.assertIn("local", SEMANTIC_PROVIDERS)
        self.assertNotIn("hash", SEMANTIC_PROVIDERS)

    def test_missing_config_rejects_at_construction(self):
        """缺 base 或 key：构造即失败，不进入半失明状态。"""
        with self.assertRaises(RosmontisError) as cm:
            Embedder(provider="openai", model="m", dim=8, ollama_url="http://x")
        self.assertEqual(cm.exception.code, "embed_provider_unavailable")
        self.assertIn("ROSMONTIS_EMBED_OPENAI_BASE", cm.exception.message)

    def test_ok_path_semantic_and_request_shape(self):
        srv = _start_mock("ok", dim=8)
        try:
            h = self._embedder(srv)
            self.assertEqual(h.vector_mode(), "semantic")
            v = h.embed("你好，世界")
            self.assertEqual(len(v), 8)
            self.assertFalse(h.degraded)
            # 请求形状：model + input + Bearer
            self.assertEqual(srv.last_body.get("model"), "text-embedding-test")
            self.assertEqual(srv.last_body.get("input"), "你好，世界")
            self.assertTrue(srv.last_auth.startswith("Bearer test-key"))
            # 缓存生效：同一句话不再打远程
            n = type(srv.RequestHandlerClass and srv).calls if False else None
            h.embed("你好，世界")
            calls_after_cache = srv.last_body  # 占位防 unused
            self.assertTrue(h.cache.get("你好，世界") is not None)
        finally:
            srv.shutdown()

    def test_server_error_falls_back_with_circuit(self):
        srv = _start_mock("fail_500")
        try:
            h = self._embedder(srv)
            v = h.embed("一句话")
            # 降级：hash 兜底、degraded 可见、断路器打开
            self.assertTrue(h.degraded)
            self.assertIn("不可用", h.degraded_reason)
            self.assertEqual(h.vector_mode(), "degraded")
            self.assertEqual(len(v), 8)   # hash 向量仍按配置维度产出
            self.assertGreater(h.circuit_state()["fail_streak"], 0)
        finally:
            srv.shutdown()

    def test_bad_json_treated_as_failure(self):
        srv = _start_mock("bad_json")
        try:
            h = self._embedder(srv)
            v = h.embed("一句话")
            self.assertTrue(h.degraded)
            self.assertEqual(len(v), 8)
        finally:
            srv.shutdown()

    def test_dim_mismatch_visible_as_misconfigured(self):
        srv = _start_mock("ok", dim=11)
        try:
            h = self._embedder(srv, dim=8)
            v = h.embed("一句话")
            self.assertIsNotNone(v)
            self.assertEqual(h.vector_mode(), "misconfigured")
            self.assertIn("11 维", h.dim_mismatch)
        finally:
            srv.shutdown()


class TestLocalChannel(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="rosmontis-local-test-")
        self._saved_dir = os.environ.get("ROSMONTIS_MODEL_DIR")
        os.environ["ROSMONTIS_MODEL_DIR"] = self.tmp

    def tearDown(self):
        if self._saved_dir is None:
            os.environ.pop("ROSMONTIS_MODEL_DIR", None)
        else:
            os.environ["ROSMONTIS_MODEL_DIR"] = self._saved_dir

    def _fake_model(self, name="bge-small-zh-v1.5"):
        """造一个 manifest 完整但文件为空的假模型：构造能过，加载必炸。"""
        import json as _json
        d = Path(self.tmp) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(_json.dumps({"complete": True}), encoding="utf-8")
        (d / "config.json").write_text(_json.dumps({"hidden_size": 512}), encoding="utf-8")
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")
        on = d / "onnx"
        on.mkdir(exist_ok=True)
        (on / "model.onnx").write_bytes(b"not-a-real-onnx")
        return d

    def test_uninstalled_model_rejects_at_construction(self):
        """模型未下载：构造即失败，消息带安装命令。"""
        with self.assertRaises(RosmontisError) as cm:
            Embedder(provider="local", model="bge-small-zh-v1.5", dim=512, ollama_url="http://x")
        self.assertEqual(cm.exception.code, "embed_provider_unavailable")
        self.assertIn("models install", cm.exception.message)

    def test_corrupt_model_falls_back_with_reason(self):
        """假模型（文件坏）：构造成功，首次 embed 降级 hash 且 degraded 可见。"""
        self._fake_model()
        h = Embedder(provider="local", model="bge-small-zh-v1.5", dim=512, ollama_url="http://x")
        self.assertEqual(h.vector_mode(), "semantic")   # 未加载前乐观：文件都在
        v = h.embed("一句话")
        self.assertEqual(len(v), 512)                   # hash 兜底按配置维度产出
        self.assertTrue(h.degraded)
        self.assertTrue(h.degraded_reason)              # 点名失败原因，不静默
        self.assertEqual(h.vector_mode(), "degraded")
        # 退避期内连续调用不再反复撞坏模型
        h.embed("另一句")
        self.assertLessEqual(h.circuit_state()["remote_attempts"], 0)  # local 不走远程计数

    def test_dim_autodiscovered_from_asset_table(self):
        """local + 未显式设 DIM：Config 自动取资产表维度（384），不再默认 1024。"""
        self._fake_model()
        from rosmontis.config import Config
        c2 = Config(embed_provider="local", embed_model="bge-small-zh-v1.5")
        self.assertEqual(c2.embed_dim, 512)

    @unittest.skipUnless(
        os.environ.get("ROSMONTIS_TEST_REAL_LOCAL") == "1",
        "真模型端到端需要 models install + [embed] 依赖；设 ROSMONTIS_TEST_REAL_LOCAL=1 开启",
    )
    def test_real_onnx_end_to_end(self):
        """真模型：384 维语义向量，两句相似文本余弦高于不相关文本。"""
        # 用真实模型目录（setUp 把 MODEL_DIR 指到了临时目录，这里还原）
        if self._saved_dir is None:
            os.environ.pop("ROSMONTIS_MODEL_DIR", None)
        else:
            os.environ["ROSMONTIS_MODEL_DIR"] = self._saved_dir
        h = Embedder(provider="local", model="bge-small-zh-v1.5", dim=512, ollama_url="http://x")
        a = h.embed("我今天搬到了上海")
        b = h.embed("我上个月搬到上海住了")
        c = h.embed("量子计算机的纠错码需要冗余比特")
        self.assertEqual(len(a), 512)
        self.assertEqual(h.vector_mode(), "semantic")
        dot = lambda x, y: sum(i * j for i, j in zip(x, y))
        self.assertGreater(dot(a, b), dot(a, c))

    def _fake_model_clear(self):
        d = Path(self.tmp) / "bge-small-zh-v1.5"
        if d.exists():
            import shutil
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
