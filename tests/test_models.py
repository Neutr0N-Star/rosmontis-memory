"""模型资产管理测试：下载、断点续传、重试、校验、安装/删除。

不联网：本地起一个 HTTP 服务模拟模型仓库（支持 Range，用于续传测试）。
运行：PYTHONIOENCODING=utf-8 python tests/test_models.py
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosmontis import models as M  # noqa: E402


class _Handler(http.server.BaseHTTPRequestHandler):
    """极简文件服务：支持 Range，可选"前 N 次失败"。"""

    files: dict[str, bytes] = {}
    fail_times: dict[str, int] = {}
    hits: dict[str, int] = {}

    def log_message(self, *a):  # 静音
        pass

    def do_GET(self):  # noqa: N802
        rel = self.path.lstrip("/")
        self.hits[rel] = self.hits.get(rel, 0) + 1
        if self.fail_times.get(rel, 0) > 0:
            self.fail_times[rel] -= 1
            self.send_response(503)
            self.end_headers()
            return
        data = self.files.get(rel)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        rng = self.headers.get("Range")
        start = 0
        if rng and rng.startswith("bytes="):
            try:
                start = int(rng.split("=")[1].split("-")[0])
            except ValueError:
                start = 0
        body = data[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
        self.end_headers()
        self.wfile.write(body)


class ModelCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = {
            "onnx/model.onnx": os.urandom(4096),
            "tokenizer.json": b'{"ok": true, "note": "fake tokenizer"}',
            "config.json": b'{"dim": 512}',
        }
        cls.prefix = "Xenova/bge-small-zh-v1.5/resolve/main/"
        _Handler.files = {cls.prefix + k: v for k, v in cls.payload.items()}
        _Handler.fail_times = {}
        _Handler.hits = {}
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.srv.server_address[1]
        cls.thr = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thr.start()
        cls.mirror = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="rosmontis_models_")
        self.addCleanup(shutil.rmtree, self.d, True)
        _Handler.fail_times = {}
        _Handler.hits = {}

    def _install(self, **kw):
        return M.install("bge-small-zh-v1.5", root=self.d, mirror=self.mirror, **kw)

    def test_available_and_describe(self):
        names = set(M.available())
        self.assertIn("bge-small-zh-v1.5", names)
        rows = {r["name"]: r for r in M.describe()}
        self.assertEqual(rows["bge-small-zh-v1.5"]["dim"], 512)

    def test_install_downloads_and_writes_manifest(self):
        man = self._install()
        self.assertTrue(man["complete"])
        base = M.model_dir("bge-small-zh-v1.5", self.d)
        for rel, data in self.payload.items():
            f = base / rel
            self.assertTrue(f.exists(), rel)
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(), man["files"][rel])
        self.assertTrue(M.is_installed("bge-small-zh-v1.5", self.d))
        self.assertTrue(M.verify("bge-small-zh-v1.5", self.d)[0])

    def test_resolve_returns_paths(self):
        self._install()
        p = M.resolve("bge-small-zh-v1.5", self.d)
        self.assertEqual(p["dim"], 512)
        self.assertTrue(Path(p["onnx"]).exists())
        self.assertTrue(Path(p["tokenizer"]).exists())

    def test_resume_from_part_file(self):
        rel = "onnx/model.onnx"
        base = M.model_dir("bge-small-zh-v1.5", self.d)
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        part = base / (rel + ".part")
        part.write_bytes(self.payload[rel][:1024])  # 假装上次下到一半
        self._install()
        self.assertEqual((base / rel).read_bytes(), self.payload[rel])
        self.assertFalse(part.exists(), ".part 应在成功后清理")

    def test_retry_after_transient_failure(self):
        _Handler.fail_times[self.prefix + "onnx/model.onnx"] = 1  # 第一次 503
        man = self._install()
        self.assertTrue(man["complete"])
        self.assertGreaterEqual(_Handler.hits.get(self.prefix + "onnx/model.onnx", 0), 2)

    def test_pinned_sha_mismatch_keeps_old_model(self):
        # 先正常装好
        self._install()
        before = M.read_manifest("bge-small-zh-v1.5", self.d)
        self.assertIsNotNone(before)
        # 篡改清单里的官方校验值 → 强制重装必须失败，且不写新的 complete manifest
        orig = {k: dict(v) for k, v in M.MODELS.items()}
        try:
            M.MODELS["bge-small-zh-v1.5"]["sha256"] = {"onnx/model.onnx": "0" * 64}
            with self.assertRaises(M.ModelError):
                self._install(force=True)
        finally:
            M.MODELS.clear()
            M.MODELS.update(orig)
        self.assertTrue(M.is_installed("bge-small-zh-v1.5", self.d), "旧模型必须保留")

    def test_unknown_model_raises(self):
        with self.assertRaises(M.ModelError):
            M.install("no-such-model", root=self.d, mirror=self.mirror)

    def test_remove(self):
        self._install()
        self.assertTrue(M.remove("bge-small-zh-v1.5", self.d))
        self.assertFalse(M.is_installed("bge-small-zh-v1.5", self.d))
        self.assertFalse(M.remove("bge-small-zh-v1.5", self.d))


if __name__ == "__main__":
    unittest.main(verbosity=2)


