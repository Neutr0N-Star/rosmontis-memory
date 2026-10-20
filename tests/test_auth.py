"""D2 本地鉴权 / 绑定安全 / 破坏性接口确认测试。

运行：PYTHONIOENCODING=utf-8 python tests/test_auth.py
不联网：起真实 uvicorn 只绑 127.0.0.1，用临时库与临时 ROSMONTIS_HOME。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosmontis import cli  # noqa: E402

try:  # FastAPI 是可选依赖；没装时跳过 HTTP 用例
    from rosmontis.service import app as service_app

    _SERVICE_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    service_app = None
    _SERVICE_IMPORT_ERROR = e


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _request(url: str, method: str = "GET", payload=None, headers: dict | None = None, timeout: float = 10.0):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    if body is not None:
        req.add_header("Content-Type", "application/json")

    def _parse(raw: str):
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError:
            return {"_raw": raw}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), _parse(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), _parse(e.read().decode("utf-8"))


def _wait_http(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5).read()
            return
        except Exception as e:
            last = e
            time.sleep(0.1)
    raise AssertionError(f"{url} 未在 {timeout}s 内就绪：{last}")


class AuthServerCase(unittest.TestCase):
    """一个带显式 token 的真实服务进程，供多个鉴权/确认用例共用。"""

    TOKEN = "test-token-123"

    @classmethod
    def setUpClass(cls):
        if service_app is None:
            raise unittest.SkipTest(f"fastapi 不可用：{_SERVICE_IMPORT_ERROR}")
        cls.tmp = tempfile.mkdtemp(prefix="rosmontis_auth_")
        cls.port = _free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.log_path = os.path.join(cls.tmp, "uvicorn.log")
        env = os.environ.copy()
        env.update({
            "ROSMONTIS_DB": os.path.join(cls.tmp, "t.db"),
            "ROSMONTIS_HOME": os.path.join(cls.tmp, "home"),
            "ROSMONTIS_EMBED": "hash",
            "ROSMONTIS_LLM_HOT": "rule",
            "ROSMONTIS_LLM_DEEP": "rule",
            "ROSMONTIS_LOG": "ERROR",
            "ROSMONTIS_TOKEN": cls.TOKEN,
            "PYTHONPATH": str(ROOT),
            "PYTHONIOENCODING": "utf-8",
        })
        cls.logf = open(cls.log_path, "w", encoding="utf-8")
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        cls.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "rosmontis.service.app:app",
             "--host", "127.0.0.1", "--port", str(cls.port), "--log-level", "warning"],
            cwd=str(ROOT), env=env, stdout=cls.logf, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        try:
            _wait_http(cls.base + "/healthz")
        except Exception:
            cls.proc.kill()
            cls.proc.wait()
            cls.logf.close()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.proc.poll() is None:
                if os.name == "nt":
                    cls.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    cls.proc.send_signal(signal.SIGTERM)
                cls.proc.wait(timeout=20)
        finally:
            if cls.proc.poll() is None:
                cls.proc.kill()
                cls.proc.wait(timeout=5)
            cls.logf.close()
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def auth(self, token: str | None = None):
        return {"Authorization": f"Bearer {token or self.TOKEN}"}

    # ── 1 无 token / 错 token / 带 token ──────────────────
    def test_api_requires_token(self):
        status, _headers, body = _request(self.base + "/v1/jobs")
        self.assertEqual(status, 401, body)
        self.assertEqual(body["ok"], False)
        self.assertIsNone(body["data"])
        self.assertEqual(body["error"]["code"], "unauthorized")
        self.assertFalse(body["error"]["retryable"])

    def test_api_with_valid_bearer(self):
        status, _headers, body = _request(self.base + "/v1/jobs", headers=self.auth())
        self.assertEqual(status, 200, body)
        self.assertTrue(body["ok"])
        self.assertIn("summary", body["data"])

    def test_wrong_and_malformed_token(self):
        for headers in (
            {"Authorization": "Bearer wrong-token"},
            {"Authorization": self.TOKEN},
            {"Authorization": "Basic dGVzdA=="},
            {},
        ):
            status, _h, body = _request(self.base + "/v1/jobs", headers=headers)
            self.assertEqual(status, 401, headers)
            self.assertEqual(body["error"]["code"], "unauthorized")

    def test_healthz_readyz_exempt_from_token(self):
        for path in ("/healthz", "/readyz"):
            status, _headers, body = _request(self.base + path)
            self.assertNotEqual(status, 401, f"{path} 不应要求 token")
            self.assertIn(status, (200, 503), body)

    def test_legacy_health_is_protected(self):
        # 只放行 healthz/readyz；旧 /health 不再免鉴权（探针已改用 healthz）
        status, _h, body = _request(self.base + "/health")
        self.assertEqual(status, 401, body)

    # ── 2 CORS 默认不含 null ──────────────────────────────
    def test_cors_default_excludes_null_origin(self):
        status, headers, _body = _request(
            self.base + "/v1/jobs", method="OPTIONS",
            headers={
                "Origin": "null",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertNotEqual(headers.get("access-control-allow-origin"), "null")
        self.assertNotEqual(status, 401, "OPTIONS 预检不应被 token 拦截")

    def test_cors_allows_local_origin(self):
        origin = "http://127.0.0.1:8770"
        status, headers, _body = _request(
            self.base + "/v1/jobs", method="OPTIONS",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(status, 200, headers)
        self.assertEqual(headers.get("access-control-allow-origin"), origin)

    # ── 3 破坏性接口确认 ──────────────────────────────────
    def test_wipe_requires_confirmation(self):
        url = self.base + "/v1/wipe?session_id=wipe-test"
        status, _h, body = _request(url, method="POST", headers=self.auth())
        self.assertEqual(status, 403, body)
        self.assertEqual(body["error"]["code"], "confirmation_required")

        status, _h, body = _request(
            url, method="POST", headers={**self.auth(), "X-Confirm": "yes"}
        )
        self.assertEqual(status, 200, body)
        self.assertTrue(body["ok"])

        status, _h, body = _request(
            self.base + "/v1/wipe?session_id=wipe-test2&confirm=yes",
            method="POST", headers=self.auth(),
        )
        self.assertEqual(status, 200, body)

    def test_settings_requires_confirmation(self):
        url = self.base + "/v1/settings"
        status, _h, body = _request(url, method="POST", payload={}, headers=self.auth())
        self.assertEqual(status, 403, body)
        self.assertEqual(body["error"]["code"], "confirmation_required")

        # 带确认后不会先被 403 拦；空 patch 由业务层返回 422（且不改 .env）
        status, _h, body = _request(
            url, method="POST", payload={},
            headers={**self.auth(), "X-Confirm": "yes"},
        )
        self.assertEqual(status, 422, body)
        self.assertEqual(body["error"]["code"], "empty")


class TokenFileAndBindCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("ROSMONTIS_HOME", "ROSMONTIS_TOKEN", "ROSMONTIS_DB")}
        self.tmp = tempfile.mkdtemp(prefix="rosmontis_token_")
        os.environ.pop("ROSMONTIS_TOKEN", None)
        os.environ["ROSMONTIS_HOME"] = os.path.join(self.tmp, "home")
        os.environ["ROSMONTIS_DB"] = os.path.join(self.tmp, "t.db")

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_token_file_generated_and_reused(self):
        if service_app is None:
            self.skipTest(f"fastapi 不可用：{_SERVICE_IMPORT_ERROR}")
        token1 = service_app.ensure_local_token()
        path = service_app.token_path()
        self.assertTrue(path.exists())
        self.assertTrue(path.read_text(encoding="utf-8").strip())
        self.assertEqual(path.read_text(encoding="utf-8").strip(), token1)
        token2 = service_app.ensure_local_token()
        self.assertEqual(token1, token2, "第二次启动必须复用同一个 token")
        if os.name != "nt":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_bind_security_helpers(self):
        self.assertTrue(cli.is_loopback_host("127.0.0.1"))
        self.assertTrue(cli.is_loopback_host("localhost"))
        self.assertTrue(cli.is_loopback_host("::1"))
        self.assertFalse(cli.is_loopback_host("0.0.0.0"))
        self.assertFalse(cli.is_loopback_host("192.168.1.10"))
        self.assertIsNone(cli.bind_security_error("127.0.0.1", None))
        self.assertIsNone(cli.bind_security_error("0.0.0.0", "explicit-token"))
        msg = cli.bind_security_error("0.0.0.0", None)
        self.assertIn("拒绝启动", msg)
        self.assertIn("--token", msg)

    def test_non_loopback_refuses_start(self):
        old = os.environ.pop("ROSMONTIS_TOKEN", None)
        import contextlib
        import io

        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                rc = cli.main(["--db", os.path.join(self.tmp, "x.db"),
                               "serve", "--host", "0.0.0.0", "--port", "1"])
        finally:
            if old is not None:
                os.environ["ROSMONTIS_TOKEN"] = old
        self.assertEqual(rc, 2)
        self.assertIn("拒绝启动", err.getvalue())


class ServePrintsTokenCase(unittest.TestCase):
    """真实 `rosmontis serve` 首次启动：生成 token、打印一次、带 token 才能访问。"""

    def test_cli_serve_prints_token_and_enforces_auth(self):
        if service_app is None:
            self.skipTest(f"fastapi 不可用：{_SERVICE_IMPORT_ERROR}")
        tmp = tempfile.mkdtemp(prefix="rosmontis_serve_")
        home = os.path.join(tmp, "home")
        log_path = os.path.join(tmp, "serve.log")
        port = _free_port()
        saved = {k: os.environ.get(k) for k in ("ROSMONTIS_HOME", "ROSMONTIS_TOKEN", "ROSMONTIS_DB")}
        os.environ.pop("ROSMONTIS_TOKEN", None)
        os.environ["ROSMONTIS_HOME"] = home
        os.environ["ROSMONTIS_DB"] = os.path.join(tmp, "t.db")
        env = os.environ.copy()
        env.update({
            "ROSMONTIS_EMBED": "hash",
            "ROSMONTIS_LLM_HOT": "rule",
            "ROSMONTIS_LLM_DEEP": "rule",
            "ROSMONTIS_LOG": "ERROR",
            "PYTHONPATH": str(ROOT),
            "PYTHONIOENCODING": "utf-8",
        })
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        logf = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "rosmontis", "serve", "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        try:
            deadline = time.time() + 20
            token = None
            while time.time() < deadline:
                logf.flush()
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
                match = re.search(r"本地访问 token.*：([A-Za-z0-9_\-]+)\s*$", text, re.M)
                if match:
                    token = match.group(1)
                    break
                if proc.poll() is not None:
                    self.fail(f"serve 进程提前退出：{text[-1500:]}")
                time.sleep(0.1)
            self.assertIsNotNone(token, "rosmontis serve 首次启动必须打印 token")
            base = f"http://127.0.0.1:{port}"
            _wait_http(base + "/healthz")
            status, _h, _b = _request(base + "/v1/jobs")
            self.assertEqual(status, 401, "不带 token 访问 /v1 必须 401")
            status, _h, body = _request(base + "/v1/jobs", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(status, 200, body)
            token_file = Path(home) / "token"
            self.assertTrue(token_file.exists())
            self.assertEqual(token_file.read_text(encoding="utf-8").strip(), token)
        finally:
            try:
                if proc.poll() is None:
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=20)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
                logf.close()
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    print("=" * 60)
    print(f"共 {res.testsRun} 项 · 失败 {len(res.failures)} · 错误 {len(res.errors)}")
    sys.exit(0 if res.wasSuccessful() else 1)
