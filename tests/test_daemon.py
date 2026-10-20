"""C2 守护进程 / jobs / 幂等键 / 优雅退出测试。

运行：PYTHONIOENCODING=utf-8 python tests/test_daemon.py
不联网、不用真睡眠 30 分钟：时间靠注入 `now`，HTTP 用临时库 + 本机回环。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosmontis import Config, cli  # noqa: E402
from rosmontis.core import db as DB  # noqa: E402
from rosmontis.core import jobs  # noqa: E402
from rosmontis.core.memory import Memory  # noqa: E402
from rosmontis.harness import Harness  # noqa: E402

try:  # FastAPI 是可选依赖；没装时 HTTP/信号测试跳过，其余单测仍可跑
    from rosmontis.service import app as service_app

    _SERVICE_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    service_app = None
    _SERVICE_IMPORT_ERROR = e


class EchoLLM:
    """可注入的假模型：每次回同一句，便于断言幂等缓存。"""

    def __call__(self, messages, tools=None):
        return {"content": "记下了。", "tool_calls": [], "usage": {}, "finish_reason": "stop"}


class DaemonCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rosmontis_daemon_")
        self.cfg = Config(
            db_path=os.path.join(self.tmp, "t.db"),
            embed_provider="hash",
            llm_hot="rule",
            llm_deep="rule",
            log_level="ERROR",
            auto_forget=False,
        )
        self.mem = Memory(self.cfg)

    def tearDown(self):
        try:
            self.mem.wait_enrich()
        except Exception:
            pass
        DB.close_thread_conns()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── 小工具 ────────────────────────────────────────────
    def q(self, sql: str, args: tuple = ()):
        with DB.reader(self.cfg.db_path) as conn:
            return conn.execute(sql, args).fetchall()

    def set_last_active(self, sid: str, ts: float) -> None:
        with DB.tx(self.cfg.db_path) as conn:
            cur = conn.execute("UPDATE session SET last_active=? WHERE id=?", (ts, sid))
            if not cur.rowcount:
                conn.execute(
                    "INSERT INTO session(id,agent_id,user_id,created_at,last_active) "
                    "VALUES(?,'default','default',?,?)",
                    (sid, ts, ts),
                )

    def seed_activity(self, sid: str, *, now: float, n_dialog: int = 6, n_notes: int = 5) -> None:
        for i in range(n_dialog):
            self.mem.ingest(f"对话 {i}：今天事情很多", session_id=sid, t_event=now - 30 + i)
        for i in range(n_notes):
            self.mem.notebook_write(what=f"速记 {i}：记得回头看", session_id=sid)
        self.set_last_active(sid, now)


class TestJobsTable(DaemonCase):
    def test_enqueue_is_deterministic(self):
        now = 1_900_000_000.0
        j1 = jobs.enqueue(self.cfg, jobs.KIND_REVIEW, "s", next_run=now, bucket=7, now=now)
        j2 = jobs.enqueue(self.cfg, jobs.KIND_REVIEW, "s", next_run=now, bucket=7, now=now)
        j3 = jobs.enqueue(self.cfg, jobs.KIND_REVIEW, "s", next_run=now, bucket=8, now=now)
        self.assertTrue(j1["created"])
        self.assertFalse(j2["created"], "同 kind+target+窗口 不能产生第二条")
        self.assertEqual(j1["job_id"], j2["job_id"])
        self.assertTrue(j3["created"])
        self.assertEqual(jobs.counts(self.cfg)["total"], 2)
        self.assertIn("job", {r[0] for r in self.q("SELECT name FROM sqlite_master")},
                      "db.py 应幂等补出 job 表")

    def test_claim_fail_backoff_and_dead(self):
        now = 1_900_000_000.0
        job = jobs.enqueue(
            self.cfg, jobs.KIND_REVIEW, "s", next_run=now, bucket=1, now=now, max_attempts=2
        )
        jid = job["job_id"]
        first = jobs.claim_due(self.cfg, owner="t", now=now)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["status"], jobs.STATUS_RUNNING)
        self.assertEqual(first[0]["attempts"], 1)
        self.assertEqual(first[0]["lease_owner"], "t")

        jobs.fail(self.cfg, jid, "boom", now=now)
        row = jobs.get(self.cfg, jid)
        self.assertEqual(row["status"], jobs.STATUS_FAILED)
        self.assertGreater(row["next_run"], now)          # 退避
        self.assertIn("boom", row["last_error"])
        self.assertEqual(jobs.claim_due(self.cfg, owner="t", now=now + 0.1), [])

        second = jobs.claim_due(self.cfg, owner="t", now=row["next_run"] + 0.1)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["attempts"], 2)
        jobs.fail(self.cfg, jid, "boom again", now=row["next_run"] + 0.2)
        self.assertEqual(jobs.get(self.cfg, jid)["status"], jobs.STATUS_DEAD)

    def test_requeue_stale_and_finish(self):
        now = 1_900_000_000.0
        job = jobs.enqueue(self.cfg, jobs.KIND_SETTLE, "s", next_run=now, bucket=2, now=now)
        claimed = jobs.claim_due(self.cfg, owner="t", now=now, lease_seconds=1.0)
        self.assertEqual(len(claimed), 1)
        stale = jobs.requeue_stale(self.cfg, now=now + 2.0)
        self.assertEqual(stale["requeued"], 1)
        row = jobs.get(self.cfg, job["job_id"])
        self.assertEqual(row["status"], jobs.STATUS_PENDING)
        self.assertIsNone(row["lease_owner"])

        again = jobs.claim_due(self.cfg, owner="t2", now=now + 3.0)
        self.assertEqual(len(again), 1)
        jobs.finish(self.cfg, job["job_id"], now=now + 3.1)
        self.assertEqual(jobs.get(self.cfg, job["job_id"])["status"], jobs.STATUS_DONE)

    def test_execute_job_handlers(self):
        now = 1_900_000_000.0
        for kind in (jobs.KIND_REVIEW, jobs.KIND_DREAM, jobs.KIND_SETTLE, jobs.KIND_FORGET):
            target = "*" if kind == jobs.KIND_FORGET else "s"
            out = jobs.execute_job(self.mem, {"kind": kind, "target": target}, now=now)
            self.assertEqual(out["kind"], kind)
            self.assertIn("result", out)
        with self.assertRaises(ValueError):
            jobs.execute_job(self.mem, {"kind": "nope", "target": "s"}, now=now)


class TestDaemonHeartbeat(DaemonCase):
    def test_idle_session_settles_exactly_once(self):
        now = 1_900_000_000.0
        self.mem.ingest("很久以前的一句话", session_id="idle", t_event=now - 200)
        self.set_last_active("idle", now - 120)

        hb = cli.daemon_heartbeat(self.mem, now=now, idle_timeout=60, owner="t")
        kinds = [d["kind"] for d in hb["discovered"]]
        self.assertIn(jobs.KIND_SETTLE, kinds, f"空闲会话应触发 settle，实际 {kinds}")
        self.assertTrue(any(d["kind"] == jobs.KIND_SETTLE for d in hb["done"]))

        rows = self.q("SELECT * FROM job WHERE kind=?", (jobs.KIND_SETTLE,))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], jobs.STATUS_DONE)
        self.assertEqual(rows[0]["target"], "idle")
        self.assertEqual(rows[0]["lease_owner"], None)

        hb2 = cli.daemon_heartbeat(self.mem, now=now, idle_timeout=60, owner="t")
        self.assertEqual(len(self.q("SELECT * FROM job WHERE kind=?", (jobs.KIND_SETTLE,))), 1,
                         "同一空闲期不能重复 settle")
        self.assertEqual(hb2["discovered"], [])

    def test_review_and_dream_jobs_run_and_dream_check(self):
        now = 1_900_000_000.0
        self.seed_activity("busy", now=now)

        hb = cli.daemon_heartbeat(self.mem, now=now, idle_timeout=99_999, owner="t")
        discovered = {d["kind"] for d in hb["discovered"]}
        done = {d["kind"] for d in hb["done"]}
        self.assertIn(jobs.KIND_REVIEW, discovered)
        self.assertIn(jobs.KIND_DREAM, discovered)
        self.assertIn(jobs.KIND_REVIEW, done)
        self.assertIn(jobs.KIND_DREAM, done)
        rows = self.q("SELECT kind, status FROM job")
        self.assertTrue(all(r["status"] == jobs.STATUS_DONE for r in rows), [tuple(r) for r in rows])

        checks = self.q("SELECT key, value FROM meta WHERE key LIKE 'dream_check:%'")
        self.assertTrue(checks, "dream 执行后必须留 dream_check（D4b）")
        self.assertIn("skipped", json.loads(checks[-1]["value"]))

    def test_forget_job_periodic_dedup(self):
        self.cfg.auto_forget = True
        now = 1_900_000_000.0
        hb = cli.daemon_heartbeat(self.mem, now=now, idle_timeout=99_999, owner="t",
                                  forget_interval=3600)
        self.assertIn(jobs.KIND_FORGET, [d["kind"] for d in hb["discovered"]])
        jobs_rows = self.q("SELECT * FROM job WHERE kind=?", (jobs.KIND_FORGET,))
        self.assertEqual(len(jobs_rows), 1)
        self.assertEqual(jobs_rows[0]["status"], jobs.STATUS_DONE)
        cli.daemon_heartbeat(self.mem, now=now + 5, idle_timeout=99_999, owner="t",
                             forget_interval=3600)
        self.assertEqual(len(self.q("SELECT * FROM job WHERE kind=?", (jobs.KIND_FORGET,))), 1)

    def test_run_daemon_loop_stops_cleanly(self):
        stop = threading.Event()
        box: dict = {}

        def _target():
            try:
                box["summary"] = cli.run_daemon(
                    self.mem, once=False, interval=0.05,
                    idle_timeout=99_999, owner="t", limit=5, stop_event=stop,
                )
            except Exception as e:  # pragma: no cover
                box["error"] = e

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            t = threading.Thread(target=_target)
            t.start()
            time.sleep(0.3)
            stop.set()
            t.join(timeout=5)
        self.assertFalse(t.is_alive(), "stop_event 后 daemon 必须退出")
        self.assertNotIn("error", box)
        self.assertGreaterEqual(box["summary"]["iterations"], 1)
        self.assertTrue(box["summary"].get("stopped"))

    def test_cli_daemon_once(self):
        db = os.path.join(self.tmp, "cli.db")
        keys = ("ROSMONTIS_DB", "ROSMONTIS_EMBED", "ROSMONTIS_LLM_HOT", "ROSMONTIS_LLM_DEEP", "ROSMONTIS_LOG")
        old = {k: os.environ.get(k) for k in keys}
        os.environ.update({
            "ROSMONTIS_DB": db, "ROSMONTIS_EMBED": "hash",
            "ROSMONTIS_LLM_HOT": "rule", "ROSMONTIS_LLM_DEEP": "rule", "ROSMONTIS_LOG": "ERROR",
        })
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["--db", db, "daemon", "--once", "--idle-timeout", "60"])
            out = buf.getvalue()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual(rc, 0, out)
        self.assertIn("[daemon] start", out)
        self.assertIn("job=", out)
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT kind, status FROM job").fetchall()
        self.assertTrue(rows, "daemon --once 必须产出 job 记录")
        self.assertTrue(all(r[1] == "done" for r in rows), rows)

    def test_cli_version_and_help(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["--version"])
        self.assertEqual(rc, 0)
        text = buf.getvalue().strip()
        self.assertRegex(text, r"^rosmontis \d+\.\d+\.\d+  \(api v\d+\)$")

        top = cli.build_parser().format_help()
        self.assertIn("daemon", top)
        sub = next(
            a for a in cli.build_parser()._actions
            if isinstance(a, argparse._SubParsersAction)
        )
        daemon_help = sub.choices["daemon"].format_help()
        self.assertIn("--once", daemon_help)


class TestDaemonSignal(unittest.TestCase):
    """真实进程：kernel 信号 → daemon 收尾退出（Windows 用 CTRL_BREAK，POSIX 用 SIGTERM）。"""

    def test_daemon_stops_on_signal(self):
        tmp = tempfile.mkdtemp(prefix="rosmontis_daemon_sig_")
        log_path = os.path.join(tmp, "daemon.log")
        env = os.environ.copy()
        env.update({
            "ROSMONTIS_DB": os.path.join(tmp, "t.db"),
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
            [sys.executable, "run.py", "daemon", "--interval", "0.5", "--idle-timeout", "60"],
            cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        try:
            deadline = time.time() + 15
            while time.time() < deadline:
                logf.flush()
                try:
                    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    text = ""
                if "[daemon] start" in text:
                    break
                time.sleep(0.05)
            else:
                self.fail("daemon 未在 15s 内启动")
            time.sleep(0.8)  # 至少跑一拍心跳
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
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        try:
            self.assertNotIn("Traceback", text, text[-2000:])
            self.assertIn("[daemon] stopped", text, text[-2000:])
            self.assertIn("[daemon] signal=", text, text[-2000:])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestHttpContracts(DaemonCase):
    """直接调用路由函数（不依赖 httpx/TestClient），覆盖幂等键与 settle/jobs。"""

    def setUp(self):
        super().setUp()
        if service_app is None:
            self.skipTest(f"fastapi 不可用：{_SERVICE_IMPORT_ERROR}")
        self._old_mem = service_app._mem
        self._old_harness = service_app._harness
        service_app._mem = self.mem
        harness = Harness(self.cfg, EchoLLM(), auto_tick=False)
        harness.mem = self.mem
        service_app._harness = harness

    def tearDown(self):
        if service_app is not None:
            service_app._mem = self._old_mem
            service_app._harness = self._old_harness
        super().tearDown()

    def endpoint(self, path: str, method: str):
        for route in service_app.app.routes:
            if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
                return route.endpoint
        raise KeyError(f"{method} {path}")

    def body(self, resp) -> dict:
        return json.loads(resp.body.decode("utf-8"))

    def test_idempotency_episodes(self):
        ep = self.endpoint("/v1/episodes", "POST")
        r1 = ep(service_app.EpisodeIn(text="幂等测试一", session_id="s"), "key-ep-1")
        r2 = ep(service_app.EpisodeIn(text="幂等测试一", session_id="s"), "key-ep-1")
        b1, b2 = self.body(r1), self.body(r2)
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(b1, b2, "同 Idempotency-Key 必须返回首次结果")
        self.assertEqual(r2.headers.get("x-idempotent-replay"), "1")
        rows = self.q("SELECT id FROM episode WHERE text=?", ("幂等测试一",))
        self.assertEqual(len(rows), 1, "同键第二次请求不能多写一条")
        r3 = ep(service_app.EpisodeIn(text="幂等测试二", session_id="s"), "key-ep-1")
        self.assertEqual(self.body(r3), b1, "同键换了 body 也应回放首次结果")

    def test_idempotency_body_field(self):
        ep = self.endpoint("/v1/episodes", "POST")
        r1 = ep(service_app.EpisodeIn(text="body 幂等", session_id="s", idempotency_key="body-k-1"))
        r2 = ep(service_app.EpisodeIn(text="body 幂等", session_id="s", idempotency_key="body-k-1"))
        self.assertEqual(self.body(r1), self.body(r2))
        self.assertEqual(len(self.q("SELECT id FROM episode WHERE text='body 幂等'")), 1)

    def test_observe_endpoint(self):
        obs = self.endpoint("/v1/observe", "POST")
        r = obs(service_app.ObserveIn(session_id="s", user_text="用户 A", assistant_text="助手 A"), None)
        data = self.body(r)["data"]
        self.assertTrue(data["queued"])
        self.assertEqual(len(data["episode_ids"]), 2)
        self.assertEqual(len(self.q("SELECT id FROM episode WHERE text='用户 A'")), 1)
        self.assertEqual(len(self.q("SELECT id FROM episode WHERE text='助手 A'")), 1)

        # 空字符串忽略；只给一个字段就只写一条
        one = obs(service_app.ObserveIn(session_id="s", user_text="只有用户", assistant_text="   "), None)
        self.assertEqual(len(self.body(one)["data"]["episode_ids"]), 1)
        self.assertEqual(len(self.q("SELECT id FROM episode WHERE text='只有用户'")), 1)

        # 两个都空 → 422；session_id 空 → 422
        empty = obs(service_app.ObserveIn(session_id="s"), None)
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(self.body(empty)["error"]["code"], "empty_text")
        no_sid = obs(service_app.ObserveIn(session_id="", user_text="x"), None)
        self.assertEqual(no_sid.status_code, 422)
        self.assertEqual(self.body(no_sid)["error"]["code"], "missing_session")

        # 显式幂等键：同键只写一条
        k = service_app.ObserveIn(session_id="s", user_text="观察幂等", idempotency_key="obs-k")
        o1 = obs(k, None)
        o2 = obs(k, None)
        self.assertEqual(self.body(o1), self.body(o2))
        self.assertEqual(len(self.q("SELECT id FROM episode WHERE text='观察幂等'")), 1)

    def test_idempotency_turn(self):
        turn = self.endpoint("/v1/turn", "POST")
        r1 = turn(service_app.TurnIn(text="你好呀", session_id="s"), "key-turn-1")
        r2 = turn(service_app.TurnIn(text="你好呀", session_id="s"), "key-turn-1")
        b1, b2 = self.body(r1), self.body(r2)
        self.assertEqual(r1.status_code, 200, b1)
        self.assertEqual(b1, b2)
        self.assertEqual(b1["data"]["reply"], "记下了。")
        self.assertEqual(len(self.q("SELECT id FROM episode WHERE text='你好呀'")), 1)
        self.assertEqual(len(self.q("SELECT id FROM episode WHERE text='记下了。'")), 1)

    def test_settle_returns_job_id_and_jobs_endpoint(self):
        settle = self.endpoint("/v1/settle", "POST")
        r = settle(service_app.SettleIn(session_id="s"), idempotency_key="key-settle-1")
        data = self.body(r)["data"]
        job_id = data["job_id"]
        self.assertTrue(job_id.startswith("job_"))
        self.assertIn(data["status"], (jobs.STATUS_PENDING, jobs.STATUS_RUNNING, jobs.STATUS_DONE))
        self.assertEqual(
            self.body(settle(service_app.SettleIn(session_id="s"), idempotency_key="key-settle-1")),
            self.body(r),
        )
        # 兼容 query 形式：/v1/settle?session_id=...
        q = self.body(settle(None, "query-session", idempotency_key="key-settle-q"))
        self.assertEqual(q["data"]["session_id"], "query-session")

        deadline = time.time() + 10
        status = None
        while time.time() < deadline:
            row = jobs.get(self.cfg, job_id)
            status = row["status"] if row else None
            if status in (jobs.STATUS_DONE, jobs.STATUS_FAILED, jobs.STATUS_DEAD):
                break
            time.sleep(0.02)
        self.assertEqual(status, jobs.STATUS_DONE)

        get_jobs = self.endpoint("/v1/jobs", "GET")
        jobs_data = self.body(get_jobs(None, 50))["data"]
        self.assertGreaterEqual(jobs_data["summary"]["total"], 1)
        self.assertIn(job_id, {j["id"] for j in jobs_data["jobs"]})


class TestServiceShutdown(unittest.TestCase):
    """起真实 uvicorn → 优雅信号 → 确认 lifespan 收尾（Windows 用 CTRL_BREAK 走同一路径）。"""

    def _free_port(self) -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def _wait_health(self, base: str, timeout: float = 20.0) -> None:
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                urllib.request.urlopen(base + "/healthz", timeout=0.5).read()
                return
            except Exception as e:
                last = e
                time.sleep(0.1)
        raise AssertionError(f"服务未在 {timeout}s 内就绪：{last}")

    def test_graceful_signal_runs_lifespan(self):
        if service_app is None:
            self.skipTest(f"fastapi 不可用：{_SERVICE_IMPORT_ERROR}")
        tmp = tempfile.mkdtemp(prefix="rosmontis_sigterm_")
        log_path = os.path.join(tmp, "uvicorn.log")
        marker = os.path.join(tmp, "shutdown.json")
        port = self._free_port()
        env = os.environ.copy()
        env.update({
            "ROSMONTIS_DB": os.path.join(tmp, "t.db"),
            "ROSMONTIS_EMBED": "hash",
            "ROSMONTIS_LLM_HOT": "rule",
            "ROSMONTIS_LLM_DEEP": "rule",
            "ROSMONTIS_LOG": "INFO",
            "ROSMONTIS_SHUTDOWN_MARKER": marker,
            "ROSMONTIS_TOKEN": "test-token",
            "PYTHONPATH": str(ROOT),
            "PYTHONIOENCODING": "utf-8",
        })
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        logf = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "rosmontis.service.app:app",
             "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"],
            cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            self._wait_health(base)
            # 建 harness（wake 会触发 get_harness），让 lifespan join 有对象可收
            req = urllib.request.Request(
                base + "/v1/session/wake?session_id=probe", data=b"", method="POST",
                headers={"Authorization": "Bearer test-token"},
            )
            urllib.request.urlopen(req, timeout=5).read()

            if os.name == "nt":
                # Windows 的 SIGTERM 是 TerminateProcess，无法跑 lifespan；
                # uvicorn 在 Windows 捕获 SIGBREAK（CTRL_BREAK_EVENT），走同一 graceful 路径。
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            logf.close()
        try:
            log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("Traceback", log_text, log_text[-2000:])
            self.assertTrue(Path(marker).exists(), f"lifespan 未写关停标记；log:\n{log_text[-2000:]}")
            payload = json.loads(Path(marker).read_text(encoding="utf-8"))
            self.assertTrue(payload.get("shutdown"))
            self.assertTrue(payload.get("harness_joined"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    print("=" * 60)
    print(f"共 {res.testsRun} 项 · 失败 {len(res.failures)} · 错误 {len(res.errors)}")
    sys.exit(0 if res.wasSuccessful() else 1)
