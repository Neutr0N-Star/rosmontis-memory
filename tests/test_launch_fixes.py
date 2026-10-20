"""D1 上线修复验收：无模型不写脏记忆 + 迁移守卫/备份/dry-run + ALAYA_HOME。

运行：`PYTHONIOENCODING=utf-8 python tests/test_launch_fixes.py`
全部临时库；不联网、不写 alaya.db。
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alaya  # noqa: E402
from alaya import Config  # noqa: E402
from alaya.api import AlayaError  # noqa: E402
from alaya.core import db as DB  # noqa: E402
from alaya.harness import Harness, LLMUnavailableError  # noqa: E402


# ── 公共 fixture ──────────────────────────────────────────────
def mk_cfg(tag: str) -> tuple[Config, str]:
    d = tempfile.mkdtemp(prefix=f"alaya_d1_{tag}_")
    cfg = Config(
        db_path=os.path.join(d, "t.db"),
        embed_provider="hash",
        llm_hot="rule",
        llm_deep="rule",
        log_level="ERROR",
        auto_forget=False,
    )
    return cfg, d


def teardown(cfg: Config | None, d: str) -> None:
    DB.close_thread_conns()
    shutil.rmtree(d, ignore_errors=True)


def roles(cfg: Config, session_id: str = "s") -> list[str]:
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            "SELECT role FROM episode WHERE session_id=? ORDER BY t_ingest", (session_id,)
        ).fetchall()
    return [r["role"] for r in rows]


def make_v3_db(path: str) -> None:
    """造一个 v3 形态库：删掉 v4 两列 + user_version=3 + 一条旧行。"""
    DB.init_db(path)
    conn = sqlite3.connect(path, isolation_level=None)
    for table in ("episode", "fact"):
        for col in ("embed_model", "embed_dim"):
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
    conn.execute(
        """INSERT INTO episode(id,session_id,role,text,t_event,t_ingest,source,tier,memory_type,author,
                               salience,emotion,state,origin)
           VALUES('ep_old','s','user','旧数据',1,1,'dialogue','retrieval','fact','system',0.5,0,'note','ingest')"""
    )
    conn.execute("PRAGMA user_version=3")
    conn.close()
    DB.close_thread_conns()


# ── P0-5：无模型不写 assistant 行 ──────────────────────────────
class TestNoModelNoDirtyMemory(unittest.TestCase):
    def _harness(self, tag: str, llm=None) -> tuple[Harness, Config, str]:
        cfg, d = mk_cfg(tag)
        h = Harness(cfg, llm, auto_tick=False)
        return h, cfg, d

    def test_harness_turn_without_llm_raises_and_writes_only_user(self):
        h, cfg, d = self._harness("nollm")
        try:
            with self.assertRaises(LLMUnavailableError):
                h.turn("今天加班到很晚", "s")
        finally:
            h.join(timeout=5)
            DB.close_thread_conns()
        try:
            r = roles(cfg)
            self.assertEqual(r, ["user"], f"无模型时库里只应有 user 一行，实际 {r}")
        finally:
            teardown(cfg, d)

    def test_llm_empty_response_raises_and_writes_only_user(self):
        h, cfg, d = self._harness("empty", lambda messages, tools=None: {"content": "", "tool_calls": []})
        try:
            with self.assertRaises(LLMUnavailableError):
                h.turn("今天加班到很晚", "s")
        finally:
            h.join(timeout=5)
        try:
            self.assertEqual(roles(cfg), ["user"])
        finally:
            teardown(cfg, d)

    def test_llm_exception_maps_to_llm_unavailable(self):
        def boom(messages, tools=None):
            raise TimeoutError("upstream timeout")

        h, cfg, d = self._harness("boom", boom)
        try:
            with self.assertRaises(LLMUnavailableError):
                h.turn("在吗", "s")
        finally:
            h.join(timeout=5)
        try:
            self.assertEqual(roles(cfg), ["user"])
        finally:
            teardown(cfg, d)

    def test_handle_turn_raises_alaya_error_code_and_retryable(self):
        d = tempfile.mkdtemp(prefix="alaya_d1_handle_")
        db_path = os.path.join(d, "t.db")
        h = alaya.open(
            db_path,
            {"embed_provider": "hash", "llm_hot": "rule", "llm_deep": "rule", "log_level": "ERROR"},
        )
        try:
            with self.assertRaises(AlayaError) as ctx:
                h.turn("u1", "c1", "今天加班到很晚")
            err = ctx.exception
            self.assertEqual(err.code, "llm_unavailable")
            self.assertTrue(err.retryable)
            self.assertEqual(err.as_dict()["error"]["code"], "llm_unavailable")
        finally:
            try:
                h.close()
            except Exception:
                pass
            DB.close_thread_conns()
        try:
            with DB.reader(db_path) as conn:
                rs = [dict(r) for r in conn.execute("SELECT role FROM episode WHERE session_id='u1'")]
            self.assertEqual([r["role"] for r in rs], ["user"], f"库里只应有 user 一行，实际 {rs}")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_turn_stream_error_event_after_meta_and_no_assistant(self):
        h, cfg, d = self._harness("stream")
        try:
            events = list(h.turn_stream("我有点累", "s"))
        finally:
            h.join(timeout=5)
        try:
            names = [e["event"] for e in events]
            self.assertEqual(names[0], "meta")
            self.assertIn("error", names)
            self.assertNotIn("done", names)
            err = next(e for e in events if e["event"] == "error")["data"]
            self.assertEqual(err["code"], "llm_unavailable")
            self.assertTrue(err["retryable"])
            self.assertEqual(err["status"], 503)
            self.assertEqual(roles(cfg), ["user"], "流式无模型时也不能落 assistant")
        finally:
            teardown(cfg, d)

    def test_turn_stream_success_still_writes_both(self):
        cfg, d = mk_cfg("streamok")
        h = Harness(cfg, lambda messages, tools=None: {"content": "我在听。"}, auto_tick=False)
        try:
            events = list(h.turn_stream("我有点累", "s"))
            h.join(timeout=5)
        finally:
            DB.close_thread_conns()
        try:
            self.assertEqual(events[-1]["event"], "done")
            self.assertEqual(roles(cfg), ["user", "assistant"])
        finally:
            teardown(cfg, d)


# ── P1-3：迁移守卫 / 自动备份 / dry-run ────────────────────────
class TestMigrationGuards(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="alaya_d1_mig_")
        self.path = os.path.join(self.dir, "t.db")

    def tearDown(self):
        DB.close_thread_conns()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _backups(self) -> list[Path]:
        return sorted(Path(self.dir).glob("t.db.*.bak"))

    def test_newer_schema_is_refused(self):
        DB.init_db(self.path)
        DB.close_thread_conns()
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute(f"PRAGMA user_version={DB.SCHEMA_VERSION + 1}")
        conn.close()
        with self.assertRaises(DB.SchemaVersionError):
            DB.init_db(self.path)
        with self.assertRaises(AlayaError) as ctx:
            alaya.open(self.path, {"embed_provider": "hash"})
        self.assertEqual(ctx.exception.code, "schema_too_new")
        self.assertFalse(ctx.exception.retryable)

    def test_auto_migration_makes_timestamped_backup(self):
        make_v3_db(self.path)
        self.assertEqual(self._backups(), [])
        meta = DB.init_db(self.path)  # auto
        self.assertTrue(meta["migrated"])
        self.assertEqual(meta["from_version"], 3)
        self.assertEqual(meta["to_version"], DB.SCHEMA_VERSION)
        backups = self._backups()
        self.assertEqual(len(backups), 1, backups)
        self.assertEqual(meta["backup"], str(backups[0]))
        # 备份仍是迁移前版本、能正常打开
        bconn = sqlite3.connect(str(backups[0]))
        try:
            self.assertEqual(bconn.execute("PRAGMA user_version").fetchone()[0], 3)
            self.assertEqual(bconn.execute("SELECT COUNT(*) FROM episode").fetchone()[0], 1)
            cols = {r[1] for r in bconn.execute("PRAGMA table_info(episode)")}
            self.assertNotIn("embed_model", cols)
        finally:
            bconn.close()
        DB.close_thread_conns()
        # 目标库已升级
        conn = DB.connect(self.path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], DB.SCHEMA_VERSION)
        # 第二次 init 不再迁移/不再新增备份
        DB.close_thread_conns()
        again = DB.init_db(self.path)
        self.assertFalse(again["migrated"])
        self.assertEqual(len(self._backups()), 1)

    def test_never_refuses_without_touching_db(self):
        make_v3_db(self.path)
        with self.assertRaises(DB.MigrationRequiredError):
            DB.init_db(self.path, migrate="never")
        self.assertEqual(self._backups(), [])
        conn = sqlite3.connect(self.path)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 3)
        finally:
            conn.close()
        with self.assertRaises(AlayaError) as ctx:
            alaya.open(self.path, {"embed_provider": "hash"}, migrate="never")
        self.assertEqual(ctx.exception.code, "migration_required")
        self.assertFalse(ctx.exception.retryable)

    def test_dry_run_reports_plan_without_changes(self):
        make_v3_db(self.path)
        plan = DB.migration_plan(self.path)
        self.assertTrue(plan["dry_run"])
        self.assertTrue(plan["needs_migration"])
        self.assertTrue(plan["will_backup"])
        self.assertGreaterEqual(len(plan["actions"]), 2)  # backup + migrate v3→v4
        self.assertTrue(any("backup:" in a for a in plan["actions"]))
        self.assertTrue(any("migrate v3" in a for a in plan["actions"]))

        plan2 = alaya.open(self.path, {"embed_provider": "hash"}, migrate="dry_run")
        self.assertEqual(plan2["current_version"], 3)
        self.assertEqual(plan2["target_version"], DB.SCHEMA_VERSION)
        self.assertTrue(plan2["dry_run"])
        # 没有迁移、没有备份
        conn = sqlite3.connect(self.path)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 3)
        finally:
            conn.close()
        self.assertEqual(self._backups(), [])

    def test_dry_run_on_new_db_reports_create_without_file(self):
        p = os.path.join(self.dir, "new.db")
        plan = DB.migration_plan(p)
        self.assertFalse(plan["exists"])
        self.assertTrue(plan["needs_migration"])
        self.assertTrue(any("create schema" in a for a in plan["actions"]))
        self.assertFalse(os.path.exists(p), "dry_run 不得创建数据库文件")

    def test_never_allows_already_current_schema(self):
        DB.init_db(self.path)  # 先建到当前版本
        DB.close_thread_conns()
        h = alaya.open(
            self.path,
            {"embed_provider": "hash", "llm_hot": "rule", "llm_deep": "rule", "log_level": "ERROR"},
            migrate="never",
        )
        try:
            self.assertEqual(Path(h.cfg.db_path), Path(self.path))
        finally:
            h.close()

    def test_plan_on_current_schema_says_noop(self):
        DB.init_db(self.path)
        DB.close_thread_conns()
        plan = DB.migration_plan(self.path)
        self.assertFalse(plan["needs_migration"])
        self.assertFalse(plan["will_backup"])


# ── P1-1：ALAYA_HOME 派生 ─────────────────────────────────────
class TestAlayaHome(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="alaya_home_")
        self._saved = {k: os.environ.get(k) for k in (
            "ALAYA_HOME", "ALAYA_DB", "ALAYA_MODEL_DIR", "ALAYA_TOKEN_FILE", "ALAYA_ENV_FILE",
        )}
        for k in self._saved:
            os.environ.pop(k, None)
        os.environ["ALAYA_HOME"] = self.home

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.home, ignore_errors=True)

    def test_home_derived_paths(self):
        home = Path(self.home)
        self.assertEqual(Config.home_dir(), home)
        cfg = Config()
        self.assertEqual(Path(cfg.db_path), home / "alaya.db")
        self.assertEqual(Config.models_dir(), home / "models")
        self.assertEqual(Config.token_file(), home / "token")
        self.assertEqual(Config.env_file(), home / ".env")

    def test_env_file_prefers_existing_home_env(self):
        home = Path(self.home)
        (home / ".env").write_text("ALAYA_TEST_MARK=home\n", encoding="utf-8")
        self.assertEqual(Config.env_file(), home / ".env")
        os.environ["ALAYA_ENV_FILE"] = str(home / "custom.env")
        self.assertEqual(Config.env_file(), home / "custom.env")

    def test_explicit_env_overrides(self):
        os.environ["ALAYA_DB"] = os.path.join(self.home, "custom.db")
        os.environ["ALAYA_MODEL_DIR"] = os.path.join(self.home, "m2")
        os.environ["ALAYA_TOKEN_FILE"] = os.path.join(self.home, "t2")
        cfg = Config()
        self.assertEqual(cfg.db_path, os.path.join(self.home, "custom.db"))
        self.assertEqual(Config.models_dir(), Path(self.home) / "m2")
        self.assertEqual(Config.token_file(), Path(self.home) / "t2")

    def test_home_expands_user(self):
        os.environ["ALAYA_HOME"] = "~/.alaya_test_d1"
        self.assertEqual(Config.home_dir(), Path.home() / ".alaya_test_d1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
