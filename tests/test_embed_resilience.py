"""C4 embedding 地基验收：断路器 / 向量身份 v4 / NULL 回算 / readyz 字段。

全部离线：远程调用由 `_embed_remote` 的测试桩替代，不联网、不依赖 Ollama。
运行：`PYTHONIOENCODING=utf-8 python tests/test_embed_resilience.py`
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alaya import Config  # noqa: E402
from alaya.core import db as DB  # noqa: E402
from alaya.core import jobs  # noqa: E402
from alaya.core import pipeline as TR  # noqa: E402
from alaya.core import recall as RC  # noqa: E402
from alaya.core import vec as V  # noqa: E402
from alaya.core.embed import BACKOFF_MAX, BACKOFF_BASE, Embedder  # noqa: E402
from alaya.core.item import MemoryItem  # noqa: E402

NOW = 1_792_368_000.0


class FakeLLM:
    provider = "fake"
    model = "fake"
    calls = 0
    failures = 0

    def __init__(self):
        self.script: list[dict] = []
        self.calls = 0
        self.failures = 0

    def queue(self, payload):
        self.script.append(payload)

    def chat_json(self, system, user, max_tokens=700):
        self.calls += 1
        return self.script.pop(0) if self.script else {
            "entities": [], "facts": [], "state_changes": [], "theme": None,
        }

    def chat(self, system, user, max_tokens=700, json_mode=False):
        return ""

    def available(self):
        return True


def mk_memory(tag: str, *, dim: int = 8, provider: str = "ollama", model_id: str = "fake-model"):
    """离线 Memory：真实 Embedder 只构造不联网，随后整体替换为测试桩。"""
    from alaya.core.memory import Memory

    d = tempfile.mkdtemp(prefix=f"alaya_{tag}_")
    cfg = Config(
        db_path=os.path.join(d, "t.db"),
        embed_provider=provider,
        embed_model="fake-model",
        embed_model_id=model_id,
        embed_dim=dim,
        llm_hot="rule",
        llm_deep="rule",
        log_level="ERROR",
        auto_forget=False,
    )
    mem = Memory(cfg)
    fake = FakeLLM()
    mem.llm_hot = fake
    mem.llm_deep = fake
    stub = StubEmbedder(dim=dim, model_id=model_id)
    mem.embedder = stub
    TR.bind(cfg, stub, fake)
    return mem, stub, fake, d


def teardown(mem=None, d: str | None = None) -> None:
    try:
        if mem is not None:
            mem.wait_enrich(timeout=10)
    except Exception:
        pass
    DB.close_thread_conns()
    if d:
        shutil.rmtree(d, ignore_errors=True)


class StubEmbedder(Embedder):
    """离线向量桩：远程调用由 _embed_remote 直接返回确定性向量。"""

    def __init__(self, *, dim: int = 8, model_id: str = "fake-model", failing: bool = False, sleep: float = 0.0):
        super().__init__(
            provider="ollama",
            model=model_id,
            dim=dim,
            ollama_url="http://127.0.0.1:9",
            timeout=0.05,
            model_id=model_id,
            probe_timeout=0.05,
        )
        self.failing = failing
        self.sleep = sleep
        self.vec_calls = 0

    def _embed_remote(self, text: str):
        self.vec_calls += 1
        if self.sleep:
            time.sleep(self.sleep)
        if self.failing:
            return None
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        seed = sum(map(ord, text or ""))
        return [((seed + i * 7) % 13 + 1.0) / 14.0 for i in range(self.dim)]


# ── 1 断路器 ──────────────────────────────────────────────────
class TestCircuitBreaker(unittest.TestCase):
    def test_consecutive_failures_probe_only_once(self):
        e = StubEmbedder(model_id="m1", failing=True)
        fake_now = [0.0]
        e._clock = lambda: fake_now[0]
        for i in range(20):
            v = e.embed(f"文本{i}")
            self.assertEqual(len(v), e.dim)
        self.assertEqual(e.remote_attempts, 1, "断路后连续调用不应再探测远程")
        self.assertEqual(e.vec_calls, 1)
        self.assertTrue(e.degraded)
        self.assertEqual(e.vector_mode(), "degraded")
        self.assertEqual(e.circuit_state()["state"], "open")

    def test_no_8s_waits_after_open(self):
        """模拟半死远端：每次远程要等 0.2s；断路后 20 次调用总耗时 < 1s。"""
        e = StubEmbedder(failing=True, sleep=0.2)
        fake_now = [0.0]
        e._clock = lambda: fake_now[0]
        t0 = time.perf_counter()
        for i in range(20):
            e.embed(f"半死{i}")
        elapsed = time.perf_counter() - t0
        self.assertEqual(e.remote_attempts, 1)
        self.assertLess(elapsed, 1.0, f"断路后仍在等远程：{elapsed:.2f}s")

    def test_exponential_backoff_then_recovery(self):
        e = StubEmbedder(failing=True)
        fake_now = [0.0]
        e._clock = lambda: fake_now[0]

        e.embed("a")                       # 第 1 次：失败 → 30s
        self.assertEqual(e.remote_attempts, 1)
        self.assertAlmostEqual(e._open_until, BACKOFF_BASE, places=3)
        fake_now[0] = BACKOFF_BASE - 0.1
        e.embed("b")                       # 窗口内：不探
        self.assertEqual(e.remote_attempts, 1)
        fake_now[0] = BACKOFF_BASE
        e.embed("c")                       # 半开重探：失败 → +60s
        self.assertEqual(e.remote_attempts, 2)
        self.assertAlmostEqual(e._open_until, BACKOFF_BASE + 60.0, places=3)
        fake_now[0] = BACKOFF_BASE + 60.0
        e.embed("d")                       # 失败 → +120s
        self.assertEqual(e.remote_attempts, 3)
        self.assertAlmostEqual(e._open_until, BACKOFF_BASE + 60.0 + 120.0, places=3)
        # 继续失败到封顶
        for step in (240.0, 480.0, 960.0):
            fake_now[0] = e._open_until
            e.embed(f"x{step}")
        self.assertLessEqual(e._open_until - fake_now[0], BACKOFF_MAX)
        self.assertTrue(e.degraded)

        # 恢复：到点后远端正常 → 自动切回 semantic
        e.failing = False
        fake_now[0] = e._open_until
        before = e.remote_attempts
        v = e.embed("recover")
        self.assertEqual(e.remote_attempts, before + 1)
        self.assertFalse(e.degraded)
        self.assertEqual(e.vector_mode(), "semantic")
        self.assertEqual(e.circuit_state()["state"], "closed")
        self.assertEqual(len(v), e.dim)

    def test_recovery_clears_degraded_cache(self):
        e = StubEmbedder(failing=True)
        fake_now = [0.0]
        e._clock = lambda: fake_now[0]
        v_fallback = e.embed("同一句话")   # 降级期缓存了哈希向量
        self.assertTrue(e.degraded)
        e.failing = False
        fake_now[0] = e._open_until
        v_semantic = e.embed("同一句话")
        self.assertEqual(v_semantic, e._vec("同一句话"), "恢复后必须重算，不能复用降级缓存")
        self.assertNotEqual(v_semantic, v_fallback)

    def test_health_probe_false_never_networks(self):
        e = StubEmbedder(failing=True)
        fake_now = [0.0]
        e._clock = lambda: fake_now[0]
        e.embed("a")
        calls_before = e.remote_attempts
        h = e.health(probe=False)
        self.assertEqual(e.remote_attempts, calls_before)
        self.assertEqual(e.health_attempts, 0)
        self.assertFalse(h["reachable"])
        self.assertEqual(h["vector_mode"], "degraded")
        self.assertEqual(h["circuit"]["state"], "open")
        # 断路窗口内 health(probe=True) 也不打远程
        h2 = e.health(probe=True)
        self.assertFalse(h2["reachable"])
        self.assertEqual(e.health_attempts, 0)


# ── 2 向量身份 v4 迁移 ────────────────────────────────────────
def _make_v3_db(path: str) -> None:
    """先建 v4 库，再退回 v3 形态（删掉 v4 两列 + user_version=3）。"""
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
    conn.execute(
        """INSERT INTO fact(id,session_id,subject_id,predicate,object_text,t_valid_from,t_ingest)
           VALUES('fa_old','s','en1','住在','北京',1,1)"""
    )
    conn.execute("PRAGMA user_version=3")
    conn.close()
    DB.close_thread_conns()


class TestVectorIdentityMigration(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="alaya_v4_")
        self.path = os.path.join(self.dir, "v3.db")

    def tearDown(self):
        DB.close_thread_conns()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_v3_to_v4_idempotent(self):
        _make_v3_db(self.path)
        DB.init_db(self.path)
        conn = DB.connect(self.path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], DB.SCHEMA_VERSION)
        for table in ("episode", "fact"):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            self.assertTrue({"embed_model", "embed_dim"} <= cols, table)
        # 旧数据：两列 NULL（未知身份），其余字段原样
        row = conn.execute("SELECT embed_model, embed_dim FROM episode WHERE id='ep_old'").fetchone()
        self.assertEqual((row["embed_model"], row["embed_dim"]), (None, None))
        row = conn.execute("SELECT embed_model, embed_dim FROM fact WHERE id='fa_old'").fetchone()
        self.assertEqual((row["embed_model"], row["embed_dim"]), (None, None))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM episode").fetchone()[0], 1)
        self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")

        DB.close_thread_conns()
        DB.init_db(self.path)  # 幂等：第二次不应报错/改数据
        conn = DB.connect(self.path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], DB.SCHEMA_VERSION)
        self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_failed_v4_migration_rolls_back(self):
        _make_v3_db(self.path)
        original = DB._migrate_v4

        def boom(conn):
            raise RuntimeError("injected v4 failure")

        DB._migrate_v4 = boom
        try:
            with self.assertRaises(RuntimeError):
                DB.init_db(self.path)
        finally:
            DB._migrate_v4 = original
            DB.close_thread_conns()
        conn = sqlite3.connect(self.path)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 3)
            for table in ("episode", "fact"):
                cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                self.assertFalse({"embed_model", "embed_dim"} & cols)
        finally:
            conn.close()

    @unittest.skipUnless((ROOT / "alaya.db").exists(), "真库 alaya.db 不在，跳过副本实测")
    def test_real_db_copy_to_v4(self):
        real = str(ROOT / "alaya.db")
        copy = os.path.join(self.dir, "alaya_copy.db")
        src = sqlite3.connect(f"file:{real.replace(os.sep, '/')}?mode=ro", uri=True)
        dst = sqlite3.connect(copy)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        probe = sqlite3.connect(copy)
        before = {
            t: probe.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("episode", "fact")
        }
        uv_before = probe.execute("PRAGMA user_version").fetchone()[0]
        probe.close()
        self.assertLess(uv_before, 4)

        DB.init_db(copy)
        conn = DB.connect(copy)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], DB.SCHEMA_VERSION)
            for table, n in before.items():
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], n, table)
                cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                self.assertTrue({"embed_model", "embed_dim"} <= cols)
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            # 旧向量身份未知，但检索侧允许使用（不报错）
            n_null = conn.execute("SELECT COUNT(*) FROM episode WHERE embed_model IS NULL").fetchone()[0]
            self.assertEqual(n_null, before["episode"])
        finally:
            DB.close_thread_conns()


# ── 3 写入记录身份 + 检索同身份过滤 ───────────────────────────
class TestIdentityRecordAndFilter(unittest.TestCase):
    def test_write_records_embed_identity(self):
        mem, stub, fake, d = mk_memory("ident")
        try:
            r = mem.ingest("我在城东住", session_id="s", t_event=NOW)
            row = q1(mem, "SELECT embedding, embed_model, embed_dim FROM episode WHERE id=?", (r["episode_id"],))
            self.assertIsNotNone(row["embedding"])
            self.assertEqual(row["embed_model"], stub.model_id)
            self.assertEqual(row["embed_dim"], stub.dim)

            note = mem.notebook_write("今天风很大", noticed="窗户响", means="该关窗了", session_id="s")
            mem.wait_enrich()
            row = q1(mem, "SELECT embedding, embed_model FROM episode WHERE id=?", (note["id"],))
            self.assertEqual(row["embed_model"], stub.model_id)
        finally:
            teardown(mem, d)

    def test_dense_only_compares_same_identity_but_accepts_unknown(self):
        mem, stub, fake, d = mk_memory("ident2")
        try:
            cfg = mem.cfg
            for eid, text in (("ep_same", "同一身份的向量"), ("ep_old", "旧数据未知身份"), ("ep_other", "别的模型的向量")):
                sub = TR.submit(
                    MemoryItem(
                        id=eid, session_id="s", text=text, origin="ingest", actor="system",
                        embedding=V.pack(stub.embed(text)), t_event=NOW, t_ingest=NOW,
                    ),
                    project=False, cfg=cfg,
                )
                self.assertTrue(sub.ok)
            with DB.tx(cfg.db_path) as conn:
                conn.execute("UPDATE episode SET embed_model=?, embed_dim=? WHERE id='ep_same'", (stub.model_id, stub.dim))
                conn.execute("UPDATE episode SET embed_model=?, embed_dim=? WHERE id='ep_other'", ("other-model", stub.dim))
                # ep_old 两列保持 NULL
            plan: dict = {}
            ids = RC._dense(cfg, stub, "任意查询", "*", as_of=NOW + 10, plan=plan)
            got = {i.split(":", 1)[1] for i in ids}
            self.assertIn("ep_same", got)
            self.assertIn("ep_old", got, "旧数据 NULL 视为未知身份，仍可用")
            self.assertNotIn("ep_other", got, "不同模型身份的向量不得参与余弦")
            self.assertEqual(plan.get("_identity_skipped"), 1)
        finally:
            teardown(mem, d)


# ── 4 NULL 回算 ───────────────────────────────────────────────
class TestEmbedBackfill(unittest.TestCase):
    def test_degraded_write_then_backfill(self):
        mem, stub, fake, d = mk_memory("backfill")
        try:
            stub.failing = True  # 降级期
            fake.queue(
                {
                    "entities": [{"name": "城东", "type": "place"}],
                    "facts": [{"subject": "我", "predicate": "住在", "object": "城东", "polarity": 1, "confidence": 0.9}],
                    "state_changes": [{"scope": "我", "slot": "city", "value": "城东"}],
                }
            )
            r1 = mem.ingest("我搬到城东了", session_id="s", t_event=NOW)
            r2 = mem.ingest("今天有点累", session_id="s", t_event=NOW + 1)
            rows = q1(mem, "SELECT COUNT(*) c FROM episode WHERE embedding IS NULL AND session_id='s'")
            self.assertGreaterEqual(rows["c"], 2, "降级期 episode 必须留 NULL 等回算")
            fact_null = q1(mem, "SELECT COUNT(*) c FROM fact WHERE embedding IS NULL")
            self.assertGreaterEqual(fact_null["c"], 1, "降级期 fact 也不得写哈希伪向量")

            self.assertGreater(mem.pending_backfill("s"), 0)
            skipped = mem.backfill_embeddings(session_id="s")
            self.assertTrue(skipped["skipped"], "降级时回算必须跳过，不能补哈希伪向量")

            # 恢复
            stub.failing = False
            stub._available = None
            stub.degraded = False
            stub.degraded_reason = None
            out = mem.backfill_embeddings(session_id="s", limit=100)
            self.assertFalse(out["skipped"], out)
            self.assertGreaterEqual(out["filled"], 2)
            row = q1(mem, "SELECT embedding, embed_model, embed_dim FROM episode WHERE id=?", (r1["episode_id"],))
            self.assertIsNotNone(row["embedding"], "回算后向量必须非空")
            self.assertEqual(row["embed_model"], stub.model_id)
            self.assertEqual(row["embed_dim"], stub.dim)
            row2 = q1(mem, "SELECT embedding FROM episode WHERE id=?", (r2["episode_id"],))
            self.assertIsNotNone(row2["embedding"])
            self.assertEqual(mem.pending_backfill("s"), 0)
            st = mem.stats()
            self.assertEqual(st["vector_mode"], "semantic")
            self.assertEqual(st["embed_model"], stub.model_id)
            self.assertEqual(st["pending_backfill"], 0)
        finally:
            teardown(mem, d)

    def test_stats_readyz_fields(self):
        mem, stub, fake, d = mk_memory("stats")
        try:
            st = mem.stats()
            for key in ("vector_mode", "embed_model", "pending_backfill", "embed_dim"):
                self.assertIn(key, st)
            self.assertEqual(st["vector_mode"], "semantic")
            self.assertEqual(st["embed_model"], stub.model_id)
            # readyz 的 providers.stats 直接摊平自 memory.stats()
            self.assertIn("vector_mode", st["embed"])
        finally:
            teardown(mem, d)

    def test_hash_provider_reports_degraded_mode(self):
        from alaya.core.memory import Memory

        d = tempfile.mkdtemp(prefix="alaya_hashmode_")
        try:
            cfg = Config(db_path=os.path.join(d, "t.db"), embed_provider="hash", llm_hot="rule",
                         llm_deep="rule", log_level="ERROR", auto_forget=False, embed_dim=8)
            mem = Memory(cfg)
            st = mem.stats()
            self.assertEqual(st["vector_mode"], "degraded")
            self.assertEqual(st["embed"]["provider"], "hash")
        finally:
            DB.close_thread_conns()
            shutil.rmtree(d, ignore_errors=True)


# ── 5 jobs：embed_backfill 执行 + daemon 自动补 ───────────────
class TestEmbedBackfillJob(unittest.TestCase):
    def test_execute_job_backfills(self):
        mem, stub, fake, d = mk_memory("job")
        try:
            for i in range(3):
                TR.submit(
                    MemoryItem(id=f"ep_null_{i}", session_id="s", text=f"待回算 {i}",
                               origin="ingest", actor="system", t_event=NOW, t_ingest=NOW),
                    project=False, cfg=mem.cfg,
                )
            self.assertGreaterEqual(mem.pending_backfill(), 3)
            job = jobs.enqueue(mem.cfg, jobs.KIND_EMBED_BACKFILL, "*", now=NOW)
            self.assertIn(jobs.KIND_EMBED_BACKFILL, jobs.KINDS)
            claimed = jobs.claim_due(mem.cfg, limit=5, owner="test", now=NOW)
            target = [j for j in claimed if j["kind"] == jobs.KIND_EMBED_BACKFILL]
            self.assertTrue(target, "embed_backfill job 应可被抢占")
            res = jobs.execute_job(mem, target[0], now=NOW)
            self.assertGreaterEqual(res["result"]["filled"], 3)
            self.assertEqual(mem.pending_backfill(), 0)
        finally:
            teardown(mem, d)

    def test_tick_job_enqueues_backfill_when_pending(self):
        mem, stub, fake, d = mk_memory("auto")
        try:
            mem.ingest("建立会话", session_id="s", t_event=NOW)
            TR.submit(
                MemoryItem(id="ep_wait", session_id="s", text="没向量的条目",
                           origin="ingest", actor="system", t_event=NOW, t_ingest=NOW),
                project=False, cfg=mem.cfg,
            )
            self.assertGreaterEqual(mem.pending_backfill(), 1)
            job = jobs.enqueue(mem.cfg, jobs.KIND_TICK, "s", now=NOW)
            claimed = jobs.claim_due(mem.cfg, limit=5, owner="test", now=NOW)
            tick_job = [j for j in claimed if j["id"] == job["job_id"]][0]
            out = jobs.execute_job(mem, tick_job, now=NOW)
            self.assertEqual(out.get("enqueued", {}).get("kind"), jobs.KIND_EMBED_BACKFILL)
            kinds = {j["kind"] for j in jobs.list_jobs(mem.cfg)}
            self.assertIn(jobs.KIND_EMBED_BACKFILL, kinds)
        finally:
            teardown(mem, d)


def q1(mem, sql: str, args: tuple = ()):
    with DB.reader(mem.cfg.db_path) as conn:
        return conn.execute(sql, args).fetchone()


if __name__ == "__main__":
    unittest.main(verbosity=2)
