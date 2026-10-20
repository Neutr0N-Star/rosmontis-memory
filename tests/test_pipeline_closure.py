"""WS-A2 收口验收：dream/forget 改道统一管线 + enrich 后重算 salience。

对应 `docs/14 §2.3` 未收口清单与评审 B4：
- 全仓扫描：`INSERT INTO episode` 只允许出现在 `rosmontis/core/pipeline.py`；
- dream 产物：origin='dream' / state='dream' / 1 条 item_state_log；
- forget 折叠摘要：origin='summary' / role='summary' / 1 条 create 日志；
- forget 归档：状态机 archive（state=archived + 1 条 archive 日志），不再直写 archived=1；
- B4：异步 enrich 完成后同事务重算 salience，同步 ingest 与 ingest_async 同输入 enrich 后差 = 0。

运行：`PYTHONIOENCODING=utf-8 python tests/test_pipeline_closure.py`
不依赖 Ollama / 网络；全部用临时库，不碰 rosmontis.db。
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosmontis import Config  # noqa: E402
from rosmontis.core import db as DB  # noqa: E402
from rosmontis.core import forget as FG  # noqa: E402
from rosmontis.core import item as IT  # noqa: E402
from rosmontis.core import pipeline as TR  # noqa: E402
from rosmontis.core import write as WR  # noqa: E402
from rosmontis.core.memory import Memory  # noqa: E402

INSERT_EPISODE_RE = re.compile(r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+episode\b", re.IGNORECASE)
UPDATE_ARCHIVED_RE = re.compile(r"UPDATE\s+episode\s+SET\s+archived\s*=\s*1", re.IGNORECASE)


class FakeLLM:
    """受控抽取器：把预设结果原样返回，让测试与模型解耦。"""

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
        if self.script:
            return self.script.pop(0)
        return {"entities": [], "facts": [], "state_changes": [], "theme": None}

    def chat(self, system, user, max_tokens=700, json_mode=False):
        return ""

    def available(self):
        return True


def mk(tag: str) -> tuple[Memory, FakeLLM, str]:
    d = tempfile.mkdtemp(prefix=f"rosmontis_{tag}_")
    cfg = Config(
        db_path=os.path.join(d, "t.db"),
        embed_provider="hash",
        llm_hot="rule",
        llm_deep="rule",
        log_level="ERROR",
        auto_forget=False,  # 闭环测试要手工控制遗忘时点
    )
    m = Memory(cfg)
    fake = FakeLLM()
    m.llm_hot = fake
    m.llm_deep = fake
    TR.bind(cfg, m.embedder, fake)
    return m, fake, d


def q1(m: Memory, sql: str, args: tuple = ()):
    with DB.reader(m.cfg.db_path) as conn:
        return conn.execute(sql, args).fetchone()


def qall(m: Memory, sql: str, args: tuple = ()):
    with DB.reader(m.cfg.db_path) as conn:
        return conn.execute(sql, args).fetchall()


def teardown(m: Memory | None, d: str) -> None:
    try:
        if m is not None:
            m.wait_enrich(timeout=30)
    except Exception:
        pass
    DB.close_thread_conns()
    shutil.rmtree(d, ignore_errors=True)


# ── 1 全仓扫描：唯一写入点 ─────────────────────────────────────
class TestFullRepoScan(unittest.TestCase):
    def _scan_roots(self) -> list[Path]:
        roots = [ROOT / "rosmontis"]
        for extra in ("bench", "examples"):
            p = ROOT / extra
            if p.exists():
                roots.append(p)
        return roots

    def test_insert_into_episode_only_in_pipeline(self):
        hits: set[str] = set()
        for root in self._scan_roots():
            for path in root.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                rel = path.relative_to(ROOT).as_posix()
                if rel.startswith("bench/_profile/"):
                    # 性能剖析脚手架故意旁路管线批量造库（一次性、不进产品路径）；
                    # 唯一写入点保证只针对产品代码与正式工具，此处显式豁免。
                    continue
                if INSERT_EPISODE_RE.search(path.read_text(encoding="utf-8")):
                    hits.add(rel)
        self.assertEqual(
            hits,
            {"rosmontis/core/pipeline.py"},
            f"存在绕过统一管线的 episode 直写点：{sorted(hits - {'rosmontis/core/pipeline.py'})}",
        )

    def test_dream_forget_have_no_direct_episode_write(self):
        for rel in ("rosmontis/core/dream.py", "rosmontis/core/forget.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIsNone(
                INSERT_EPISODE_RE.search(text),
                f"{rel} 仍存在 INSERT INTO episode",
            )
        forget_text = (ROOT / "rosmontis/core/forget.py").read_text(encoding="utf-8")
        self.assertIsNone(
            UPDATE_ARCHIVED_RE.search(forget_text),
            "forget.py 仍在直写 archived=1，应改走 pipeline.transition('archive')",
        )


# ── 2 dream 产物 ──────────────────────────────────────────────
class TestDreamClosure(unittest.TestCase):
    def test_dream_product_has_origin_state_and_log(self):
        m, fake, d = mk("dream")
        try:
            now = time.time()
            t = now - 100
            for i in range(5):
                m.ingest(f"今天第{i}件小事，天气不错", session_id="s", t_event=t + i)
            out = m.dream("s", force=True)
            self.assertTrue(out.get("dream_id"), out)

            row = q1(
                m,
                "SELECT id, origin, state, tier, role, source, salience FROM episode WHERE id=?",
                (out["dream_id"],),
            )
            self.assertEqual(row["origin"], IT.ORIGIN_DREAM)
            self.assertEqual(row["state"], IT.STATE_DREAM)
            self.assertEqual(row["tier"], "dream")
            self.assertEqual(row["role"], "dream")
            self.assertEqual(row["source"], "dream")
            self.assertAlmostEqual(row["salience"], 0.4, places=6)

            logs = qall(m, "SELECT action, from_state, to_state FROM item_state_log WHERE item_id=?", (row["id"],))
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]["action"], "create")
            self.assertEqual(logs[0]["to_state"], IT.STATE_DREAM)

            n = q1(m, "SELECT COUNT(*) c FROM dream_log WHERE session_id='s'")["c"]
            self.assertEqual(n, 1)
        finally:
            teardown(m, d)

    def test_dream_is_idempotent_within_same_second(self):
        m, fake, d = mk("dream2")
        try:
            from rosmontis.core import dream as DM

            now = time.time()
            for i in range(5):
                m.ingest(f"重放材料{i}", session_id="s", t_event=now - 100 + i)
            first = DM.dream(m.cfg, m.llm_deep, m.embedder, "s", now=now)
            self.assertTrue(first.get("dream_id"), first)
            # 同一时刻再跑一次：pipeline 幂等，不产生第二条 episode / 日志
            second = DM.dream(m.cfg, m.llm_deep, m.embedder, "s", now=now)
            self.assertEqual(second.get("dream_id"), first.get("dream_id"))
            n_ep = q1(m, "SELECT COUNT(*) c FROM episode WHERE origin='dream'")["c"]
            self.assertEqual(n_ep, 1)
            n_log = q1(m, "SELECT COUNT(*) c FROM item_state_log WHERE item_id=?", (first["dream_id"],))["c"]
            self.assertEqual(n_log, 1)
        finally:
            teardown(m, d)


# ── 3 forget：折叠摘要 + 归档 ─────────────────────────────────
class TestForgetClosure(unittest.TestCase):
    def test_fold_summary_has_origin_state_and_log(self):
        m, fake, d = mk("fold")
        try:
            now = time.time()
            old = now - 86400 * 365 * 5
            payload = {
                "entities": [{"name": "城东", "type": "place"}],
                "facts": [],
                "state_changes": [],
            }
            for i in range(5):
                fake.queue(payload)
                m.ingest(f"关于城东的第{i}条旧事", session_id="s", t_event=old + i)

            rep = m.forget("s")
            self.assertGreaterEqual(rep["folded"], 4, rep)

            summaries = qall(
                m,
                "SELECT id, origin, state, tier, role, source, salience FROM episode WHERE origin='summary'",
            )
            self.assertEqual(len(summaries), 1)
            s = summaries[0]
            self.assertEqual(s["state"], IT.STATE_NOTE)
            self.assertEqual(s["tier"], "retrieval")
            self.assertEqual(s["role"], "summary")
            self.assertEqual(s["source"], "summary")
            self.assertAlmostEqual(s["salience"], 0.55, places=6)

            logs = qall(m, "SELECT action, to_state FROM item_state_log WHERE item_id=?", (s["id"],))
            self.assertEqual([(x["action"], x["to_state"]) for x in logs], [("create", IT.STATE_NOTE)])

            members = qall(
                m, "SELECT id, folded_into, state FROM episode WHERE session_id='s' AND origin='ingest'"
            )
            self.assertEqual(len(members), 5)
            for r in members:
                self.assertEqual(r["folded_into"], s["id"])
                self.assertEqual(r["state"], IT.STATE_ARCHIVED)
                mlog = qall(m, "SELECT action FROM item_state_log WHERE item_id=?", (r["id"],))
                self.assertEqual([x["action"] for x in mlog][-1], "archive")

            member = q1(
                m,
                "SELECT weight FROM hyperedge_member WHERE member_type='episode' AND member_id=?",
                (s["id"],),
            )
            self.assertIsNotNone(member)
            self.assertAlmostEqual(member["weight"], 1.2, places=6)

            # 重复跑：不再产生摘要（幂等）
            rep2 = m.forget("s")
            self.assertEqual(rep2["folded"], 0)
            self.assertEqual(
                q1(m, "SELECT COUNT(*) c FROM episode WHERE origin='summary'")["c"], 1
            )
        finally:
            teardown(m, d)

    def test_decay_archive_uses_state_machine(self):
        m, fake, d = mk("decay")
        try:
            now = time.time()
            old = now - 86400 * 365 * 5
            r1 = m.ingest("一条很久以前的琐事", session_id="s", t_event=old)
            r2 = m.ingest("另一条很久以前的琐事", session_id="s", t_event=old + 10)

            n = FG.run_decay_only(m.cfg, "s", now)
            self.assertGreaterEqual(n, 2)

            for rid in (r1["episode_id"], r2["episode_id"]):
                row = q1(m, "SELECT state, archived, tier FROM episode WHERE id=?", (rid,))
                self.assertEqual(row["state"], IT.STATE_ARCHIVED)
                self.assertEqual(row["tier"], "retrieval")
                self.assertEqual(row["archived"], 1)
                logs = qall(m, "SELECT action, from_state, to_state, gate_json FROM item_state_log WHERE item_id=?", (rid,))
                self.assertEqual(logs[-1]["action"], "archive")
                self.assertEqual(logs[-1]["from_state"], IT.STATE_NOTE)
                self.assertEqual(logs[-1]["to_state"], IT.STATE_ARCHIVED)
                self.assertIn("3.0.0", logs[-1]["gate_json"])
        finally:
            teardown(m, d)


# ── 4 评审 B4：enrich 后重算 salience ─────────────────────────
class TestSalienceRecompute(unittest.TestCase):
    def test_sync_and_async_ingest_equal_after_enrich(self):
        m, fake, d = mk("salience")
        try:
            payload = {
                "entities": [{"name": "猫", "type": "thing"}],
                "facts": [
                    {"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.9}
                ],
                "state_changes": [],
            }
            fake.queue(payload)
            fake.queue(payload)
            t = 1_700_000_000.0

            r_sync = m.ingest("我喜欢猫", session_id="sync", t_event=t)
            r_async = m.ingest_async("我喜欢猫", session_id="async", t_event=t)
            # 旧 B4 的差：异步入口在 enrich 前没有结构证据，salience 偏低
            self.assertLess(r_async["salience"], r_sync["salience"])

            m.wait_enrich()
            s_sync = q1(m, "SELECT salience FROM episode WHERE id=?", (r_sync["episode_id"],))["salience"]
            s_async = q1(m, "SELECT salience FROM episode WHERE id=?", (r_async["episode_id"],))["salience"]
            self.assertAlmostEqual(s_async, s_sync, places=9, msg="两入口 enrich 后 salience 必须相等")
            self.assertEqual(
                q1(m, "SELECT COUNT(*) c FROM item_state_log WHERE item_id=?", (r_async["episode_id"],))["c"], 1
            )
            self.assertGreaterEqual(s_async, 0.0)
            self.assertLessEqual(s_async, 1.0)
        finally:
            teardown(m, d)

    def test_enrich_does_not_change_state_or_duplicate_logs(self):
        m, fake, d = mk("salience2")
        try:
            r = m.notebook_write("今天风很大", noticed="窗户响", means="她不太舒服", session_id="s")
            m.wait_enrich()
            row = q1(m, "SELECT state, tier, meta_json FROM episode WHERE id=?", (r["id"],))
            self.assertEqual(row["state"], IT.STATE_NOTE)
            self.assertEqual(row["tier"], "retrieval")
            self.assertEqual(
                q1(m, "SELECT COUNT(*) c FROM item_state_log WHERE item_id=?", (r["id"],))["c"], 1
            )
            self.assertGreaterEqual(
                q1(m, "SELECT salience FROM episode WHERE id=?", (r["id"],))["salience"], 0.0
            )
        finally:
            teardown(m, d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
