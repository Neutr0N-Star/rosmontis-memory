"""统一写入管线 / 状态机 / schema v3 迁移的验收测试。

运行：`PYTHONIOENCODING=utf-8 python tests/test_items.py`
不依赖 Ollama / 网络；迁移测试只用临时副本库，真库 `rosmontis.db` 全程只读。

覆盖（对应 A2 M1/M3/M5、I6/I9/I10/I12）：
- 写入单点：`INSERT INTO episode` 只允许出现在 core/pipeline.py（其它线遗留的
  dream.py / forget.py 在报告里点名，测试只做守门不放行新违规）。
- 幂等：重复 submit 不产生第二条 episode、不重复写日志。
- 状态机：inbox/note/resident/important/muted/archived 的转移与 gate 拒绝。
- 迁移：v2→v3 幂等、回填容错（tier='dream'、t_event=1e18、异常 session_id）、
  失败 ROLLBACK 保持 v2、真库副本实测。
- 基线：subprocess 跑 tests/test_all.py，必须 15 项 OK。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosmontis import Config  # noqa: E402
from rosmontis.core import db as DB  # noqa: E402
from rosmontis.core import gate as GATE  # noqa: E402
from rosmontis.core import item as IT  # noqa: E402
from rosmontis.core import pipeline as TR  # noqa: E402
from rosmontis.core import write as WR  # noqa: E402
from rosmontis.core.memory import Memory  # noqa: E402

CORE_DIR = ROOT / "rosmontis" / "core"

# 不在本工作线写入范围的遗留直写点（报告已点名，等工作线收口）
LEGACY_EPISODE_INSERTERS = {"dream.py", "forget.py"}


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


def mk(tag: str = "items") -> tuple[Memory, FakeLLM, str]:
    d = tempfile.mkdtemp(prefix=f"rosmontis_{tag}_")
    cfg = Config(
        db_path=os.path.join(d, "t.db"),
        embed_provider="hash",
        llm_hot="rule",
        llm_deep="rule",
        log_level="ERROR",
        auto_forget=False,  # forget.py 仍是直写 archived 的遗留入口，单独测状态机时关掉
    )
    m = Memory(cfg)
    fake = FakeLLM()
    m.llm_hot = fake
    m.llm_deep = fake
    TR.bind(cfg, m.embedder, fake)  # 异步 enrich 也走 fake，测试可复现
    return m, fake, d


def teardown(m: Memory | None, d: str) -> None:
    try:
        if m is not None:
            m.wait_enrich(timeout=30)
    except Exception:
        pass
    DB.close_thread_conns()
    shutil.rmtree(d, ignore_errors=True)


def q1(m: Memory, sql: str, args: tuple = ()):
    with DB.reader(m.cfg.db_path) as conn:
        return conn.execute(sql, args).fetchone()


def qall(m: Memory, sql: str, args: tuple = ()):
    with DB.reader(m.cfg.db_path) as conn:
        return conn.execute(sql, args).fetchall()


# ── 1 对象模型 ────────────────────────────────────────────────
class TestMemoryItem(unittest.TestCase):
    def test_contract_fields_and_projection(self):
        item = IT.MemoryItem(
            id="ep_x",
            session_id="s",
            text="我在城东住",
            origin=IT.ORIGIN_INGEST,
            actor=IT.ACTOR_USER,
            memory_type="fact",
            state=IT.STATE_RESIDENT,
            segments={"what": "我在城东住"},
            salience=0.6,
            emotion=0.2,
            t_event=1000.0,
            t_ingest=1001.0,
        )
        self.assertEqual(IT.tier_of(IT.STATE_RESIDENT), "resident")
        self.assertEqual(IT.projection(IT.STATE_RESIDENT), ("resident", 0, 0))
        self.assertEqual(IT.projection(IT.STATE_ARCHIVED), ("retrieval", 1, 0))
        self.assertEqual(IT.projection(IT.STATE_MUTED), ("retrieval", 0, 1))
        self.assertEqual(item.effective_role(), "user")
        self.assertEqual(item.effective_author(), "teacher")
        self.assertTrue(IT.recall_visible(IT.STATE_NOTE))
        self.assertFalse(IT.recall_visible(IT.STATE_INBOX))

    def test_legacy_compat_state(self):
        self.assertEqual(IT.compat_state("dream"), IT.STATE_DREAM)
        self.assertEqual(IT.compat_state("inbox"), IT.STATE_INBOX)
        self.assertEqual(IT.compat_state("retrieval"), IT.STATE_NOTE)
        self.assertEqual(IT.compat_state("retrieval", archived=1), IT.STATE_ARCHIVED)
        self.assertEqual(IT.compat_state("resident", muted=1), IT.STATE_MUTED)

    def test_from_row_falls_back_to_projection(self):
        row = {
            "id": "ep_1",
            "session_id": "s",
            "text": "x",
            "tier": "dream",
            "state": None,
            "origin": None,
            "source": "dream",
            "role": "dream",
            "meta_json": None,
        }
        item = IT.MemoryItem.from_row(row)
        self.assertEqual(item.state, IT.STATE_DREAM)
        self.assertEqual(item.origin, IT.ORIGIN_DREAM)


# ── 2 写入单点 ────────────────────────────────────────────────
class TestSingleWritePoint(unittest.TestCase):
    _INSERT_RE = re.compile(r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+episode\b", re.IGNORECASE)

    def test_insert_into_episode_only_in_pipeline(self):
        hits = set()
        for path in CORE_DIR.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            if self._INSERT_RE.search(text):
                hits.add(path.name)
        self.assertIn("pipeline.py", hits, "管线模块必须持有唯一 episode 写入点")
        unexpected = hits - {"pipeline.py"} - LEGACY_EPISODE_INSERTERS
        self.assertEqual(unexpected, set(), f"新增了绕过管线的直写点：{sorted(unexpected)}")

    def test_every_write_has_state_log_and_projection(self):
        m, fake, d = mk("single")
        try:
            fake.queue(
                {
                    "entities": [{"name": "猫", "type": "thing"}],
                    "facts": [
                        {"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.9}
                    ],
                    "state_changes": [],
                }
            )
            r = m.ingest("我喜欢猫", session_id="s", t_event=1000.0)
            note = m.notebook_write("今天风很大", session_id="s")
            m.wait_enrich()

            rows = qall(m, "SELECT id, state, tier, archived, muted FROM episode")
            self.assertGreaterEqual(len(rows), 2)
            for row in rows:
                self.assertIn(row["state"], IT.STATES, f"{row['id']} 没有状态")
                self.assertEqual(row["tier"], IT.tier_of(row["state"]))
                self.assertEqual(bool(row["archived"]), bool(IT.archived_of(row["state"])))
                self.assertEqual(bool(row["muted"]), bool(IT.muted_of(row["state"])))
                logs = qall(m, "SELECT action, to_state FROM item_state_log WHERE item_id=?", (row["id"],))
                self.assertEqual(len(logs), 1, f"{row['id']} 应有且只有 1 条 create 日志")
                self.assertEqual(logs[0]["action"], "create")
                self.assertEqual(logs[0]["to_state"], row["state"])

            ing = q1(m, "SELECT origin, role, author FROM episode WHERE id=?", (r["episode_id"],))
            self.assertEqual(ing["origin"], IT.ORIGIN_INGEST)
            nb = q1(m, "SELECT origin, role, author FROM episode WHERE id=?", (note["id"],))
            self.assertEqual(nb["origin"], IT.ORIGIN_TOOL)
            self.assertEqual(nb["author"], "self")
        finally:
            teardown(m, d)

    def test_salience_single_implementation(self):
        """M3：同字段输入两条入口（不同 origin）salience 必须相等。"""
        m, fake, d = mk("salience")
        try:
            a = TR.submit(
                IT.MemoryItem(
                    session_id="s",
                    text="同一段内容用于比较打分",
                    origin=IT.ORIGIN_INGEST,
                    actor=IT.ACTOR_SYSTEM,
                    t_event=1000.0,
                ),
                project=False,
                cfg=m.cfg,
            )
            b = TR.submit(
                IT.MemoryItem(
                    session_id="s",
                    text="同一段内容用于比较打分",
                    origin=IT.ORIGIN_TOOL,
                    actor=IT.ACTOR_SYSTEM,
                    t_event=2000.0,
                ),
                project=False,
                cfg=m.cfg,
            )
            self.assertTrue(a.ok and b.ok)
            self.assertEqual(a.salience, b.salience)
            self.assertAlmostEqual(
                WR.salience_of("同一段内容用于比较打分", 0.0),
                a.salience,
                places=6,
            )
        finally:
            teardown(m, d)


# ── 3 幂等 ────────────────────────────────────────────────────
class TestPipelineIdempotency(unittest.TestCase):
    def test_submit_same_item_twice(self):
        m, fake, d = mk("idem2")
        try:
            item = IT.MemoryItem(
                id="ep_fixed_1",
                session_id="s",
                text="重复写入只应有一条",
                origin=IT.ORIGIN_INGEST,
                actor=IT.ACTOR_USER,
                t_event=1000.0,
            )
            a = TR.submit(item, project=False, cfg=m.cfg)
            b = TR.submit(item, project=False, cfg=m.cfg)
            self.assertTrue(a.ok)
            self.assertFalse(a.duplicate)
            self.assertTrue(b.duplicate)
            self.assertEqual(a.item_id, b.item_id)
            self.assertEqual(q1(m, "SELECT COUNT(*) c FROM episode")["c"], 1)
            self.assertEqual(q1(m, "SELECT COUNT(*) c FROM item_state_log")["c"], 1)
        finally:
            teardown(m, d)

    def test_ingest_duplicate_or_raises_flat(self):
        m, fake, d = mk("idem3")
        try:
            t = 1_700_000_000.0
            r1 = m.ingest("我喜欢猫", session_id="s", t_event=t)
            r2 = m.ingest("我喜欢猫", session_id="s", t_event=t)
            self.assertFalse(r1.get("duplicate"))
            self.assertTrue(r2.get("duplicate"))
            self.assertEqual(m.stats()["episode"], 1)
        finally:
            teardown(m, d)

    def test_notebook_repeat_write_idempotent(self):
        """I8：5 分钟内主动重复写 5 次只产生 1 条 item / 1 条日志。"""
        m, fake, d = mk("idem4")
        try:
            results = [
                m.notebook_write("同一件小事", noticed="反复写", session_id="s") for _ in range(5)
            ]
            self.assertTrue(all(r["ok"] for r in results))
            self.assertEqual(len({r["id"] for r in results}), 1)
            self.assertEqual(q1(m, "SELECT COUNT(*) c FROM episode WHERE origin='tool'")["c"], 1)
            self.assertEqual(q1(m, "SELECT COUNT(*) c FROM item_state_log")["c"], 1)
        finally:
            teardown(m, d)


# ── 4 状态机 ──────────────────────────────────────────────────
class TestStateMachine(unittest.TestCase):
    def test_note_lifecycle(self):
        m, fake, d = mk("state")
        try:
            r = m.notebook_write(
                "今天在城东看了房子", noticed="离地铁近", means="考虑搬家", next_step="问预算", session_id="s"
            )
            self.assertEqual(r["state"], IT.STATE_NOTE)
            self.assertEqual(m.item_state(r["id"]), IT.STATE_NOTE)

            up = m.promote(r["id"], "resident")
            self.assertTrue(up["ok"])
            self.assertEqual(up["to"], "resident")
            self.assertEqual(q1(m, "SELECT tier FROM episode WHERE id=?", (r["id"],))["tier"], "resident")

            down = TR.transition(r["id"], "demote", cfg=m.cfg)
            self.assertTrue(down.ok)
            self.assertEqual(down.to_state, IT.STATE_NOTE)
            self.assertEqual(m.item_state(r["id"]), IT.STATE_NOTE)

            skip = TR.transition(r["id"], "skip", cfg=m.cfg)
            self.assertTrue(skip.ok)
            self.assertEqual(skip.to_state, IT.STATE_MUTED)
            proj = q1(m, "SELECT tier, archived, muted FROM episode WHERE id=?", (r["id"],))
            self.assertEqual((proj["tier"], proj["archived"], proj["muted"]), ("retrieval", 0, 1))

            unmute = TR.transition(r["id"], "unmute", cfg=m.cfg)
            self.assertTrue(unmute.ok)
            self.assertEqual(unmute.to_state, IT.STATE_NOTE)

            logs = m.state_log(r["id"])
            actions = [x["action"] for x in logs]
            self.assertEqual(actions, ["create", "promote_resident", "demote", "skip", "unmute"])
            self.assertTrue(all(json.loads(x["gate_json"])["version"] == GATE.GATE_VERSION for x in logs))
        finally:
            teardown(m, d)

    def test_inbox_is_not_promotable_before_review(self):
        m, fake, d = mk("inbox")
        try:
            r = m.notebook_write("随手记一句", session_id="s")
            self.assertEqual(r["state"], IT.STATE_INBOX)

            denied = m.promote(r["id"], "resident")
            self.assertFalse(denied["ok"])
            self.assertIn("inbox", denied["reason"])

            ok = TR.transition(r["id"], "review_note", cfg=m.cfg)
            self.assertTrue(ok.ok)
            self.assertEqual(m.item_state(r["id"]), IT.STATE_NOTE)
            self.assertTrue(m.promote(r["id"], "resident")["ok"])
        finally:
            teardown(m, d)

    def test_tool_behavior_cannot_be_resident(self):
        m, fake, d = mk("gate2")
        try:
            r = m.ingest("grep 又返回空了，这个工具行为要记一下", session_id="s", t_event=1000.0)
            res = m.promote(r["episode_id"], "resident")
            self.assertFalse(res["ok"], "工具行为不该能升常驻（PR.can_promote 必须仍在闸内）")
            self.assertIn("工具行为", res["reason"])
        finally:
            teardown(m, d)

    def test_system_origin_cannot_persona(self):
        m, fake, d = mk("sys")
        try:
            sub = TR.submit(
                IT.MemoryItem(
                    session_id="s",
                    text="系统整理的候选",
                    origin=IT.ORIGIN_SYSTEM,
                    source="notebook",
                    segments={"what": "系统整理的候选", "means": "候选"},
                ),
                project=False,
                cfg=m.cfg,
            )
            tr = TR.transition(sub.item_id, "promote_persona", cfg=m.cfg)
            self.assertFalse(tr.ok)
            self.assertIn("系统产物", tr.reason)
        finally:
            teardown(m, d)

    def test_persona_frequency_and_amplitude(self):
        m, fake, d = mk("persona_gate")
        try:
            for i in range(3):
                r = m.notebook_write(f"她喜欢第{i}种靠窗的座位", means=f"靠窗才安心{i}", session_id="p")
                resp = m.notebook_promote(r["id"], "persona", section="taste")
                if i < 2:
                    self.assertTrue(resp["ok"], resp)
                    self.assertEqual(m.item_state(r["id"]), IT.STATE_NOTE)
                else:
                    self.assertFalse(resp["ok"])
                    self.assertIn("最多改", resp["reason"])

            long_note = m.notebook_write("她有一条很长很长的自我要求", means="长" * 260, session_id="q")
            too_long = m.notebook_promote(long_note["id"], "persona")
            self.assertFalse(too_long["ok"])
            self.assertIn("超过", too_long["reason"])
        finally:
            teardown(m, d)

    def test_archive_and_log(self):
        m, fake, d = mk("archive")
        try:
            a = m.notebook_write("凑不成组的速记一", session_id="r")
            b = m.notebook_write("凑不成组的速记二", session_id="r")
            m.wait_enrich()
            rep = m.notebook_review("r")
            self.assertEqual(rep["scanned"], 2)
            for nid in (a["id"], b["id"]):
                self.assertEqual(m.item_state(nid), IT.STATE_ARCHIVED)
                logs = m.state_log(nid)
                self.assertEqual(logs[-1]["action"], "review_archive")
                self.assertEqual(logs[-1]["to_state"], IT.STATE_ARCHIVED)
        finally:
            teardown(m, d)


# ── 5 异步 enrich（主动条目）────────────────────────────────────
class TestEnrich(unittest.TestCase):
    def test_active_item_enriched_with_segments_preserved(self):
        m, fake, d = mk("enrich")
        try:
            r = m.notebook_write(
                "今天在城东看了房子", noticed="离地铁近", means="考虑搬家", next_step="问预算", session_id="s"
            )
            self.assertTrue(r["enrich_queued"])
            m.wait_enrich()
            row = q1(m, "SELECT meta_json FROM episode WHERE id=?", (r["id"],))
            meta = json.loads(row["meta_json"])
            self.assertTrue(meta.get("enriched"), "主动条目必须进异步 enrich")
            self.assertEqual(meta.get("means"), "考虑搬家", "enrich 不得覆盖主动条目的四段")
            self.assertEqual(meta.get("segments", {}).get("next"), "问预算")
            self.assertEqual(q1(m, "SELECT COUNT(*) c FROM item_state_log WHERE item_id=?", (r["id"],))["c"], 1)
        finally:
            teardown(m, d)

    def test_ingest_async_uses_pipeline_enrich(self):
        m, fake, d = mk("async")
        try:
            r = m.ingest_async("我在字节工作", session_id="s", t_event=1000.0)
            self.assertTrue(r["queued"])
            m.wait_enrich()
            self.assertEqual(q1(m, "SELECT COUNT(*) c FROM item_state_log WHERE item_id=?", (r["episode_id"],))["c"], 1)
            row = q1(m, "SELECT state, origin FROM episode WHERE id=?", (r["episode_id"],))
            self.assertEqual(row["origin"], IT.ORIGIN_INGEST)
            self.assertEqual(row["state"], IT.STATE_NOTE)
        finally:
            teardown(m, d)


# ── 6 迁移：v2 → v3 ───────────────────────────────────────────
_V2_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE session (id TEXT PRIMARY KEY, agent_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL DEFAULT 'default', title TEXT, created_at REAL NOT NULL, last_active REAL NOT NULL);
CREATE TABLE episode (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, seq INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL, text TEXT NOT NULL, t_event REAL NOT NULL, t_ingest REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'dialogue', tier TEXT NOT NULL DEFAULT 'retrieval',
    memory_type TEXT NOT NULL DEFAULT 'fact', author TEXT NOT NULL DEFAULT 'system',
    confirmed INTEGER NOT NULL DEFAULT 0, salience REAL NOT NULL DEFAULT 0.5,
    emotion REAL NOT NULL DEFAULT 0, access_count INTEGER NOT NULL DEFAULT 0,
    last_access REAL, embedding BLOB, folded_into TEXT, archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0, muted INTEGER NOT NULL DEFAULT 0, cooldown_until REAL,
    frozen_from TEXT, correction_of TEXT, superseded_by TEXT, chunk_key TEXT, meta_json TEXT);
CREATE TABLE fact (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL, object_text TEXT NOT NULL, object_entity_id TEXT,
    polarity INTEGER NOT NULL DEFAULT 1, confidence REAL NOT NULL DEFAULT 0.7,
    author TEXT NOT NULL DEFAULT 'system', memory_type TEXT NOT NULL DEFAULT 'fact',
    t_valid_from REAL NOT NULL, t_valid_to REAL, t_ingest REAL NOT NULL, t_invalid REAL,
    status TEXT NOT NULL DEFAULT 'active', source_episode TEXT, salience REAL NOT NULL DEFAULT 0.5,
    embedding BLOB, access_count INTEGER NOT NULL DEFAULT 0, last_access REAL,
    correction_of TEXT, superseded_by TEXT);
"""


def make_v2_db(path: str) -> None:
    """最小的 v2 库 + 真实库里已知的三类脏数据。"""
    import sqlite3

    conn = sqlite3.connect(path, isolation_level=None)
    conn.executescript(_V2_SQL)
    conn.executemany(
        """INSERT INTO episode(id,session_id,role,text,t_event,t_ingest,source,tier,muted,archived,meta_json)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        [
            # 真库测试残留：tier='dream' 不在 PR.TIERS 内
            ("ep_dream_1", "main", "dream", "梦的内容", 1000.0, 1000.0, "dream", "dream", 0, 0,
             json.dumps({"evidence": ["x"]}, ensure_ascii=False)),
            # t_event=1e18 的坏时间戳
            ("ep_big", "tz2", "user", "坏时间戳", 1e18, 1e18, "dialogue", "retrieval", 0, 0, None),
            # 异常 session_id
            ("ep_path", "../../etc/passwd", "user", "异常会话", 2000.0, 2000.0, "dialogue", "retrieval", 0, 0, None),
            # 静音 + 归档
            ("ep_muted", "s", "user", "静音的", 3000.0, 3000.0, "dialogue", "retrieval", 1, 0, None),
            ("ep_arch", "s", "user", "归档的", 4000.0, 4000.0, "dialogue", "retrieval", 0, 1, None),
            # 主动写入（v3 前也允许存在的历史 tier=retrieval）
            ("ep_nb", "s", "self", "笔记", 5000.0, 5000.0, "notebook", "retrieval", 0, 0, None),
        ],
    )
    conn.execute(
        """INSERT INTO fact(id,session_id,subject_id,predicate,object_text,t_valid_from,t_ingest,source_episode)
           VALUES('fa_1','s','en_1','住在','北京',1,1,'ep_nb')"""
    )
    conn.execute("PRAGMA user_version=2")
    conn.close()


class TestMigration(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rosmontis_mig_")
        self.path = os.path.join(self.dir, "v2.db")
        make_v2_db(self.path)

    def tearDown(self):
        DB.close_thread_conns()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_v2_to_v3_backfill(self):
        DB.init_db(self.path)
        conn = DB.connect(self.path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], DB.SCHEMA_VERSION)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(episode)")}
        self.assertTrue({"origin", "state", "state_at", "gate_version"} <= cols)
        fcols = {r[1] for r in conn.execute("PRAGMA table_info(fact)")}
        self.assertTrue({"origin", "archived"} <= fcols)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM episode").fetchone()[0], 6)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM fact").fetchone()[0], 1)

        rows = {r["id"]: dict(r) for r in conn.execute("SELECT id, origin, state FROM episode")}
        self.assertEqual(rows["ep_dream_1"]["state"], "dream")
        self.assertEqual(rows["ep_dream_1"]["origin"], "dream")
        self.assertEqual(rows["ep_big"]["origin"], "ingest")
        self.assertEqual(rows["ep_big"]["state"], "note")
        self.assertEqual(rows["ep_path"]["origin"], "ingest")
        self.assertEqual(rows["ep_muted"]["state"], "muted")
        self.assertEqual(rows["ep_arch"]["state"], "archived")
        self.assertEqual(rows["ep_nb"]["origin"], "tool")

        fact = conn.execute("SELECT origin, archived FROM fact WHERE id='fa_1'").fetchone()
        self.assertEqual(fact["origin"], "tool", "来源 episode 是主动写入，fact.origin 应为 tool")
        self.assertEqual(fact["archived"], 0)

        # 每条 episode 恰好 1 条 backfill 日志
        n_ep = conn.execute("SELECT COUNT(*) FROM episode").fetchone()[0]
        n_log = conn.execute("SELECT COUNT(*) FROM item_state_log").fetchone()[0]
        self.assertEqual(n_ep, n_log)
        dup = conn.execute(
            "SELECT COUNT(*) FROM (SELECT item_id,COUNT(*) c FROM item_state_log GROUP BY item_id HAVING c>1)"
        ).fetchone()[0]
        self.assertEqual(dup, 0)

    def test_idempotent_second_run(self):
        DB.init_db(self.path)
        DB.close_thread_conns()
        conn = DB.connect(self.path)
        before = {
            "ep": conn.execute("SELECT COUNT(*) FROM episode").fetchone()[0],
            "log": conn.execute("SELECT COUNT(*) FROM item_state_log").fetchone()[0],
            "state_at": conn.execute("SELECT state_at FROM episode WHERE id='ep_nb'").fetchone()[0],
        }
        DB.close_thread_conns()
        DB.init_db(self.path)
        conn = DB.connect(self.path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], DB.SCHEMA_VERSION)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM episode").fetchone()[0], before["ep"])
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM item_state_log").fetchone()[0], before["log"])
        self.assertEqual(conn.execute("SELECT state_at FROM episode WHERE id='ep_nb'").fetchone()[0], before["state_at"])
        self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_failure_rolls_back_to_v2(self):
        original = DB._backfill_episode

        def boom(conn, now):
            raise RuntimeError("injected migration failure")

        DB._backfill_episode = boom
        try:
            with self.assertRaises(RuntimeError):
                DB.init_db(self.path)
        finally:
            DB._backfill_episode = original
            DB.close_thread_conns()

        import sqlite3

        conn = sqlite3.connect(self.path)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 2, "失败必须回滚保持 v2")
            cols = {r[1] for r in conn.execute("PRAGMA table_info(episode)")}
            self.assertEqual({"origin", "state", "state_at", "gate_version"} & cols, set())
            n = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='item_state_log'"
            ).fetchone()[0]
            self.assertEqual(n, 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM episode").fetchone()[0], 6)
        finally:
            conn.close()

    @unittest.skipUnless((ROOT / "rosmontis.db").exists(), "真库 rosmontis.db 不在，跳过副本实测")
    def test_real_db_copy_migrates(self):
        """真库只读备份 → 副本上跑迁移（真库一个字节都不写）。"""
        import sqlite3

        real = str(ROOT / "rosmontis.db")
        copy = os.path.join(self.dir, "rosmontis_copy.db")
        src = sqlite3.connect(f"file:{real.replace(os.sep, '/')}?mode=ro", uri=True)
        dst = sqlite3.connect(copy)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

        probe = sqlite3.connect(copy)
        probe.row_factory = sqlite3.Row
        counts_before = {
            t: probe.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("episode", "fact", "entity", "state_slot", "hyperedge", "persona")
        }
        uv_before = probe.execute("PRAGMA user_version").fetchone()[0]
        weirds = {
            r["id"]: dict(r)
            for r in probe.execute(
                "SELECT id, session_id, t_event FROM episode WHERE t_event > 1e17 OR session_id LIKE '%/%'"
            )
        }
        probe.close()
        self.assertLess(uv_before, 3, "真库应停在 v2 供迁移实测")

        DB.close_thread_conns()
        DB.init_db(copy)
        conn = DB.connect(copy)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], DB.SCHEMA_VERSION)
            for table, n in counts_before.items():
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], n, table)
            n_ep = conn.execute("SELECT COUNT(*) FROM episode").fetchone()[0]
            n_log = conn.execute("SELECT COUNT(*) FROM item_state_log").fetchone()[0]
            self.assertEqual(n_ep, n_log)
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")

            dream = conn.execute(
                "SELECT COUNT(*) c FROM episode WHERE source='dream' AND state='dream' AND origin='dream'"
            ).fetchone()["c"]
            self.assertGreaterEqual(dream, 1, "真库里的 tier='dream' 必须映射到 state='dream'")
            for eid, old in weirds.items():
                now = dict(conn.execute("SELECT id,session_id,t_event,origin,state FROM episode WHERE id=?", (eid,)).fetchone())
                self.assertEqual(now["session_id"], old["session_id"], "异常 session_id 必须原样保留")
                self.assertEqual(now["t_event"], old["t_event"], "1e18 必须原样保留")
                self.assertIn(now["origin"], IT.ORIGINS)
                self.assertIn(now["state"], IT.STATES)
        finally:
            DB.close_thread_conns()


# ── 7 15 项基线不回归 ─────────────────────────────────────────
class TestBaseline(unittest.TestCase):
    def test_test_all_still_green(self):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "test_all.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=300,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        self.assertIn("Ran 15 tests", out, out[-2000:])
        self.assertIn("OK", out, out[-2000:])
        self.assertNotIn("FAILED", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
