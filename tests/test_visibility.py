"""C2 可见性接线测试：recall 按状态过滤（inbox 默认不可检索、archived/muted 恒不可见）。

运行：PYTHONIOENCODING=utf-8 python tests/test_visibility.py
不依赖 Ollama / 网络；v2 旧库用例只建临时库，不碰真库 `rosmontis.db`。
"""

from __future__ import annotations

import json
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

from rosmontis import Config  # noqa: E402
from rosmontis.core import db as DB  # noqa: E402
from rosmontis.core import item as IT  # noqa: E402
from rosmontis.core import pipeline as TR  # noqa: E402
from rosmontis.core import recall as RC  # noqa: E402
from rosmontis.core import text as TX  # noqa: E402
from rosmontis.core.embed import Embedder  # noqa: E402
from rosmontis.core.memory import Memory  # noqa: E402


class FakeLLM:
    """受控抽取器：把预设结果原样返回。"""

    provider = "fake"
    model = "fake"

    def __init__(self):
        self.script: list[dict] = []

    def queue(self, payload):
        self.script.append(payload)

    def chat_json(self, system, user, max_tokens=700):
        return self.script.pop(0) if self.script else {"entities": [], "facts": [], "state_changes": []}

    def chat(self, system, user, max_tokens=700, json_mode=False):
        return ""

    def available(self):
        return True


def mk(tag: str = "vis") -> tuple[Memory, FakeLLM, str]:
    d = tempfile.mkdtemp(prefix=f"rosmontis_vis_{tag}_")
    cfg = Config(
        db_path=os.path.join(d, "t.db"),
        embed_provider="hash",
        llm_hot="rule",
        llm_deep="rule",
        log_level="ERROR",
        auto_forget=False,
    )
    m = Memory(cfg)
    fake = FakeLLM()
    m.llm_hot = fake
    m.llm_deep = fake
    TR.bind(cfg, m.embedder, fake)
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


def submit_tool_item(m: Memory, fake: FakeLLM, text: str, session_id: str = "s") -> str:
    """走管线提交一条只有 what 的主动条目 → gate 判 inbox。"""
    now = time.time()
    item = IT.MemoryItem(
        session_id=session_id,
        text=text,
        origin=IT.ORIGIN_TOOL,
        actor=IT.ACTOR_MODEL,
        memory_type="fact",
        salience=0.8,
        emotion=0.0,
        t_event=now,
        t_ingest=now,
    )
    r = TR.submit(item, project=False, cfg=m.cfg, embedder=m.embedder, llm=fake)
    assert r.ok, r.reason
    return r.item_id


def item_ids(result: dict) -> list[str]:
    return [i["id"] for i in result["items"]]


class TestInboxVisibility(unittest.TestCase):
    """C2=B：inbox 速记默认不参与召回，开关打开后才可见。"""

    def test_inbox_hidden_by_default(self):
        m, fake, d = mk("inbox_off")
        try:
            iid = submit_tool_item(m, fake, "速记：周五交周报，主题是海王星迁移")
            self.assertEqual(q1(m, "SELECT state FROM episode WHERE id=?", (iid,))["state"], "inbox")
            r = m.recall("海王星迁移", session_id="s")
            self.assertNotIn(iid, item_ids(r), "inbox 默认不得出现在召回里")
            self.assertTrue(r["abstained"], "只有一条 inbox 时应拒答")
            self.assertFalse(r["plan"]["inbox_retrievable"])
            self.assertGreaterEqual(r["plan"]["filtered_by_state"], 1, "应记录被状态过滤的候选数")
            with DB.reader(m.cfg.db_path) as conn:
                row = conn.execute("SELECT plan_json FROM retrieval_log ORDER BY id DESC LIMIT 1").fetchone()
            logged = json.loads(row["plan_json"])
            self.assertIn("filtered_by_state", logged, "plan_json 必须留审计计数")
            self.assertIn("inbox_retrievable", logged)
        finally:
            teardown(m, d)

    def test_inbox_visible_when_switch_on(self):
        m, fake, d = mk("inbox_on")
        try:
            iid = submit_tool_item(m, fake, "速记：周五交周报，主题是海王星迁移")
            m.cfg.inbox_retrievable = True
            r = m.recall("海王星迁移", session_id="s")
            self.assertIn(iid, item_ids(r), "开关打开后 inbox 应可被检索")
            self.assertTrue(r["plan"]["inbox_retrievable"])
        finally:
            teardown(m, d)

    def test_switch_default_is_false(self):
        cfg = Config(log_level="ERROR")
        self.assertFalse(cfg.inbox_retrievable)

    def test_note_still_visible(self):
        m, fake, d = mk("note")
        try:
            r0 = m.ingest("我喜欢天狼星的颜色", session_id="s", t_event=time.time())
            eid = r0["episode_id"]
            self.assertEqual(q1(m, "SELECT state FROM episode WHERE id=?", (eid,))["state"], "note")
            r = m.recall("天狼星", session_id="s")
            self.assertIn(eid, item_ids(r), "note 是默认可见层，不能被 C2 误伤")
        finally:
            teardown(m, d)


class TestArchivedMuted(unittest.TestCase):
    """archived / muted 恒不可见，包括开关打开时。"""

    def _make_note(self, m: Memory, text: str) -> str:
        return m.ingest(text, session_id="s", t_event=time.time())["episode_id"]

    def test_archived_hidden(self):
        m, fake, d = mk("archived")
        try:
            eid = self._make_note(m, "我的旧地址在猎户座旋臂")
            self.assertIn(eid, item_ids(m.recall("猎户座旋臂", session_id="s")))
            res = TR.transition(eid, "archive", cfg=m.cfg)
            self.assertTrue(res.ok, res.reason)
            r = m.recall("猎户座旋臂", session_id="s")
            self.assertNotIn(eid, item_ids(r), "archived 恒不可见")
            self.assertGreaterEqual(r["plan"]["filtered_by_state"], 1)
        finally:
            teardown(m, d)

    def test_muted_hidden(self):
        m, fake, d = mk("muted")
        try:
            eid = self._make_note(m, "我说过要去看仙女座流星雨")
            self.assertIn(eid, item_ids(m.recall("仙女座流星雨", session_id="s")))
            res = TR.transition(eid, "skip", reason="测试静音", cfg=m.cfg)
            self.assertTrue(res.ok, res.reason)
            r = m.recall("仙女座流星雨", session_id="s")
            self.assertNotIn(eid, item_ids(r), "muted 恒不可见")
            self.assertGreaterEqual(r["plan"]["filtered_by_state"], 1)
        finally:
            teardown(m, d)

    def test_archived_still_hidden_with_inbox_switch_on(self):
        m, fake, d = mk("arch_switch")
        try:
            eid = self._make_note(m, "旧稿写在船底座星云")
            TR.transition(eid, "archive", cfg=m.cfg)
            m.cfg.inbox_retrievable = True
            r = m.recall("船底座星云", session_id="s")
            self.assertNotIn(eid, item_ids(r), "开关只放开 inbox，不放 archived")
        finally:
            teardown(m, d)


# ── v2 旧库（无 episode.state 列）兜底 ──────────────────────
_V2_DDL = """
CREATE TABLE episode (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, seq INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL, text TEXT NOT NULL, t_event REAL NOT NULL, t_ingest REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'dialogue', tier TEXT NOT NULL DEFAULT 'retrieval',
    memory_type TEXT NOT NULL DEFAULT 'fact', author TEXT NOT NULL DEFAULT 'system',
    confirmed INTEGER NOT NULL DEFAULT 0, salience REAL NOT NULL DEFAULT 0.5,
    emotion REAL NOT NULL DEFAULT 0.0, access_count INTEGER NOT NULL DEFAULT 0,
    last_access REAL, embedding BLOB, folded_into TEXT, archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0, muted INTEGER NOT NULL DEFAULT 0,
    cooldown_until REAL, frozen_from TEXT, correction_of TEXT, superseded_by TEXT,
    chunk_key TEXT, meta_json TEXT
);
CREATE VIRTUAL TABLE episode_fts USING fts5(
    episode_id UNINDEXED, tokens, tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE entity (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, name TEXT NOT NULL, norm_name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'concept', aliases TEXT, attrs TEXT,
    salience REAL NOT NULL DEFAULT 0.5, first_seen REAL NOT NULL, last_seen REAL NOT NULL,
    mentions INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE fact (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL, object_text TEXT NOT NULL, object_entity_id TEXT,
    polarity INTEGER NOT NULL DEFAULT 1, confidence REAL NOT NULL DEFAULT 0.7,
    t_valid_from REAL NOT NULL, t_valid_to REAL, t_ingest REAL NOT NULL, t_invalid REAL,
    status TEXT NOT NULL DEFAULT 'active', source_episode TEXT, salience REAL NOT NULL DEFAULT 0.5,
    embedding BLOB, access_count INTEGER NOT NULL DEFAULT 0, last_access REAL,
    correction_of TEXT, superseded_by TEXT
);
CREATE VIRTUAL TABLE fact_fts USING fts5(
    fact_id UNINDEXED, tokens, tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE state_slot (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, scope TEXT NOT NULL, slot_key TEXT NOT NULL,
    value_text TEXT NOT NULL, since REAL NOT NULL, until REAL, source_fact TEXT,
    confidence REAL NOT NULL DEFAULT 0.7
);
CREATE TABLE hyperedge (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'theme',
    label TEXT NOT NULL, summary TEXT, salience REAL NOT NULL DEFAULT 0.5,
    t_from REAL NOT NULL, t_to REAL, created_at REAL NOT NULL,
    member_count INTEGER NOT NULL DEFAULT 0, embedding BLOB
);
CREATE TABLE hyperedge_member (
    edge_id TEXT NOT NULL, member_type TEXT NOT NULL, member_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0, added_at REAL NOT NULL,
    PRIMARY KEY (edge_id, member_type, member_id)
);
CREATE TABLE retrieval_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, session_id TEXT, query TEXT NOT NULL,
    plan_json TEXT, selected_json TEXT, excluded_json TEXT, latency_ms REAL,
    abstained INTEGER NOT NULL DEFAULT 0, budget_used INTEGER, budget_total INTEGER
);
PRAGMA user_version=2;
"""


def _insert_episode(conn, eid: str, text: str, *, tier="retrieval", archived=0, muted=0) -> None:
    now = time.time()
    conn.execute(
        """INSERT INTO episode(id,session_id,seq,role,text,t_event,t_ingest,source,tier,
           memory_type,author,salience,emotion,access_count,archived,muted)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (eid, "v2s", 1, "user", text, now - 3600, now - 3600, "dialogue", tier,
         "fact", "teacher", 0.7, 0.0, 0, archived, muted),
    )
    conn.execute("INSERT INTO episode_fts(episode_id,tokens) VALUES(?,?)", (eid, " ".join(TX.tokenize(text))))


class TestLegacyV2Fallback(unittest.TestCase):
    """v2 旧库没有 state 列：用 tier/archived/muted 兼容投影，不得抛错。"""

    def test_v2_recall_does_not_crash(self):
        d = tempfile.mkdtemp(prefix="rosmontis_vis_v2_")
        db_path = os.path.join(d, "v2.db")
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(_V2_DDL)
            _insert_episode(conn, "ep_v2_live", "旧库里的火星基地还开着")
            _insert_episode(conn, "ep_v2_arch", "旧库里的火星基地旧稿", archived=1)
            _insert_episode(conn, "ep_v2_muted", "旧库里的火星基地禁语", muted=1)
            _insert_episode(conn, "ep_v2_inbox", "旧库里的火星基地速记", tier="inbox")
            conn.commit()
        finally:
            conn.close()

        cfg = Config(db_path=db_path, embed_provider="hash", log_level="ERROR", auto_forget=False)
        emb = Embedder("hash", "hash", cfg.embed_dim, "")
        try:
            r = RC.recall(cfg, emb, "火星基地", session_id="v2s")
        finally:
            DB.close_thread_conns()
            shutil.rmtree(d, ignore_errors=True)
        self.assertFalse(r.abstained, "旧库可见条目应能召回")
        ids = [i.ref_id for i in r.items]
        self.assertIn("ep_v2_live", ids)
        self.assertNotIn("ep_v2_arch", ids, "旧库 archived=1 必须靠兼容投影隐藏")
        self.assertNotIn("ep_v2_muted", ids, "旧库 muted=1 必须靠兼容投影隐藏")
        self.assertNotIn("ep_v2_inbox", ids, "旧库 tier=inbox 必须靠兼容投影隐藏")
        self.assertGreaterEqual(r.plan["filtered_by_state"], 3, "三个隐藏条目都应被计数")
        self.assertFalse(r.plan["inbox_retrievable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
