"""C1 forget 批量化验收：语义等价 + 日志语义 + N+1 消除 + 幂等。

对照方法：同一份 fixture 库复制成两份，**同一个 now** 分别跑
  - 新实现：`forget.run()`（批量 transition_many + 批量超边读取）
  - 旧实现：本文件内逐字保留的 `_legacy_forget()`（逐条 transition + 每条一次超边查询）
然后比对归档集合、folded_into、摘要行、state 投影与 item_state_log 指纹。

运行：`PYTHONIOENCODING=utf-8 python tests/test_forget_batch.py`
不依赖 Ollama / 网络；只用临时库，不碰 alaya.db。
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
from alaya.core import forget as FG  # noqa: E402
from alaya.core import ids as IDS  # noqa: E402
from alaya.core import item as IT  # noqa: E402
from alaya.core import memory_field as MF  # noqa: E402
from alaya.core import pipeline as TR  # noqa: E402

NOW = 1_792_368_000.0          # 与 bench/_profile 固定时钟一致
OLD = NOW - 86400 * 365 * 3    # 3 年前 → retention ≈ 0，必然 cold


# ── fixture ───────────────────────────────────────────────────
def _cfg(path: str) -> Config:
    return Config(
        db_path=path,
        embed_provider="hash",
        llm_hot="rule",
        llm_deep="rule",
        log_level="ERROR",
        auto_forget=False,
    )


def _submit(cfg: Config, item_id: str, text: str, *, t_event: float = OLD, salience: float = 0.3) -> None:
    sub = TR.submit(
        IT.MemoryItem(
            id=item_id,
            session_id="s",
            text=text,
            origin=IT.ORIGIN_INGEST,
            actor=IT.ACTOR_SYSTEM,
            memory_type="fact",
            salience=salience,
            emotion=0.0,
            t_event=t_event,
            t_ingest=t_event,
            role="user",
            source="dialogue",
            author="system",
        ),
        project=False,
        cfg=cfg,
    )
    assert sub.ok, sub


def _add_edge(path: str, edge_id: str, member_ids: list[str], label: str = "主题") -> None:
    with DB.tx(path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO hyperedge(id,session_id,kind,label,summary,salience,t_from,t_to,created_at,member_count)
               VALUES(?,?,'theme',?,NULL,0.5,?,NULL,?,0)""",
            (edge_id, "s", label, OLD, OLD),
        )
        for mid in member_ids:
            conn.execute(
                "INSERT OR IGNORE INTO hyperedge_member(edge_id,member_type,member_id,weight,added_at) VALUES(?,?,?,1.0,?)",
                (edge_id, "episode", mid, OLD),
            )
        conn.execute(
            "UPDATE hyperedge SET member_count=(SELECT COUNT(*) FROM hyperedge_member WHERE edge_id=?) WHERE id=?",
            (edge_id, edge_id),
        )


def build_fixture(path: str) -> None:
    """确定 id 的 fixture：3 个可折叠超边组 + 多边归属 + 120 条无边冷条 + 保护项。"""
    cfg = _cfg(path)
    DB.init_db(path)
    TR.bind(cfg, None, None)

    # 3 个超边组，每组 5 条（>= fold_min_members=4）
    for g in range(3):
        members = []
        for i in range(5):
            eid = f"ep_g{g}_{i}"
            _submit(cfg, eid, f"第{g}组关于城东的旧事 {i}")
            members.append(eid)
        _add_edge(path, f"he_g{g}", members, f"组{g}")

    # 多边归属：ep_multi 同时属于 he_g0(5) 与 he_extra(1)，按规则归给成员数最多的边
    _submit(cfg, "ep_multi", "同时在两个主题里的旧事")
    _add_edge(path, "he_extra", ["ep_multi"], "小边")
    with DB.tx(path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO hyperedge_member(edge_id,member_type,member_id,weight,added_at) VALUES('he_g0','episode','ep_multi',1.0,?)",
            (OLD,),
        )
    # 小组（3 < 4）：不折叠，只归档
    small = []
    for i in range(3):
        eid = f"ep_small_{i}"
        _submit(cfg, eid, f"小组旧事 {i}")
        small.append(eid)
    _add_edge(path, "he_small", small, "小组")

    # 120 条无边冷条目：专门放大旧实现的 N+1
    for i in range(120):
        _submit(cfg, f"ep_plain_{i:03d}", f"没有超边的旧事 {i}")

    # 保护项：resident（不应进扫描）、pinned、高 salience
    _submit(cfg, "ep_resident", "常驻的旧事", salience=0.3)
    TR.transition("ep_resident", "promote_resident", reason="fixture", actor=IT.ACTOR_SYSTEM, cfg=cfg)
    _submit(cfg, "ep_pinned", "钉住的旧事", salience=0.3)
    with DB.tx(path) as conn:
        conn.execute("UPDATE episode SET pinned=1 WHERE id='ep_pinned'")
    _submit(cfg, "ep_protect_sal", "高显著的旧事", salience=0.95)

    # 墓碑：删除真值不受 forget 影响
    _submit(cfg, "ep_tomb", "已删除的旧事")
    with DB.tx(path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tombstone(chunk_key,line_ids,session_id,deleted_at,reason,author) VALUES(?,?,?,?,?,?)",
            ("ck_tomb", '["ep_tomb"]', "s", NOW, "manual", "teacher"),
        )

    DB.close_thread_conns()


def clone_db(src: str, dst: str) -> None:
    DB.close_thread_conns()
    conn = sqlite3.connect(src)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    shutil.copy2(src, dst)


def qall(cfg: Config, sql: str, args: tuple = ()):
    with DB.reader(cfg.db_path) as conn:
        return conn.execute(sql, args).fetchall()


def q1(cfg: Config, sql: str, args: tuple = ()):
    with DB.reader(cfg.db_path) as conn:
        return conn.execute(sql, args).fetchone()


# ── 旧实现（C1 改造前逐字逻辑）：逐条 transition + 每条冷条目一次超边查询 ──
def _legacy_forget(cfg: Config, sid: str, now: float) -> FG.ForgetReport:
    rep = FG.ForgetReport()
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT id, session_id, text, t_event, access_count, salience, pinned,
                      tier, source, memory_type
               FROM episode WHERE (?='*' OR session_id=?)
                 AND archived=0 AND folded_into IS NULL AND tier!='resident' AND pinned=0
               ORDER BY t_event ASC LIMIT 5000""",
            (sid, sid),
        ).fetchall()
    rep.scanned = len(rows)
    cold: list[dict] = []
    for r in rows:
        if r["pinned"] or r["salience"] >= cfg.forget_protect_salience:
            rep.protected += 1
            continue
        space = MF.classify_space("episode", r["tier"], r["source"], r["memory_type"])
        rr = FG.retention(cfg, r["t_event"], r["access_count"], r["salience"], now, space=space)
        if rr < cfg.forget_archive_r:
            cold.append({"row": r, "r": rr})

    with DB.reader(cfg.db_path) as conn:
        edge_map: dict[str, list] = {}
        for item in cold:
            r = item["row"]
            hits = conn.execute(
                "SELECT edge_id FROM hyperedge_member WHERE member_type='episode' AND member_id=?",
                (r["id"],),
            ).fetchall()
            for h in hits:
                edge_map.setdefault(h["edge_id"], []).append(r)
        best_edge: dict[str, str] = {}
        for eid, members in edge_map.items():
            for r in members:
                cur = best_edge.get(r["id"])
                if cur is None or len(members) > len(edge_map[cur]):
                    best_edge[r["id"]] = eid
        deduped: dict[str, list] = {}
        for r_id, eid in best_edge.items():
            deduped.setdefault(eid, []).append(next(r for r in edge_map[eid] if r["id"] == r_id))
        edge_map = deduped

    for eid, members in edge_map.items():
        if len(members) < cfg.fold_min_members:
            continue
        text = FG._summarize(None, members)
        summary_id = "ep_fold_" + IDS._digest(eid, int(now))
        summary_item = IT.MemoryItem(
            id=summary_id, session_id=members[0]["session_id"], text=text,
            origin=IT.ORIGIN_SUMMARY, actor=IT.ACTOR_SYSTEM, memory_type="fact",
            salience=0.55, emotion=0.0, t_event=members[-1]["t_event"], t_ingest=now,
            role="summary", source="summary", meta={"is_summary": True},
        )
        sub = TR.submit(summary_item, project=False, cfg=cfg)
        if not sub.ok:
            rep.errors.append(f"fold submit failed for {eid}: {sub.reason}")
            continue
        summary_id = sub.item_id
        with DB.tx(cfg.db_path) as conn:
            for m in members:
                conn.execute("UPDATE episode SET folded_into=? WHERE id=?", (summary_id, m["id"]))
            conn.execute(
                "INSERT OR IGNORE INTO hyperedge_member(edge_id,member_type,member_id,weight,added_at) VALUES(?,?,?,1.2,?)",
                (eid, "episode", summary_id, now),
            )
        rep.folded += len(members)
        rep.folded_into.append(summary_id)

    if cold:
        cold_ids = [item["row"]["id"] for item in cold]
        marks = ",".join("?" * len(cold_ids))
        with DB.reader(cfg.db_path) as conn:
            cur_rows = {
                r["id"]: dict(r)
                for r in conn.execute(
                    f"SELECT id, folded_into, archived FROM episode WHERE id IN ({marks})", cold_ids
                )
            }
        for item in cold:
            r = item["row"]
            cur = cur_rows.get(r["id"])
            if cur is None or cur["archived"]:
                continue
            tr = TR.transition(r["id"], "archive", reason="cold after fold", actor=IT.ACTOR_SYSTEM, cfg=cfg)
            if tr.ok and tr.from_state != tr.to_state:
                rep.archived += 1
    return rep


# ── fingerprint ───────────────────────────────────────────────
def episode_fingerprint(cfg: Config):
    rows = qall(
        cfg,
        "SELECT id,state,tier,archived,muted,folded_into FROM episode ORDER BY id",
    )
    return [tuple(r) for r in rows]


def log_fingerprint(cfg: Config):
    rows = qall(
        cfg,
        """SELECT item_id,action,from_state,to_state,target,reason,gate_json,actor
           FROM item_state_log ORDER BY item_id, id""",
    )
    return [tuple(r) for r in rows]


def summary_rows(cfg: Config):
    rows = qall(
        cfg,
        """SELECT id,text,origin,state,tier,role,source,salience
           FROM episode WHERE origin='summary' ORDER BY id""",
    )
    return [tuple(r) for r in rows]


class ForgeBatchBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="alaya_forget_batch_")
        cls.template = os.path.join(cls.tmp, "template.db")
        build_fixture(cls.template)

    @classmethod
    def tearDownClass(cls):
        DB.close_thread_conns()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="case_", dir=self.tmp)
        self.new_db = os.path.join(self.dir, "new.db")
        self.old_db = os.path.join(self.dir, "old.db")
        clone_db(self.template, self.new_db)
        clone_db(self.template, self.old_db)
        self.cfg_new = _cfg(self.new_db)
        self.cfg_old = _cfg(self.old_db)

    def tearDown(self):
        DB.close_thread_conns()
        shutil.rmtree(self.dir, ignore_errors=True)


# ── 1 归档集合等价（新旧对照）──────────────────────────────────
class TestArchiveEquivalence(ForgeBatchBase):
    def test_archive_set_and_report_equal_to_legacy(self):
        rep_new = FG.run(self.cfg_new, None, "s", now=NOW)
        rep_old = _legacy_forget(self.cfg_old, "s", NOW)

        self.assertEqual(rep_new.scanned, rep_old.scanned)
        self.assertEqual(rep_new.archived, rep_old.archived)
        self.assertEqual(rep_new.folded, rep_old.folded)
        self.assertEqual(rep_new.protected, rep_old.protected)
        self.assertEqual(len(rep_new.folded_into), len(rep_old.folded_into))
        self.assertEqual(rep_new.errors, [])
        self.assertEqual(rep_old.errors, [])

        new_arch = {r["id"] for r in qall(self.cfg_new, "SELECT id FROM episode WHERE archived=1")}
        old_arch = {r["id"] for r in qall(self.cfg_old, "SELECT id FROM episode WHERE archived=1")}
        self.assertEqual(new_arch, old_arch, "归档集合必须逐条等价")
        self.assertGreater(len(new_arch), 120 + 15)

        self.assertEqual(episode_fingerprint(self.cfg_new), episode_fingerprint(self.cfg_old))
        self.assertEqual(summary_rows(self.cfg_new), summary_rows(self.cfg_old))
        frozen_new = {r["id"]: r["folded_into"] for r in qall(self.cfg_new, "SELECT id,folded_into FROM episode")}
        frozen_old = {r["id"]: r["folded_into"] for r in qall(self.cfg_old, "SELECT id,folded_into FROM episode")}
        self.assertEqual(frozen_new, frozen_old, "folded_into 映射必须一致")

    def test_state_log_equal_to_legacy(self):
        FG.run(self.cfg_new, None, "s", now=NOW)
        _legacy_forget(self.cfg_old, "s", NOW)
        self.assertEqual(log_fingerprint(self.cfg_new), log_fingerprint(self.cfg_old), "每条日志内容必须一致")

        # 每条被归档条目恰好 1 条 archive 日志；摘要恰好 1 条 create
        for r in qall(self.cfg_new, "SELECT id FROM episode WHERE archived=1"):
            logs = qall(
                self.cfg_new,
                "SELECT action FROM item_state_log WHERE item_id=? ORDER BY id", (r["id"],),
            )
            self.assertEqual([x["action"] for x in logs].count("archive"), 1, r["id"])
        for r in qall(self.cfg_new, "SELECT id FROM episode WHERE origin='summary'"):
            n = q1(self.cfg_new, "SELECT COUNT(*) c FROM item_state_log WHERE item_id=? AND action='create'", (r["id"],))["c"]
            self.assertEqual(n, 1)


# ── 2 保护语义 ────────────────────────────────────────────────
class TestProtection(ForgeBatchBase):
    def test_resident_pinned_high_salience_protected(self):
        rep = FG.run(self.cfg_new, None, "s", now=NOW)
        # 高 salience 进入扫描后被计 protected；resident 与 pinned 由扫描 SQL 直接排除
        self.assertGreaterEqual(rep.protected, 1)

        for eid, expected_state in (("ep_resident", "resident"), ("ep_pinned", "note"), ("ep_protect_sal", "note")):
            row = q1(self.cfg_new, "SELECT state,archived FROM episode WHERE id=?", (eid,))
            self.assertEqual(row["archived"], 0, f"{eid} 不该被归档")
            self.assertEqual(row["state"], expected_state)
            n_log = q1(self.cfg_new, "SELECT COUNT(*) c FROM item_state_log WHERE item_id=? AND action='archive'", (eid,))["c"]
            self.assertEqual(n_log, 0, f"{eid} 不该有 archive 日志")

    def test_tombstone_unaffected(self):
        FG.run(self.cfg_new, None, "s", now=NOW)
        row = q1(self.cfg_new, "SELECT chunk_key,line_ids,reason FROM tombstone WHERE chunk_key='ck_tomb'")
        self.assertIsNotNone(row)
        self.assertEqual(row["line_ids"], '["ep_tomb"]')
        self.assertEqual(row["reason"], "manual")
        # 墓碑是删除真值：forget 不改它、也不物理删除 episode
        self.assertIsNotNone(q1(self.cfg_new, "SELECT id FROM episode WHERE id='ep_tomb'"))


# ── 3 折叠语义 ────────────────────────────────────────────────
class TestFold(ForgeBatchBase):
    def test_fold_only_hyperedge_members_and_best_edge(self):
        rep = FG.run(self.cfg_new, None, "s", now=NOW)
        summaries = qall(self.cfg_new, "SELECT id FROM episode WHERE origin='summary'")
        self.assertEqual(len(summaries), 3, "只有 3 个 >=4 成员的超边组应折叠")
        self.assertEqual(len(rep.folded_into), 3)

        # 多边归属：ep_multi 应归给 he_g0（与 he_g1 同为 5 人，先遇到者），同一份摘要
        fold = {r["id"]: r["folded_into"] for r in qall(self.cfg_new, "SELECT id,folded_into FROM episode")}
        self.assertIsNotNone(fold.get("ep_multi"))
        self.assertEqual(fold["ep_multi"], fold["ep_g0_0"])
        # 小组（3 人）只归档、不折叠
        for i in range(3):
            self.assertIsNone(fold[f"ep_small_{i}"])
        # 无边冷条目只归档
        self.assertIsNone(fold["ep_plain_000"])

        # 摘要以 role/source=summary 落库，并挂回原超边（weight 1.2）
        for s in summaries:
            row = q1(
                self.cfg_new,
                "SELECT role,source,state,tier,salience FROM episode WHERE id=?", (s["id"],),
            )
            self.assertEqual((row["role"], row["source"], row["state"], row["tier"]), ("summary", "summary", "note", "retrieval"))
            self.assertAlmostEqual(row["salience"], 0.55, places=6)
            m = q1(
                self.cfg_new,
                "SELECT weight FROM hyperedge_member WHERE member_type='episode' AND member_id=?",
                (s["id"],),
            )
            self.assertIsNotNone(m)
            self.assertAlmostEqual(m["weight"], 1.2, places=6)


# ── 4 幂等 ────────────────────────────────────────────────────
class TestIdempotency(ForgeBatchBase):
    def test_second_run_matches_legacy_and_does_not_duplicate(self):
        rep1 = FG.run(self.cfg_new, None, "s", now=NOW)
        rep1_old = _legacy_forget(self.cfg_old, "s", NOW)
        self.assertEqual((rep1.scanned, rep1.archived, rep1.folded), (rep1_old.scanned, rep1_old.archived, rep1_old.folded))

        rep2 = FG.run(self.cfg_new, None, "s", now=NOW)
        rep2_old = _legacy_forget(self.cfg_old, "s", NOW)
        self.assertEqual(
            (rep2.scanned, rep2.archived, rep2.folded, rep2.protected),
            (rep2_old.scanned, rep2_old.archived, rep2_old.folded, rep2_old.protected),
            "第二次遗忘也必须与旧实现一致",
        )
        self.assertEqual(
            q1(self.cfg_new, "SELECT COUNT(*) c FROM episode WHERE origin='summary'")["c"],
            q1(self.cfg_old, "SELECT COUNT(*) c FROM episode WHERE origin='summary'")["c"],
        )
        # 不重复写 archive 日志：每条 item 的 archive 日志 <= 1
        bad = qall(
            self.cfg_new,
            """SELECT item_id, COUNT(*) c FROM item_state_log
               WHERE action='archive' GROUP BY item_id HAVING c > 1""",
        )
        self.assertEqual(bad, [])
        self.assertEqual(episode_fingerprint(self.cfg_new), episode_fingerprint(self.cfg_old))


# ── 5 SQL 条数：N+1 消除 ──────────────────────────────────────
_FTS_INTERNAL = ("fts_content", "fts_docsize", "fts_idx", "data_version")


def count_sql(cfg: Config, fn) -> int:
    conn = DB.connect(cfg.db_path)
    n = {"v": 0}

    def cb(stmt: str) -> None:
        s = stmt.lstrip()
        if s.startswith("--") or any(k in s for k in _FTS_INTERNAL):
            return
        n["v"] += 1

    conn.set_trace_callback(cb)
    try:
        fn()
    finally:
        conn.set_trace_callback(None)
    return n["v"]


class TestQueryCount(ForgeBatchBase):
    def test_no_n_plus_one(self):
        new_sql = count_sql(self.cfg_new, lambda: FG.run(self.cfg_new, None, "s", now=NOW))
        old_sql = count_sql(self.cfg_old, lambda: _legacy_forget(self.cfg_old, "s", NOW))
        self.assertLess(new_sql, 120, f"新实现 SQL 条数应远低于 120，实际 {new_sql}")
        self.assertGreater(old_sql, 250, f"旧实现（N+1）SQL 条数应明显更大，实际 {old_sql}")
        self.assertGreater(old_sql / max(1, new_sql), 4.0, f"提速比不足：new={new_sql} old={old_sql}")


# ── 6 批量入口与逐条 transition 等价 ──────────────────────────
class TestTransitionMany(unittest.TestCase):
    def test_batch_equals_per_item(self):
        d = tempfile.mkdtemp(prefix="alaya_tmany_")
        try:
            batch_db = os.path.join(d, "batch.db")
            single_db = os.path.join(d, "single.db")
            cfg = _cfg(batch_db)
            DB.init_db(batch_db)
            TR.bind(cfg, None, None)
            for i in range(6):
                _submit(cfg, f"ep_t{i}", f"批量转移 {i}")
            TR.transition("ep_t2", "archive", reason="pre", actor=IT.ACTOR_SYSTEM, cfg=cfg)
            TR.transition("ep_t3", "skip", reason="pre", actor=IT.ACTOR_SYSTEM, cfg=cfg)
            DB.close_thread_conns()
            shutil.copy2(batch_db, single_db)

            ids = ["ep_t0", "ep_t1", "ep_t2", "ep_t3", "ep_missing"]
            cfg_b = _cfg(batch_db)
            cfg_s = _cfg(single_db)
            batch = TR.transition_many(ids, "archive", reason="cmp", actor=IT.ACTOR_SYSTEM, cfg=cfg_b)
            single = [TR.transition(i, "archive", reason="cmp", actor=IT.ACTOR_SYSTEM, cfg=cfg_s) for i in ids]

            self.assertEqual(len(batch), len(single))
            for b, s in zip(batch, single):
                self.assertEqual((b.ok, b.item_id, b.from_state, b.to_state, b.reason),
                                 (s.ok, s.item_id, s.from_state, s.to_state, s.reason))
            self.assertEqual(episode_fingerprint(cfg_b), episode_fingerprint(cfg_s))
            self.assertEqual(log_fingerprint(cfg_b), log_fingerprint(cfg_s))
        finally:
            DB.close_thread_conns()
            shutil.rmtree(d, ignore_errors=True)


# ── 7 报告结构 ────────────────────────────────────────────────
class TestReportShape(ForgeBatchBase):
    def test_report_keys_and_decay_only(self):
        rep = FG.run(self.cfg_new, None, "s", now=NOW)
        d = rep.to_dict()
        self.assertEqual(
            set(d),
            {"scanned", "archived", "folded", "folded_into", "protected", "errors"},
        )
        self.assertIsInstance(d["folded_into"], list)

        # run_decay_only 也走批量：冷条目归档、resident/pinned 不动
        n = FG.run_decay_only(self.cfg_new, "s", NOW + 10)
        self.assertGreaterEqual(n, 0)
        for eid in ("ep_resident", "ep_pinned"):
            row = q1(self.cfg_new, "SELECT archived,state FROM episode WHERE id=?", (eid,))
            self.assertEqual(row["archived"], 0)
        row = q1(self.cfg_new, "SELECT state,archived FROM episode WHERE id='ep_plain_000'")
        self.assertEqual((row["state"], row["archived"]), ("archived", 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
