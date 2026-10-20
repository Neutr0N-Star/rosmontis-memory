"""v1.1.1 发布收口门禁：把审查定稿的每条问题钉成"以后不会再犯"的断言。

对应规格：`.agent-runs/2026-10-20/task-002/implementation-spec.md`。
每条用例的 docstring 里写明它锁的是哪条验收断言，改实现时先读这一行。

运行：
    PYTHONIOENCODING=utf-8 python -m unittest tests.test_release_gate -v

离线：`ROSMONTIS_EMBED=hash` + 假 LLM，不依赖 Ollama / 网络。
本文件只读 `rosmontis/`，不改任何业务代码——红了说明实现有问题，不是测试写错了。
"""

from __future__ import annotations

import ast
import math
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosmontis  # noqa: E402
from rosmontis import api as API  # noqa: E402
from rosmontis import Config  # noqa: E402
from rosmontis.core import db as DB  # noqa: E402
from rosmontis.core import pipeline as TR  # noqa: E402
from rosmontis.core import recall as RC  # noqa: E402
from rosmontis.core import write as WR  # noqa: E402
from rosmontis.core.memory import Memory  # noqa: E402

CORE_DIR = ROOT / "rosmontis" / "core"

# `(?='*' OR session_id=?)` 这种"用哨兵值绕开动态 SQL"的写法是有意的历史折中，
# 但它让索引选择性消失、也让 as_of 类查询难审计。本轮只锁住**不新增**（棘轮）：
# 规格没把清理排进这一批，直接断言 0 会让发布门禁永久红，而门禁永久红等于没有门禁。
# 基线是用下面的 AST 扫描器（只数字符串常量、剔除注释与文档字符串）量出来的，
# 不是用 grep 量的——两套口径对不上会让棘轮自己先失真。
SID_STAR_BASELINE = {
    "field.py": 5,
    "forget.py": 1,
    "memory.py": 5,
    "recall.py": 1,
}
# audit.py 本轮不修，走白名单
SID_STAR_WHITELIST = {"audit.py"}


# ── 假件 ──────────────────────────────────────────────────────
class FakeLLM:
    """受控抽取器：把预设结果原样返回，让测试与模型解耦。"""

    provider = "fake"
    model = "fake"

    def __init__(self):
        self.script: list[dict] = []
        self.calls = 0
        self.failures = 0
        self.last_error = None

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


class StubEmbedder:
    """语义模式的假向量器：只为验证"向量身份"判据，不做真实语义。

    为什么不能用 `ROSMONTIS_EMBED=hash`：hash 是降级通道，写路径刻意不落伪向量
    （embedding 留 NULL）。而 A-5 要复现的正是"旧身份**有**向量、换模型后不重算"，
    所以需要一个 degraded=False、vector_mode()=='semantic' 的替身。
    """

    provider = "stub"

    def __init__(self, model_id: str, dim: int = 8):
        self.model = model_id
        self.model_id = model_id
        self.dim = dim
        self.degraded = False
        self.degraded_reason = None

    def vector_mode(self):
        return "semantic"

    def identity(self):
        return {"model": self.model_id, "model_name": self.model_id, "dim": self.dim}

    def health(self, probe=False):
        return {"ok": True, "provider": self.provider, "degraded": False}

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for ch in text or "":
            vec[ord(ch) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_blob(self, text: str) -> bytes:
        return struct.pack(f"<{self.dim}f", *self.embed(text))


# ── 夹具 ──────────────────────────────────────────────────────
def state_payload(subject: str, predicate: str, obj: str, slot: str | None = None) -> dict:
    """一条"我住 X"式的抽取结果。

    `slot` 给了就同时模拟 LLM 产出的 state_changes——真实提示词要求它两个字段都填，
    只给 facts 只能走到写入管线的确定性兜底路径，覆盖不到消息驱动的那条。
    """
    out = {
        "entities": [{"name": subject, "type": "person"}, {"name": obj, "type": "place"}],
        "facts": [
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "polarity": 1,
                "confidence": 0.9,
            }
        ],
        "state_changes": [],
        "theme": None,
    }
    if slot:
        out["state_changes"].append({"scope": subject, "slot": slot, "value": obj})
    return out


class GateCase(unittest.TestCase):
    """每个用例一个临时库；收尾按 LIFO 等 enrich、关连接、删目录。"""

    def mk(self, tag: str, **cfg) -> tuple[Memory, FakeLLM, str]:
        d = tempfile.mkdtemp(prefix=f"rosmontis_gate_{tag}_")
        conf = {
            "db_path": os.path.join(d, "t.db"),
            "embed_provider": "hash",
            "llm_hot": "rule",
            "llm_deep": "rule",
            "log_level": "ERROR",
            "auto_forget": False,
        }
        conf.update(cfg)
        cfg_obj = Config(**conf)
        m = Memory(cfg_obj)
        fake = FakeLLM()
        m.llm_hot = fake
        m.llm_deep = fake
        TR.bind(cfg_obj, m.embedder, fake)
        self.addCleanup(shutil.rmtree, d, True)
        self.addCleanup(DB.close_thread_conns)
        self.addCleanup(self._quiet_wait, m)
        return m, fake, d

    @staticmethod
    def _quiet_wait(m: Memory) -> None:
        try:
            m.wait_enrich(timeout=30)
        except Exception:
            pass

    @staticmethod
    def qall(m: Memory, sql: str, args: tuple = ()):
        with DB.reader(m.cfg.db_path) as conn:
            return conn.execute(sql, args).fetchall()

    def ingest_at(self, m: Memory, fake: FakeLLM, text: str, payload: dict,
                  t_event: float, session_id: str = "s") -> str:
        """写入一条并固定 t_ingest=t_event。

        `now=` 必须一起钉住：as_of 检索靠 `t_ingest <= as_of` 过滤，
        t_ingest 留真实时钟的话"回到过去"永远查不到任何东西。
        """
        fake.queue(payload)
        r = WR.ingest(
            m.cfg, m.embedder, fake, text,
            session_id=session_id, t_event=t_event, now=t_event,
        )
        self.assertFalse(r.duplicate, f"夹具写入被当成重复：{text}")
        return r.episode_id


# ── A-2 订正与状态联动 ────────────────────────────────────────
class TestStateSlotLinkage(GateCase):
    """锁 write.py 的 `state_slot.source_fact` 必须真的写进去（否则后面两条都是瞎的）。"""

    def test_city_slot_has_single_open_value_after_move(self):
        """说"我住北京"→说"我搬到上海了"：city 槽只有一条 until IS NULL，且值是上海。"""
        m, fake, _ = self.mk("a2move")
        self.ingest_at(m, fake, "我住北京", state_payload("我", "住在", "北京", "city"), 100.0)
        self.ingest_at(m, fake, "我搬到上海了", state_payload("我", "住在", "上海", "city"), 200.0)

        open_rows = self.qall(
            m,
            "SELECT value_text FROM state_slot WHERE session_id='s' AND slot_key='city' AND until IS NULL",
        )
        self.assertEqual(len(open_rows), 1, "同一 (scope, slot_key) 只许有一条未关闭的槽")
        self.assertEqual(open_rows[0]["value_text"], "上海")

    def test_correct_closes_state_slot_of_source_fact(self):
        """对"我住北京"那条 episode 执行 correct() → 对应 state_slot.until 必须非空。"""
        m, fake, _ = self.mk("a2corr")
        ep_bj = self.ingest_at(
            m, fake, "我住北京", state_payload("我", "住在", "北京"), 100.0
        )
        # 只给 facts 不给 state_changes：槽必须由写入管线的确定性兜底建立，
        # 这样 source_fact 才指向「这条 episode 派生的事实」，正是订正要清理的那条。
        slots = self.qall(
            m,
            "SELECT id, value_text, until FROM state_slot WHERE slot_key='city'",
        )
        self.assertEqual([s["value_text"] for s in slots], ["北京"], "夹具没造出北京槽")

        r = m.correct(ep_bj, "我其实一直住在北京")
        self.assertTrue(r.get("ok"), r)

        after = self.qall(m, "SELECT id, value_text, until FROM state_slot WHERE slot_key='city'")
        self.assertEqual(len(after), 1, "订正不该凭空多出一条槽")
        self.assertIsNotNone(
            after[0]["until"],
            "订正已把源事实作废，但状态槽还开着——source_fact 没写进去，"
            "memory.correct() 的 UPDATE 命中空集",
        )

    def test_audit_reports_stale_state_when_source_fact_dead(self):
        """造一条"源事实已失效但槽还开着"的脏数据 → audit 的 stale_state 计数必须 > 0。"""
        m, fake, _ = self.mk("a2audit")
        self.ingest_at(m, fake, "我住北京", state_payload("我", "住在", "北京"), 100.0)

        dead = self.qall(
            m,
            "SELECT id FROM fact WHERE predicate='住在' AND object_text='北京'",
        )
        self.assertEqual(len(dead), 1, "夹具没造出北京事实")
        # 模拟其它路径（skip / forget / 手工订正）把源事实作废，而槽没来得及联动。
        with DB.tx(m.cfg.db_path) as conn:
            conn.execute(
                "UPDATE fact SET status='invalidated', t_invalid=?, t_valid_to=? WHERE id=?",
                (150.0, 150.0, dead[0]["id"]),
            )

        report = m.audit()
        stale = [i for i in report["issues"] if i["type"] == "stale_state"]
        self.assertEqual(len(stale), 1, f"安全网没看见脏数据：{report['issues']}")
        self.assertIn("1 个", stale[0]["detail"])


# ── A-3 乱序/补录写入不许制造负区间 ────────────────────────────
class TestOutOfOrderWrite(GateCase):
    """锁 write.py `_apply_state` 的补录分支：历史时刻只能插历史槽，不能抢"当前真理"。"""

    def _seed(self, tag: str):
        m, fake, _ = self.mk(tag)
        self.ingest_at(m, fake, "我住北京", state_payload("我", "住在", "北京", "city"), 100.0)
        self.ingest_at(m, fake, "我搬到上海了", state_payload("我", "住在", "上海", "city"), 200.0)
        # 补录一条更早的历史状态：t=150 在广州
        self.ingest_at(m, fake, "我那时在广州", state_payload("我", "住在", "广州", "city"), 150.0)
        return m

    def test_backfill_never_creates_negative_interval(self):
        """t=100 北京、t=200 上海、补录 t=150 广州 → 所有槽 until >= since。"""
        m = self._seed("a3neg")
        rows = self.qall(m, "SELECT value_text, since, until FROM state_slot WHERE slot_key='city'")
        self.assertEqual(len(rows), 3, f"三条状态应各留一条槽：{rows}")
        for r in rows:
            if r["until"] is None:
                continue
            self.assertGreaterEqual(
                r["until"], r["since"],
                f"「{r['value_text']}」产生负区间：since={r['since']} until={r['until']}"
                "（补录事件时间早于当前槽 since 时，不能拿它去关当前槽）",
            )

    def test_current_state_still_latest_after_backfill(self):
        """补录 t=150 之后：until IS NULL 的仍是上海，当前状态查询也返回上海。"""
        m = self._seed("a3open")
        open_rows = self.qall(
            m, "SELECT value_text FROM state_slot WHERE slot_key='city' AND until IS NULL"
        )
        self.assertEqual([r["value_text"] for r in open_rows], ["上海"],
                         "补录的历史状态抢走了「当前」")

        snap = m.state_snapshot("我", session_id="s")
        self.assertEqual([s["value"] for s in snap["slots"]], ["上海"])


# ── A-4 as_of 时态 ────────────────────────────────────────────
class TestAsOfTemporal(GateCase):
    """锁 as_of 在「过滤」与「出口渲染」两处都成立。"""

    def _seed(self, tag: str):
        m, fake, _ = self.mk(tag)
        self.ingest_at(m, fake, "我住北京", state_payload("我", "住在", "北京", "city"), 1000.0)
        self.ingest_at(m, fake, "我搬到上海了", state_payload("我", "住在", "上海", "city"), 2000.0)
        return m

    def test_as_of_past_returns_old_value(self):
        """as_of 指到搬家前 → 检索返回北京那条，且不提上海。"""
        m = self._seed("a4past")
        ctx = m.build_context("我住在哪", "s", budget=400, as_of=1500.0)
        text = ctx["parts"]["memory"]["text"]
        self.assertIn("北京", text, f"as_of=1500 应看得见北京：{text!r}")
        self.assertNotIn("上海", text, f"as_of=1500 不该看见搬家的结果：{text!r}")

    def test_as_of_past_block_title_not_current_state(self):
        """as_of 落在过去时，注入块标题不许再写「当前状态」。"""
        m = self._seed("a4title")
        ctx = m.build_context("我住在哪", "s", budget=400, as_of=1500.0)
        text = ctx["parts"]["memory"]["text"]
        self.assertNotIn(
            "当前状态", text,
            "as_of 指向过去，标题还标「当前状态」——模型会把历史值当真理",
        )

    def test_as_of_is_reproducible(self):
        """同一 as_of 在两个真实时刻各查一次 → 结果一致（冷却不能跟 as_of 挂钩）。"""
        m = self._seed("a4repeat")
        first = m.build_context("我住在哪", "s", budget=400, as_of=1500.0)
        second = m.build_context("我住在哪", "s", budget=400, as_of=1500.0)
        self.assertEqual(
            first["parts"]["memory"]["text"], second["parts"]["memory"]["text"],
            "同一 as_of 两次查询结果不同：冷却判定多半用了 as_of 而不是真实时钟",
        )

    def test_now_still_returns_new_value(self):
        """现在时刻检索仍返回上海（不能把当前查询改坏）。"""
        m = self._seed("a4now")
        ctx = m.build_context("我住在哪", "s", budget=400)
        text = ctx["parts"]["memory"]["text"]
        self.assertIn("上海", text, f"当前时刻查不到新住址：{text!r}")
        self.assertIn("当前状态", text, "当前时刻的标题仍应是「当前状态」")


# ── A-5 换模型后可恢复 ────────────────────────────────────────
class TestIdentityRecovery(GateCase):
    """锁 backfill 判据从「embedding IS NULL」扩到「身份不符」，且跳过要计数。"""

    N = 20

    def _two_identities(self, tag: str):
        """身份 A 写 20 条 → 换身份 B 再写 20 条，返回 (身份 B 的 Memory, 库路径)。"""
        m1, fake, d = self.mk(tag)
        m1.embedder = StubEmbedder("model-A")
        for i in range(self.N):
            m1.ingest(f"第一身份的记忆条目{i}", session_id="s", t_event=1000.0 + i)
        DB.close_thread_conns()

        m2 = Memory(m1.cfg)
        m2.llm_hot = fake
        m2.llm_deep = fake
        TR.bind(m1.cfg, m2.embedder, fake)
        m2.embedder = StubEmbedder("model-B")
        for i in range(self.N):
            m2.ingest(f"第二身份的记忆条目{i}", session_id="s", t_event=2000.0 + i)
        return m2, m1.cfg.db_path

    def test_identity_switch_counts_stale_vectors(self):
        """写 20 条（身份 A）→ 换身份写 20 条 → pending_backfill > 0 且 stale_identity == 20。"""
        m, _ = self._two_identities("a5count")
        total = pending_total(m)
        self.assertGreater(
            total, 0,
            "换模型后旧身份向量仍非 NULL，pending_backfill 报 0 —— /readyz 会全绿而库半失明",
        )
        detail = pending_detail(m, m.pending_backfill())
        self.assertIn(
            "stale_identity", detail,
            f"没人报出「身份不符」的条数，只看到了 {sorted(detail)}",
        )
        self.assertEqual(detail["stale_identity"], self.N)
        self.assertEqual(int(detail.get("null_vectors", 0)), 0, "这批不是从未算过，是身份旧了")

    def test_backfill_rewrites_old_identity(self):
        """跑 backfill → 旧 20 条被重算成新身份，pending 归零。"""
        m, db_path = self._two_identities("a5fill")
        before = self.qall(
            m, "SELECT COUNT(*) c FROM episode WHERE embed_model='model-A'"
        )[0]["c"]
        self.assertEqual(before, self.N, "夹具里身份 A 的条目数不对")

        out = m.backfill_embeddings(limit=500)
        self.assertFalse(out.get("skipped"), out)
        self.assertEqual(int(out.get("filled", 0)), self.N, out)
        self.assertEqual(pending_total(m), 0, "回算后仍有待补条目")

        rows = self.qall(m, "SELECT DISTINCT embed_model FROM episode")
        self.assertEqual([r["embed_model"] for r in rows], ["model-B"])

    def test_dim_mismatch_counted_as_identity_skipped(self):
        """维度不符被跳过的条数必须进 identity_skipped（不能先 continue 掉计数）。"""
        m, db_path = self._two_identities("a5skip")
        # 造 20 条维度不符的向量：len(blob) != embed_dim*4，命中 _dense 的第一道 continue
        blob = struct.pack("<4f", 0.5, 0.5, 0.5, 0.5)
        with DB.tx(db_path) as conn:
            conn.execute(
                "UPDATE episode SET embedding=?, embed_dim=4 WHERE embed_model='model-A'", (blob,)
            )

        res = RC.recall(m.cfg, m.embedder, "记忆条目", "s", budget=400)
        self.assertEqual(
            int(res.plan.get("identity_skipped") or 0), self.N,
            "维度不符的条目被静默剔除了：审计看不到，就永远不知道该回算多少条",
        )


# ── A-7 未知 provider 启动即失败 + 降级信号外露 ────────────────
class TestProviderGate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rosmontis_gate_a7_")
        self.db = os.path.join(self.dir, "t.db")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(DB.close_thread_conns)

    def test_unknown_provider_fails_fast(self):
        """ROSMONTIS_EMBED=local → rosmontis.open() 抛错，code 为 embed_provider_unavailable。"""
        with mock.patch.dict(os.environ, {"ROSMONTIS_EMBED": "local"}):
            with self.assertRaises(API.RosmontisError) as cm:
                rosmontis.open(self.db, {"llm_hot": "rule", "llm_deep": "rule", "log_level": "ERROR"})
        self.assertEqual(
            cm.exception.code, "embed_provider_unavailable",
            f"未知 provider 被静默换成 hash，而不是开不起来的显式失败：{cm.exception.code}",
        )
        self.assertIn("local", cm.exception.message, "报错要带原值，否则用户不知道是哪来的配置")

    def test_hash_provider_reports_degraded_in_context(self):
        """ROSMONTIS_EMBED=hash → 正常打开，context() 的 degraded=True 且 degraded_reason 非空。"""
        with mock.patch.dict(os.environ, {"ROSMONTIS_EMBED": "hash"}):
            h = rosmontis.open(self.db, {"llm_hot": "rule", "llm_deep": "rule", "log_level": "ERROR"})
            self.addCleanup(h.close)
        ctx = h.context("u1", "c1", "随便问一句", budget=200)

        self.assertIn("degraded", ctx, f"context() 没外露降级字段：{sorted(ctx)}")
        self.assertTrue(ctx["degraded"], "hash 是降级通道，必须报 degraded=True")
        self.assertTrue(
            (ctx.get("degraded_reason") or "").strip(),
            "degraded=True 却不给理由，接入方只能猜",
        )
        self.assertEqual(ctx.get("vector_mode"), "degraded")


# ── A-1 备份不许静默产出空文件/坏文件 ────────────────────────
class TestBackupIntegrity(GateCase):
    """锁备份：并发读者在场时也必须是可打开的、行数与主库一致的库。"""

    def _run_cli(self, *args) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [sys.executable, "-m", "rosmontis", "backup", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=180,
        )

    def test_backup_survives_concurrent_reader(self):
        """持读事务的连接（断点落后于写入）→ 备份文件能 count episode 且与主库一致。"""
        m, fake, d = self.mk("a1")
        for i in range(2):
            self.ingest_at(m, fake, f"旧条目{i}", {"entities": [], "facts": [], "state_changes": []}, 1000.0 + i)

        reader = sqlite3.connect(m.cfg.db_path, timeout=5.0, isolation_level=None)
        self.addCleanup(reader.close)
        reader.execute("BEGIN")
        self.assertEqual(reader.execute("SELECT COUNT(*) FROM episode").fetchone()[0], 2)
        # 读者快照停在 2 条之后继续写：WAL 里多出来的帧 checkpoint 不进去，
        # "checkpoint + 复制主库文件"正是在这里丢掉尾部数据（真实现场是 daemon 在写）。
        for i in range(3):
            self.ingest_at(m, fake, f"新条目{i}", {"entities": [], "facts": [], "state_changes": []}, 2000.0 + i)

        out = os.path.join(d, "manual.bak")
        proc = self._run_cli("--db", m.cfg.db_path, "--out", out)
        self.assertEqual(proc.returncode, 0, f"备份失败：{proc.stdout}{proc.stderr}")
        self.assertTrue(os.path.exists(out), "命令报成功但没有产物")

        main_n = self._count(m.cfg.db_path, required=True)
        self.assertGreater(main_n, 0, "主库都读不到 episode，夹具坏了")
        bak_n = self._count(out, required=False)
        self.assertIsNotNone(
            bak_n,
            "备份文件打不开 / 缺 episode 表：命令 rc=0 还打印了 sha256，这是静默产出坏备份",
        )
        self.assertEqual(bak_n, main_n, "备份行数与主库不一致（WAL 尾部没进 .bak）")

    def test_backup_failure_is_not_silent(self):
        """备份写不出去时必须非零退出，不能 rc=0 假装成功。"""
        m, fake, d = self.mk("a1fail")
        self.ingest_at(m, fake, "随便一条", {"entities": [], "facts": [], "state_changes": []}, 1000.0)

        out = os.path.join(d, "no-such-dir", "x.bak")
        proc = self._run_cli("--db", m.cfg.db_path, "--out", out)
        self.assertNotEqual(proc.returncode, 0, f"备份失败却退出码为 0：{proc.stdout}")
        self.assertFalse(os.path.exists(out))

    @staticmethod
    def _count(path: str, *, required: bool):
        try:
            conn = sqlite3.connect(path)
            try:
                return int(conn.execute("SELECT COUNT(*) FROM episode").fetchone()[0])
            finally:
                conn.close()
        except sqlite3.Error:
            if required:
                raise
            return None


# ── A-21 性能不变式（源码级合规锁）────────────────────────────
class TestPerformanceInvariant(unittest.TestCase):
    def test_entity_seeds_has_no_instr(self):
        """field.py 的 `_entity_seeds` 不再含 `instr(`：子串扫全表是 200× 的损耗源。

        `field.py:196` 那条 instr 不在 recall 路径上，规格明确要求**不动**，
        所以只锁这一个函数体，不做全文断言。
        """
        src = (CORE_DIR / "field.py").read_text(encoding="utf-8")
        m = re.search(r"^def _entity_seeds\b.*?(?=^def |\Z)", src, re.S | re.M)
        self.assertIsNotNone(m, "field.py 里找不到 _entity_seeds——函数被改名了？锁会失效")
        body = m.group(0)
        self.assertNotIn(
            "instr(", body,
            "`_entity_seeds` 又用 instr 做子串匹配了（20k entity 下 2.06ms → 0.01ms 的差距）",
        )


# ── 哨兵式 session 过滤扫描（门禁与被测自身共用一套口径）──────
# 只认字符串常量，绝不扫源码文本：一句「我们不用 (?='*' OR …)」的解释性注释
# 就能让棘轮误红。门禁一旦会因为"解释为什么不用"而变红，就会被当成玄学绕开。
# 反之，写宽一点：`? = '*'`（带空格）、`session_id = '*'`（不带占位符）、
# 双引号版本都算——审查定稿里那处漏网的就是 `(? = '*' OR s.session_id = ?)`。
_SID_STAR = re.compile(r"(?:\?|\b[A-Za-z_][\w.]*)\s*=\s*['\"]\*['\"]\s+OR")


def _module_strings(src: str) -> list[str]:
    """取模块里的字符串常量；文档字符串单独剔除（那是散文，不是 SQL）。"""
    tree = ast.parse(src)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docs.add(id(body[0].value))
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
    ]


def count_sid_star(src: str) -> int:
    """数源码里真正写在 SQL 字符串中的 `= '*' OR` 哨兵式过滤。

    f-string 的字面片段也算：`f"(?='*' OR {col}=?)"` 会被拆成常量片段，
    含哨兵的那一片照样命中。
    已知盲区：先把哨兵存进变量再拼（`SID_ANY = "*"` + `+`）静态看不出来。
    那种只能靠 review——写清盲区比假装覆盖了安全。
    """
    return sum(len(_SID_STAR.findall(s)) for s in _module_strings(src))


def count_sid_star_file(path: Path) -> int:
    # utf-8-sig：仓库里有带 BOM 的模块（core/item.py），而 ast.parse 不吃 BOM
    return count_sid_star(path.read_text(encoding="utf-8-sig"))


# ── 其他门禁 ──────────────────────────────────────────────────
class TestReleaseHygiene(unittest.TestCase):
    def test_no_new_sid_star_or_antipattern(self):
        """`rosmontis/core/` 不再**新增** `(?='*' OR ...)` 反模式（audit.py 本轮白名单）。

        棘轮而非归零：规格没把清理排进本批，断言 0 会让门禁永久红。
        真清理时同步下调 SID_STAR_BASELINE（下面的提示会提醒，但不算失败）。
        """
        current = {}
        for path in sorted(CORE_DIR.glob("*.py")):
            if path.name in SID_STAR_WHITELIST:
                continue
            n = count_sid_star_file(path)
            if n:
                current[path.name] = n

        grew = {k: v for k, v in current.items() if v > SID_STAR_BASELINE.get(k, 0)}
        self.assertEqual(
            grew, {},
            f"新增了哨兵式 session 过滤（基线 {SID_STAR_BASELINE}，现在 {current}）",
        )

        # 基线只许"够高"，不许变成装饰：真清理了就得把数字降下来，
        # 否则下次有人加回去时棘轮还是绿的。不降不算失败，只提醒。
        lowered = {k: current.get(k, 0) for k in SID_STAR_BASELINE
                   if current.get(k, 0) < SID_STAR_BASELINE[k]}
        if lowered:
            warnings.warn(
                f"棘轮基线已过期：{lowered} 已低于 SID_STAR_BASELINE，"
                "请把基线调到当前值，否则这门禁挡不住回潮",
                stacklevel=2,
            )

    def test_comment_and_docstring_mentions_are_not_counted(self):
        """注释/文档字符串里**解释**该写法不算违规（误报会把门禁变成玄学）。"""
        code = (
            "# 这里按需拼 session 条件，不用 (?='*' OR session_id=?) 这种哨兵\n"
            '"""把 (? = \'*\' OR col=?) 换成按需拼接，别照抄。"""\n'
            'SQL = "SELECT id FROM episode WHERE archived=1"\n'
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mod.py"
            path.write_text(code, encoding="utf-8")
            self.assertEqual(count_sid_star_file(path), 0)

    def test_real_sentinel_is_counted(self):
        """真写了哨兵（含带空格与 f-string 两种写法）→ 计数 +1，棘轮本身不瞎。"""
        spellings = [
            'SQL = "SELECT 1 FROM episode WHERE (?=\'*\' OR session_id=?)"',
            'SQL = f"SELECT 1 FROM episode WHERE (? = \'*\' OR {col}=?)"',
            'SQL = "SELECT 1 FROM episode WHERE (session_id = \'*\' OR id=?)"',
        ]
        with tempfile.TemporaryDirectory() as d:
            for i, code in enumerate(spellings):
                path = Path(d) / f"mod{i}.py"
                path.write_text(code + "\n", encoding="utf-8")
                with self.subTest(code=code):
                    self.assertEqual(count_sid_star_file(path), 1)

    def test_version_consistency_is_covered_elsewhere(self):
        """版本三处一致（pyproject / rosmontis.__version__ / openapi.json）已有测试覆盖。

        `tests/test_contract.py:27-31` 已逐项断言，这里不重复造第二条——
        重复的断言只会让"改了一处忘了另一处"时出现两个红点。
        """
        self.assertTrue((ROOT / "tests" / "test_contract.py").exists())


# ── 待补接口探测（规格只说"拆成两个数"，没说落点）─────────────
def pending_total(m) -> int:
    """整型总量：兼容 `pending_backfill()` 返回 int 或 dict。"""
    got = m.pending_backfill()
    if isinstance(got, dict):
        for k in ("pending_backfill", "total", "remaining"):
            if k in got:
                return int(got[k] or 0)
        return int(got.get("null_vectors", 0) or 0) + int(got.get("stale_identity", 0) or 0)
    return int(got or 0)


def pending_detail(m, already=None) -> dict:
    """取含 `null_vectors` / `stale_identity` 的明细。

    规格只说"拆成两个数返回"，没定落点。这里按三个最可能的入口探测，
    都不存在就返回空 dict，让用例以"没人报出身份不符条数"失败，
    避免把实现口径的差异误报成测试错误。
    """
    if isinstance(already, dict):
        return already
    for name in ("pending_backfill_detail", "backfill_pending", "pending_vectors"):
        fn = getattr(m, name, None)
        if callable(fn):
            got = fn()
            if isinstance(got, dict):
                return got
    try:
        st = m.stats()
    except Exception:
        return {}
    return st if isinstance(st, dict) else {}


if __name__ == "__main__":
    unittest.main(verbosity=2)
