"""语义场测试：激活 / 扩散 / 空间倍率 / 冷启动退化 / 场接线。

运行：PYTHONIOENCODING=utf-8 python tests/test_field.py
不依赖 Ollama / 网络：embed_provider=hash + FakeLLM 注入受控抽取。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alaya import Config  # noqa: E402
from alaya.core import db as DB  # noqa: E402
from alaya.core import field as F  # noqa: E402
from alaya.core import forget as FG  # noqa: E402
from alaya.core import memory_field as MF  # noqa: E402
from alaya.core import score as SC  # noqa: E402
from alaya.core.memory import Memory  # noqa: E402


class FakeLLM:
    """受控抽取器：把预设结果原样返回，让测试与模型解耦。"""

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


@contextmanager
def scenario(tag: str):
    """建一个临时库；退出时关连接、删目录。"""
    d = tempfile.mkdtemp(prefix=f"alaya_field_{tag}_")
    cfg = Config(
        db_path=os.path.join(d, "t.db"),
        embed_provider="hash",
        llm_hot="rule",
        llm_deep="rule",
        log_level="ERROR",
    )
    m = Memory(cfg)
    fake = FakeLLM()
    m.llm_hot = fake
    try:
        yield m, fake, cfg
    finally:
        try:
            m.wait_enrich()
        except Exception:
            pass
        DB.close_thread_conns()
        shutil.rmtree(d, ignore_errors=True)


def seed_cat(m: Memory, fake: FakeLLM, now: float) -> tuple[str, str]:
    """两条共享「猫/团子」实体与主题的 episode，用来测扩散。"""
    fake.queue(
        {
            "entities": [{"name": "猫", "type": "thing"}, {"name": "团子", "type": "thing"}],
            "facts": [
                {"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.9}
            ],
            "state_changes": [],
        }
    )
    r1 = m.ingest("我喜欢猫，猫叫团子", session_id="s", t_event=now - 86400)
    fake.queue(
        {
            "entities": [{"name": "猫", "type": "thing"}, {"name": "团子", "type": "thing"}],
            "facts": [
                {"subject": "我", "predicate": "养", "object": "团子", "polarity": 1, "confidence": 0.9}
            ],
            "state_changes": [],
        }
    )
    r2 = m.ingest("我养了一只猫叫团子", session_id="s", t_event=now - 3600)
    m.wait_enrich()
    return r1["episode_id"], r2["episode_id"]


class TestActivation(unittest.TestCase):
    """ACT-R 幂律激活：单调性、时间衰减、边界。"""

    def test_recent_beats_old(self):
        now = 1_700_000_000.0
        encode = now - 86400 * 30
        recent = F.activation_from_parts(encode, now - 60, 1, 0.5, now)
        old = F.activation_from_parts(encode, now - 86400 * 20, 1, 0.5, now)
        self.assertGreater(recent, old, "同样一次访问，最近的激活更高")

    def test_more_traces_beat_fewer(self):
        now = 1_700_000_000.0
        encode = now - 86400 * 30
        few = F.activation_from_parts(encode, now - 3600, 1, 0.5, now)
        many = F.activation_from_parts(encode, now - 3600, 8, 0.5, now)
        self.assertGreater(many, few, "访问次数多（痕迹密）的激活更高")

    def test_time_decay(self):
        encode = 1_700_000_000.0
        last = encode + 86400
        early = F.activation_from_parts(encode, last, 3, 0.5, last + 3600)
        late = F.activation_from_parts(encode, last, 3, 0.5, last + 86400 * 60)
        self.assertGreater(early, late, "同一条记忆，越往后激活越低")

    def test_empty_traces_base(self):
        now = 1_700_000_000.0
        self.assertAlmostEqual(MF.activation_value([], 0.5, now), 1.0, places=6)
        self.assertAlmostEqual(MF.activation_value([], 0.0, now), 0.0, places=6)

    def test_odds_bounded_and_monotone(self):
        vals = [MF.activation_odds(a) for a in (-20, -8, -3, 0, 3, 10)]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))
        self.assertEqual(vals, sorted(vals), "激活越大，召回强度单调不减")
        self.assertGreater(vals[-1], 0.99)


class TestSpace(unittest.TestCase):
    """三空间归属与衰减倍率。"""

    def test_classification(self):
        self.assertEqual(F.space_of({"kind": "persona"}), "identity")
        self.assertEqual(F.space_of({"kind": "fact"}), "knowledge")
        self.assertEqual(F.space_of({"kind": "state"}), "knowledge")
        self.assertEqual(
            F.space_of({"kind": "episode", "tier": "retrieval", "source": "dialogue", "memory_type": "fact"}),
            "operational",
        )
        self.assertEqual(F.space_of({"kind": "episode", "tier": "resident"}), "identity")
        self.assertEqual(
            F.space_of({"kind": "episode", "source": "summary", "memory_type": "lesson"}), "knowledge"
        )
        self.assertEqual(
            F.space_of({"kind": "episode", "tier": "retrieval", "memory_type": "tool_behavior"}),
            "operational",
        )

    def test_decay_factors(self):
        self.assertAlmostEqual(MF.decay_factor("identity"), 0.1)
        self.assertAlmostEqual(MF.decay_factor("knowledge"), 1.0)
        self.assertAlmostEqual(MF.decay_factor("operational"), 3.0)
        self.assertAlmostEqual(MF.space_weight("identity"), 0.1)
        self.assertAlmostEqual(MF.space_weight("operational"), 3.0)

    def test_forget_retention_by_space(self):
        cfg = Config(log_level="ERROR")
        now = 1_700_000_000.0
        t_ref = now - 86400 * 30
        r_id = FG.retention(cfg, t_ref, 0, 0.5, now, space="identity")
        r_kn = FG.retention(cfg, t_ref, 0, 0.5, now, space="knowledge")
        r_op = FG.retention(cfg, t_ref, 0, 0.5, now, space="operational")
        self.assertGreater(r_id, r_kn, "identity 几乎不衰减，保留率更高")
        self.assertGreater(r_kn, r_op, "operational 烧得快，保留率更低")
        # 默认空间是 knowledge，旧调用方（audit 等）行为不变
        self.assertAlmostEqual(r_kn, FG.retention(cfg, t_ref, 0, 0.5, now), places=12)


class TestScoreWiring(unittest.TestCase):
    """打分接线：ACT-R 替换 access_count 加成，空间倍率生效。"""

    def _cand(self, space: str, activation: float) -> SC.Candidate:
        now = 1_700_000_000.0
        return SC.Candidate(
            kind="fact",
            ref_id="fa_x",
            subject="我",
            predicate="喜欢",
            object_text="猫",
            t_ref=now - 3600,
            salience=0.5,
            emotion=0.2,
            access_count=3,
            last_recall=now - 600,
            rel=0.7,
            graph_prox=0.2,
            time_fit=1.0,
            space=space,
            activation=activation,
        )

    def test_space_multiplier_applies(self):
        now = 1_700_000_000.0
        w = Config(log_level="ERROR").weights
        id_s = SC.score_one(self._cand("identity", 0.5), now, w)
        kn_s = SC.score_one(self._cand("knowledge", 0.5), now, w)
        op_s = SC.score_one(self._cand("operational", 0.5), now, w)
        self.assertLess(id_s, kn_s, "identity 的激活加成最低（人格不靠最近想起排序）")
        self.assertLess(kn_s, op_s, "operational 的激活加成最高")

    def test_activation_replaces_access_count(self):
        now = 1_700_000_000.0
        w = Config(log_level="ERROR").weights
        low = SC.score_one(self._cand("knowledge", -10.0), now, w)
        high = SC.score_one(self._cand("knowledge", 1.0), now, w)
        self.assertGreater(high, low, "激活高的条目分数更高")
        # 加成项始终 ≥1（不会把 raw 压低），且上限受 1.5 总上限约束
        self.assertLessEqual(high, 1.5)


class TestPropagation(unittest.TestCase):
    """稀疏扩散：有界、可复现、迭代上限、冷启动退化、空种子。"""

    def test_bounded_and_deterministic(self):
        with scenario("prop") as (m, fake, cfg):
            now = time.time()
            e1, e2 = seed_cat(m, fake, now)
            F.bind(cfg, m.embedder)
            seeds = F.seed_nodes("团子", "s", routes={"vec": [], "fts": [f"episode:{e2}"]})
            self.assertTrue(seeds, "至少要点着一个种子")
            phi1 = F.propagate(seeds, "s", iterations=9)  # 超上限会被截到 3
            phi2 = F.propagate(seeds, "s", iterations=9)
            self.assertLessEqual(phi1.iterations, F.MAX_ITERATIONS)
            self.assertGreaterEqual(phi1.iterations, 1)
            self.assertTrue(0.0 <= min(phi1.values()) <= max(phi1.values()) <= 1.0)
            self.assertEqual(dict(phi1), dict(phi2), "同输入扩散结果必须可复现")
            self.assertTrue(set(phi1.edge_types) <= set(F.EDGE_KINDS))
            self.assertIn("structural", phi1.edge_types)
            self.assertIn(f"episode:{e1}", phi1, "同主题的另一条 episode 应被扩散到")

    def test_cold_start_degrades_to_ignition(self):
        with scenario("cold") as (m, fake, cfg):
            now = time.time()
            fake.queue({"entities": [], "facts": [], "state_changes": []})
            eid = m.ingest("今天有点累，想早点睡", session_id="s", t_event=now - 3600)["episode_id"]
            F.bind(cfg, m.embedder)
            seeds = F.seed_nodes("今天有点累", "s")
            self.assertTrue(seeds, "没有结构边时也应能靠 FTS/向量点火")
            phi = F.propagate(seeds, "s", iterations=3)
            self.assertEqual(phi.iterations, 0, "无结构边应跳过扩散，不得报错")
            self.assertEqual(phi.edge_types, ())
            self.assertEqual(set(phi), {s.key for s in seeds}, "退化为纯点火")

    def test_empty_seeds(self):
        with scenario("empty") as (m, fake, cfg):
            F.bind(cfg, m.embedder)
            phi = F.propagate([], "s", iterations=3)
            self.assertEqual(len(phi), 0)
            self.assertEqual(phi.iterations, 0)
            self.assertEqual(F.seed_nodes("量子色动力学渐近自由", "s"), [])

    def test_vector_route_unavailable(self):
        with scenario("novec") as (m, fake, cfg):
            now = time.time()
            seed_cat(m, fake, now)
            F.bind(cfg, None)
            seeds = F.seed_nodes("团子", "s", cfg=cfg, embedder=None)
            self.assertTrue(seeds, "向量路不可用时仍可用 FTS+实体点火")
            sources = " ".join(s.source for s in seeds)
            self.assertTrue(("fts" in sources) or ("entity" in sources), f"点火来源异常: {sources}")
            phi = F.propagate(seeds, "s", cfg=cfg, iterations=3)
            self.assertGreater(len(phi), 0)

    def test_laplacian_step_bounded(self):
        adj = {"a": {"b": 1.0}, "b": {"a": 1.0}, "c": {}}
        phi = {"a": 1.0, "b": 0.0, "c": 0.5}
        for _ in range(3):
            phi = MF.laplacian_step(phi, adj, 0.5)
        self.assertLessEqual(max(phi.values()), 1.0)
        self.assertAlmostEqual(phi["a"], phi["b"], places=9)
        self.assertAlmostEqual(phi["c"], 0.5, places=9, msg="孤立节点不参与扩散也不丢")


class TestRecallFieldWiring(unittest.TestCase):
    """recall 接线：场作为第 4 路候选，plan_json 留痕，预算/拒答不动。"""

    def test_plan_records_field(self):
        with scenario("plan") as (m, fake, cfg):
            seed_cat(m, fake, time.time())
            r = m.recall("团子是谁", session_id="s")
            self.assertFalse(r["abstained"])
            for key in ("seeds", "iterations", "edge_types", "phi_top"):
                self.assertIn(key, r["plan"], f"plan 缺少 {key}")
            self.assertGreater(r["plan"]["seeds"], 0)
            self.assertLessEqual(r["tokens_used"], r["budget"] + 40)
            with DB.reader(cfg.db_path) as conn:
                row = conn.execute("SELECT plan_json FROM retrieval_log ORDER BY id DESC LIMIT 1").fetchone()
            logged = json.loads(row["plan_json"])
            for key in ("seeds", "iterations", "edge_types", "phi_top"):
                self.assertIn(key, logged, f"retrieval_log.plan_json 缺少 {key}")

    def test_abstain_when_unrelated(self):
        with scenario("abstain") as (m, fake, cfg):
            seed_cat(m, fake, time.time())
            r = m.recall("量子色动力学的渐近自由怎么证明", session_id="s")
            self.assertTrue(r["abstained"], "场扩散不得把无关查询撑成有记忆")
            self.assertEqual(r["items"], [])

    def test_switch_off(self):
        with scenario("off") as (m, fake, cfg):
            seed_cat(m, fake, time.time())
            old = os.environ.get("ALAYA_FIELD")
            os.environ["ALAYA_FIELD"] = "0"
            try:
                r = m.recall("团子是谁", session_id="s")
            finally:
                if old is None:
                    os.environ.pop("ALAYA_FIELD", None)
                else:
                    os.environ["ALAYA_FIELD"] = old
            self.assertFalse(r["plan"]["field_enabled"])
            self.assertEqual(r["plan"]["seeds"], 0)
            self.assertEqual(r["plan"]["iterations"], 0)
            self.assertEqual(r["plan"]["phi_top"], [])


class TestFieldMetrics(unittest.TestCase):
    """场指标可产出，供 WS-D 采集。"""

    def test_metrics_shape(self):
        with scenario("metric") as (m, fake, cfg):
            seed_cat(m, fake, time.time())
            met = F.field_metrics("s", cfg=cfg)
            self.assertEqual(met["session_id"], "s")
            self.assertGreater(met["nodes"]["episode"], 0)
            self.assertGreater(met["nodes"]["entity"], 0)
            self.assertGreater(met["edges"]["total"], 0)
            self.assertGreaterEqual(met["degree_entropy"], 0.0)
            self.assertEqual(set(met["spaces"]), {"identity", "knowledge", "operational"})
            self.assertGreater(sum(met["spaces"].values()), 0)

    def test_activation_from_db(self):
        with scenario("actdb") as (m, fake, cfg):
            now = time.time()
            e1, _e2 = seed_cat(m, fake, now)
            F.bind(cfg, m.embedder)
            a_now = F.activation(f"episode:{e1}", now)
            a_later = F.activation(f"episode:{e1}", now + 86400 * 90)
            self.assertIsInstance(a_now, float)
            self.assertGreater(a_now, a_later, "时间越往后，库内条目的激活越低")


if __name__ == "__main__":
    unittest.main(verbosity=2)
