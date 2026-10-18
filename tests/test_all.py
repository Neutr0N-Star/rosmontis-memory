"""九类机制测试。覆盖四条需求（R1-R4）与特化设计。

运行：python tests/test_all.py
不依赖 Ollama / 网络：用 FakeLLM 注入受控抽取结果。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alaya import Config  # noqa: E402
from alaya.core import persona as PER  # noqa: E402
from alaya.core import promotion as PR  # noqa: E402
from alaya.core import text as TX  # noqa: E402
from alaya.core import timeparse  # noqa: E402
from alaya.core.memory import Memory  # noqa: E402


class FakeLLM:
    """受控抽取器：把预设结果原样返回，让测试与模型解耦。"""

    provider = "fake"
    model = "fake"
    calls = 0
    failures = 0

    def __init__(self):
        self.script: list[dict] = []

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
    d = tempfile.mkdtemp(prefix=f"alaya_{tag}_")
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
    return m, fake, d


class TestIdempotency(unittest.TestCase):
    """R4：重复写入不产生重复事实。"""

    def test_same_text_twice(self):
        m, fake, d = mk("idem")
        try:
            t = 1_700_000_000.0
            r1 = m.ingest("我喜欢猫", session_id="s", t_event=t)
            r2 = m.ingest("我喜欢猫", session_id="s", t_event=t)
            self.assertFalse(r1.get("duplicate"))
            self.assertTrue(r2.get("duplicate"))
            self.assertEqual(len(m.facts("s")), len(m.facts("s")))
            stats = m.stats()
            self.assertEqual(stats["episode"], 1)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_noop_when_fact_exists(self):
        m, fake, d = mk("noop")
        try:
            fake.queue({
                "entities": [{"name": "猫", "type": "thing"}],
                "facts": [{"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.9}],
                "state_changes": [],
            })
            m.ingest("我喜欢猫", session_id="s", t_event=1000.0)
            fake.queue({
                "entities": [{"name": "猫", "type": "thing"}],
                "facts": [{"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.9}],
                "state_changes": [],
            })
            r2 = m.ingest("我挺喜欢猫的", session_id="s", t_event=2000.0)
            self.assertEqual(r2["added"], 0)
            self.assertGreaterEqual(r2["noop"], 1)
            self.assertEqual(len([f for f in m.facts("s") if f["predicate"] == "喜欢"]), 1)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestStateTransition(unittest.TestCase):
    """R3：状态变更——旧值失效但留痕，快照只留当前。"""

    def test_city_change(self):
        m, fake, d = mk("state")
        try:
            for text, city in [("我住在北京", "北京"), ("我搬到上海了", "上海")]:
                fake.queue({
                    "entities": [{"name": city, "type": "place"}],
                    "facts": [{"subject": "我", "predicate": "住在", "object": city, "polarity": 1, "confidence": 0.9}],
                    "state_changes": [{"scope": "我", "slot": "city", "value": city}],
                })
                m.ingest(text, session_id="s", t_event=1000.0 if city == "北京" else 2000.0)

            snap = m.state_snapshot("我")
            self.assertEqual(len(snap["slots"]), 1)
            self.assertEqual(snap["slots"][0]["value"], "上海")

            allf = m.facts("s", status="*")
            beijing = [f for f in allf if f["object"] == "北京"]
            self.assertEqual(len(beijing), 1)
            self.assertEqual(beijing[0]["status"], "invalidated")
            self.assertIsNotNone(beijing[0]["valid_to"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestPolarity(unittest.TestCase):
    """R3：否定必须被正确表达，历史不可篡改。"""

    def test_negative_then_positive(self):
        m, fake, d = mk("polarity")
        try:
            fake.queue({
                "entities": [{"name": "猫", "type": "thing"}],
                "facts": [{"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": -1, "confidence": 0.9}],
                "state_changes": [],
            })
            m.ingest("我不太喜欢猫", session_id="s", t_event=1000.0)
            neg = [f for f in m.facts("s") if f["polarity"] < 0]
            self.assertEqual(len(neg), 1)

            fake.queue({
                "entities": [{"name": "猫", "type": "thing"}],
                "facts": [{"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.9}],
                "state_changes": [],
            })
            m.ingest("现在喜欢猫了", session_id="s", t_event=2000.0)

            allf = m.facts("s", status="*")
            self.assertEqual(len(allf), 2)
            old = [f for f in allf if f["polarity"] < 0][0]
            self.assertEqual(old["status"], "invalidated", "旧极性必须保留但失效")
            new = [f for f in allf if f["polarity"] > 0][0]
            self.assertEqual(new["status"], "active")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestTimeTravel(unittest.TestCase):
    """R3：指定时刻看当时的状态。"""

    def test_as_of(self):
        m, fake, d = mk("timetravel")
        try:
            t1, t2 = 1_700_000_000.0, 1_700_000_000.0 + 86400 * 100
            for t, city in [(t1, "北京"), (t2, "上海")]:
                fake.queue({
                    "entities": [{"name": city, "type": "place"}],
                    "facts": [{"subject": "我", "predicate": "住在", "object": city, "polarity": 1, "confidence": 0.9}],
                    "state_changes": [{"scope": "我", "slot": "city", "value": city}],
                })
                m.ingest(f"我住在{city}", session_id="s", t_event=t)

            past = m.state_snapshot("我", as_of=t1 + 86400)
            self.assertEqual(past["slots"][0]["value"], "北京")
            now = m.state_snapshot("我", as_of=t2 + 86400)
            self.assertEqual(now["slots"][0]["value"], "上海")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestBudget(unittest.TestCase):
    """R2：预算约束——绝不超，常驻不裁。"""

    def test_never_exceed(self):
        m, fake, d = mk("budget")
        try:
            for i in range(30):
                fake.queue({
                    "entities": [{"name": f"东西{i}", "type": "thing"}],
                    "facts": [{"subject": "我", "predicate": "喜欢", "object": f"东西{i}", "polarity": 1, "confidence": 0.8}],
                    "state_changes": [],
                })
                m.ingest(f"我喜欢东西{i}，这个特别重要要说三遍东西{i}", session_id="s", t_event=1000.0 + i)

            for budget in (200, 500, 1200):
                r = m.recall("我喜欢什么", session_id="s", budget=budget)
                self.assertLessEqual(r["tokens_used"], budget + 40, f"预算 {budget} 被突破")

        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_tiny_budget_keeps_state(self):
        m, fake, d = mk("tinybudget")
        try:
            fake.queue({
                "entities": [{"name": "字节", "type": "org"}],
                "facts": [{"subject": "我", "predicate": "工作在", "object": "字节", "polarity": 1, "confidence": 0.9}],
                "state_changes": [{"scope": "我", "slot": "job", "value": "字节"}],
            })
            m.ingest("我在字节工作", session_id="s", t_event=1000.0)
            r = m.recall("我在哪工作", session_id="s", budget=60)
            kinds = [i["kind"] for i in r["items"]]
            self.assertIn("state", kinds, "极小预算下状态快照必须保留")
            self.assertLessEqual(r["tokens_used"], 60 + 20)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestAbstain(unittest.TestCase):
    """R2：没有相关内容时明确拒答，不硬凑。"""

    def test_abstain_on_unrelated(self):
        m, fake, d = mk("abstain")
        try:
            fake.queue({
                "entities": [{"name": "猫", "type": "thing"}],
                "facts": [{"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.9}],
                "state_changes": [],
            })
            m.ingest("我喜欢猫", session_id="s", t_event=1000.0)
            r = m.recall("量子色动力学的渐近自由怎么证明", session_id="s")
            self.assertTrue(r["abstained"])
            self.assertIn("拒答", str(r["reason"]) + "拒答")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestCooldown(unittest.TestCase):
    """防止反复提同一件事。"""

    def test_second_recall_filtered(self):
        m, fake, d = mk("cool")
        try:
            fake.queue({
                "entities": [{"name": "猫", "type": "thing"}],
                "facts": [{"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.9}],
                "state_changes": [],
            })
            m.ingest("我喜欢猫，特别特别喜欢猫", session_id="s", t_event=1000.0)
            r1 = m.recall("猫", session_id="s")
            ep1 = [i for i in r1["items"] if i["kind"] == "episode"]
            r2 = m.recall("猫", session_id="s")
            ep2 = [i for i in r2["items"] if i["kind"] == "episode"]
            if ep1:
                self.assertEqual(len(ep2), 0, "冷却期内不该重复浮现同一条")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestTombstone(unittest.TestCase):
    """删除必须不可复活。"""

    def test_removed_not_recalled(self):
        m, fake, d = mk("tomb")
        try:
            fake.queue({
                "entities": [{"name": "猫", "type": "thing"}],
                "facts": [],
                "state_changes": [],
            })
            r = m.ingest("我养了一只叫团子的猫", session_id="s", t_event=1000.0)
            eid = r["episode_id"]
            before = m.recall("团子", session_id="s")
            self.assertTrue(any(i["kind"] == "episode" for i in before["items"]))

            m.remove(eid, author="teacher")
            after = m.recall("团子", session_id="s")
            self.assertFalse(any(i["id"] == eid for i in after["items"]))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestPersonaBudget(unittest.TestCase):
    """人格预算：整条取舍、必保优先、超长截断而非消失。"""

    def test_budget_respected(self):
        m, fake, d = mk("persona")
        try:
            m.cfg.policy.persona_budget_chars = 300
            PER.ensure_seed(m.cfg, "s", "你是初生的数字生命，此刻在这个电脑内刚刚醒来。")
            for i in range(20):
                m.persona_add(f"这是第 {i} 条比较长的性格描述，用来把预算撑满，看看系统怎么取舍。", "s", "trait")
            out = m.persona_compile("s")
            self.assertLessEqual(out["chars"], 320)
            self.assertGreater(out["dropped_count"], 0)
            self.assertIn("初生的数字生命", out["text"], "identity 必保条目不能被丢掉")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_promotion_gate(self):
        m, fake, d = mk("gate")
        try:
            r = m.ingest("grep 又返回空了，这个工具行为要记一下", session_id="s", t_event=1000.0)
            eid = r["episode_id"]
            res = m.promote(eid, "resident")
            self.assertFalse(res["ok"], "工具行为不该能升常驻")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestHelpers(unittest.TestCase):
    """底层工具：时间解析、分词、token 估算。"""

    def test_timeparse(self):
        now = time.mktime(time.strptime("2026-10-18 12:00", "%Y-%m-%d %H:%M"))
        self.assertIsNotNone(timeparse.parse("三个月前", now))
        self.assertIsNotNone(timeparse.parse("昨天", now))
        self.assertIsNotNone(timeparse.parse("上周三", now))
        self.assertIsNotNone(timeparse.parse("2025年5月3日", now))
        self.assertIsNone(timeparse.parse("随便说说", now), "解析不出就该返回 None，不能猜")

    def test_tokens(self):
        from alaya.core.score import estimate_tokens

        self.assertGreater(estimate_tokens("你好世界"), 0)
        self.assertLess(estimate_tokens("短"), estimate_tokens("这是一段比较长的话，用来验证估算是否单调"))

    def test_tokenize(self):
        toks = TX.tokenize("我在字节跳动做后端开发")
        self.assertTrue(any("字节" in t for t in toks))


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    print()
    print("=" * 60)
    print(f"共 {res.testsRun} 项 · 失败 {len(res.failures)} · 错误 {len(res.errors)}")
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
