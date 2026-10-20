"""WS-C 接口测试：公共 API / 四钩子 / 工具链 / 纯函数 context / 工具调用透传。

运行：PYTHONIOENCODING=utf-8 python tests/test_api.py
不依赖网络与真实模型：假 LLM + 假 HTTP 响应。
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosmontis  # noqa: E402
from rosmontis import api  # noqa: E402
from rosmontis.core import db as DB  # noqa: E402
from rosmontis.harness import chat_with_tools, make_llm_callable, normalize_tool_calls  # noqa: E402
from rosmontis.tools import dispatch, item_state, transition_item  # noqa: E402

TOOL_KEYS = {"ok", "id", "state", "from", "to", "reason"}


class FakeExtractor:
    """受控抽取器：把预设结果原样返回，让测试与模型解耦。"""

    provider = "fake"
    model = "fake"

    def __init__(self, script=None):
        self.script = list(script or [])
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


class RecordingLLM:
    """按剧本返回的假模型，记录每次收到的 messages。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def __call__(self, messages, tools=None):
        self.calls.append(copy.deepcopy(messages))
        if self.responses:
            return self.responses.pop(0)
        return {"content": "好。"}


def db_counts(db_path: str) -> dict:
    out = {}
    with DB.reader(db_path) as conn:
        for t in ("episode", "fact", "persona", "retrieval_log", "notebook_promotion"):
            try:
                out[t] = int(conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"])
            except Exception:
                out[t] = -1
    return out


class ApiCase(unittest.TestCase):
    """公共测试基类：每个用例独立临时库，收尾时关句柄再删目录。"""

    def mk(self, tag: str, llm_callable=None, **cfg):
        d = tempfile.mkdtemp(prefix=f"rosmontis_api_{tag}_")
        db_path = os.path.join(d, "t.db")
        conf = {
            "db_path": db_path,
            "embed_provider": "hash",
            "llm_hot": "rule",
            "llm_deep": "rule",
            "log_level": "ERROR",
        }
        conf.update(cfg)
        if llm_callable is not None:
            conf["llm_callable"] = llm_callable
        h = rosmontis.open(db_path, conf)
        h.mem.llm_hot = FakeExtractor()
        self.addCleanup(shutil.rmtree, d, True)
        self.addCleanup(h.close)
        return h, db_path


class TestPublicApi(ApiCase):
    def test_open_wake_idempotent(self):
        h, db_path = self.mk("wake")
        w1 = h.wake("u1", "c1")
        w2 = h.wake("u1", "c1")
        self.assertTrue(w1["ok"])
        self.assertEqual(w1["session_id"], "u1")
        self.assertEqual(w1["persona_chars"], w2["persona_chars"])
        self.assertEqual(w1["inbox"], w2["inbox"])
        self.assertGreaterEqual(w1["persona_items"], 1)
        for key in ("persona_text", "resident_text", "inbox", "degraded"):
            self.assertIn(key, w1)
        self.assertEqual(db_counts(db_path)["persona"], 1, "wake 重复调用不能多播一条人格种子")

    def test_context_is_pure_and_within_budget(self):
        h, db_path = self.mk("pure")
        h.wake("u1", "c1")
        h.mem.llm_hot.queue({
            "entities": [{"name": "猫", "type": "thing"}],
            "facts": [{"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.9}],
            "state_changes": [],
        })
        h.mem.ingest("我喜欢猫，特别喜欢猫", session_id="u1", t_event=1000.0)

        before = db_counts(db_path)
        for _ in range(3):
            ctx = h.context("u1", "c1", "我喜欢什么", budget=300)
            self.assertTrue(ctx["pure"])
            self.assertLessEqual(ctx["memory_tokens"], 300)
            self.assertEqual(ctx["tokens"], ctx["memory_tokens"])
            self.assertIn("parts", ctx)
        after = db_counts(db_path)
        self.assertEqual(before["episode"], after["episode"], "context 绝不能写 episode")
        self.assertEqual(before["fact"], after["fact"], "context 绝不能写 fact")
        self.assertEqual(before["retrieval_log"], after["retrieval_log"], "context 绝不能写 retrieval_log")

    def test_context_budget_hard_cap(self):
        h, _ = self.mk("budget")
        for i in range(24):
            h.mem.ingest(f"我喜欢东西{i}，这个很重要", session_id="u1", t_event=1000.0 + i)
        big = h.context("u1", "c1", "东西", budget=600)
        self.assertGreater(big["memory_tokens"], 0, "有内容时应能召回")
        self.assertLessEqual(big["memory_tokens"], 600)
        for budget in (60, 120, 200):
            small = h.context("u1", "c1", "东西", budget=budget)
            self.assertLessEqual(small["memory_tokens"], budget, f"预算 {budget} 被突破")
            self.assertEqual(small["budget"], budget)

    def test_context_bad_budget_and_closed(self):
        h, _ = self.mk("bad")
        with self.assertRaises(api.RosmontisError) as cm:
            h.context("u1", "c1", "x", budget=0)
        self.assertEqual(cm.exception.code, "invalid_argument")
        with self.assertRaises(api.RosmontisError):
            h.context("", "c1", "x")
        h.close()
        with self.assertRaises(api.RosmontisError) as cm:
            h.context("u1", "c1", "x")
        self.assertEqual(cm.exception.code, "closed")
        self.assertFalse(api.RosmontisError("x", "y").as_dict()["ok"])

    def test_turn_empty_text_raises(self):
        h, _ = self.mk("empty")
        with self.assertRaises(api.RosmontisError) as cm:
            h.turn("u1", "c1", "   ")
        self.assertEqual(cm.exception.code, "invalid_argument")

    def test_observe_idempotent(self):
        h, _ = self.mk("observe")
        h.mem.wait_enrich()
        h.mem.llm_hot = FakeExtractor()
        r1 = h.observe("u1", "c1", "我先睡了", "晚安")
        n1 = db_counts(h.cfg.db_path)["episode"]
        r2 = h.observe("u1", "c1", "我先睡了", "晚安")
        n2 = db_counts(h.cfg.db_path)["episode"]
        self.assertTrue(r1["ok"])
        self.assertEqual(r1["written"], ["user", "assistant"])
        self.assertTrue(r2["idempotent"], "同一文本在窗口内必须幂等")
        self.assertEqual(n1, n2, "幂等调用不能多写记忆")


class TestToolsContract(ApiCase):
    def _remember(self, h, **kw):
        args = {"what": kw.get("what", "他说他换工作了"), "noticed": kw.get("noticed", "说的时候像松了口气"),
                "means": kw.get("means", "他可能想逃"), "next": kw.get("next", "下次别问好不好，问累不累"),
                "topic": kw.get("topic", "工作")}
        return dispatch(h.mem, "rosmontis_remember", args, "u1")

    def test_remember_payload_shape_and_gate(self):
        h, _ = self.mk("remember")
        full = self._remember(h)
        self.assertEqual(set(full.keys()), TOOL_KEYS, "工具返回必须是统一六键")
        self.assertTrue(full["ok"])
        self.assertTrue(full["id"].startswith("ep_"))
        self.assertEqual(full["state"], "note")
        self.assertEqual(full["from"], "new")
        self.assertEqual(full["to"], "note")

        only_what = dispatch(h.mem, "rosmontis_remember", {"what": "今天下雨了"}, "u1")
        self.assertTrue(only_what["ok"])
        self.assertEqual(only_what["state"], "inbox", "只有 what 的条目应由 gate 留在速记层")

    def test_promote_gate_decides(self):
        h, _ = self.mk("promote")
        ids = [self._remember(h, what=f"第 {i} 条判断，反复想到")["id"] for i in range(3)]
        r1 = dispatch(h.mem, "rosmontis_promote", {"note_id": ids[0], "section": "trait"}, "u1")
        r2 = dispatch(h.mem, "rosmontis_promote", {"note_id": ids[1], "section": "trait"}, "u1")
        r3 = dispatch(h.mem, "rosmontis_promote", {"note_id": ids[2], "section": "trait"}, "u1")
        self.assertEqual(set(r1.keys()), TOOL_KEYS)
        self.assertTrue(r1["ok"], r1["reason"])
        self.assertEqual(r1["state"], r1["to"], "成功时 state 必须等于 gate 判定的 to")
        self.assertIn(r1["state"], {"note", "resident", "important"})
        self.assertTrue(r2["ok"], r2["reason"])
        self.assertFalse(r3["ok"], "单会话第 3 次改人设必须被闸门拒绝")
        self.assertIn("最多", r3["reason"])
        # 被拒时状态不迁移，且 reason 原样可读（模型能据此自我修正）
        self.assertEqual(r3["from"], r3["state"])

    def test_skip_applies_to_episode_and_fact(self):
        h, _ = self.mk("skip")
        h.mem.llm_hot.queue({
            "entities": [{"name": "猫", "type": "thing"}],
            "facts": [{"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.9}],
            "state_changes": [],
        })
        ep = h.mem.ingest("我喜欢猫", session_id="u1", t_event=1000.0)["episode_id"]
        r_ep = dispatch(h.mem, "rosmontis_skip", {"item_id": ep, "reason": "不想提"}, "u1")
        self.assertTrue(r_ep["ok"])
        self.assertEqual(r_ep["state"], "muted")
        self.assertEqual(item_state(h.mem, ep)["state"], "muted")

        fact_id = h.mem.facts("u1")[0]["id"]
        r_fa = dispatch(h.mem, "rosmontis_skip", {"item_id": fact_id, "reason": "不想提"}, "u1")
        self.assertEqual(set(r_fa.keys()), TOOL_KEYS)
        self.assertTrue(r_fa["ok"], r_fa["reason"])
        self.assertEqual(r_fa["state"], "muted")
        rows = h.mem.facts("u1", status="*")
        row = [f for f in rows if f["id"] == fact_id][0]
        self.assertEqual(row["status"], "invalidated", "fact 在现有 schema 下用 invalidated 停止检索")
        self.assertEqual(item_state(h.mem, fact_id)["state"], "muted", "静音与归档要能区分")

    def test_skip_missing_and_failure_reason(self):
        h, _ = self.mk("skipmiss")
        r = dispatch(h.mem, "rosmontis_skip", {"item_id": "ep_nope"}, "u1")
        self.assertFalse(r["ok"])
        self.assertIn("不存在", r["reason"])
        r2 = dispatch(h.mem, "rosmontis_remember", {"what": "   "}, "u1")
        self.assertFalse(r2["ok"])
        self.assertIn("what", r2["reason"])

    def test_transition_archive_restore(self):
        h, _ = self.mk("trans")
        note = self._remember(h)["id"]
        r = transition_item(h.mem, note, "archive", reason="test")
        self.assertTrue(r["ok"])
        self.assertEqual(r["state"], "archived")
        self.assertEqual(item_state(h.mem, note)["state"], "archived")
        r2 = transition_item(h.mem, note, "restore", reason="test")
        self.assertTrue(r2["ok"])
        self.assertEqual(item_state(h.mem, note)["state"], "note")
        bad = transition_item(h.mem, note, "explode")
        self.assertFalse(bad["ok"])
        self.assertIn("未知动作", bad["reason"])
        missing = transition_item(h.mem, "ep_nope", "archive")
        self.assertFalse(missing["ok"])
        self.assertIn("不存在", missing["reason"])


class TestHarnessContract(ApiCase):
    def test_turn_history_tool_loop_and_usage(self):
        record = RecordingLLM([
            {
                "content": "",
                "tool_calls": [{"name": "rosmontis_remember", "arguments": {"what": "他说他今天很累", "next": "先问睡够没有"}}],
                "usage": {"prompt_tokens": 180, "completion_tokens": 64},
                "finish_reason": "tool_calls",
            },
            {
                "content": "嗯，记下了。今天早点睡。",
                "tool_calls": [],
                "usage": {"prompt_tokens": 240, "completion_tokens": 18},
                "finish_reason": "stop",
            },
        ])
        h, db_path = self.mk("loop", llm_callable=record)
        h.wake("u1", "c1")
        history = [
            {"role": "user", "content": "昨天我加班到很晚"},
            {"role": "assistant", "content": "那今天要早点休息"},
        ]
        r = h.turn("u1", "c1", "今天还是有点累", history=history)

        self.assertEqual(r["reply"], "嗯，记下了。今天早点睡。")
        self.assertEqual(r["rounds"], 2)
        self.assertEqual(r["finish_reason"], "stop")
        self.assertEqual(r["usage"]["prompt_tokens"], 420)
        self.assertEqual(r["usage"]["completion_tokens"], 82)
        self.assertEqual(len(r["tools_called"]), 1)
        tool = r["tools_called"][0]
        self.assertEqual(tool["tool"], "rosmontis_remember")
        self.assertTrue(tool["ok"], tool["reason"])
        self.assertEqual(tool["state"], "note")
        self.assertTrue(tool["tool_call_id"].startswith("call_"))

        # 第一次调用：system + history + 当轮输入（历史缺口已补）
        first = record.calls[0]
        contents = [(m.get("role"), m.get("content")) for m in first]
        self.assertEqual(contents[1], ("user", "昨天我加班到很晚"))
        self.assertEqual(contents[2], ("assistant", "那今天要早点休息"))
        self.assertEqual(contents[-1], ("user", "今天还是有点累"))

        # 第二次调用：助手 tool_calls 带 id，工具结果带 tool_call_id
        second = record.calls[1]
        assistant = [m for m in second if m.get("role") == "assistant" and m.get("tool_calls")][-1]
        call_id = assistant["tool_calls"][0]["id"]
        self.assertTrue(call_id)
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "rosmontis_remember")
        tool_msgs = [m for m in second if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], call_id)
        self.assertTrue(tool_msgs[0]["content"].startswith("成功"))

        # 工具真的落库了
        with DB.reader(db_path) as conn:
            n = conn.execute("SELECT COUNT(*) c FROM episode WHERE source='notebook'").fetchone()["c"]
        self.assertEqual(n, 1)

    def test_missing_usage_and_finish_reason_defaults(self):
        record = RecordingLLM([{"content": "好。"}])
        h, _ = self.mk("nousage", llm_callable=record)
        r = h.turn("u1", "c1", "在吗")
        self.assertEqual(r["reply"], "好。")
        self.assertEqual(r["usage"], {})
        self.assertIsNone(r["finish_reason"])
        self.assertEqual(r["tools_called"], [])

    def test_normalize_tool_calls_variants(self):
        openai_raw = [{
            "id": "call_x",
            "type": "function",
            "function": {"name": "rosmontis_recall", "arguments": '{"query": "猫"}'},
        }]
        flat = [{"name": "rosmontis_skip", "arguments": {"item_id": "ep_1"}}]
        n1 = normalize_tool_calls(openai_raw)
        n2 = normalize_tool_calls(flat)
        self.assertEqual(n1[0]["id"], "call_x")
        self.assertEqual(n1[0]["arguments"], {"query": "猫"})
        self.assertTrue(n2[0]["id"].startswith("call_"))
        self.assertEqual(n2[0]["arguments"], {"item_id": "ep_1"})
        self.assertEqual(normalize_tool_calls(None), [])
        self.assertEqual(normalize_tool_calls([{"bad": 1}]), [])


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpenAILLM:
    """只提供 chat_with_tools 需要的属性，不发真实请求。"""

    provider = "openai"
    model = "fake-model"
    openai_base = "http://fake.local/v1"
    openai_key = "sk-test"
    openai_model = "fake-model"
    ollama_url = "http://fake.local"
    timeout = 1.0
    temperature = 0.1
    calls = 0
    failures = 0
    last_error = None


class TestOpenAICompatiblePassthrough(ApiCase):
    """证明「工具链真能透传」：OpenAI 兼容响应 → 解析 tool_calls → 执行工具。"""

    def test_openai_tool_calls_reach_dispatcher(self):
        first = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "rosmontis_remember",
                            "arguments": json.dumps(
                                {"what": "透传测试：他今天没吃饭", "next": "先问饿不饿，别问想吃什么"},
                                ensure_ascii=False,
                            ),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
        second = {
            "choices": [{
                "message": {"role": "assistant", "content": "记好了。"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 140, "completion_tokens": 8, "total_tokens": 148},
        }
        responses = [first, second]
        payloads: list[dict] = []
        fake_llm = FakeOpenAILLM()

        def fake_urlopen(req, timeout=None):
            payloads.append(json.loads(req.data.decode("utf-8")))
            return FakeHTTPResponse(responses.pop(0))

        def llm_callable(messages, tools=None):
            # 与 service/app.py 完全同一条构造路径
            return make_llm_callable(fake_llm, clean=lambda s: s.strip(), max_tokens=400)(messages, tools)

        h, db_path = self.mk("openai", llm_callable=llm_callable)
        with mock.patch("rosmontis.harness.urllib.request.urlopen", side_effect=fake_urlopen):
            r = h.turn("u1", "c1", "我今天没吃饭")

        self.assertEqual(r["reply"], "记好了。")
        self.assertEqual(r["rounds"], 2)
        self.assertEqual(r["finish_reason"], "stop")
        self.assertEqual(r["usage"]["prompt_tokens"], 240)
        self.assertEqual(len(r["tools_called"]), 1)
        self.assertEqual(r["tools_called"][0]["tool"], "rosmontis_remember")
        self.assertTrue(r["tools_called"][0]["ok"], r["tools_called"][0]["reason"])
        self.assertEqual(r["tools_called"][0]["state"], "note")

        # 首次请求：带 tools 与 tool_choice，说明底层真的拿到了工具协议
        self.assertEqual(len(payloads[0]["tools"]), 5)
        self.assertEqual(payloads[0]["tool_choice"], "auto")

        # 第二次请求：assistant.tool_calls 与 tool.tool_call_id 对得上（OpenAI 硬要求）
        assistant = [m for m in payloads[1]["messages"] if m.get("role") == "assistant" and m.get("tool_calls")][-1]
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_abc")
        self.assertIn("透传测试", assistant["tool_calls"][0]["function"]["arguments"])
        last = payloads[1]["messages"][-1]
        self.assertEqual(last["role"], "tool")
        self.assertEqual(last["tool_call_id"], "call_abc")
        self.assertEqual(last["name"], "rosmontis_remember")

        with DB.reader(db_path) as conn:
            rows = conn.execute("SELECT text FROM episode WHERE source='notebook'").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertIn("透传测试", rows[0]["text"])

    def test_make_llm_callable_fallback_text_channel(self):
        """上游没有工具能力时：安静降级为纯文本通道，tool 轮被压成可读对话。"""

        class PlainLLM:
            provider = "rule"
            timeout = 1.0
            temperature = 0.1

            def __init__(self):
                self.seen = None

            def chat(self, system, user, max_tokens=700, json_mode=False):
                self.seen = (system, user)
                return "（保持自然回应）纯文本回复"

        llm = PlainLLM()
        call = make_llm_callable(llm, clean=lambda s: s.replace("（保持自然回应）", "").strip())
        messages = [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "你好"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "rosmontis_recall", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "rosmontis_recall", "content": "成功：found=True"},
        ]
        resp = call(messages, [{"type": "function"}])
        self.assertEqual(resp["tool_calls"], [])
        self.assertEqual(resp["usage"], {})
        self.assertIsNone(resp["finish_reason"])
        self.assertEqual(resp["content"], "纯文本回复")
        self.assertIn("系统提示", llm.seen[0])
        self.assertIn("（工具结果）成功", llm.seen[1])
        self.assertIn("rosmontis_recall", llm.seen[1])

    def test_chat_with_tools_quiet_degrade(self):
        class NoToolsLLM:
            provider = "rule"
            timeout = 1.0
            temperature = 0.1

        self.assertIsNone(chat_with_tools(NoToolsLLM(), [{"role": "user", "content": "hi"}], []))

        class NoKeyLLM:
            provider = "openai"
            openai_base = "http://fake.local/v1"
            openai_key = ""
            timeout = 1.0

        self.assertIsNone(chat_with_tools(NoKeyLLM(), [{"role": "user", "content": "hi"}], []))
        self.assertIsNone(chat_with_tools(None, [], []))


class TestFourHooks(ApiCase):
    def test_four_hooks_run_through(self):
        record = RecordingLLM([{"content": "嗯。"}])
        h, db_path = self.mk("hooks", llm_callable=record)
        space, conv = "u1", "c1"

        # session_start
        w = h.wake(space, conv)
        self.assertTrue(w["ok"])

        # pre_prompt：纯函数
        before = db_counts(db_path)
        ctx = h.context(space, conv, "他最近怎么样", budget=400)
        after = db_counts(db_path)
        self.assertLessEqual(ctx["memory_tokens"], 400)
        self.assertEqual(before["retrieval_log"], after["retrieval_log"])

        # post_turn：只记录，不生成
        obs = h.observe(space, conv, "今天挺累的", "那早点休息。")
        self.assertEqual(obs["written"], ["user", "assistant"])
        h.mem.wait_enrich()

        # session_end：立即返回 job_id，后台完成
        t0 = time.perf_counter()
        job = h.settle(space, conv)
        accept_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(accept_ms, 500, "settle 必须立即返回，不允许同步等结算")
        self.assertTrue(job["job_id"].startswith("job_"))
        done = h.wait(job["job_id"], timeout=60)
        self.assertEqual(done["state"], "done")
        self.assertIn("steps", done["result"] or {})
        self.assertGreaterEqual(db_counts(db_path)["episode"], 2)

    def test_settle_deduplicates_and_job_errors(self):
        h, _ = self.mk("settlejoin")
        gate = threading.Event()
        h.harness.sleep = lambda sid: (gate.wait(10), {"session_id": sid, "steps": []})[1]
        try:
            j1 = h.settle("u1", "c1")
            j2 = h.settle("u1", "c1")
            self.assertEqual(j1["job_id"], j2["job_id"])
            self.assertTrue(j2["deduplicated"])
            gate.set()
            self.assertEqual(h.wait(j1["job_id"], timeout=10)["state"], "done")
        finally:
            gate.set()
            del h.harness.sleep
        with self.assertRaises(api.RosmontisError) as cm:
            h.job("job_nope")
        self.assertEqual(cm.exception.code, "not_found")


class TestPackageSurface(unittest.TestCase):
    def test_lazy_exports(self):
        self.assertEqual(rosmontis.api_version, "1")
        self.assertEqual(api.__version__, "1")
        self.assertTrue(callable(rosmontis.__getattr__("open")))
        self.assertTrue(callable(rosmontis.__getattr__("Harness")))
        self.assertTrue(callable(rosmontis.__getattr__("dispatch")))
        self.assertEqual(len(rosmontis.__getattr__("TOOLS")), 5)
        with self.assertRaises(AttributeError):
            rosmontis.__getattr__("nope")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    print("=" * 60)
    print(f"共 {res.testsRun} 项 · 失败 {len(res.failures)} · 错误 {len(res.errors)}")
    sys.exit(0 if res.wasSuccessful() else 1)
