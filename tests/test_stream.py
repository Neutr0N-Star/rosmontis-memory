"""流式接口测试：`Harness.turn_stream` 的事件顺序、内容一致性与写回时机。

运行：PYTHONIOENCODING=utf-8 python tests/test_stream.py
不依赖网络与真实模型。
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
import tempfile
import unittest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosmontis.config import Config  # noqa: E402
from rosmontis.core import db as DB  # noqa: E402
from rosmontis.harness import Harness  # noqa: E402


class RecordingLLM:
    """按剧本返回的假模型。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def __call__(self, messages, tools=None):
        self.calls.append(copy.deepcopy(messages or []))
        if self.responses:
            return self.responses.pop(0)
        return {"content": "好。"}


def make_harness(tag: str, responses) -> tuple[Harness, str, str]:
    d = tempfile.mkdtemp(prefix=f"rosmontis_stream_{tag}_")
    db_path = os.path.join(d, "t.db")
    cfg = Config()
    cfg.db_path = db_path
    cfg.embed_provider = "hash"
    cfg.llm_hot = "rule"
    cfg.llm_deep = "rule"
    cfg.log_level = "ERROR"
    h = Harness(cfg, RecordingLLM(responses))
    return h, db_path, d


def episode_rows(db_path: str, session_id: str) -> list[dict]:
    with DB.reader(db_path) as conn:
        rows = conn.execute(
            "SELECT role, text FROM episode WHERE session_id=? ORDER BY t_ingest", (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


class StreamCase(unittest.TestCase):
    def cleanup(self, h: Harness, d: str) -> None:
        try:
            h.join(timeout=5)
        except Exception:
            pass
        DB.close_thread_conns()
        shutil.rmtree(d, ignore_errors=True)

    def test_event_order_and_reply_identity(self):
        h, db_path, d = make_harness("order", [{"content": "我记得你说过喜欢猫。"}])
        try:
            events = list(h.turn_stream("我喜欢猫", "u1"))
        finally:
            self.cleanup(h, d)

        names = [e["event"] for e in events]
        self.assertEqual(names[0], "meta")
        self.assertEqual(names[-1], "done")
        self.assertTrue(all(n == "delta" for n in names[1:-1]), names)

        text = "".join(e["data"]["text"] for e in events[1:-1])
        self.assertEqual(text, events[-1]["data"]["reply"])
        self.assertIn("persona_chars", events[0]["data"])
        self.assertIn("memory_tokens", events[0]["data"])

    def test_tool_event_then_reply(self):
        h, db_path, d = make_harness(
            "tool",
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "rosmontis_remember",
                            "arguments": {"what": "他说他搬到上海了", "noticed": "提到新城市"},
                        }
                    ],
                },
                {"content": "记住了，上海。"},
            ],
        )
        try:
            events = list(h.turn_stream("我搬到上海了", "u1"))
        finally:
            self.cleanup(h, d)

        names = [e["event"] for e in events]
        self.assertIn("tool", names)
        self.assertLess(names.index("tool"), names.index("done"))
        done = events[-1]["data"]
        self.assertEqual(done["reply"], "记住了，上海。")
        self.assertEqual(len(done["tools_called"]), 1)
        self.assertEqual(done["tools_called"][0]["tool"], "rosmontis_remember")

    def test_assistant_text_written_only_after_full_stream(self):
        h, db_path, d = make_harness("write", [{"content": "嗯，我在听。"}])
        try:
            gen = h.turn_stream("我有点累", "u1")
            next(gen)  # meta
            next(gen)  # 第一个 delta
            gen.close()  # 模拟客户端中途断开
            h.join(timeout=10)
            rows = episode_rows(db_path, "u1")
        finally:
            self.cleanup(h, d)
        roles = [r["role"] for r in rows]
        self.assertIn("user", roles, "用户输入应已落库")
        self.assertNotIn("assistant", roles, "断流后不得把半截回复当记忆")

    def test_full_stream_writes_both_sides(self):
        h, db_path, d = make_harness("full", [{"content": "那就别撑着。"}])
        try:
            events = list(h.turn_stream("我今天有点累", "u1"))
            h.join(timeout=10)
            rows = episode_rows(db_path, "u1")
        finally:
            self.cleanup(h, d)
        roles = [r["role"] for r in rows]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        self.assertIn("别撑着", "".join(r["text"] for r in rows))
        self.assertEqual(events[-1]["event"], "done")

    def test_empty_text_raises(self):
        h, db_path, d = make_harness("empty", [{"content": "x"}])
        try:
            with self.assertRaises(ValueError):
                list(h.turn_stream("   ", "u1"))
        finally:
            self.cleanup(h, d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
