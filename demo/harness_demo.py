"""Harness 演示：一个完整的对话循环，不需要任何真实模型。

跑法：python demo/harness_demo.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alaya import Config
from alaya.harness import Harness


class ScriptedLLM:
    """按剧本走的假模型。用"做过什么"驱动，不用轮次计数——
    一次 turn 里可能调两次模型，用计数会错位。
    """

    def __init__(self):
        import re

        self._re = re
        self.done: set[str] = set()
        self.note_id: str | None = None
        self.stage = 0

    def __call__(self, messages, tools):
        last = messages[-1]

        # 工具结果回来了：从里面找笔记 id，然后往下走
        if last.get("role") == "tool":
            content = last.get("content", "")
            m = self._re.search(r"id=(ep_[\w]+)", content)
            if m:
                self.note_id = m.group(1)
            return {"content": "", "tool_calls": []}

        # 普通对话
        if self.stage < 3:
            self.stage += 1
            return {"content": "嗯，我在听。", "tool_calls": []}

        if "alaya_remember" not in self.done:
            self.done.add("alaya_remember")
            return {
                "content": "",
                "tool_calls": [
                    {
                        "name": "alaya_remember",
                        "arguments": {
                            "what": "他说他换工作了",
                            "noticed": "说的时候没高兴，倒像松了口气",
                            "means": "他可能不是想换，是想逃",
                            "next": "下次提新公司，别问好不好，问累不累",
                            "topic": "工作",
                        },
                    }
                ],
            }

        if "alaya_recall" not in self.done:
            self.done.add("alaya_recall")
            return {"content": "", "tool_calls": [{"name": "alaya_recall", "arguments": {"query": "他工作的事"}}]}

        if "alaya_promote" not in self.done and self.note_id:
            self.done.add("alaya_promote")
            return {
                "content": "",
                "tool_calls": [
                    {"name": "alaya_promote", "arguments": {"note_id": self.note_id, "section": "trait"}}
                ],
            }

        return {"content": "今天先到这儿吧。", "tool_calls": []}


def main() -> int:
    db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_demo_harness.db")
    if os.path.exists(db):
        for suf in ("", "-wal", "-shm"):
            p = db + suf
            if os.path.exists(p):
                os.remove(p)

    cfg = Config(db_path=db, embed_provider="hash", llm_hot="rule", llm_deep="rule", log_level="ERROR")
    llm = ScriptedLLM()
    h = Harness(cfg, llm, auto_tick=False)

    print("=" * 60)
    print("Harness 演示 · 醒 → 一息 × N → 眠")
    print("=" * 60)

    w = h.wake("demo")
    print(f"\n[醒] 人格 {w['persona_chars']}/{w['persona_budget']} 字 · 锦囊 {w['inbox']['count']} 条")

    lines = [
        "我今天有点累",
        "其实我最近在考虑换工作",
        "不太确定要不要走",
        "算了先不聊这个",
    ]
    for i, text in enumerate(lines):
        r = h.turn(text, "demo")
        flag = f" 工具 {[t['tool'] for t in r['tools_called']]}" if r["tools_called"] else ""
        print(f"\n[一息 {i+1}] 用户：{text}")
        print(f"         她：{r['reply']}{flag}")
        print(f"         token {r['memory_used']['memory_tokens']} · 深度 {r['memory_used']['depth']}")

    # 继续推进，让剧本把三个工具都演一遍
    for i in range(4):
        r = h.turn("（继续）", "demo")
        for t in r["tools_called"]:
            print(f"\n[工具] {t['tool']} → {'成功' if t['ok'] else '被拒：' + t['reason']}")
        h.join(timeout=3)

    print("\n" + "-" * 60)
    print("锦囊与笔记")
    st = h.mem.inbox("demo")
    print(f"  速记 {st['count']} 条 · 该整理：{st['should_review']}")
    notes = h.mem.notebook_list("demo")
    for n in notes[:3]:
        print(f"  [{n['tier']}] {n['what'][:30]}")
        if n["next"]:
            print(f"         下次：{n['next']}")

    print("\n" + "-" * 60)
    print("[眠]")
    s = h.sleep("demo")
    print(f"  动作：{s.get('steps')}")
    if s.get("close"):
        c = s["close"]
        print(f"  峰终：峰值情绪 {c.get('peak', {}).get('emotion')} · 触及 {c.get('touched')} 条")
    if s.get("dream"):
        d = s["dream"]
        print(f"  梦：重放 {d.get('replayed')} 条 · 抽象 {d.get('abstracts')} 条 · 脱敏 {d.get('decoupled')} 条")
        if d.get("body"):
            print(f"      「{d['body'][:80]}」")
    elif s.get("dream_skipped"):
        print(f"  没做梦：{s['dream_skipped']}")

    print("\n" + "-" * 60)
    print("人格（升格后）")
    p = h.mem.persona_compile("demo")
    print(f"  {p['chars']}/{p['budget']} 字，{p['included']} 条")
    if p["text"]:
        print("  " + p["text"].replace("\n", "\n  ")[:300])

    print("\n" + "=" * 60)
    print("演示结束。整个流程只用了 4 行接入代码：wake / turn / sleep。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
