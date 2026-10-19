"""四钩子接入示例：session_start / pre_prompt / post_turn / session_end。

跑法（不需要网络、不需要模型）：

    python examples/hooks_demo.py

这个示例演示的是 **宿主自己拥有对话循环** 的接入形态：
宿主每轮按四个钩子调 Alaya，而不是把整个循环交给 Harness。
钩子契约见 docs/_wip/A3-产品形态与接入.md §3.1。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alaya  # noqa: E402
from alaya.core import db as DB  # noqa: E402


class FakeLLM:
    """假模型：第一轮发起一次工具调用，拿到工具结果后正常收尾。

    故意用扁平 `{"name", "arguments"}` 写法——harness 会把它归一成
    OpenAI 兼容格式并补 `tool_call_id`。返回值里带 usage / finish_reason，
    验证这两个字段缺省也不会崩。
    """

    def __init__(self):
        self.calls = 0
        self.saw_history = False
        self.saw_tool_call_id = False

    def __call__(self, messages, tools=None):
        self.calls += 1
        if any(m.get("content") == "昨天我加班到很晚" for m in messages if m.get("role") == "user"):
            self.saw_history = True

        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if not tool_msgs:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "name": "alaya_remember",
                        "arguments": {
                            "what": "他说最近累，但不是身体累",
                            "noticed": "说到『加班』时语气很轻，像早就认了",
                            "means": "他可能在硬撑，不想让人担心",
                            "next": "先别劝他休息，问他最近有没有睡够",
                            "topic": "状态",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 180, "completion_tokens": 64},
                "finish_reason": "tool_calls",
            }

        self.saw_tool_call_id = bool(tool_msgs[-1].get("tool_call_id"))
        return {
            "content": "嗯，我记下了。你今天先早点睡，别想太多。",
            "tool_calls": [],
            "usage": {"prompt_tokens": 240, "completion_tokens": 18},
            "finish_reason": "stop",
        }


def counts(db_path: str) -> dict:
    out = {}
    with DB.reader(db_path) as conn:
        for t in ("episode", "fact", "persona", "retrieval_log", "notebook_promotion"):
            try:
                out[t] = int(conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"])
            except Exception:
                out[t] = -1
    return out


def main() -> int:
    work = tempfile.mkdtemp(prefix="alaya_hooks_demo_")
    db_path = os.path.join(work, "hooks.db")
    space, conversation = "u1", "c1"
    llm = FakeLLM()
    h = alaya.open(
        db_path,
        {
            "embed_provider": "hash",   # 不联网
            "llm_hot": "rule",          # 内存/抽取走规则
            "llm_deep": "rule",
            "log_level": "ERROR",
            "llm_callable": llm,
            "memory_budget": 700,
        },
    )

    print("=" * 64)
    print("Alaya 四钩子接入示例 · session_start → pre_prompt → post_turn → session_end")
    print("=" * 64)

    try:
        # ── ① session_start ──────────────────────────────
        w = h.wake(space, conversation)
        print("\n[session_start] 醒来")
        print(f"  space={space} conversation={conversation} session={w['session_id']}")
        print(f"  人格 {w['persona_chars']}/{w['persona_budget']} 字（{w['persona_items']} 条）")
        print(f"  常驻 {w['resident_tokens']} token · 锦囊积压 {w['inbox']['count']} 条 · 降级={w['degraded']}")
        print(f"  人格首句：{(w['persona_text'] or '').splitlines()[0] if w['persona_text'] else '（空）'}")

        # ── ② pre_prompt ─────────────────────────────────
        before = counts(db_path)
        ctx = h.context(space, conversation, "他最近状态怎么样", budget=400)
        after = counts(db_path)
        print("\n[pre_prompt] 取注入上下文（纯函数：不写库、不超预算）")
        print(f"  system_prefix {ctx['persona_tokens']} token · memory {ctx['memory_tokens']}/{ctx['budget']} token")
        print(f"  abstained={ctx['abstained']} · depth={ctx['parts']['memory'].get('depth')}")
        print(f"  memory_text：{(ctx['memory_text'] or '（本轮没有相关记忆）').replace(chr(10), ' / ')[:120]}")
        unchanged = all(before[t] == after[t] for t in ("episode", "retrieval_log"))
        print(f"  写库检查：episode {before['episode']}→{after['episode']}，retrieval_log "
              f"{before['retrieval_log']}→{after['retrieval_log']} → {'未写库 ✓' if unchanged else '写了库 ✗'}")
        if not unchanged:
            print("  [失败] context 必须是纯函数")
            return 1
        if ctx["memory_tokens"] > (ctx["budget"] or 0):
            print("  [失败] context 超预算")
            return 1

        # ── ③ post_turn（一轮完整对话 + 工具链）───────────
        history = [
            {"role": "user", "content": "昨天我加班到很晚"},
            {"role": "assistant", "content": "那今天要早点休息"},
        ]
        r = h.turn(space, conversation, "今天还是有点累", history=history)
        print("\n[post_turn] 一轮对话（Harness 管循环的形态，用于演示工具链）")
        print(f"  history 已带入：{'是' if llm.saw_history else '否'}")
        print(f"  她：{r['reply']}")
        print(f"  轮次={r['rounds']} · usage={r['usage']} · finish_reason={r['finish_reason']}")
        for t in r["tools_called"]:
            print(f"  工具 {t['tool']} → ok={t['ok']} state={t['state']} "
                  f"from={t['from'] if 'from' in t else ''} tool_call_id={t['tool_call_id']}")
        print(f"  工具轮补 tool_call_id：{'是' if llm.saw_tool_call_id else '否'}")

        # 下一轮的 pre_prompt：这一轮开始真的能注入刚才记下的东西
        ctx2 = h.context(space, conversation, "他最近累不累", budget=400)
        print("\n[pre_prompt / 第二轮] 同一对 space+conversation 已经能想起刚才写下的事")
        print(f"  memory {ctx2['memory_tokens']}/{ctx2['budget']} token · abstained={ctx2['abstained']}")
        print(f"  memory_text：{(ctx2['memory_text'] or '（空）').replace(chr(10), ' / ')[:160]}")
        if ctx2["memory_tokens"] > 0 and not ctx2["abstained"]:
            print(f"  归因：工具写下的笔记被下一轮上下文检索到（命中 "
                  f"{ctx2['parts']['memory'].get('items')} 条）")
        else:
            print("  [提示] 本轮没有命中（hash 伪向量 + 短查询的正常表现，不视为失败）")

        # 宿主自己管循环时，post_turn 只调 observe（不生成回答）
        obs = h.observe(space, conversation, "我先去睡了", "好，晚安，明早记得喝水。")
        dup = h.observe(space, conversation, "我先去睡了", "好，晚安，明早记得喝水。")
        print("\n[post_turn / observe] 宿主回写一来一回")
        print(f"  第一次写入：{obs['written']} · 第二次幂等={dup['idempotent']}")

        # ── ④ session_end ────────────────────────────────
        t0 = time.perf_counter()
        job = h.settle(space, conversation)
        accept_ms = (time.perf_counter() - t0) * 1000
        print("\n[session_end] 打烊（立即返回 job_id，不挂住宿主退出流程）")
        print(f"  job_id={job['job_id']} · 受理耗时 {accept_ms:.1f}ms · queued={job['queued']}")
        done = h.wait(job["job_id"], timeout=60)
        steps = (done.get("result") or {}).get("steps")
        print(f"  状态={done['state']} · 完成动作={steps} · 后台耗时 "
              f"{(done.get('finished_at') or 0) - (done.get('created_at') or 0):.2f}s")
        if done["state"] != "done":
            print(f"  [失败] 打烊任务未完成：{done.get('error')}")
            return 1

        # ── 收尾体检 ─────────────────────────────────────
        final = counts(db_path)
        print("\n最终库状态")
        print(f"  episode={final['episode']} fact={final['fact']} persona={final['persona']} "
              f"retrieval_log={final['retrieval_log']}")
        note_states = h.context(space, conversation, "他累不累", budget=300)
        print(f"  打烊后仍可检索：memory_tokens={note_states['memory_tokens']} "
              f"abstained={note_states['abstained']}")
        print("\n四个钩子全部跑通 ✓")
        return 0
    finally:
        h.close()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
