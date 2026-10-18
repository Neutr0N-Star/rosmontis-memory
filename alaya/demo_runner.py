"""三个月对话演示：看记忆库怎么演化。

不依赖模型：用脚本化抽取结果，保证每次跑结果一致、可复现。
"""

from __future__ import annotations

import os
import shutil
import time

from .config import Config
from .core.memory import Memory

DAY = 86400.0

# (第几天, 说话人, 内容, 抽取结果)
SCRIPT = [
    (0, "user", "我叫晓星，在哈尔滨上学。", {
        "entities": [{"name": "晓星", "type": "person"}, {"name": "哈尔滨", "type": "place"}],
        "facts": [
            {"subject": "我", "predicate": "叫", "object": "晓星", "polarity": 1, "confidence": 0.95},
            {"subject": "我", "predicate": "住在", "object": "哈尔滨", "polarity": 1, "confidence": 0.9},
        ],
        "state_changes": [{"scope": "我", "slot": "city", "value": "哈尔滨"}],
        "theme": "初次见面",
    }),
    (1, "user", "我在一家小公司做后端，主要是写接口。", {
        "entities": [{"name": "小公司", "type": "org"}],
        "facts": [{"subject": "我", "predicate": "工作在", "object": "小公司", "polarity": 1, "confidence": 0.85}],
        "state_changes": [{"scope": "我", "slot": "job", "value": "小公司"}],
        "theme": "工作",
    }),
    (3, "user", "说实话我不太喜欢猫，主要是怕毛。", {
        "entities": [{"name": "猫", "type": "thing"}],
        "facts": [{"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": -1, "confidence": 0.9}],
        "state_changes": [],
        "theme": "偏好",
    }),
    (5, "user", "不过狗我可以，最好是大一点的。", {
        "entities": [{"name": "狗", "type": "thing"}],
        "facts": [{"subject": "我", "predicate": "喜欢", "object": "狗", "polarity": 1, "confidence": 0.8}],
        "state_changes": [],
        "theme": "偏好",
    }),
    (20, "user", "我换了工作，去了字节跳动，还是做后端。", {
        "entities": [{"name": "字节跳动", "type": "org"}],
        "facts": [{"subject": "我", "predicate": "工作在", "object": "字节跳动", "polarity": 1, "confidence": 0.95}],
        "state_changes": [{"scope": "我", "slot": "job", "value": "字节跳动"}],
        "theme": "工作变动",
    }),
    (35, "user", "今天特别难受，感觉什么都做不好。", {
        "entities": [], "facts": [], "state_changes": [], "theme": "情绪",
    }),
    (40, "user", "我养了一只狗，叫团子。", {
        "entities": [{"name": "团子", "type": "thing"}, {"name": "狗", "type": "thing"}],
        "facts": [{"subject": "我", "predicate": "有", "object": "团子", "polarity": 1, "confidence": 0.95}],
        "state_changes": [],
        "theme": "养狗",
    }),
    (60, "user", "我搬到上海了，新公司在那边。", {
        "entities": [{"name": "上海", "type": "place"}],
        "facts": [{"subject": "我", "predicate": "住在", "object": "上海", "polarity": 1, "confidence": 0.95}],
        "state_changes": [{"scope": "我", "slot": "city", "value": "上海"}],
        "theme": "搬家",
    }),
    (75, "user", "其实现在也能接受猫了，团子跟猫玩得挺好。", {
        "entities": [{"name": "猫", "type": "thing"}],
        "facts": [{"subject": "我", "predicate": "喜欢", "object": "猫", "polarity": 1, "confidence": 0.8}],
        "state_changes": [],
        "theme": "偏好变化",
    }),
]

QUERIES = [
    ("我住在哪", "问当前状态"),
    ("我喜欢猫吗", "问偏好（中间变过）"),
    ("我养了什么", "问拥有的东西"),
    ("我换过工作吗", "问变化"),
]


def run_demo(args) -> int:
    db = getattr(args, "db", None) or "demo.db"
    if getattr(args, "reset", False) and os.path.exists(db):
        for suffix in ("", "-wal", "-shm"):
            p = db + suffix
            if os.path.exists(p):
                os.remove(p)

    cfg = Config(db_path=db, embed_provider="hash", llm_hot="rule", llm_deep="rule", log_level="ERROR")
    m = Memory(cfg)

    print("=" * 64)
    print("藏识 Alaya · 三个月对话演示")
    print(f"数据库：{os.path.abspath(db)}")
    print("=" * 64)

    base = time.time() - 90 * DAY
    fake = _FakeLLM()


    m.llm_hot = fake

    for day, role, text, payload in SCRIPT:
        fake.queue(payload)
        t = base + day * DAY
        r = m.ingest(text, session_id="demo", role=role, t_event=t)
        mark = "+" if r["added"] else ("~" if r["updated"] else "·")
        print(f"  D{day:>3} {mark} {text[:38]:<40} 新增 {r['added']} 改 {r['updated']}")

    print("\n" + "-" * 64)
    print("① 当前状态快照")
    snap = m.state_snapshot("我")
    for s in snap["slots"]:
        print(f"   {s['slot']} = {s['value']}")

    print("\n② 时间旅行：站在第 30 天看")
    past = m.state_snapshot("我", as_of=base + 30 * DAY)
    for s in past["slots"]:
        print(f"   {s['slot']} = {s['value']}")

    print("\n③ 事实的完整历史（注意失效的仍然留着）")
    for f in m.facts("demo", status="*"):
        flag = "有效" if f["status"] == "active" else "已失效"
        neg = "不" if f["polarity"] < 0 else ""
        print(f"   [{flag}] 我 {neg}{f['predicate']} {f['object']}")

    print("\n④ 生成常驻活文档")
    m.rebuild_pages("demo")
    block = m.resident_block("demo")
    print(f"   {block['tokens']} token / {block['pages']} 页")
    print("   " + block["text"].replace("\n", "\n   ")[:600])

    print("\n⑤ 检索四问")
    for q, why in QUERIES:
        r = m.recall(q, session_id="demo", budget=400)
        head = "拒答" if r["abstained"] else f"{len(r['items'])} 条 / {r['tokens_used']}t / {r['latency_ms']}ms"
        print(f"\n   「{q}」（{why}）→ {head}")
        if not r["abstained"]:
            for i in r["items"][:4]:
                print(f"     {i['score']:.2f} L{i['level']} {i['kind']:<8} {i['text'][:56]}")

    print("\n⑥ 体检")
    a = m.audit("demo")
    h = a["health"]
    print(f"   健康度 {h['score']}/100（{h['label']}）  高危 {h['high']} 中危 {h['medium']}")
    for i in a["issues"][:4]:
        print(f"   - [{i['severity']}] {i['detail']}")

    print("\n" + "=" * 64)
    print("演示结束。想看界面：python run.py serve --port 8770")
    print("=" * 64)
    return 0


class _FakeLLM:
    """脚本化抽取器：按预设顺序返回结果，保证演示可复现。"""

    provider = "fake"
    model = "scripted"
    calls = 0
    failures = 0

    def __init__(self):
        self.script = []

    def queue(self, payload):
        self.script.append(payload)

    def chat_json(self, system, user, max_tokens=700):
        self.calls += 1
        return self.script.pop(0) if self.script else {"entities": [], "facts": [], "state_changes": [], "theme": None}

    def chat(self, system, user, max_tokens=700, json_mode=False):
        return ""

    def available(self):
        return True
