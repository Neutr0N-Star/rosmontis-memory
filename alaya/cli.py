"""命令行入口。

python -m alaya.cli <命令>
或者 python run.py <命令>
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .config import Config
from .core.memory import Memory


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("缺少 uvicorn。装一下：pip install uvicorn fastapi")
        return 1
    print(f"藏识 Alaya 启动中 → http://{args.host}:{args.port}")
    print(f"接口文档 → http://{args.host}:{args.port}/docs")
    uvicorn.run("alaya.service.app:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


def cmd_ingest(args) -> int:
    m = Memory(Config())
    r = m.ingest(args.text, session_id=args.session, role=args.role)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


def cmd_recall(args) -> int:
    m = Memory(Config())
    r = m.recall(args.query, session_id=args.session, budget=args.budget)
    if args.raw:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0
    print(f"\n查询：{r['query']}")
    print(f"耗时 {r['latency_ms']}ms ｜ token {r['tokens_used']}/{r['budget']} ｜ 条目 {len(r['items'])}")
    if r["abstained"]:
        print(f"\n[拒答] {r['reason']}")
        return 0
    print("\n" + "-" * 56)
    print(r["prompt"])
    print("-" * 56)
    print("\n打分明细：")
    for i in r["items"]:
        print(f"  {i['score']:.3f}  L{i['level']}  {i['tokens']:>3}t  {i['kind']:<8} {i['text'][:60]}")
    if r["excluded"]:
        print("\n被排除：")
        for e in r["excluded"][:6]:
            print(f"  - {e['kind']} {e['id'][:14]} … {e['why']}")
    return 0


def cmd_stats(args) -> int:
    m = Memory(Config())
    print(json.dumps(m.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_audit(args) -> int:
    m = Memory(Config())
    d = m.audit(args.session)
    h = d["health"]
    print(f"\n健康度 {h['score']}/100（{h['label']}）  高危 {h['high']} 中危 {h['medium']}\n")
    print("规模：", json.dumps(d["counts"], ensure_ascii=False))
    if not d["issues"]:
        print("\n没有发现问题。")
        return 0
    print("\n问题清单：")
    for i in d["issues"]:
        print(f"  [{i['severity']:>6}] {i['type']}: {i['detail']}")
        print(f"           → {i['hint']}")
    return 0


def cmd_forget(args) -> int:
    m = Memory(Config())
    r = m.forget(args.session)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


def cmd_entities(args) -> int:
    m = Memory(Config())
    rows = m.entities(args.session, args.type, args.limit)
    if not rows:
        print("没有实体。")
        return 0
    print(f"{'名字':<16}{'类型':<10}{'事实':<6}{'提及':<6}最近出现")
    for r in rows:
        print(f"{r['name'][:15]:<16}{r['type']:<10}{r['fact_count']:<6}{r['mentions']:<6}{_fmt_ts(r['last_seen'])}")
    return 0


def cmd_state(args) -> int:
    m = Memory(Config())
    d = m.state_snapshot(args.scope, args.as_of)
    print(f"\n{d['scope']} 的状态（截至 {_fmt_ts(d['as_of'])}）")
    if not d["slots"]:
        print("  没有记录")
    for s in d["slots"]:
        print(f"  {s['slot']:<14} = {s['value']}   （自 {_fmt_ts(s['since'])}）")
    return 0


def cmd_timeline(args) -> int:
    m = Memory(Config())
    d = m.entity_detail(args.entity_id)
    if not d:
        print("实体不存在")
        return 1
    print(f"\n{d['entity']['name']} 的事实时间线")
    for f in reversed(d["facts"]):
        mark = "✓" if f["status"] == "active" else "×"
        print(f"  {mark} [{_fmt_ts(f['valid_from'])}] {f['predicate']} {f['object']}")
        if f["invalid_at"]:
            print(f"      于 {_fmt_ts(f['invalid_at'])} 失效")
    return 0


def cmd_demo(args) -> int:
    from .demo_runner import run_demo

    return run_demo(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alaya", description="藏识 Alaya — 长期记忆系统")
    p.add_argument("--db", default=None, help="数据库路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="启动 HTTP 服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("ingest", help="写入一句话")
    s.add_argument("text")
    s.add_argument("--session", default="default")
    s.add_argument("--role", default="user")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("recall", help="检索")
    s.add_argument("query")
    s.add_argument("--session", default=None)
    s.add_argument("--budget", type=int, default=None)
    s.add_argument("--raw", action="store_true", help="输出完整 JSON")
    s.set_defaults(func=cmd_recall)

    s = sub.add_parser("stats", help="库统计")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("audit", help="体检")
    s.add_argument("--session", default=None)
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("forget", help="跑一次遗忘周期")
    s.add_argument("--session", default=None)
    s.set_defaults(func=cmd_forget)

    s = sub.add_parser("entities", help="实体列表")
    s.add_argument("--session", default=None)
    s.add_argument("--type", default=None)
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(func=cmd_entities)

    s = sub.add_parser("state", help="某人当前状态")
    s.add_argument("scope")
    s.add_argument("--as-of", type=float, default=None)
    s.set_defaults(func=cmd_state)

    s = sub.add_parser("timeline", help="实体事实时间线")
    s.add_argument("entity_id")
    s.set_defaults(func=cmd_timeline)

    s = sub.add_parser("demo", help="跑三个月对话演示")
    s.add_argument("--db", default="demo.db", help="演示专用库，避免污染主库")
    s.add_argument("--reset", action="store_true")
    s.set_defaults(func=cmd_demo, db="demo.db")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "db", None):
        import os

        os.environ["ALAYA_DB"] = args.db
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
