"""一键质量评测：抽取 P/R/F1、召回 nDCG@10、四类准确率、拒答误召/漏召、token、延迟。

用法
    PYTHONIOENCODING=utf-8 python bench/run_eval.py
    PYTHONIOENCODING=utf-8 python bench/run_eval.py --field off --out bench/results/field_off.json
    PYTHONIOENCODING=utf-8 python bench/run_eval.py --backend rule      # 强制规则兜底
    PYTHONIOENCODING=utf-8 python bench/run_eval.py --backend llm       # 有 Ollama 才真跑

约束
- 只读 `alaya/`；评测库建在系统临时目录，跑完删除；不碰 `alaya.db`。
- 缺 Ollama / 缺网络不报错：自动回退「脚本化金标 + 规则抽取 + 哈希向量」，并在
  JSON 的 ``meta.extraction_backend`` / ``meta.embed_backend`` 里显式标注降级。
- 所有数字由本脚本产出，禁止手工填写。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import common as C


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="alaya-memory 质量基准评测")
    ap.add_argument("--backend", choices=["scripted", "rule", "llm"], default="scripted",
                    help="抽取后端；默认 scripted=脚本化金标（确定性，用于隔离检索链路）")
    ap.add_argument("--field", choices=["on", "off"], default=None,
                    help="语义场开关；默认读环境变量 ALAYA_FIELD（缺省 on）")
    ap.add_argument("--embed", choices=["hash", "ollama"], default="hash",
                    help="embedding 后端；默认 hash（无 Ollama 时的确定性降级）")
    ap.add_argument("--budget", type=int, default=1400, help="检索注入 token 预算（默认 1400）")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "latest.json"), help="JSON 输出路径")
    ap.add_argument("--limit", type=int, default=0, help="冒烟用：只写入前 N 段对话（0=全部）")
    ap.add_argument("--extraction-limit", type=int, default=0, help="抽取质量最多评测多少段（0=全部）")
    ap.add_argument("--regen", action="store_true", help="重新生成数据集后评测")
    ap.add_argument("--quiet", action="store_true", help="只写 JSON，少打印")
    return ap.parse_args(argv)


def _resolve_field(value: str | None) -> str:
    if value:
        return value
    env = os.environ.get("ALAYA_FIELD", "on").strip().lower()
    return "off" if env in ("0", "false", "off", "no") else "on"


def _print_summary(payload: dict) -> None:
    meta = payload["meta"]
    ds = payload["dataset"]
    ext = payload["extraction"]
    wf = payload["write_fidelity"]
    ret = payload["retrieval"]
    print("=" * 72)
    print("alaya-bench · 质量评测")
    print("=" * 72)
    print(f"数据集        : {ds['dialogues']} 段对话 / {ds['turns']} turn / {ds['questions']} 题（种子 {ds['seed']}）")
    print(f"抽取后端      : {meta['extraction_backend']}  ·  embedding: {meta['embed_backend']}")
    print(f"语义场        : ALAYA_FIELD={meta['field_env']}（原生开关 {meta['field_backend']['native_switch']}）")
    print(f"核心版本      : git {str(meta['code'].get('git_head') or '?')[:10]} · dirty core {len(meta['code'].get('dirty_core_files') or [])} 个")
    print("-" * 72)
    print("[抽取质量] 规则兜底（离线真实能力）")
    rule = ext["offline_rule"]
    for name in ("entity", "fact", "state_slot"):
        m = rule[name]
        print(f"  {name:<10} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
              f"  (pred={m['predicted']} gold={m['gold']} match={m['matched']})")
    active = ext["active"]
    am = active["fact"]
    print(f"[抽取质量] 当前后端 {active['backend']}：fact F1={am['f1']:.3f}（P={am['precision']:.3f} R={am['recall']:.3f}）")
    print(f"[写入保真] 脚本化金标落库：active fact P={wf['fact']['precision']:.3f} R={wf['fact']['recall']:.3f}"
          f" F1={wf['fact']['f1']:.3f}；当前槽命中 {wf['state_slot_current']['matched']}/{wf['state_slot_current']['expected']}")
    print("-" * 72)
    print("[召回质量]")
    print(f"  nDCG@10            : {ret['ndcg_at_10']:.4f}")
    for cat in ("temporal", "multi_hop", "update", "preference"):
        c = ret["by_category"][cat]
        print(f"  {cat:<18}: n={c['n']:<3} acc={c['accuracy']:.3f} nDCG@10={c['ndcg_at_10']:.4f}"
              f" hit@10={c['hit_at_10']:.3f} 漏召={c['miss_rate']:.3f}")
    ab = ret["abstain"]
    print(f"  拒答               : n={ab['n']} 误召率={ab['false_recall_rate']:.3f} 拒答精度={ab['precision']:.3f}")
    print(f"  可作答漏召率       : {ret['answerable']['miss_rate']:.3f} · 可作答 acc={ret['answerable']['accuracy']:.3f}")
    print("-" * 72)
    print("[成本]")
    tk, la = ret["tokens"], ret["latency_ms"]
    print(f"  注入 token         : mean={tk['mean']:.1f} p50={tk['p50']:.1f} p95={tk['p95']:.1f} max={tk['max']} 预算={tk['budget']}")
    print(f"  检索延迟           : mean={la['mean']:.2f}ms p50={la['p50']:.2f}ms p95={la['p95']:.2f}ms")
    print(f"  构建/错误          : DB {meta['build_elapsed_s']}s · 题目错误 {ret['errors']} · 写入错误 {meta['build_errors']}")
    print("-" * 72)
    print(f"JSON → {payload['meta']['out_path']}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    field = _resolve_field(args.field)
    data = C.load_dataset()
    fingerprint = C.dataset_fingerprint()
    t_build = time.perf_counter()
    db = C.build_eval_db(data, backend=args.backend, embed_provider=args.embed, limit=args.limit)
    build_elapsed = round(time.perf_counter() - t_build, 3)
    try:
        ec = C.EvalConfig(
            name="product_default",
            field=field,
            gamma=None,
            hops=None,
            budget=int(args.budget),
            uniform_depth=False,
            note="产品默认反射深度与预算倍率，只按 ALAYA_FIELD 切换场",
        )
        ext_active = C.measure_extraction(
            data["conversations"],
            backend=db.backend,
            limit=(args.extraction_limit or (20 if db.backend == "llm" else 0)),
        )
        ext_rule = ext_active if db.backend == "rule" else C.measure_extraction(
            data["conversations"], backend="rule", limit=args.extraction_limit
        )
        write_fidelity = C.measure_write_fidelity(db.cfg, data["sessions"])
        metrics, results, applied = C.evaluate_config(db, data, ec)
    finally:
        db.close()

    payload = {
        "ok": metrics["errors"] == 0,
        "meta": {
            "bench": data["manifest"]["name"],
            "seed": data["manifest"]["seed"],
            "run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "out_path": str(Path(args.out).resolve()),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "build_elapsed_s": build_elapsed,
            "build_errors": db.build.get("errors", 0),
            "extraction_backend": db.backend,
            "extraction_note": db.backend_note,
            "embed_backend": args.embed,
            "embed_health": db.build.get("embed_health"),
            "field_env": field,
            "field_backend": C.field_backend_info(),
            "environment": C.environment_info(),
            "code": C.code_fingerprint(),
            "db_build": db.build,
            "config": ec.to_dict(),
            "config_applied": applied,
            "command": "PYTHONIOENCODING=utf-8 python bench/run_eval.py"
            + (f" --field {field}" if args.field else "")
            + (f" --backend {args.backend}" if args.backend != "scripted" else ""),
        },
        "dataset": {
            "name": data["manifest"]["name"],
            "seed": data["manifest"]["seed"],
            "sessions": data["manifest"]["sessions"],
            "dialogues": data["manifest"]["dialogues"],
            "turns": data["manifest"]["turns"],
            "questions": data["manifest"]["questions"],
            "questions_by_category": data["manifest"]["questions_by_category"],
            "fingerprint_verified": fingerprint["verified"],
            "files": fingerprint["files"],
            "labels": data["manifest"].get("labels"),
        },
        "extraction": {
            "active": ext_active,
            "offline_rule": ext_rule,
        },
        "write_fidelity": write_fidelity,
        "retrieval": {
            "ndcg_at_10": metrics["ndcg_at_10"],
            "by_category": metrics["by_category"],
            "answerable": metrics["answerable"],
            "abstain": metrics["abstain"],
            "tokens": metrics["tokens"],
            "latency_ms": metrics["latency_ms"],
            "field": metrics["field"],
            "budget": metrics["budget"],
            "errors": metrics["errors"],
            "questions": results,
        },
    }
    out = Path(args.out)
    C.write_json(out, payload)
    if not args.quiet:
        _print_summary(payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
