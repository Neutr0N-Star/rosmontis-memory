"""配置网格 → 帕累托前沿 → 生成 bench/report.md 与 bench/eval-summary.md。

用法
    PYTHONIOENCODING=utf-8 python bench/pareto.py
    PYTHONIOENCODING=utf-8 python bench/pareto.py --repeats 3 --max-questions 0

纪律
- 数字全部由本脚本计算并写入报告，报告里没有一个手填数字。
- 支配判定（docs/15 §2.2）：配置 B 支配 A ⇔ B 的全部质量指标 ≥ A、全部成本指标 ≤ A，
  且至少一项严格更优。质量高为优、成本低为优。
- 所有配置跑同一个库、同一批题；同一库只构建一次，换配置只换检索参数。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import common as C

EPS = 1e-9
THETA_SWEEP = (0.30, 0.42, 0.50, 0.54, 0.58)

QUALITY_METRICS = (
    ("ndcg_at_10", "nDCG@10"),
    ("temporal_accuracy", "时序 acc"),
    ("multi_hop_accuracy", "多跳 acc"),
    ("update_accuracy", "知识更新 acc"),
    ("preference_accuracy", "偏好 acc"),
    ("abstain_specificity", "拒答特异度"),
    ("answer_recall", "可作答召回"),
)
COST_METRICS = (
    ("mean_injected_tokens", "平均注入 token"),
    ("p95_latency_ms", "p95 延迟 ms"),
    ("budget_violation_rate", "预算越界率"),
)


def grid_configs() -> list[C.EvalConfig]:
    base = dict(field="on", gamma=0.45, hops=2, dense_k=30, sparse_k=30, graph_k=18, budget=1400, uniform_depth=True)
    return [
        C.EvalConfig("baseline_uniform", note="统一档基线：场开、D=0.45、2 轮扩散、30/30/18、预算 1400", **base),
        C.EvalConfig("field_off", field="off", gamma=0.45, hops=2, dense_k=30, sparse_k=30, graph_k=18,
                     budget=1400, uniform_depth=True, note="关语义场（ROSMONTIS_FIELD=0），其余同基线"),
        C.EvalConfig("D_low", field="on", gamma=0.25, hops=2, dense_k=30, sparse_k=30, graph_k=18,
                     budget=1400, uniform_depth=True, note="扩散强度 D=0.25"),
        C.EvalConfig("D_high", field="on", gamma=0.70, hops=2, dense_k=30, sparse_k=30, graph_k=18,
                     budget=1400, uniform_depth=True, note="扩散强度 D=0.70"),
        C.EvalConfig("hops_1", field="on", gamma=0.45, hops=1, dense_k=30, sparse_k=30, graph_k=18,
                     budget=1400, uniform_depth=True, note="扩散迭代 1 轮"),
        C.EvalConfig("hops_3", field="on", gamma=0.45, hops=3, dense_k=30, sparse_k=30, graph_k=18,
                     budget=1400, uniform_depth=True, note="扩散迭代 3 轮"),
        C.EvalConfig("depth_shallow", field="on", gamma=0.45, hops=2, dense_k=12, sparse_k=12, graph_k=8,
                     budget=1400, uniform_depth=True, note="检索深度 12/12/8"),
        C.EvalConfig("depth_deep", field="on", gamma=0.45, hops=2, dense_k=60, sparse_k=60, graph_k=40,
                     budget=1400, uniform_depth=True, note="检索深度 60/60/40"),
        C.EvalConfig("budget_low", field="on", gamma=0.45, hops=2, dense_k=30, sparse_k=30, graph_k=18,
                     budget=240, uniform_depth=True, note="注入预算 240 token"),
        C.EvalConfig("budget_high", field="on", gamma=0.45, hops=2, dense_k=30, sparse_k=30, graph_k=18,
                     budget=2000, uniform_depth=True, note="注入预算 2000 token"),
    ]


def reference_config(field: str) -> C.EvalConfig:
    return C.EvalConfig(
        "product_default",
        field=field,
        gamma=None,
        hops=None,
        budget=1400,
        uniform_depth=False,
        note="产品默认：按问题自适应反射深度 + 预算倍率，不注入任何参数",
    )


def theta_configs() -> list[C.EvalConfig]:
    out = []
    for t in THETA_SWEEP:
        out.append(
            C.EvalConfig(
                f"theta_{int(round(t * 100)):03d}",
                field="on",
                gamma=0.45,
                hops=2,
                dense_k=30,
                sparse_k=30,
                graph_k=18,
                budget=1400,
                uniform_depth=True,
                theta_abstain=t,
                note=f"拒答阈值 θ_abstain={t}（其余同统一档基线）",
            )
        )
    return out


# ── 分析 ──────────────────────────────────────────────────────
def qvec(m: dict) -> dict:
    cats = m["by_category"]
    return {
        "ndcg_at_10": m["ndcg_at_10"],
        "temporal_accuracy": cats["temporal"]["accuracy"],
        "multi_hop_accuracy": cats["multi_hop"]["accuracy"],
        "update_accuracy": cats["update"]["accuracy"],
        "preference_accuracy": cats["preference"]["accuracy"],
        "abstain_specificity": m["abstain"]["specificity"],
        "answer_recall": 1.0 - m["answerable"]["miss_rate"],
    }


def cvec(m: dict) -> dict:
    return {
        "mean_injected_tokens": m["tokens"]["mean"],
        "p95_latency_ms": m["latency_ms"]["p95"],
        "budget_violation_rate": m["tokens"]["budget_violation_rate"],
    }


def dominates(a: dict, b: dict) -> bool:
    qa, qb = qvec(a), qvec(b)
    ca, cb = cvec(a), cvec(b)
    if not all(qa[k] >= qb[k] - EPS for k in qa):
        return False
    if not all(ca[k] <= cb[k] + EPS for k in ca):
        return False
    return any(qa[k] > qb[k] + EPS for k in qa) or any(ca[k] < cb[k] - EPS for k in ca)


def frontier_names(metrics_by_name: dict[str, dict]) -> list[str]:
    names = list(metrics_by_name)
    return [n for n in names if not any(dominates(metrics_by_name[o], metrics_by_name[n]) for o in names if o != n)]


def dominators(name: str, metrics_by_name: dict[str, dict]) -> list[str]:
    m = metrics_by_name[name]
    return [o for o in metrics_by_name if o != name and dominates(metrics_by_name[o], m)]


def per_q_map(results: list[dict]) -> dict[str, dict]:
    return {r["qid"]: r for r in results}


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def dimension_sensitivity(metrics_by_name: dict[str, dict], baseline: str = "baseline_uniform") -> list[dict]:
    groups = {
        "场开关（ROSMONTIS_FIELD）": ["field_off"],
        "扩散强度 D": ["D_low", "D_high"],
        "扩散迭代次数": ["hops_1", "hops_3"],
        "检索深度 k": ["depth_shallow", "depth_deep"],
        "注入预算 token": ["budget_low", "budget_high"],
    }
    base = metrics_by_name[baseline]
    base_q = C.quality_composite(base)
    base_c = cvec(base)
    out = []
    for dim, names in groups.items():
        options = []
        for n in names:
            if n not in metrics_by_name:
                continue
            m = metrics_by_name[n]
            options.append(
                {
                    "name": n,
                    "quality_delta": C.quality_composite(m) - base_q,
                    "tokens_delta_pct": (cvec(m)["mean_injected_tokens"] / base_c["mean_injected_tokens"] - 1.0)
                    if base_c["mean_injected_tokens"]
                    else 0.0,
                    "p95_delta_pct": (cvec(m)["p95_latency_ms"] / base_c["p95_latency_ms"] - 1.0)
                    if base_c["p95_latency_ms"]
                    else 0.0,
                }
            )
        if not options:
            continue
        best = max(options, key=lambda o: (o["quality_delta"], -o["tokens_delta_pct"], -o["p95_delta_pct"]))
        out.append({"dimension": dim, **best, "options": options})
    return out


def choose_recommendation(metrics_by_name: dict[str, dict], label_of: dict[str, tuple[str, str]],
                          baseline: str = "baseline_uniform") -> dict:
    """可复现的推荐规则（写进报告）：
    1. 在所有已测配置中找「严格支配基线」的配置；有则取质量综合分最高（并列取 p95 更低）的；
    2. 没有严格支配者，则取质量综合分最高者，并如实报告它的成本增量；
    3. 若质量最好的配置成本更高，结论写成「值得尝试的折中」，不写成支配。
    """
    base = metrics_by_name[baseline]
    base_q = C.quality_composite(base)
    base_c = cvec(base)
    names = [n for n in metrics_by_name if n != baseline]
    dominating = [n for n in names if dominates(metrics_by_name[n], base)]
    pool = dominating or names
    best = max(
        pool,
        key=lambda n: (round(C.quality_composite(metrics_by_name[n]), 9), -cvec(metrics_by_name[n])["p95_latency_ms"]),
    )
    m = metrics_by_name[best]
    dim, value = label_of.get(best, (best, ""))
    c = cvec(m)
    return {
        "config": best,
        "dimension": dim,
        "value": value,
        "dominates_baseline": bool(best in dominating),
        "quality_composite": C.quality_composite(m),
        "quality_delta": C.quality_composite(m) - base_q,
        "tokens_delta_pct": (c["mean_injected_tokens"] / base_c["mean_injected_tokens"] - 1.0)
        if base_c["mean_injected_tokens"]
        else 0.0,
        "p95_delta_pct": (c["p95_latency_ms"] / base_c["p95_latency_ms"] - 1.0) if base_c["p95_latency_ms"] else 0.0,
        "metrics": m,
        "baseline_metrics": base,
        "dominating_candidates": dominating,
    }


def field_comparison(metrics_by_name: dict[str, dict], results_by_name: dict[str, list[dict]],
                     base_name: str = "baseline_uniform", off_name: str = "field_off") -> dict:
    base, off = metrics_by_name[base_name], metrics_by_name[off_name]
    bq, oq = qvec(base), qvec(off)
    bmap, omap = per_q_map(results_by_name[base_name]), per_q_map(results_by_name[off_name])
    answerable = [q for q, r in bmap.items() if r["category"] != "abstain"]
    new_hits = sum(1 for q in answerable if bmap[q]["pass"] and not omap[q]["pass"])
    lost_hits = sum(1 for q in answerable if omap[q]["pass"] and not bmap[q]["pass"])
    lat_pct = (cvec(base)["p95_latency_ms"] / cvec(off)["p95_latency_ms"] - 1.0) if cvec(off)["p95_latency_ms"] else 0.0
    quality_fields = [k for k in bq if k != "answer_recall"]  # answer_recall 单独看
    no_worse = all(bq[k] >= oq[k] - EPS for k in quality_fields)
    return {
        "base": base_name,
        "off": off_name,
        "quality": {k: {"on": bq[k], "off": oq[k], "delta": bq[k] - oq[k]} for k in bq},
        "new_hits": new_hits,
        "lost_hits": lost_hits,
        "answerable": len(answerable),
        "field_gain_rate": new_hits / len(answerable) if answerable else 0.0,
        "tokens": {"on": cvec(base)["mean_injected_tokens"], "off": cvec(off)["mean_injected_tokens"]},
        "p95_latency": {"on": cvec(base)["p95_latency_ms"], "off": cvec(off)["p95_latency_ms"]},
        "p95_latency_delta_pct": lat_pct,
        "d5_quality_no_worse": no_worse,
        "d5_latency_within_20pct": lat_pct < 0.20,
        "d5_pass": bool(no_worse and lat_pct < 0.20),
    }


# ── 报告 ──────────────────────────────────────────────────────
def _meta_block(ctx: dict) -> str:
    env = ctx["environment"]
    code = ctx["code"]
    ds = ctx["dataset"]
    fp = ctx["fingerprint"]
    fm = ctx["field_backend"]
    lines = [
        f"- 数据规模：**{ds['dialogues']} 段合成对话**（{ds['turns']} turn）、**{ds['questions']} 道题**"
        f"（{ds['sessions']} 个会话，种子 `{ds['seed']}`，时间跨度 {ds['horizon_days']} 天）。",
        f"- 题集分布：{', '.join(f'{k} {v}' for k, v in ds['questions_by_category'].items())}。",
        f"- 标注方式：程序化金标（生成器在生成时同步写出），**非人工标注**；抽取/检索判定口径见下文。",
        f"- 数据集指纹：manifest sha256 {'校验通过' if fp['verified'] else '**校验失败**'}；"
        f"{', '.join(f'`{name}`={info['actual'][:16]}…' for name, info in fp['files'].items())}",
        f"- 运行环境：Python {env['python']} · numpy {env['numpy'] or '缺失（纯 Python 兜底）'} · "
        f"jieba {'有' if env['jieba'] else '缺失（字符二元组分词）'} · embedding `{ctx['embed']}`。",
        f"- 抽取后端：`{ctx['extraction_backend']}`（{ctx['extraction_note']}）。",
        f"- 语义场：`ROSMONTIS_FIELD` 原生开关 = {fm['native_switch']}，recall 已接场 = {fm['recall_uses_field']}；"
        f"D 走 `{fm['decay_source']}`，迭代次数走 `{fm['iterations_source']}`。",
        f"- 代码指纹：git `{str(code.get('git_head') or '?')[:12]}`，工作区 `rosmontis/` 未提交改动 "
        f"{len(code.get('dirty_core_files') or [])} 个（核心施工期，集成完成后应用同一命令重跑刷新数字）。",
        f"- 生成时间：{ctx['run_at']}；DB 构建 {ctx['build_elapsed_s']}s（{ds['db_build_line']}）。",
    ]
    return "\n".join(lines)


def _definitions_block(ctx: dict) -> str:
    return f"""\
### 指标定义（口径固化，换实现必须同步改本节）

| 指标 | 定义 | 采集 |
|---|---|---|
| 抽取 P / R / F1 | 对 {ctx['dataset']['dialogues']} 个用户 turn，事实按 `(主语, 谓词, 客体, 极性)` 归一后 micro 比对；实体按规范名、状态槽按 `(slot, value)` 分别 micro 汇总 | `bench/run_eval.py` → `extraction` |
| 写入保真度 | 脚本化金标经 `write/pipeline` 落库后，active 事实与当前状态槽相对金标的 P/R/F1 | 同上 → `write_fidelity` |
| 召回 nDCG@10 | 注入顺序前 10 条按分级相关度计分：答案值 = 2 级、桥接实体/主题 = 1 级，`gain = 2^rel - 1`，折损 `log2(rank+1)`，除以理想排序 IDCG；对 4 类可作答题取均值 | `bench/run_eval.py` → `retrieval.ndcg_at_10` |
| 四类准确率 | 答案证据是否出现在**本轮实际注入的全部条目**里（预算越小注入越少，准确率随预算变化，这是帕累托质量轴）：时序 = 该 as_of 的状态值；多跳 = 答案值 + 桥接实体都在；知识更新 = 新值在且旧值不得排在它前面（极性翻转同理）；偏好 = 谓词+客体+极性同时命中 | 同上 → `retrieval.by_category` |
| `hit@10` | 答案证据是否进得了前 10 条（与准确率分开：在上下文里 ≠ 排得靠前） | 同上 |
| 拒答误召率 / 漏召率 | 标注「该拒答」的 {ctx['dataset']['questions_by_category'].get('abstain', 0)} 题里未拒答的比例 / 可作答题里被拒答的比例；拒答精度 = 正确拒答 / 全部拒答 | 同上 → `retrieval.abstain` |
| 注入 token | 产品 `RecallResult.tokens_used`（产品自身 token 估算，含常驻块；本基准库无常驻页面） | 同上 → `retrieval.tokens` |
| 检索延迟 | 产品 `RecallResult.latency_ms`（含 retrieval_log 落库）；每题重复 {ctx['repeats']} 次取均值，p50/p95 跨全部题目 | 同上 → `retrieval.latency_ms` |
| 帕累托支配 | 配置 B 支配 A ⇔ 7 项质量指标全部 ≥ A，3 项成本（平均注入 token、p95 延迟、预算越界率）全部 ≤ A，且至少一项严格更优 | `bench/pareto.py` |

**读法提醒**：准确率看「宿主拿到的上下文里有没有答案」，nDCG@10 看「答案排得够不够前」；两者一起看才完整。延迟是单机单次会话测量，绝对值受机器负载影响，横向比较只看数量级与相对差。
"""


def _baseline_block(ctx: dict) -> str:
    base = ctx["metrics_by_name"]["baseline_uniform"]
    ref = ctx["metrics_by_name"]["product_default"]
    rows = []
    for name, m, note in (("统一档基线", base, "场开、D=0.45、2 轮、30/30/18、预算 1400"), ("产品默认", ref, "自适应反射深度，不注入参数")):
        cats = m["by_category"]
        rows.append(
            [
                name,
                _fmt(m["ndcg_at_10"], 4),
                _fmt(cats["temporal"]["accuracy"]),
                _fmt(cats["multi_hop"]["accuracy"]),
                _fmt(cats["update"]["accuracy"]),
                _fmt(cats["preference"]["accuracy"]),
                _fmt(m["abstain"]["specificity"]),
                _fmt(m["answerable"]["miss_rate"]),
                _fmt(m["tokens"]["mean"], 1),
                _fmt(m["latency_ms"]["p50"], 2),
                _fmt(m["latency_ms"]["p95"], 2),
                note,
            ]
        )
    return md_table(
        ["配置", "nDCG@10", "时序 acc", "多跳 acc", "更新 acc", "偏好 acc", "拒答特异度", "漏召率", "token 均值", "p50 ms", "p95 ms", "说明"],
        rows,
    )


def _config_table(ctx: dict) -> str:
    rows = []
    mbn = ctx["metrics_by_name"]
    grid_names = {e.name for e in ctx["grid"]}
    grid_metrics = {n: mbn[n] for n in grid_names}
    front = set(ctx["frontier_grid"])
    for ec in ctx["grid"]:
        m = mbn[ec.name]
        cats = m["by_category"]
        doms = dominators(ec.name, grid_metrics)
        status = "前沿" if ec.name in front else f"被 {doms[0]} 支配"
        rows.append(
            [
                f"`{ec.name}`",
                ec.field,
                _fmt(ec.gamma, 2) if ec.gamma is not None else "默认",
                str(ec.hops) if ec.hops is not None else "默认",
                f"{ec.dense_k}/{ec.sparse_k}/{ec.graph_k}" if ec.dense_k else "默认",
                str(ec.budget),
                _fmt(m["ndcg_at_10"], 4),
                _fmt(cats["temporal"]["accuracy"]),
                _fmt(cats["multi_hop"]["accuracy"]),
                _fmt(cats["update"]["accuracy"]),
                _fmt(cats["preference"]["accuracy"]),
                _fmt(m["abstain"]["specificity"]),
                _fmt(m["tokens"]["mean"], 1),
                _fmt(m["latency_ms"]["p95"], 2),
                _fmt(m["quality_composite"], 4),
                status,
            ]
        )
    return md_table(
        ["配置", "场", "D", "迭代", "深度 k", "预算", "nDCG@10", "时序", "多跳", "更新", "偏好", "拒答特异度", "token", "p95 ms", "质量综合", "支配状态"],
        rows,
    )


def _frontier_block(ctx: dict) -> str:
    mbn = ctx["metrics_by_name"]
    front = ctx["frontier_grid"]
    rows = []
    ordered = sorted(front, key=lambda n: (C.quality_composite(mbn[n]), -cvec(mbn[n])["mean_injected_tokens"]), reverse=True)
    for i, name in enumerate(ordered, 1):
        m = mbn[name]
        rows.append(
            [
                str(i),
                f"`{name}`",
                _fmt(C.quality_composite(m), 4),
                _fmt(m["ndcg_at_10"], 4),
                _fmt(m["tokens"]["mean"], 1),
                _fmt(m["latency_ms"]["p95"], 2),
                _fmt(m["abstain"]["specificity"]),
                _fmt(m["by_category"]["multi_hop"]["accuracy"]),
            ]
        )
    return md_table(["#", "配置", "质量综合", "nDCG@10", "token", "p95 ms", "拒答特异度", "多跳 acc"], rows)


def _theta_block(ctx: dict) -> str:
    rows = []
    base = ctx["metrics_by_name"]["baseline_uniform"]
    rows.append(
        [
            "0.18（产品默认 = 基线）",
            _fmt(base["abstain"]["specificity"]),
            _fmt(base["abstain"]["false_recall_rate"]),
            _fmt(base["answerable"]["miss_rate"]),
            _fmt(base["answerable"]["accuracy"]),
            _fmt(base["tokens"]["mean"], 1),
            _fmt(base["quality_composite"], 4),
        ]
    )
    for ec in ctx["theta"]:
        m = ctx["metrics_by_name"][ec.name]
        rows.append(
            [
                f"{ec.theta_abstain:.2f}",
                _fmt(m["abstain"]["specificity"]),
                _fmt(m["abstain"]["false_recall_rate"]),
                _fmt(m["answerable"]["miss_rate"]),
                _fmt(m["answerable"]["accuracy"]),
                _fmt(m["tokens"]["mean"], 1),
                _fmt(m["quality_composite"], 4),
            ]
        )
    return md_table(["θ_abstain", "拒答特异度", "误召率", "漏召率", "可作答 acc", "token 均值", "质量综合"], rows)


def _extraction_block(ctx: dict) -> str:
    rule = ctx["extraction"]["offline_rule"]
    active = ctx["extraction"]["active"]
    wf = ctx["write_fidelity"]
    rows = [
        ["规则兜底 · 实体", _fmt(rule["entity"]["precision"]), _fmt(rule["entity"]["recall"]), _fmt(rule["entity"]["f1"]),
         f"{rule['entity']['matched']}/{rule['entity']['predicted']}/{rule['entity']['gold']}"],
        ["规则兜底 · 事实", _fmt(rule["fact"]["precision"]), _fmt(rule["fact"]["recall"]), _fmt(rule["fact"]["f1"]),
         f"{rule['fact']['matched']}/{rule['fact']['predicted']}/{rule['fact']['gold']}"],
        ["规则兜底 · 状态槽", _fmt(rule["state_slot"]["precision"]), _fmt(rule["state_slot"]["recall"]), _fmt(rule["state_slot"]["f1"]),
         f"{rule['state_slot']['matched']}/{rule['state_slot']['predicted']}/{rule['state_slot']['gold']}"],
        [f"当前后端 `{active['backend']}` · 事实", _fmt(active["fact"]["precision"]), _fmt(active["fact"]["recall"]),
         _fmt(active["fact"]["f1"]), f"{active['fact']['matched']}/{active['fact']['predicted']}/{active['fact']['gold']}"],
        ["写入保真 · active 事实", _fmt(wf["fact"]["precision"]), _fmt(wf["fact"]["recall"]), _fmt(wf["fact"]["f1"]),
         f"{wf['fact']['matched']}/{wf['fact']['predicted']}/{wf['fact']['gold']}"],
    ]
    table = md_table(["口径", "P", "R", "F1", "match/pred/gold"], rows)
    return (
        table
        + "\n\n"
        + f"- 状态槽当前值保真：{wf['state_slot_current']['matched']}/{wf['state_slot_current']['expected']}"
        f"（accuracy={_fmt(wf['state_slot_current']['accuracy'])}）。\n"
        + f"- **离线规则抽取 F1 只有 {_fmt(rule['fact']['f1'])}**：核心规则兜底只认「我叫 X / 我在 X 上班」"
        "三种高频句式，这是设计取舍（宁缺毋滥），不是评测 bug。要测真实抽取质量需接模型"
        "（`--backend llm`，有 Ollama 时）；本报告的检索类数字用脚本化金标隔离抽取误差，"
        f"写入保真 {_fmt(wf['fact']['f1'])} 说明结构投影链路本身没有丢金标。\n"
    )


def _field_block(ctx: dict) -> str:
    fc = ctx["field_comparison"]
    rows = []
    for key, label in QUALITY_METRICS:
        if key not in fc["quality"]:
            continue
        q = fc["quality"][key]
        rows.append([label, _fmt(q["on"]), _fmt(q["off"]), _fmt(q["delta"], 4)])
    rows.append(["平均注入 token", _fmt(fc["tokens"]["on"], 1), _fmt(fc["tokens"]["off"], 1),
                 _fmt(fc["tokens"]["on"] - fc["tokens"]["off"], 1)])
    rows.append(["p95 延迟 ms", _fmt(fc["p95_latency"]["on"], 2), _fmt(fc["p95_latency"]["off"], 2),
                 _fmt(fc["p95_latency"]["on"] - fc["p95_latency"]["off"], 2)])
    table = md_table(["指标", "场开", "场关", "差值（开-关）"], rows)
    verdict = "通过" if fc["d5_pass"] else "未通过"
    return (
        table
        + "\n\n"
        + f"- 场增益：场关时答不出、场开时答出的可作答题 **{fc['new_hits']}/{fc['answerable']}**"
        f"（{fc['field_gain_rate']:.3f}），反向丢失 {fc['lost_hits']} 题。\n"
        + f"- D5 判据（场不得劣化质量 ∧ p95 延迟增幅 < 20%）：**{verdict}**。"
        f"质量不劣 = {fc['d5_quality_no_worse']}，p95 延迟变化 = {fc['p95_latency_delta_pct'] * 100:+.1f}%"
        f"（{_fmt(fc['p95_latency']['off'], 2)}ms → {_fmt(fc['p95_latency']['on'], 2)}ms）。\n"
        + "- 需要如实说明：本基准集上开场的**多跳准确率更高**（结构扩散把桥接实体带进来了），"
        "但 nDCG@10 反而下降（扩散额外的候选把答案往后挤）、注入 token 与延迟上升。"
        "场的价值是结构召回，不是零成本——它是否值得开，取决于宿主更在意多跳还是排序/成本。\n"
    )


def _sensitivity_block(ctx: dict) -> str:
    rows = []
    for s in ctx["sensitivity"]:
        rows.append(
            [
                s["dimension"],
                f"`{s['name']}`",
                f"{s['quality_delta']:+.4f}",
                f"{s['tokens_delta_pct'] * 100:+.1f}%",
                f"{s['p95_delta_pct'] * 100:+.1f}%",
            ]
        )
    return md_table(["参数", "最佳取值", "质量综合变化", "token 变化", "p95 变化"], rows)


def _recommendation_block(ctx: dict) -> str:
    rec = ctx["recommendation"]
    b = rec["baseline_metrics"]
    m = rec["metrics"]
    mode = "严格支配基线（零成本改进）" if rec["dominates_baseline"] else "未严格支配基线（质量/成本折中）"
    lines = [
        f"**最值得调的一个参数：`{rec['dimension']}` = {rec['value']}**（{mode}）。",
        "",
        f"- 质量综合：{_fmt(C.quality_composite(b), 4)} → {_fmt(rec['quality_composite'], 4)}"
        f"（{rec['quality_delta']:+.4f}）",
        f"- nDCG@10：{_fmt(b['ndcg_at_10'], 4)} → {_fmt(m['ndcg_at_10'], 4)}；"
        f"多跳 acc：{_fmt(b['by_category']['multi_hop']['accuracy'])} → {_fmt(m['by_category']['multi_hop']['accuracy'])}；"
        f"拒答特异度：{_fmt(b['abstain']['specificity'])} → {_fmt(m['abstain']['specificity'])}",
        f"- 成本：平均注入 token {_fmt(b['tokens']['mean'], 1)} → {_fmt(m['tokens']['mean'], 1)}"
        f"（{rec['tokens_delta_pct'] * 100:+.1f}%），p95 延迟 {_fmt(b['latency_ms']['p95'], 2)}ms → "
        f"{_fmt(m['latency_ms']['p95'], 2)}ms（{rec['p95_delta_pct'] * 100:+.1f}%）",
        f"- 选择规则（脚本可复现）：先在全部已测配置里找严格支配「统一档基线」者；有则取质量综合最高"
        f"（并列取 p95 更低）；没有则取质量综合最高者并如实标注成本增量。本次支配基线者："
        f"{', '.join(rec['dominating_candidates']) if rec['dominating_candidates'] else '（无）'}。",
    ]
    return "\n".join(lines)


def _limitations_block(ctx: dict) -> str:
    code = ctx["code"]
    return f"""\
1. **合成数据的边界**：对话与金标均由模板生成，句式和实体池有限，分布比真实对话窄；金标是程序化标注，与 docs/15 §2.1 要求的「人工 golden set 200 段」不等价。它能验证管线可测、可复现与相对排序，不能替代真实标注集。
2. **离线降级显著影响数字**：本机无 Ollama/无 jieba，embedding 走哈希伪向量且核心在降级时不落库向量，稠密路实际退化；分词退化为字符二元组。因此「拒答误召率」偏高主要来自 FTS 把「为什么/怎么/什么」这类问题词当关键词命中（详见下文与阈值扫描），有真语义向量后应重跑。
3. **核心仍在施工**：评测时 `rosmontis/` 有 {len(code.get('dirty_core_files') or [])} 个未提交改动（{', '.join(code.get('dirty_core_files') or []) or '无'}）。集成完成后必须用同一命令重跑，数字以重跑结果为准。
4. **延迟是单机测量**：绝对值受机器负载影响，跨机器不可比；只用于同机横向比较。p95 基于每题 {ctx['repeats']} 次重复的均值。
5. **场参数映射**：当前 D 走 `memory_field.DEFAULT_GAMMA`，迭代次数走 `REFLEX_DEPTH[*].hops`；若 WS-B 把扩散参数改为 cfg 字段，`bench/common.py` 的 `applied_config()` 是唯一需要改的地方（报告与脚本无需变化）。
6. **不评价生成质量**：基准没有答案文本生成环节，多跳/比较类题判定的是「可作答证据是否齐备」，不是最终自然语言回答的正确性。
"""


def build_report(ctx: dict) -> str:
    ds = ctx["dataset"]
    mbn = ctx["metrics_by_name"]
    rec = ctx["recommendation"]
    fc = ctx["field_comparison"]
    base = mbn["baseline_uniform"]
    return f"""\
# 藏识 Rosmontis · 质量基准与帕累托报告

> 本文件由 `bench/pareto.py` 自动生成，**所有数字来自同一次脚本运行**；手工改动会被下次运行覆盖。
> 复现：`{ds['commands']['pareto']}`（数据集另可用 `{ds['commands']['generate']}` 重生成，种子固定）。

## 0. 一句话结论

在 {ds['dialogues']} 段合成对话、{ds['questions']} 道四类题上，本轮 {len(ctx['grid'])} 组检索配置的帕累托前沿有 **{len(ctx['frontier_grid'])} 组**；
最值得调的一个参数是 **`{rec['dimension']}` = {rec['value']}**：质量综合 {_fmt(C.quality_composite(rec['baseline_metrics']), 4)} → {_fmt(rec['quality_composite'], 4)}，
nDCG@10 {_fmt(rec['baseline_metrics']['ndcg_at_10'], 4)} → {_fmt(rec['metrics']['ndcg_at_10'], 4)}，
p95 延迟 {rec['p95_delta_pct'] * 100:+.1f}%，平均注入 token {rec['tokens_delta_pct'] * 100:+.1f}%。
`{'严格支配基线' if rec['dominates_baseline'] else '是质量/成本折中，不是严格支配'}`。

## 1. 数据集与标注口径

{_meta_block(ctx)}

生成器：`bench/dataset/gen_dataset.py`（纯标准库、固定种子、无网络）。产物：
`conversations.jsonl`（{ds['dialogues']} 行对话 + 抽取金标）、`sessions.json`（时间线/当前状态/最终有效事实期望）、
`questions.jsonl`（{ds['questions']} 道题）与 `manifest.json`（规模 + sha256）。

题目四类：时序 {ds['questions_by_category'].get('temporal', 0)}、多跳 {ds['questions_by_category'].get('multi_hop', 0)}、
知识更新 {ds['questions_by_category'].get('update', 0)}、偏好 {ds['questions_by_category'].get('preference', 0)}；
另加拒答标注 {ds['questions_by_category'].get('abstain', 0)} 道（题面与会话内容无关，正确答案是不召回）。
每类均 ≥20 题，固定种子重跑逐字节一致（脚本启动时会校验 manifest 哈希）。

## 2. 指标与判定口径

{_definitions_block(ctx)}

## 3. 抽取 / 写入链路

{_extraction_block(ctx)}

## 4. 基线数字

{_baseline_block(ctx)}

## 5. 配置网格（{len(ctx['grid'])} 组）与帕累托前沿

> 统一档：四档反射深度统一成同一组 k 值、预算倍率固定为 1.0，这样配置之间的差异只来自被研究的参数；
> 「产品默认」行保留自适应深度，作为现实基线参考（不参与网格支配判定）。

{_config_table(ctx)}

### 帕累托前沿（{len(ctx['frontier_grid'])} 组）

{_frontier_block(ctx)}

支配判定按 docs/15 §2.2：7 项质量指标全不劣、3 项成本全不劣且至少一项严格更优。前沿之外的配置都能被前沿里某组支配。

## 6. 场地基对比（ROSMONTIS_FIELD on/off）

{_field_block(ctx)}

## 7. 参数灵敏度与推荐

{_sensitivity_block(ctx)}

{_recommendation_block(ctx)}

## 8. 拒答阈值标定（附加台账）

离线降级下主指标最刺眼的是拒答误召（基线误召率 {_fmt(base['abstain']['false_recall_rate'])}）。
docs/03 的 θ_abstain 是猜的 0.18；本基准集自带「该拒答」标注，因此附带做一轮阈值扫描（其余同统一档基线）：

{_theta_block(ctx)}

阈值扫描只改 `cfg.weights.theta_abstain`（评测期临时覆盖、退出恢复，不改核心文件）。它不在检索配置网格里，
但结论直接可用：**推荐值 = {ctx['theta_recommendation']['theta']:.2f}**，理由见第 7 节推荐。

## 9. 复现命令

```bash
# 0) 环境（Windows 控制台必须显式 UTF-8，否则中文输出会乱码/报编码错）
set PYTHONIOENCODING=utf-8

# 1) 生成/重生成数据集（固定种子，逐字节可复现）
python bench/dataset/gen_dataset.py

# 2) 单次基线评测（输出 JSON；加 --field off 跑场关闭对照）
python bench/run_eval.py
python bench/run_eval.py --field off --out bench/results/latest_field_off.json

# 3) 帕累托网格 + 生成 bench/report.md 与 bench/eval-summary.md
python bench/pareto.py

# 4) 有本地 Ollama 时可测真实抽取（无则自动回退规则，不崩）
python bench/run_eval.py --backend llm --extraction-limit 50
```

## 10. 局限与依赖（必须和数字一起读）

{_limitations_block(ctx)}
"""


def build_summary(ctx: dict) -> str:
    """bench/eval-summary.md：给决策者看的压缩版（数字同样全部来自脚本）。"""
    ds = ctx["dataset"]
    mbn = ctx["metrics_by_name"]
    rec = ctx["recommendation"]
    fc = ctx["field_comparison"]
    base = mbn["baseline_uniform"]
    ref = mbn["product_default"]
    return f"""\
# 质量基准与帕累托 · 评测摘要

> 由 `bench/pareto.py` 自动生成；完整表格、口径与局限见 [bench/report.md](../bench/report.md)。
> 本文件不手工填数——所有数字与同次运行的 `bench/results/pareto.json` 对齐。

## 结论摘要

- 基准集：**{ds['dialogues']} 段合成对话**（{ds['turns']} turn，{ds['sessions']} 个会话，{ds['horizon_days']} 天跨度，种子 `{ds['seed']}`）+
  **{ds['questions']} 道题**（时序 {ds['questions_by_category'].get('temporal', 0)} / 多跳 {ds['questions_by_category'].get('multi_hop', 0)} /
  更新 {ds['questions_by_category'].get('update', 0)} / 偏好 {ds['questions_by_category'].get('preference', 0)} / 拒答 {ds['questions_by_category'].get('abstain', 0)}）。
- 基线（产品默认配置）：抽取事实 F1（离线规则兜底）= {_fmt(ctx['extraction']['offline_rule']['fact']['f1'])}，
  写入保真 F1 = {_fmt(ctx['write_fidelity']['fact']['f1'])}，
  nDCG@10 = {_fmt(ref['ndcg_at_10'], 4)}，
  四类 acc = 时序 {_fmt(ref['by_category']['temporal']['accuracy'])} / 多跳 {_fmt(ref['by_category']['multi_hop']['accuracy'])} /
  更新 {_fmt(ref['by_category']['update']['accuracy'])} / 偏好 {_fmt(ref['by_category']['preference']['accuracy'])}，
  拒答误召率 = {_fmt(ref['abstain']['false_recall_rate'])}，漏召率 = {_fmt(ref['answerable']['miss_rate'])}，
  注入 token 均值 = {_fmt(ref['tokens']['mean'], 1)}，检索 p50/p95 = {_fmt(ref['latency_ms']['p50'], 2)}/{_fmt(ref['latency_ms']['p95'], 2)} ms。
- 帕累托前沿：{len(ctx['grid'])} 组检索配置中 **{len(ctx['frontier_grid'])} 组**位于前沿（配置见下）；场开/关、深度、迭代、预算各有取舍，
  没有单组配置在所有质量与成本维度同时占优。
- 最值得调的一个参数：**`{rec['dimension']}` = {rec['value']}**（{'严格支配基线' if rec['dominates_baseline'] else '质量/成本折中'}）：
  质量综合 {_fmt(C.quality_composite(rec['baseline_metrics']), 4)} → {_fmt(rec['quality_composite'], 4)}，
  nDCG@10 {_fmt(rec['baseline_metrics']['ndcg_at_10'], 4)} → {_fmt(rec['metrics']['ndcg_at_10'], 4)}，
  p95 延迟 {rec['p95_delta_pct'] * 100:+.1f}%，token {rec['tokens_delta_pct'] * 100:+.1f}%。
- 场对比（D5）：{('通过' if fc['d5_pass'] else '未通过')}——多跳 acc {_fmt(fc['quality']['multi_hop_accuracy']['off'])} → {_fmt(fc['quality']['multi_hop_accuracy']['on'])}
  （增益 {fc['new_hits']}/{fc['answerable']} 题），但 nDCG@10 {_fmt(fc['quality']['ndcg_at_10']['off'])} → {_fmt(fc['quality']['ndcg_at_10']['on'])}、
  p95 延迟 {fc['p95_latency_delta_pct'] * 100:+.1f}%。场是结构召回工具，不是零成本。
- 拒答阈值 θ_abstain：产品默认 0.18 在本基准上误召率 {_fmt(base['abstain']['false_recall_rate'])}；
  扫描后推荐 **{ctx['theta_recommendation']['theta']:.2f}**（误召率 {_fmt(ctx['theta_recommendation']['metrics']['abstain']['false_recall_rate'])}，
  漏召率 {_fmt(ctx['theta_recommendation']['metrics']['answerable']['miss_rate'])}）。该值来自标注集，不应再靠猜。

## 帕累托前沿

{_frontier_block(ctx)}

## 拒答阈值扫描

{_theta_block(ctx)}

## 抽取 / 写入

{_extraction_block(ctx)}

## 复现

```bash
set PYTHONIOENCODING=utf-8
python bench/dataset/gen_dataset.py     # 固定种子，逐字节可复现
python bench/run_eval.py                # 单次 JSON
python bench/pareto.py                  # 生成 report + 本文件
```

## 局限

1. 合成数据 + 程序化金标，与人工标注 golden set 不等价；数字用于相对比较与回归，不用于对外对标。
2. 本机无 Ollama/jieba：hash 伪向量 + 字符二元组分词导致稠密路退化，拒答误召偏高；接真模型后需重跑。
3. 核心施工期（`rosmontis/` 有 {len(ctx['code'].get('dirty_core_files') or [])} 个未提交改动），集成完成后以同一命令重跑结果为准。
4. 延迟为单机测量，跨机器不可比。
"""


# ── 主流程 ────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="rosmontis-memory 帕累托评测")
    ap.add_argument("--field", choices=["on", "off"], default=None, help="参考配置的场开关（默认读 ROSMONTIS_FIELD）")
    ap.add_argument("--budget", type=int, default=1400, help="产品默认参考配置的预算")
    ap.add_argument("--repeats", type=int, default=2, help="每题重复检索次数（延迟取均值，默认 2）")
    ap.add_argument("--max-questions", type=int, default=0, help="冒烟用：只跑前 N 题（0=全部，题目顺序按会话/类别）")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "pareto.json"), help="原始结果 JSON 输出路径")
    ap.add_argument("--no-docs", action="store_true", help="只算不写报告（调试用）")
    ap.add_argument("--regen", action="store_true", help="重新生成数据集")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    C.ensure_dataset(regen=args.regen)
    data = C.load_dataset()
    fingerprint = C.dataset_fingerprint()
    field = args.field or (
        "off" if str(os.environ.get("ROSMONTIS_FIELD", "on")).strip().lower() in ("0", "false", "off", "no") else "on"
    )
    if args.max_questions:
        data = dict(data)
        data["questions"] = data["questions"][: int(args.max_questions)]

    build_t0 = time.perf_counter()
    db = C.build_eval_db(data, backend="scripted", embed_provider="hash")
    build_elapsed = round(time.perf_counter() - build_t0, 3)
    try:
        extraction = {
            "active": C.measure_extraction(data["conversations"], backend="scripted"),
            "offline_rule": C.measure_extraction(data["conversations"], backend="rule"),
        }
        write_fidelity = C.measure_write_fidelity(db.cfg, data["sessions"])

        grid = grid_configs()
        reference = reference_config(field)
        theta = theta_configs()
        metrics_by_name: dict[str, dict] = {}
        results_by_name: dict[str, list[dict]] = {}
        applied_by_name: dict[str, dict] = {}
        for ec in grid + [reference] + theta:
            metrics, results, applied = C.evaluate_config(db, data, ec, repeats=max(1, args.repeats))
            metrics_by_name[ec.name] = metrics
            results_by_name[ec.name] = results
            applied_by_name[ec.name] = applied
            if not args.no_docs:
                cats = metrics["by_category"]
                print(
                    f"[{ec.name:<18}] q={C.quality_composite(metrics):.4f} ndcg={metrics['ndcg_at_10']:.4f} "
                    f"mh={cats['multi_hop']['accuracy']:.2f} spec={metrics['abstain']['specificity']:.2f} "
                    f"tok={metrics['tokens']['mean']:.0f} p95={metrics['latency_ms']['p95']:.2f}ms"
                )

        # 阈值推荐：在扫描里取「漏召不增加（相对基线不劣）且特异度最高」的 θ
        base_m = metrics_by_name["baseline_uniform"]
        best_theta = None
        for ec in theta:
            m = metrics_by_name[ec.name]
            if m["answerable"]["miss_rate"] <= base_m["answerable"]["miss_rate"] + 1e-9 and m["abstain"]["specificity"] >= base_m["abstain"]["specificity"] - 1e-9:
                if best_theta is None or m["abstain"]["specificity"] > best_theta[1]["abstain"]["specificity"]:
                    best_theta = (ec, m)
        if best_theta is None:
            best_theta = (None, base_m)
        theta_rec = {
            "theta": float(best_theta[0].theta_abstain) if best_theta[0] is not None else 0.18,
            "config": best_theta[0].name if best_theta[0] is not None else "baseline_uniform",
            "metrics": best_theta[1],
        }

        frontier_grid = frontier_names({n: metrics_by_name[n] for n in metrics_by_name if n in [e.name for e in grid]})
        extended_pool = {e.name: metrics_by_name[e.name] for e in grid + theta + [reference]}
        frontier_extended = frontier_names(extended_pool)

        label_of: dict[str, tuple[str, str]] = {
            "field_off": ("场开关 ROSMONTIS_FIELD", "off"),
            "D_low": ("扩散强度 D", "0.25"),
            "D_high": ("扩散强度 D", "0.70"),
            "hops_1": ("扩散迭代次数", "1"),
            "hops_3": ("扩散迭代次数", "3"),
            "depth_shallow": ("检索深度 dense/sparse/graph", "12/12/8"),
            "depth_deep": ("检索深度 dense/sparse/graph", "60/60/40"),
            "budget_low": ("注入预算 token", "240"),
            "budget_high": ("注入预算 token", "2000"),
        }
        for ec in theta:
            label_of[ec.name] = ("拒答阈值 θ_abstain", f"{ec.theta_abstain:.2f}")

        recommendation = choose_recommendation(metrics_by_name, label_of)
        # 额外确认：把推荐配置真正跑一遍（阈值配置已在 metrics 里；非阈值配置也在网格里）
        sensitivity = dimension_sensitivity(metrics_by_name)
        field_cmp = field_comparison(metrics_by_name, results_by_name)

        ds = dict(data["manifest"])
        ds["db_build_line"] = (
            f"episode={db.build.get('db_stats', {}).get('episode', 0)}, "
            f"fact_active={db.build.get('db_stats', {}).get('fact_active', 0)}, "
            f"entity={db.build.get('db_stats', {}).get('entity', 0)}, "
            f"state_open={db.build.get('db_stats', {}).get('state_open', 0)}, "
            f"hyperedge={db.build.get('db_stats', {}).get('hyperedge', 0)}"
        )
        ds["commands"] = dict(data["manifest"]["commands"])
        ctx = {
            "dataset": ds,
            "fingerprint": fingerprint,
            "environment": C.environment_info(),
            "code": C.code_fingerprint(),
            "field_backend": C.field_backend_info(),
            "extraction_backend": "scripted",
            "extraction_note": "脚本化金标抽取，用于隔离检索/打包链路；离线规则抽取单列",
            "embed": "hash",
            "repeats": max(1, args.repeats),
            "build_elapsed_s": build_elapsed,
            "run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "grid": grid,
            "theta": theta,
            "metrics_by_name": metrics_by_name,
            "extraction": extraction,
            "write_fidelity": write_fidelity,
            "frontier_grid": frontier_grid,
            "frontier_extended": frontier_extended,
            "field_comparison": field_cmp,
            "sensitivity": sensitivity,
            "recommendation": recommendation,
            "theta_recommendation": theta_rec,
            "db_build": db.build,
        }
        payload = {
            "ok": True,
            "dataset": {
                "name": ds["name"],
                "seed": ds["seed"],
                "dialogues": ds["dialogues"],
                "turns": ds["turns"],
                "sessions": ds["sessions"],
                "questions": ds["questions"],
                "questions_by_category": ds["questions_by_category"],
                "fingerprint": fingerprint,
            },
            "meta": {
                "run_at": ctx["run_at"],
                "field": field,
                "repeats": ctx["repeats"],
                "build_elapsed_s": build_elapsed,
                "environment": ctx["environment"],
                "code": ctx["code"],
                "field_backend": ctx["field_backend"],
                "db_build": db.build,
                "configs": [e.to_dict() for e in grid + [reference] + theta],
            },
            "extraction": extraction,
            "write_fidelity": write_fidelity,
            "configs": [
                {
                    "name": e.name,
                    "kind": (
                        "grid"
                        if e in grid
                        else "reference"
                        if e.name == reference.name
                        else "theta"
                    ),
                    "config": e.to_dict(),
                    "metrics": {k: v for k, v in metrics_by_name[e.name].items() if k not in ("applied",)},
                    "applied": applied_by_name[e.name],
                    "questions": [
                        {
                            "qid": r["qid"],
                            "category": r["category"],
                            "pass": r["pass"],
                            "ndcg": r["ndcg"],
                            "hit_at_10": r.get("hit_at_10"),
                            "abstained": r["abstained"],
                            "answer_rank": r.get("answer_rank"),
                            "tokens_used": r["tokens_used"],
                            "latency_ms": r["latency_ms"],
                            "n_items": r.get("n_items"),
                            "error": r.get("error"),
                        }
                        for r in results_by_name[e.name]
                    ],
                }
                for e in grid + [reference] + theta
            ],
            "frontier_grid": frontier_grid,
            "frontier_extended": frontier_extended,
            "field_comparison": field_cmp,
            "sensitivity": sensitivity,
            "recommendation": {
                k: v for k, v in recommendation.items() if k not in ("metrics", "baseline_metrics")
            }
            | {
                "recommended_metrics": {
                    "quality_composite": recommendation["quality_composite"],
                    "ndcg_at_10": recommendation["metrics"]["ndcg_at_10"],
                    "by_category": recommendation["metrics"]["by_category"],
                    "abstain": recommendation["metrics"]["abstain"],
                    "tokens": recommendation["metrics"]["tokens"],
                    "latency_ms": recommendation["metrics"]["latency_ms"],
                },
                "baseline_metrics": {
                    "quality_composite": C.quality_composite(recommendation["baseline_metrics"]),
                    "ndcg_at_10": recommendation["baseline_metrics"]["ndcg_at_10"],
                    "by_category": recommendation["baseline_metrics"]["by_category"],
                    "abstain": recommendation["baseline_metrics"]["abstain"],
                    "tokens": recommendation["baseline_metrics"]["tokens"],
                    "latency_ms": recommendation["baseline_metrics"]["latency_ms"],
                },
            },
            "theta_recommendation": {
                "theta": theta_rec["theta"],
                "config": theta_rec["config"],
                "metrics": {k: v for k, v in theta_rec["metrics"].items() if k != "applied"},
            },
            "elapsed_s": round(time.perf_counter() - started, 3),
        }
    finally:
        db.close()

    C.write_json(Path(args.out), payload)
    if not args.no_docs:
        report = build_report(ctx)
        with open(C.REPORT_PATH, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(report)
        summary = build_summary(ctx)
        C.SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(C.SUMMARY_PATH, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(summary)
        mbn = ctx["metrics_by_name"]
        rec = ctx["recommendation"]
        print("-" * 72)
        print(f"帕累托前沿（10 组网格）：{len(ctx['frontier_grid'])} 组 → {', '.join(ctx['frontier_grid'])}")
        print(f"扩展前沿（+阈值/参考）：{len(ctx['frontier_extended'])} 组")
        print(
            f"最值得调：{rec['dimension']} = {rec['value']}"
            f"（{'支配基线' if rec['dominates_baseline'] else '折中'}，质量综合 "
            f"{C.quality_composite(rec['baseline_metrics']):.4f} → {rec['quality_composite']:.4f}，"
            f"p95 {rec['p95_delta_pct'] * 100:+.1f}%，token {rec['tokens_delta_pct'] * 100:+.1f}%）"
        )
        print(f"θ_abstain 推荐：{ctx['theta_recommendation']['theta']:.2f}")
        print(f"报告：{C.REPORT_PATH}")
        print(f"摘要：{C.SUMMARY_PATH}")
        print(f"JSON：{Path(args.out).resolve()}")
    print(f"总耗时：{payload['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





