"""评测公共设施：数据集加载、可插拔后端、临时库构建、配置注入、指标计算。

设计要点
- **只读核心**：本模块只 import / 调用 ``rosmontis/``，不写其中任何文件。
- **可插拔**：抽取后端 ``scripted | rule | llm``；embedding ``hash | ollama``。
  没有模型时走脚本化金标 + 规则抽取 + 哈希向量，绝不因缺 Ollama / 网络而崩。
- **可复现**：脚本化金标与哈希向量都是确定性的；时间旅行通过给
  ``write.ingest(now=t_event)`` 传入事件时间来实现，因此不依赖评测当天日期。
- **配置注入**：扩散开关走 ``ROSMONTIS_FIELD``（核心 field.enabled 原生支持）；
  扩散强度 D 改 ``memory_field.DEFAULT_GAMMA``；迭代次数/检索深度/预算倍率
  通过替换 ``recall.REFLEX_DEPTH`` 的副本实现（评测结束恢复原对象）。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
DATASET_DIR = BENCH_DIR / "dataset"
RESULTS_DIR = BENCH_DIR / "results"
REPORT_PATH = BENCH_DIR / "report.md"
SUMMARY_PATH = ROOT / "bench" / "eval-summary.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

import gen_dataset  # noqa: E402  (bench/dataset/gen_dataset.py)

from rosmontis import __version__ as ROSMONTIS_VERSION  # noqa: E402
from rosmontis.config import Config  # noqa: E402
from rosmontis.core import db as DB  # noqa: E402
from rosmontis.core import extract as EX  # noqa: E402
from rosmontis.core import field as FD  # noqa: E402
from rosmontis.core import ids as IDS  # noqa: E402
from rosmontis.core import memory_field as MF  # noqa: E402
from rosmontis.core import pages as PAG  # noqa: E402
from rosmontis.core import recall as RC  # noqa: E402
from rosmontis.core import write as WR  # noqa: E402
from rosmontis.core.embed import Embedder  # noqa: E402
from rosmontis.core.llm import LLM  # noqa: E402

# 评测是脚本，不需要核心的 INFO/WARNING 日志；降级状态写进 JSON（embed_health / notes）。
logging.getLogger("rosmontis").setLevel(logging.ERROR)
logging.getLogger("rosmontis.embed").setLevel(logging.ERROR)

EMPTY_GOLD = {"entities": [], "facts": [], "state_changes": [], "theme": None}
CATEGORIES = ("temporal", "update", "preference", "multi_hop", "abstain")
DAY = 86400.0


# ── 数据集 ────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dataset(*, regen: bool = False) -> None:
    """缺文件或显式 regen 时重新生成（固定种子，重跑逐字节一致）。"""
    paths = [gen_dataset.CONV_PATH, gen_dataset.SESS_PATH, gen_dataset.QUEST_PATH, gen_dataset.MANIFEST_PATH]
    if regen or not all(p.exists() for p in paths):
        gen_dataset.generate()


def load_dataset() -> dict:
    ensure_dataset()
    with open(gen_dataset.MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(gen_dataset.SESS_PATH, "r", encoding="utf-8") as fh:
        sessions = json.load(fh)
    return {
        "manifest": manifest,
        "sessions": sessions,
        "conversations": load_jsonl(gen_dataset.CONV_PATH),
        "questions": load_jsonl(gen_dataset.QUEST_PATH),
    }


def dataset_fingerprint() -> dict:
    """校验当前数据集文件与 manifest 记录的 sha256 是否一致。"""
    ensure_dataset()
    with open(gen_dataset.MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    files = {}
    for name, expected in (manifest.get("files") or {}).items():
        path = DATASET_DIR / name
        actual = _sha256(path)
        files[name] = {"expected": expected, "actual": actual, "match": actual == expected}
    return {
        "verified": bool(files) and all(v["match"] for v in files.values()),
        "files": files,
        "manifest": manifest,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, sort_keys=False, indent=2)
        fh.write("\n")


# ── 小数/统计工具（numpy 可选，这里全部纯 Python）────────────────
def mean(values) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def percentile(values, p: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * float(p) / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return vals[int(k)]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def _fmt(value, nd: int = 4) -> float:
    return round(float(value), nd)


# ── 可插拔 LLM 后端 ───────────────────────────────────────────
class ScriptedLLM:
    """脚本化抽取器：按写入顺序返回金标 payload，保证可复现（demo_runner 同款思路）。

    provider 不能是 "rule"，否则 extract.extract() 会走规则兜底。
    """

    provider = "scripted"

    def __init__(self) -> None:
        self.model = "gold-script"
        self.calls = 0
        self.failures = 0
        self._queue: list[dict] = []

    def queue(self, payload: dict) -> None:
        self._queue.append(payload)

    def clear(self) -> None:
        self._queue.clear()

    def chat_json(self, system: str, user: str, max_tokens: int = 700):
        self.calls += 1
        return self._queue.pop(0) if self._queue else dict(EMPTY_GOLD)

    def chat(self, system: str, user: str, max_tokens: int = 700, json_mode: bool = False):
        return ""

    def available(self) -> bool:
        return True


class OneShotLLM(ScriptedLLM):
    """一次性抽取用：直接返回给定 payload（测抽取后处理/归一化）。"""

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.queue(payload)


def _make_llm(backend: str, cfg: Config) -> tuple[object | None, str, str]:
    """返回 (llm, 实际后端名, 说明)。"""
    if backend == "scripted":
        return ScriptedLLM(), "scripted", "脚本化金标抽取（确定性，用于隔离检索/打包）"
    if backend == "rule":
        return None, "rule", "核心规则兜底抽取（无模型，覆盖有限）"
    if backend == "llm":
        llm = LLM(
            provider="ollama",
            model=cfg.llm_hot_model,
            ollama_url=cfg.ollama_url,
            timeout=cfg.llm_timeout,
            temperature=cfg.llm_temperature,
        )
        if llm.available():
            return llm, "llm", "本地 Ollama 真实抽取"
        return None, "rule", "Ollama 不可用 → 自动回退规则抽取（不崩）"
    raise ValueError(f"未知抽取后端: {backend}")


# ── 配置注入 ──────────────────────────────────────────────────
@dataclass
class EvalConfig:
    """一组可比较的检索配置。None 表示沿用产品默认值。"""

    name: str
    field: str = "on"             # on / off（ROSMONTIS_FIELD）
    gamma: float | None = None    # 扩散强度 D（memory_field.DEFAULT_GAMMA）
    hops: int | None = None       # 扩散迭代次数（REFLEX_DEPTH[*].hops）
    dense_k: int | None = None    # 向量路 top-k（统一档位时生效）
    sparse_k: int | None = None   # FTS 路 top-k
    graph_k: int | None = None    # 图路 top-k
    budget: int = 1400            # 生效 token 预算（统一档位时绕过倍率）
    uniform_depth: bool = False   # True = 四档反射深度统一成上面的 k 值，预算倍率固定 1.0
    theta_abstain: float | None = None  # 拒答阈值覆盖（None = 产品默认 0.18）
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def field_backend_info() -> dict:
    rec = RC
    return {
        "module": "rosmontis.core.field",
        "native_switch": hasattr(FD, "enabled"),
        "has_propagate": hasattr(FD, "propagate"),
        "has_seed_nodes": hasattr(FD, "seed_nodes"),
        "recall_uses_field": hasattr(rec, "_field_route"),
        "iterations_source": "REFLEX_DEPTH[*].hops",
        "decay_source": "memory_field.DEFAULT_GAMMA",
        "note": "场代码已落地时 ROSMONTIS_FIELD 由核心 field.enabled 原生支持；未落地时本脚本退化为关掉图扩散",
    }


@contextlib.contextmanager
def applied_config(ec: EvalConfig, cfg: Config | None = None):
    """临时应用配置，退出时恢复（不修改任何源文件）。"""
    applied = {"field_env": None, "gamma_patched": False, "depth_patched": False, "theta_patched": False}
    old_env = os.environ.get("ROSMONTIS_FIELD")
    old_depth = getattr(RC, "REFLEX_DEPTH", None)
    old_gamma = getattr(MF, "DEFAULT_GAMMA", None)
    old_theta = getattr(getattr(cfg, "weights", None), "theta_abstain", None) if cfg is not None else None
    try:
        os.environ["ROSMONTIS_FIELD"] = "0" if ec.field == "off" else "1"
        applied["field_env"] = os.environ["ROSMONTIS_FIELD"]
        if ec.theta_abstain is not None and cfg is not None and old_theta is not None:
            cfg.weights.theta_abstain = float(ec.theta_abstain)
            applied["theta_patched"] = True
        if ec.gamma is not None and ec.field != "off" and old_gamma is not None:
            MF.DEFAULT_GAMMA = float(ec.gamma)
            applied["gamma_patched"] = True
        if old_depth is not None and (ec.uniform_depth or ec.hops is not None):
            new_depth = {k: dict(v) for k, v in old_depth.items()}
            for _k, v in new_depth.items():
                if ec.uniform_depth:
                    if ec.dense_k is not None:
                        v["dense"] = int(ec.dense_k)
                    if ec.sparse_k is not None:
                        v["sparse"] = int(ec.sparse_k)
                    if ec.graph_k is not None:
                        v["graph"] = int(ec.graph_k)
                    v["budget"] = 1.0  # 预算按传入值生效，便于跨深度比较
                if ec.hops is not None:
                    v["hops"] = int(ec.hops)
            RC.REFLEX_DEPTH = new_depth
            applied["depth_patched"] = True
        yield applied
    finally:
        if old_theta is not None and cfg is not None:
            cfg.weights.theta_abstain = old_theta
        if old_depth is not None:
            RC.REFLEX_DEPTH = old_depth
        if old_gamma is not None:
            MF.DEFAULT_GAMMA = old_gamma
        if old_env is None:
            os.environ.pop("ROSMONTIS_FIELD", None)
        else:
            os.environ["ROSMONTIS_FIELD"] = old_env


def clear_retrieval_log(cfg: Config) -> None:
    try:
        with DB.tx(cfg.db_path) as conn:
            conn.execute("DELETE FROM retrieval_log")
    except Exception:
        pass


# ── 临时评测库 ────────────────────────────────────────────────
@dataclass
class EvalDB:
    cfg: Config
    embedder: Embedder
    llm: object | None
    path: str
    tempdir: str
    backend: str
    backend_note: str
    build: dict = field(default_factory=dict)

    def close(self) -> None:
        shutil.rmtree(self.tempdir, ignore_errors=True)


def make_config(db_path: str, *, embed_provider: str = "hash", embed_dim: int = 1024, budget: int = 1400) -> Config:
    return Config(
        db_path=str(db_path),
        embed_provider=embed_provider,
        embed_dim=int(embed_dim),
        llm_hot="rule",
        llm_deep="rule",
        log_level="ERROR",
        auto_forget=False,
        budget_tokens=int(budget),
    )


def build_eval_db(data: dict, *, backend: str = "scripted", embed_provider: str = "hash", limit: int = 0) -> EvalDB:
    """把对话写入临时 SQLite 库。time-travel 通过 t_ingest=t_event+120s 实现。"""
    tempdir = tempfile.mkdtemp(prefix="rosmontis_bench_")
    db_path = os.path.join(tempdir, "eval.db")
    cfg = make_config(db_path, embed_provider=embed_provider)
    t0 = time.perf_counter()

    DB.init_db(cfg.db_path)
    with contextlib.suppress(Exception):
        PAG.ensure_tables(cfg)

    embedder = Embedder(
        provider=cfg.embed_provider,
        model=cfg.embed_model,
        dim=cfg.embed_dim,
        ollama_url=cfg.ollama_url,
        timeout=cfg.embed_timeout,
    )
    llm, backend_used, backend_note = _make_llm(backend, cfg)

    convs = data["conversations"]
    if limit:
        convs = convs[: int(limit)]
    stats = {
        "dialogues": 0,
        "turns": 0,
        "added": 0,
        "updated": 0,
        "invalidated": 0,
        "noop": 0,
        "states_changed": 0,
        "entities_new": 0,
        "errors": 0,
        "time_travel": True,
    }
    for conv in convs:
        turns = (
            ("user", conv["user_text"], conv["gold"]),
            ("assistant", conv["assistant_text"], dict(EMPTY_GOLD)),
        )
        for role, text, payload in turns:
            if isinstance(llm, ScriptedLLM):
                llm.queue(payload)
            try:
                r = WR.ingest(
                    cfg,
                    embedder,
                    llm,
                    text=text,
                    session_id=conv["session_id"],
                    role=role,
                    t_event=conv["t_event"],
                    now=conv["t_event"] + 120.0,
                    auto_forget=False,
                )
                result = r.to_dict() if hasattr(r, "to_dict") else dict(r)
            except TypeError:
                # 核心签名变动时的兜底：没有 now 参数就只能用真实时间（时间旅行失效）
                stats["time_travel"] = False
                r = WR.ingest(
                    cfg, embedder, llm, text=text, session_id=conv["session_id"], role=role, t_event=conv["t_event"]
                )
                result = r.to_dict() if hasattr(r, "to_dict") else dict(r)
            except Exception:
                stats["errors"] += 1
                result = {}
            finally:
                if isinstance(llm, ScriptedLLM):
                    llm.clear()
            stats["turns"] += 1
            for key in ("added", "updated", "invalidated", "noop", "states_changed", "entities_new"):
                stats[key] += int(result.get(key) or 0)
        stats["dialogues"] += 1

    stats["elapsed_s"] = round(time.perf_counter() - t0, 3)
    stats["db_stats"] = db_stats(cfg)
    stats["embed_health"] = embedder.health()
    return EvalDB(
        cfg=cfg,
        embedder=embedder,
        llm=llm,
        path=db_path,
        tempdir=tempdir,
        backend=backend_used,
        backend_note=backend_note,
        build=stats,
    )


def db_stats(cfg: Config) -> dict:
    out: dict = {}
    with contextlib.suppress(Exception):
        with DB.reader(cfg.db_path) as conn:
            def c(sql: str) -> int:
                return int(conn.execute(sql).fetchone()[0])

            out = {
                "episode": c("SELECT COUNT(*) FROM episode"),
                "fact_all": c("SELECT COUNT(*) FROM fact"),
                "fact_active": c("SELECT COUNT(*) FROM fact WHERE status='active'"),
                "entity": c("SELECT COUNT(*) FROM entity"),
                "state_open": c("SELECT COUNT(*) FROM state_slot WHERE until IS NULL"),
                "hyperedge": c("SELECT COUNT(*) FROM hyperedge"),
                "hyperedge_member": c("SELECT COUNT(*) FROM hyperedge_member"),
            }
    return out


# ── 抽取质量（P/R/F1）────────────────────────────────────────
def _fact_key_from_gold(f: dict) -> tuple:
    return (
        IDS.normalize(f.get("subject", "")),
        EX._canon_predicate(str(f.get("predicate", ""))),
        IDS.normalize(f.get("object", "")),
        int(f.get("polarity", 1)),
    )


def _fact_key_from_cand(f) -> tuple:
    return (
        IDS.normalize(getattr(f, "subject", "")),
        EX._canon_predicate(str(getattr(f, "predicate", ""))),
        IDS.normalize(getattr(f, "obj", "")),
        int(getattr(f, "polarity", 1)),
    )


def _prf(matched: int, predicted: int, gold: int) -> dict:
    p = matched / predicted if predicted else 0.0
    r = matched / gold if gold else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "precision": _fmt(p),
        "recall": _fmt(r),
        "f1": _fmt(f1),
        "matched": int(matched),
        "predicted": int(predicted),
        "gold": int(gold),
    }


def measure_extraction(conversations: list[dict], *, backend: str = "rule", limit: int = 0) -> dict:
    """对 200 个用户 turn 跑抽取，与生成器金标比对（micro 汇总）。"""
    convs = conversations[: int(limit)] if limit else conversations
    pred_facts = gold_facts = match_facts = 0
    pred_ents = gold_ents = match_ents = 0
    pred_states = gold_states = match_states = 0
    failed = 0

    for conv in convs:
        text = conv["user_text"]
        gold = conv["gold"]
        try:
            if backend == "scripted":
                ext = EX.extract(text, OneShotLLM(gold), conv["t_event"], role="user", use_llm=True)
            elif backend == "llm":
                llm = LLM(
                    provider="ollama",
                    model="minicpm5-2b:q8-64k",
                    ollama_url=os.environ.get("ROSMONTIS_OLLAMA", "http://127.0.0.1:11434"),
                    timeout=10.0,
                )
                if not llm.available():
                    backend = "rule"
                    ext = EX.extract(text, None, conv["t_event"], role="user", use_llm=False)
                else:
                    ext = EX.extract(text, llm, conv["t_event"], role="user", use_llm=True)
            else:
                ext = EX.extract(text, None, conv["t_event"], role="user", use_llm=False)
        except Exception:
            failed += 1
            continue

        g_facts = {_fact_key_from_gold(f) for f in gold.get("facts") or []}
        p_facts = {_fact_key_from_cand(f) for f in ext.facts}
        g_ents = {IDS.normalize(e.get("name", "")) for e in gold.get("entities") or [] if e.get("name")}
        p_ents = {IDS.normalize(e.name) for e in ext.entities if e.name}
        g_states = {(s.get("slot"), IDS.normalize(s.get("value", ""))) for s in gold.get("state_changes") or []}
        p_states = {(s.slot, IDS.normalize(s.value)) for s in ext.state_changes}
        gold_facts += len(g_facts)
        pred_facts += len(p_facts)
        match_facts += len(g_facts & p_facts)
        gold_ents += len(g_ents)
        pred_ents += len(p_ents)
        match_ents += len(g_ents & p_ents)
        gold_states += len(g_states)
        pred_states += len(p_states)
        match_states += len(g_states & p_states)

    return {
        "backend": backend,
        "dialogues": len(convs),
        "failed_turns": failed,
        "entity": _prf(match_ents, pred_ents, gold_ents),
        "fact": _prf(match_facts, pred_facts, gold_facts),
        "state_slot": _prf(match_states, pred_states, gold_states),
    }


def measure_write_fidelity(cfg: Config, sessions: dict) -> dict:
    """写入保真度：脚本化金标经 write/pipeline 落库后，active 事实与当前槽是否还在。"""
    stored_facts: set[tuple] = set()
    with contextlib.suppress(Exception):
        with DB.reader(cfg.db_path) as conn:
            rows = conn.execute(
                """SELECT f.predicate, f.polarity, f.object_text, e.name AS subject_name
                   FROM fact f LEFT JOIN entity e ON e.id = f.subject_id
                   WHERE f.status='active'"""
            ).fetchall()
            for r in rows:
                stored_facts.add(
                    (
                        IDS.normalize(r["subject_name"] or ""),
                        EX._canon_predicate(r["predicate"]),
                        IDS.normalize(r["object_text"]),
                        int(r["polarity"]),
                    )
                )

    expected_facts: set[tuple] = set()
    expected_slots = 0
    matched_slots = 0
    for sid, sess in sessions.items():
        for f in sess.get("expected_active_facts") or []:
            expected_facts.add(_fact_key_from_gold(f))
        exp_state = sess.get("expected_state_now") or {}
        with contextlib.suppress(Exception):
            with DB.reader(cfg.db_path) as conn:
                for slot, value in exp_state.items():
                    expected_slots += 1
                    row = conn.execute(
                        """SELECT s.value_text FROM state_slot s
                           JOIN entity e ON e.id = s.scope
                           WHERE s.session_id=? AND e.norm_name='我' AND s.slot_key=?
                             AND s.until IS NULL ORDER BY s.since DESC LIMIT 1""",
                        (sid, slot),
                    ).fetchone()
                    if row is not None and IDS.normalize(row["value_text"]) == IDS.normalize(value):
                        matched_slots += 1

    matched_facts = len(stored_facts & expected_facts)
    fact_prf = _prf(matched_facts, len(stored_facts), len(expected_facts))
    return {
        "fact": fact_prf,
        "state_slot_current": {
            "accuracy": _fmt(matched_slots / expected_slots if expected_slots else 0.0),
            "matched": matched_slots,
            "expected": expected_slots,
        },
        "note": "脚本化 gold 经 write/pipeline 后的落库保真度，用于隔离检索链路的误差",
    }


# ── 判定与指标 ────────────────────────────────────────────────
def _norm_text(s) -> str:
    return "".join(str(s or "").lower().split())


def _atom_in(atom: str, text: str) -> bool:
    atom_n = _norm_text(atom)
    return bool(atom_n) and atom_n in _norm_text(text)


def _fact_match(text: str, predicate: str, obj: str, polarity: int) -> bool:
    t = _norm_text(text)
    obj_n = _norm_text(obj)
    pred_n = _norm_text(predicate)
    if not obj_n or obj_n not in t:
        return False
    neg = _norm_text("不" + str(predicate))
    if int(polarity) >= 0:
        if pred_n not in t:
            return False
        if neg and neg in t:
            return False
        return True
    return bool(neg and neg in t) or pred_n in ("讨厌", "害怕")


def _first_rank(texts: list[str], atoms: list[str]) -> int | None:
    for i, t in enumerate(texts):
        if any(_atom_in(a, t) for a in atoms):
            return i
    return None


def _first_fact_rank(texts: list[str], predicate: str, obj: str, polarity: int) -> int | None:
    for i, t in enumerate(texts):
        if _fact_match(t, predicate, obj, polarity):
            return i
    return None


def _ndcg_value(q: dict, texts: list[str]) -> float:
    answer = q.get("answer_atoms") or []
    bridge = q.get("bridge_atoms") or []
    gains: list[int] = []
    for t in texts[:10]:
        if any(_atom_in(a, t) for a in answer):
            gains.append(2)
        elif any(_atom_in(b, t) for b in bridge):
            gains.append(1)
        else:
            gains.append(0)
    ideal = sorted([2] * max(1, len(answer)) + [1] * len(bridge), reverse=True)[:10]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _ndcg_fact(q: dict, texts: list[str]) -> float:
    m = q["match"]
    gains: list[int] = []
    for t in texts[:10]:
        if _fact_match(t, m["predicate"], m["object"], m["polarity"]):
            gains.append(2)
        elif _atom_in(m["object"], t):
            gains.append(1)
        else:
            gains.append(0)
    ideal = [2]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def judge_question(q: dict, items: list[dict], abstained: bool, tokens_used: int, latency_ms: float) -> dict:
    """判定单题：答案是否在**实际注入的上下文**里；nDCG@10 只看前 10 条排名。

    口径
    - 四类准确率 = 答案证据是否出现在本轮注入的全部条目中（这就是宿主真正看到的
      上下文；预算越小注入越少，准确率会随预算变化，这是帕累托的质量轴）。
    - nDCG@10 = 前 10 条注入顺序上的排序质量（答案排得越靠前越高）。
    - ``hit_at_10`` 单独记录，用来区分「答案在上下文里」与「答案进得了前 10」。
    """
    top = items[:10]
    texts = [str(i.get("text") or "") for i in top]
    all_texts = [str(i.get("text") or "") for i in items]
    kind = (q.get("match") or {}).get("type", "value")
    out = {
        "qid": q["id"],
        "category": q["category"],
        "abstained": bool(abstained),
        "tokens_used": int(tokens_used or 0),
        "latency_ms": float(latency_ms or 0.0),
        "n_items": len(items),
        "pass": False,
        "hit_at_10": False,
        "ndcg": None,
        "answer_rank": None,
        "bridge_rank": None,
        "stale_rank": None,
        "error": None,
    }
    if kind == "abstain":
        out["pass"] = bool(abstained)
        return out
    if abstained:
        out["ndcg"] = 0.0
        return out

    if kind in ("value", "multi_hop"):
        answer_atoms = q.get("answer_atoms") or []
        bridge_atoms = q.get("bridge_atoms") or []
        answer_ranks = [_first_rank(all_texts, [a]) for a in answer_atoms]
        bridge_ranks = [_first_rank(all_texts, [b]) for b in bridge_atoms]
        answers_ok = bool(answer_ranks) and all(r is not None for r in answer_ranks)
        bridges_ok = all(r is not None for r in bridge_ranks)
        out["answer_rank"] = max(answer_ranks) if answers_ok else None
        out["bridge_rank"] = max(bridge_ranks) if (bridge_ranks and bridges_ok) else None
        stale_rank = None
        for atom in q.get("stale_atoms") or []:
            r = _first_rank(all_texts, [atom])
            if r is not None and (stale_rank is None or r < stale_rank):
                stale_rank = r
        out["stale_rank"] = stale_rank
        stale_ok = not (stale_rank is not None and out["answer_rank"] is not None and stale_rank < out["answer_rank"])
        out["pass"] = bool(answers_ok and bridges_ok and stale_ok)
        top10_answers = [_first_rank(texts, [a]) for a in answer_atoms]
        top10_bridges = [_first_rank(texts, [b]) for b in bridge_atoms]
        out["hit_at_10"] = bool(
            bool(top10_answers)
            and all(r is not None for r in top10_answers)
            and all(r is not None for r in top10_bridges)
        )
        out["ndcg"] = _ndcg_value(q, texts)
        return out

    if kind == "fact":
        m = q["match"]
        rank = _first_fact_rank(all_texts, m["predicate"], m["object"], m["polarity"])
        stale_rank = None
        if m.get("stale_polarity") is not None:
            stale_rank = _first_fact_rank(all_texts, m["predicate"], m["object"], int(m["stale_polarity"]))
        out["answer_rank"] = rank
        out["stale_rank"] = stale_rank
        ok = rank is not None
        if ok and stale_rank is not None and stale_rank < rank:
            ok = False
        out["pass"] = bool(ok)
        out["hit_at_10"] = _first_fact_rank(texts, m["predicate"], m["object"], m["polarity"]) is not None
        out["ndcg"] = _ndcg_fact(q, texts)
        return out

    return out


def _agg_category(rows: list[dict], category: str) -> dict:
    ok_rows = [r for r in rows if not r.get("error")]
    n = len(ok_rows)
    if category == "abstain":
        correct = sum(1 for r in ok_rows if r["abstained"])
        return {
            "n": n,
            "accuracy": _fmt(correct / n if n else 0.0),
            "ndcg_at_10": None,
            "hit_at_10": None,
            "miss_rate": None,
            "errors": len(rows) - n,
        }
    ndcgs = [r["ndcg"] for r in ok_rows if r.get("ndcg") is not None]
    return {
        "n": n,
        "accuracy": _fmt(mean(1.0 if r["pass"] else 0.0 for r in ok_rows)),
        "ndcg_at_10": _fmt(mean(ndcgs)) if ndcgs else 0.0,
        "hit_at_10": _fmt(mean(1.0 if r.get("hit_at_10") else 0.0 for r in ok_rows)),
        "miss_rate": _fmt(mean(1.0 if r["abstained"] else 0.0 for r in ok_rows)),
        "errors": len(rows) - n,
    }


def aggregate_results(results: list[dict], *, budget: int, plan_meta: list[dict] | None = None) -> dict:
    errs = [r for r in results if r.get("error")]
    answerable = [r for r in results if r["category"] != "abstain" and not r.get("error")]
    abstain = [r for r in results if r["category"] == "abstain" and not r.get("error")]
    by_cat = {cat: _agg_category([r for r in results if r["category"] == cat], cat) for cat in CATEGORIES}

    ndcgs = [r["ndcg"] for r in answerable if r.get("ndcg") is not None]
    tokens = [r["tokens_used"] for r in answerable + abstain]
    lat = [r["latency_ms"] for r in answerable + abstain]
    correct_abstain = sum(1 for r in abstain if r["abstained"])
    false_recall = sum(1 for r in abstain if not r["abstained"])
    all_abstained = sum(1 for r in answerable + abstain if r["abstained"])

    plan_meta = plan_meta or []
    return {
        "ndcg_at_10": _fmt(mean(ndcgs)),
        "by_category": by_cat,
        "answerable": {
            "n": len(answerable),
            "accuracy": _fmt(mean(1.0 if r["pass"] else 0.0 for r in answerable)),
            "ndcg_at_10": _fmt(mean(ndcgs)),
            "hit_at_10": _fmt(mean(1.0 if r.get("hit_at_10") else 0.0 for r in answerable)),
            "miss_rate": _fmt(mean(1.0 if r["abstained"] else 0.0 for r in answerable)),
        },
        "abstain": {
            "n": len(abstain),
            "false_recall_rate": _fmt(false_recall / len(abstain) if abstain else 0.0),
            "specificity": _fmt(correct_abstain / len(abstain) if abstain else 0.0),
            "precision": _fmt(correct_abstain / all_abstained if all_abstained else 1.0),
            "correct": correct_abstain,
        },
        "tokens": {
            "mean": _fmt(mean(tokens), 1),
            "p50": _fmt(percentile(tokens, 50), 1),
            "p95": _fmt(percentile(tokens, 95), 1),
            "max": int(max(tokens)) if tokens else 0,
            "budget": int(budget),
            "budget_violation_rate": _fmt(mean(1.0 if t > budget else 0.0 for t in tokens)),
        },
        "latency_ms": {
            "mean": _fmt(mean(lat), 2),
            "p50": _fmt(percentile(lat, 50), 2),
            "p95": _fmt(percentile(lat, 95), 2),
            "max": _fmt(max(lat) if lat else 0.0, 2),
        },
        "field": {
            "active_question_rate": _fmt(mean(1.0 if p.get("iterations") else 0.0 for p in plan_meta)) if plan_meta else 0.0,
            "mean_iterations": _fmt(mean(p.get("iterations") or 0 for p in plan_meta), 2) if plan_meta else 0.0,
            "mean_seeds": _fmt(mean(p.get("seeds") or 0 for p in plan_meta), 2) if plan_meta else 0.0,
        },
        "budget": int(budget),
        "errors": len(errs),
    }


def evaluate_questions(
    db: EvalDB, data: dict, ec: EvalConfig, *, repeats: int = 1
) -> tuple[list[dict], list[dict], dict]:
    """在同一临时库上按一组配置跑全部题目。返回 (单题结果, plan 元信息, 配置注入记录)。

    ``repeats`` > 1 时同一题重复检索并取延迟均值（降低 p95 的偶然波动）；判定用第一次结果。
    """
    questions = data["questions"]
    repeats = max(1, int(repeats))
    clear_retrieval_log(db.cfg)
    results: list[dict] = []
    plan_meta: list[dict] = []
    with applied_config(ec, db.cfg) as applied:
        # 预热一次，避免首题把惰性导入/连接开销算进 p95
        try:
            warm = questions[0]
            RC.recall(
                db.cfg,
                db.embedder,
                "预热",
                session_id=warm["session_id"],
                as_of=warm["as_of"],
                budget=ec.budget,
            )
        except Exception:
            pass
        for q in questions:
            try:
                latencies: list[float] = []
                res = None
                for _ in range(repeats):
                    res = RC.recall(
                        db.cfg,
                        db.embedder,
                        q["query"],
                        session_id=q["session_id"],
                        as_of=q["as_of"],
                        budget=ec.budget,
                    )
                    latencies.append(float(res.latency_ms or 0.0))
                items = res.to_dict().get("items") or []
                latency = mean(latencies) if latencies else 0.0
                row = judge_question(q, items, res.abstained, res.tokens_used, latency)
                row["latency_samples"] = len(latencies)
                plan = getattr(res, "plan", {}) or {}
                plan_meta.append(
                    {
                        "depth": plan.get("depth"),
                        "iterations": plan.get("iterations"),
                        "seeds": plan.get("seeds"),
                        "field_enabled": plan.get("field_enabled"),
                    }
                )
            except Exception as exc:  # 单题失败不拖垮整轮
                row = {
                    "qid": q["id"],
                    "category": q["category"],
                    "abstained": True,
                    "tokens_used": 0,
                    "latency_ms": 0.0,
                    "n_items": 0,
                    "pass": False,
                    "ndcg": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(row)
    applied_info = {
        **applied,
        "config": ec.to_dict(),
        "repeats": repeats,
        "field_backend": field_backend_info(),
    }
    return results, plan_meta, applied_info


def evaluate_config(
    db: EvalDB, data: dict, ec: EvalConfig, *, repeats: int = 1
) -> tuple[dict, list[dict], dict]:
    results, plan_meta, applied = evaluate_questions(db, data, ec, repeats=repeats)
    metrics = aggregate_results(results, budget=ec.budget, plan_meta=plan_meta)
    metrics["applied"] = applied
    metrics["quality_composite"] = _fmt(quality_composite(metrics))
    return metrics, results, applied


# ── 环境/代码指纹 ────────────────────────────────────────────
def environment_info() -> dict:
    try:
        import numpy as np  # noqa: F401

        numpy_version = np.__version__
    except Exception:
        numpy_version = None
    try:
        from rosmontis.core.text import _HAS_JIEBA  # noqa: PLC0415

        jieba = bool(_HAS_JIEBA)
    except Exception:
        jieba = False
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "numpy": numpy_version,
        "jieba": jieba,
        "rosmontis_version": ROSMONTIS_VERSION,
        "embed_default": "hash（无 Ollama 时哈希伪向量，核心不落库向量 → 稠密路退化）",
        "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", ""),
        "ROSMONTIS_FIELD": os.environ.get("ROSMONTIS_FIELD", ""),
    }


def code_fingerprint() -> dict:
    out: dict = {"git_head": None, "dirty_core_files": [], "git_ok": False, "core_files": {}}
    for p in sorted((ROOT / "rosmontis").rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        out["core_files"][rel] = _sha256(p)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=10
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True, timeout=10
        )
        if head.returncode == 0:
            out["git_head"] = head.stdout.strip()
            out["git_ok"] = True
        if status.returncode == 0:
            out["dirty_core_files"] = [
                line[3:].strip() for line in status.stdout.splitlines() if line[3:].strip().startswith("rosmontis/")
            ]
    except Exception as exc:  # git 不可用不影响评测
        out["git_error"] = f"{type(exc).__name__}: {exc}"
    return out


def quality_composite(metrics: dict) -> float:
    """帕累托比较用的质量综合分：7 项质量指标等权平均（全部 0-1，越高越好）。"""
    cats = metrics["by_category"]
    return mean(
        [
            metrics["ndcg_at_10"],
            cats["temporal"]["accuracy"],
            cats["multi_hop"]["accuracy"],
            cats["update"]["accuracy"],
            cats["preference"]["accuracy"],
            metrics["abstain"]["specificity"],
            1.0 - metrics["answerable"]["miss_rate"],
        ]
    )


