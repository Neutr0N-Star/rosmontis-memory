"""遗忘：降权 → 折叠 → 归档。

三级都**不删数据**。遗忘是降低可及性，不是消灭。
保护规则优先于一切：salience 高或 pinned 的条目永不折叠、永不归档。
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from ..config import Config
from . import db as DB
from . import ids as IDS
from .llm import LLM

log = logging.getLogger("alaya.forget")


@dataclass
class ForgetReport:
    scanned: int = 0
    archived: int = 0
    folded: int = 0
    folded_into: list[str] = field(default_factory=list)
    protected: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def retention(cfg: Config, t_ref: float, access_count: int, salience: float, now: float) -> float:
    """R = exp(-Δt / S)。S 随召回次数与重要度增长，所以重要的事衰减更慢。"""
    s = cfg.decay_base_strength * (1.0 + math.log(1 + max(0, access_count))) * (0.5 + salience)
    if s <= 0:
        return 0.0
    dt = max(0.0, now - t_ref)
    return math.exp(-dt / s)


def run_decay_only(cfg: Config, session_id: str, now: float | None = None) -> int:
    """轻量衰减：只归档已经彻底凉掉的条目。写入路径调用这个，开销小。"""
    now = now or time.time()
    n = 0
    with DB.tx(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT id, t_event, access_count, salience FROM episode
               WHERE session_id=? AND archived=0 AND folded_into IS NULL
                 AND pinned=0 AND tier!='resident'
               LIMIT 2000""",
            (session_id,),
        ).fetchall()
        for r in rows:
            if r["salience"] >= cfg.forget_protect_salience:
                continue
            if retention(cfg, r["t_event"], r["access_count"], r["salience"], now) < cfg.forget_archive_r:
                conn.execute("UPDATE episode SET archived=1 WHERE id=?", (r["id"],))
                n += 1
    return n


def run(cfg: Config, llm: LLM | None, session_id: str | None = None, now: float | None = None, embedder=None) -> ForgetReport:
    """完整遗忘周期：折叠同主题的低价值情节，再归档冷条目。"""
    now = now or time.time()
    rep = ForgetReport()
    sid = session_id or "default"

    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT id, session_id, text, t_event, access_count, salience, pinned
               FROM episode WHERE (?='*' OR session_id=?)
                 AND archived=0 AND folded_into IS NULL AND tier!='resident' AND pinned=0
               ORDER BY t_event ASC LIMIT 5000""",
            (sid, sid),
        ).fetchall()

    rep.scanned = len(rows)
    cold: list[dict] = []
    for r in rows:
        if r["pinned"] or r["salience"] >= cfg.forget_protect_salience:
            rep.protected += 1
            continue
        rr = retention(cfg, r["t_event"], r["access_count"], r["salience"], now)
        if rr < cfg.forget_archive_r:
            cold.append({"row": r, "r": rr})

    # ── 折叠：按超边分组，冷却条目够多就合成摘要 ──
    with DB.tx(cfg.db_path) as conn:
        edge_map: dict[str, list] = {}
        for item in cold:
            r = item["row"]
            hits = conn.execute(
                """SELECT edge_id FROM hyperedge_member
                   WHERE member_type='episode' AND member_id=?""",
                (r["id"],),
            ).fetchall()
            for h in hits:
                edge_map.setdefault(h["edge_id"], []).append(r)

        # 一条 episode 可能同时挂在多个超边下。不加这一步，它会被折叠两次，
        # 生成两份摘要，其中一份成孤儿，另一份的 folded_into 覆盖掉前者。
        # 规则：只归给成员数最多的那个超边。
        best_edge: dict[str, str] = {}
        for eid, members in edge_map.items():
            for r in members:
                cur = best_edge.get(r["id"])
                if cur is None or len(members) > len(edge_map[cur]):
                    best_edge[r["id"]] = eid
        deduped: dict[str, list] = {}
        for r_id, eid in best_edge.items():
            deduped.setdefault(eid, []).append(
                next(r for r in edge_map[eid] if r["id"] == r_id)
            )
        edge_map = deduped

        for eid, members in edge_map.items():
            if len(members) < cfg.fold_min_members:
                continue
            text = _summarize(llm, members)
            summary_id = "ep_fold_" + IDS._digest(eid, int(now))  # noqa: SLF001
            conn.execute(
                """INSERT OR IGNORE INTO episode(id,session_id,seq,role,text,t_event,t_ingest,salience,meta_json)
                   VALUES(?,?,0,'summary',?,?,?,0.55,?)""",
                (
                    summary_id,
                    members[0]["session_id"],
                    text,
                    members[-1]["t_event"],
                    now,
                    '{"is_summary": true}',
                ),
            )
            try:
                from .text import tokenize

                toks = " ".join(tokenize(text))
            except Exception:
                toks = text[:200]
            conn.execute(
                "INSERT INTO episode_fts(episode_id,tokens) VALUES(?,?)",
                (summary_id, toks),
            )
            for m in members:
                conn.execute("UPDATE episode SET folded_into=? WHERE id=?", (summary_id, m["id"]))
            conn.execute(
                "INSERT OR IGNORE INTO hyperedge_member(edge_id,member_type,member_id,weight,added_at) VALUES(?,?,?,1.2,?)",
                (eid, "episode", summary_id, now),
            )
            rep.folded += len(members)
            rep.folded_into.append(summary_id)

        # ── 归档：折叠之后仍然极冷的条目 ──
        for item in cold:
            r = item["row"]
            cur = conn.execute("SELECT folded_into, archived FROM episode WHERE id=?", (r["id"],)).fetchone()
            if cur is None or cur["archived"]:
                continue
            conn.execute("UPDATE episode SET archived=1 WHERE id=?", (r["id"],))
            rep.archived += 1

    log.info("forget: scanned=%d folded=%d archived=%d", rep.scanned, rep.folded, rep.archived)
    return rep


def _summarize(llm: LLM | None, members: list) -> str:
    """折叠摘要：LLM 可用就总结，否则用原文拼接（保真度降低但不丢信息）。"""
    texts = [m["text"] for m in members if m["text"]]
    if llm is not None and llm.provider != "rule" and texts:
        joined = "\n".join(f"- {t[:200]}" for t in texts[:12])
        out = llm.chat(
            "你是记忆压缩器。把下面若干条同一主题的对话片段，压成一段不超过 120 字的事实性摘要。"
            "只保留关于人的持久信息与情感节点，去掉寒暄。直接输出摘要，不要解释。",
            joined,
            max_tokens=220,
        )
        if out and len(out.strip()) > 8:
            return out.strip()[:400]
    joined = "；".join(t[:60] for t in texts[:6])
    return f"（{len(texts)} 条情节折叠）{joined}"[:400]
