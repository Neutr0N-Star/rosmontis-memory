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
from . import item as IT
from . import memory_field as MF
from . import pipeline as TR
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


def retention(
    cfg: Config,
    t_ref: float,
    access_count: int,
    salience: float,
    now: float,
    *,
    space: str = "knowledge",
) -> float:
    """R = exp(-Δt / S)。S 随召回次数与重要度增长，所以重要的事衰减更慢。

    空间衰减倍率（docs/15 §1.1）乘在 Δt 上：
    `Δt_eff = Δt × SPACE_DECAY[space]`，identity 0.1（几乎不衰减）、
    knowledge 1.0（不变）、operational 3.0（烧得快）。
    默认 knowledge，所以旧调用方（audit 等）行为不变。
    """
    s = cfg.decay_base_strength * (1.0 + math.log(1 + max(0, access_count))) * (0.5 + salience)
    if s <= 0:
        return 0.0
    dt = max(0.0, now - t_ref) * MF.decay_factor(space)
    return math.exp(-dt / s)


def run_decay_only(cfg: Config, session_id: str, now: float | None = None) -> int:
    """轻量衰减：只归档已经彻底凉掉的条目。写入路径调用这个，开销小。

    归档走 `pipeline.transition('archive')`：state 与 `item_state_log` 同事务同步，
    不再直接 UPDATE archived=1（M1/M4 收口）。
    """
    now = now or time.time()
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT id, t_event, access_count, salience, tier, source, memory_type FROM episode
               WHERE session_id=? AND archived=0 AND folded_into IS NULL
                 AND pinned=0 AND tier!='resident'
               LIMIT 2000""",
            (session_id,),
        ).fetchall()
        cold: list[str] = []
        for r in rows:
            if r["salience"] >= cfg.forget_protect_salience:
                continue
            space = MF.classify_space("episode", r["tier"], r["source"], r["memory_type"])
            if retention(cfg, r["t_event"], r["access_count"], r["salience"], now, space=space) < cfg.forget_archive_r:
                cold.append(r["id"])

    n = 0
    for item_id in cold:
        try:
            tr = TR.transition(
                item_id, "archive", reason="decay below archive threshold",
                actor=IT.ACTOR_SYSTEM, cfg=cfg,
            )
        except Exception as e:  # 单条失败不影响整轮衰减
            log.warning("decay archive failed for %s: %s", item_id, e)
            continue
        if tr.ok and tr.from_state != tr.to_state:
            n += 1
    return n


def run(cfg: Config, llm: LLM | None, session_id: str | None = None, now: float | None = None, embedder=None) -> ForgetReport:
    """完整遗忘周期：折叠同主题的低价值情节，再归档冷条目。"""
    now = now or time.time()
    rep = ForgetReport()
    sid = session_id or "default"

    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT id, session_id, text, t_event, access_count, salience, pinned,
                      tier, source, memory_type
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
        space = MF.classify_space("episode", r["tier"], r["source"], r["memory_type"])
        rr = retention(cfg, r["t_event"], r["access_count"], r["salience"], now, space=space)
        if rr < cfg.forget_archive_r:
            cold.append({"row": r, "r": rr})

    # ── 折叠：按超边分组，冷却条目够多就合成摘要 ──
    # 摘要创建走 `pipeline.submit`（全仓唯一 episode 写入点），分组/关联读在事务外。
    with DB.reader(cfg.db_path) as conn:
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
        summary_item = IT.MemoryItem(
            id=summary_id,
            session_id=members[0]["session_id"],
            text=text,
            origin=IT.ORIGIN_SUMMARY,
            actor=IT.ACTOR_SYSTEM,
            memory_type="fact",
            salience=0.55,          # 旧口径保留
            emotion=0.0,
            t_event=members[-1]["t_event"],
            t_ingest=now,
            role="summary",         # role='summary' 语义保留
            source="summary",
            meta={"is_summary": True},
        )
        sub = TR.submit(summary_item, project=False, cfg=cfg, embedder=embedder, llm=llm)
        if not sub.ok:
            rep.errors.append(f"fold submit failed for {eid}: {sub.reason}")
            continue
        summary_id = sub.item_id

        with DB.tx(cfg.db_path) as conn:
            for m in members:
                conn.execute("UPDATE episode SET folded_into=? WHERE id=?", (summary_id, m["id"]))
            conn.execute(
                "INSERT OR IGNORE INTO hyperedge_member(edge_id,member_type,member_id,weight,added_at) VALUES(?,?,?,1.2,?)",
                (eid, "episode", summary_id, now),
            )
        rep.folded += len(members)
        rep.folded_into.append(summary_id)

    # ── 归档：折叠之后仍然极冷的条目（走状态机，写 item_state_log）──
    if cold:
        cold_ids = [item["row"]["id"] for item in cold]
        marks = ",".join("?" * len(cold_ids))
        with DB.reader(cfg.db_path) as conn:
            cur_rows = {
                r["id"]: dict(r)
                for r in conn.execute(
                    f"SELECT id, folded_into, archived FROM episode WHERE id IN ({marks})",
                    cold_ids,
                )
            }
        for item in cold:
            r = item["row"]
            cur = cur_rows.get(r["id"])
            if cur is None or cur["archived"]:
                continue
            try:
                tr = TR.transition(
                    r["id"], "archive", reason="cold after fold",
                    actor=IT.ACTOR_SYSTEM,
                    cfg=cfg,
                )
            except Exception as e:
                rep.errors.append(f"archive failed for {r['id']}: {e}")
                continue
            if tr.ok and tr.from_state != tr.to_state:
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
