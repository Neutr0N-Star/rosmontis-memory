"""Memory：进程内库形态的门面。

服务层和 CLI 都通过它调用，保证两套入口行为完全一致。
"""

from __future__ import annotations

import logging
import os
import time

from ..config import Config
from . import audit as AUD
from . import db as DB
from . import forget as FG
from . import ids as IDS
from . import item as IT
from . import notebook as NB
from . import context as CTX
from . import dream as DM
from . import persona as PER
from . import pages as PAG
from . import pipeline as TR
from . import promotion as PR
from . import score as SC
from . import recall as RC
from . import text as TX
from . import timeparse
from . import write as WR
from .embed import Embedder
from .llm import LLM

log = logging.getLogger("alaya.memory")


class Memory:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        logging.basicConfig(level=getattr(logging, self.cfg.log_level, logging.INFO))
        for noisy in ("jieba", "urllib3", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        DB.init_db(self.cfg.db_path)
        # 建表只做一次。放在检索路径里会让每次 recall 都抢一次写锁。
        PAG.ensure_tables(self.cfg)
        self.embedder = Embedder(
            provider=self.cfg.embed_provider,
            model=self.cfg.embed_model,
            dim=self.cfg.embed_dim,
            ollama_url=self.cfg.ollama_url,
            timeout=self.cfg.embed_timeout,
            model_id=self.cfg.embed_model_id,  # 向量身份（v4）
        )
        self.llm_hot = LLM(
            provider=self.cfg.llm_hot,
            model=self.cfg.llm_hot_model,
            ollama_url=self.cfg.ollama_url,
            timeout=self.cfg.llm_timeout,
            temperature=self.cfg.llm_temperature,
            openai_base=self.cfg.openai_base,
            openai_key=self.cfg.openai_key,
            openai_model=self.cfg.openai_model,
        )
        self.llm_deep = LLM(
            provider=self.cfg.llm_deep,
            model=self.cfg.llm_deep_model,
            ollama_url=self.cfg.ollama_url,
            timeout=self.cfg.llm_timeout,
            temperature=self.cfg.llm_temperature,
            openai_base=self.cfg.openai_base,
            openai_key=self.cfg.openai_key,
            openai_model=self.cfg.openai_model,
        )
        # 统一写入管线的进程内上下文：契约签名不带 cfg，由门面绑定
        TR.bind(self.cfg, self.embedder, self.llm_hot)

    # ── 写 ────────────────────────────────────────────────
    def ingest(
        self,
        text: str,
        session_id: str = "default",
        role: str = "user",
        t_event: float | None = None,
    ) -> dict:
        r = WR.ingest(
            self.cfg,
            self.embedder,
            self.llm_hot,
            text=text,
            session_id=session_id,
            role=role,
            t_event=t_event,
        )
        return r.to_dict()

    def ingest_async(
        self,
        text: str,
        session_id: str = "default",
        role: str = "user",
        t_event: float | None = None,
    ) -> dict:
        """快速写入：原文立刻落库，抽取与事实写入放后台。

        对话场景必须用这个——本地 2B 模型抽一条要 2-10 秒，同步等待会让
        对话卡住。原文先存保证不丢，后台补结构。
        """
        import time as _t

        t0 = _t.perf_counter()
        r = WR.ingest(
            self.cfg,
            self.embedder,
            self.llm_hot,
            text=text,
            session_id=session_id,
            role=role,
            t_event=t_event,
            skip_extract=True,
        )
        took = (_t.perf_counter() - t0) * 1000
        d = r.to_dict()
        d["queued"] = True
        d["accept_ms"] = round(took, 1)
        if r.episode_id and not r.duplicate:
            self._enqueue(r.episode_id)
        return d

    def _enqueue(self, episode_id: str) -> None:
        """后台结构投影。池在 pipeline 里（主动/被动共用同一异步通道）。"""
        TR.enqueue_enrich(episode_id, cfg=self.cfg, embedder=self.embedder, llm=self.llm_hot)

    def _run_enrich(self, episode_id: str) -> None:
        try:
            TR.enrich(episode_id, cfg=self.cfg, embedder=self.embedder, llm=self.llm_hot)
        except Exception as e:  # pragma: no cover
            log.debug("enrich failed for %s: %s", episode_id, e)

    def wait_enrich(self, timeout: float = 60.0) -> None:
        """测试与批处理用：等后台抽取跑完。"""
        TR.wait_enrich(timeout=timeout)

    def ingest_batch(self, items: list[dict], session_id: str = "default") -> dict:
        results = []
        for it in items:
            results.append(
                self.ingest(
                    text=it.get("text", ""),
                    session_id=it.get("session_id", session_id),
                    role=it.get("role", "user"),
                    t_event=it.get("t_event"),
                )
            )
        added = sum(r.get("added", 0) for r in results)
        updated = sum(r.get("updated", 0) for r in results)
        return {"count": len(results), "added": added, "updated": updated, "results": results}

    # ── 读 ────────────────────────────────────────────────
    def recall(
        self,
        query: str,
        session_id: str | None = None,
        as_of: float | None = None,
        budget: int | None = None,
        exclude_ids: list[str] | None = None,
    ) -> dict:
        r = RC.recall(
            self.cfg,
            self.embedder,
            query=query,
            session_id=session_id,
            as_of=as_of,
            budget=budget,
            exclude_ids=exclude_ids,
        )
        self._touch([i.ref_id for i in r.items if i.kind in ("fact", "episode")])
        return r.to_dict()

    def _touch(self, ids: list[str]) -> None:
        """采纳一次召回：计数 + 进冷却。

        常驻条目跳过——它从不"想起"，而是"一直知道"，不该被冷却也不该
        影响陈旧度计算。
        """
        if not ids:
            return
        now = time.time()
        pol = self.cfg.policy
        try:
            with DB.tx(self.cfg.db_path) as conn:
                for i in ids:
                    if i.startswith("ep_"):
                        row = conn.execute("SELECT tier FROM episode WHERE id=?", (i,)).fetchone()
                        if row is None or row["tier"] == "resident":
                            continue
                        cool = (
                            now + pol.cooldown_important_hours * 3600
                            if row["tier"] == "important"
                            else now + pol.cooldown_normal_days * 86400
                        )
                        conn.execute(
                            "UPDATE episode SET access_count=access_count+1, last_access=?, cooldown_until=? WHERE id=?",
                            (now, cool, i),
                        )
                    elif i.startswith("fa_"):
                        conn.execute(
                            "UPDATE fact SET access_count=access_count+1, last_access=? WHERE id=?",
                            (now, i),
                        )
        except Exception as e:  # pragma: no cover
            log.debug("touch failed: %s", e)

    # ── 查询接口 ──────────────────────────────────────────
    def entities(self, session_id: str | None = None, etype: str | None = None, limit: int = 100) -> list[dict]:
        sid = session_id or "*"
        with DB.reader(self.cfg.db_path) as conn:
            sql = """SELECT e.*, 
                            (SELECT COUNT(*) FROM fact f WHERE f.subject_id=e.id AND f.status='active') AS fact_count
                     FROM entity e WHERE (?='*' OR e.session_id=?)"""
            args: list = [sid, sid]
            if etype:
                sql += " AND e.type=?"
                args.append(etype)
            sql += " ORDER BY e.salience DESC, e.mentions DESC LIMIT ?"
            args.append(limit)
            rows = conn.execute(sql, args).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "salience": round(r["salience"], 3),
                "mentions": r["mentions"],
                "fact_count": r["fact_count"],
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
            }
            for r in rows
        ]

    def entity_detail(self, entity_id: str, as_of: float | None = None) -> dict | None:
        as_of = as_of or time.time()
        with DB.reader(self.cfg.db_path) as conn:
            e = conn.execute("SELECT * FROM entity WHERE id=?", (entity_id,)).fetchone()
            if e is None:
                return None
            facts = conn.execute(
                """SELECT f.*, s.name AS subject_name FROM fact f
                   LEFT JOIN entity s ON s.id=f.subject_id
                   WHERE f.subject_id=? AND f.t_ingest <= ?
                   ORDER BY f.t_valid_from DESC LIMIT 200""",
                (entity_id, as_of),
            ).fetchall()
            slots = conn.execute(
                """SELECT * FROM state_slot WHERE scope=? AND since <= ?
                   AND (until IS NULL OR until > ?) ORDER BY since DESC""",
                (entity_id, as_of, as_of),
            ).fetchall()

        return {
            "entity": dict(e),
            "facts": [
                {
                    "id": f["id"],
                    "predicate": f["predicate"],
                    "object": f["object_text"],
                    "polarity": f["polarity"],
                    "status": f["status"],
                    "valid_from": f["t_valid_from"],
                    "valid_to": f["t_valid_to"],
                    "invalid_at": f["t_invalid"],
                    "salience": round(f["salience"], 3),
                    "confidence": round(f["confidence"], 2),
                    "source_episode": f["source_episode"],
                }
                for f in facts
            ],
            "states": [
                {
                    "slot": s["slot_key"],
                    "value": s["value_text"],
                    "since": s["since"],
                    "until": s["until"],
                }
                for s in slots
            ],
        }

    def facts(self, session_id: str | None = None, status: str = "active", limit: int = 200) -> list[dict]:
        sid = session_id or "*"
        with DB.reader(self.cfg.db_path) as conn:
            rows = conn.execute(
                """SELECT f.*, s.name AS subject_name FROM fact f
                   LEFT JOIN entity s ON s.id=f.subject_id
                   WHERE (?='*' OR f.session_id=?) AND (?='*' OR f.status=?)
                   ORDER BY f.t_valid_from DESC LIMIT ?""",
                (sid, sid, status, status, limit),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "subject": r["subject_name"] or r["subject_id"],
                "predicate": r["predicate"],
                "object": r["object_text"],
                "polarity": r["polarity"],
                "status": r["status"],
                "valid_from": r["t_valid_from"],
                "valid_to": r["t_valid_to"],
                "salience": round(r["salience"], 3),
            }
            for r in rows
        ]

    def timeline(self, entity_id: str) -> list[dict]:
        with DB.reader(self.cfg.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM fact WHERE subject_id=?
                   ORDER BY t_valid_from ASC""",
                (entity_id,),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "predicate": r["predicate"],
                    "object": r["object_text"],
                    "status": r["status"],
                    "changed_at": timeparse.humanize(r["t_valid_from"]),
                    "invalid_at": timeparse.humanize(r["t_invalid"]) if r["t_invalid"] else None,
                    "t": r["t_valid_from"],
                }
            )
        return out

    _WIPE_TABLES = (
        "retrieval_log", "hyperedge_member", "hyperedge", "state_slot",
        "tombstone", "fact", "entity", "persona",
        "episode_fts", "episode", "observation", "page",
    )

    def wipe(self, session_id: str = "default") -> dict:
        """删掉一个角色的全部数据。

        只动带 session_id 的表。这个操作不可逆，调用方必须先确认。
        """
        counts = {}
        with DB.tx(self.cfg.db_path) as conn:
            # 审计表没有 session_id，先按 episode 归属清掉，避免删库后留孤儿日志
            try:
                cur = conn.execute(
                    "DELETE FROM item_state_log WHERE item_id IN (SELECT id FROM episode WHERE session_id=?)",
                    (session_id,),
                )
                counts["item_state_log"] = cur.rowcount
            except Exception:
                pass
            for t in self._WIPE_TABLES:
                try:
                    col = "episode_id" if t == "episode_fts" else "session_id"
                    cur = conn.execute(f"DELETE FROM {t} WHERE {col}=?", (session_id,))
                    counts[t] = cur.rowcount
                except Exception:
                    pass
        return {"session_id": session_id, "wiped": counts}

    def state_snapshot(
        self, scope: str, as_of: float | None = None, session_id: str | None = None
    ) -> dict:
        # session_id 必须过滤。state_slot 表本来就有这个字段，早期单会话时
        # 漏了过滤条件——加了角色卡之后才暴露：A 角色说"我叫晓星"，
        # B 角色说"我叫小深"，两边会互相覆盖。
        as_of = as_of or time.time()
        with DB.reader(self.cfg.db_path) as conn:
            # entity 查询必须带 session：库里每个会话都有一条 norm_name=「我」，
            # 不过滤的话 OR norm_name 永远命中第一条（别的会话的），
            # 拿着别人的 scope_id 再查本会话的槽 = 永远查无。
            # session_id=None 时不带过滤（向后兼容老调用），此时可能错配。
            if session_id:
                e = conn.execute(
                    "SELECT * FROM entity WHERE (id=? OR norm_name=?) AND session_id=?",
                    (scope, scope.lower(), session_id),
                ).fetchone()
            else:
                e = conn.execute(
                    "SELECT * FROM entity WHERE id=? OR norm_name=?",
                    (scope, scope.lower()),
                ).fetchone()
            scope_id = e["id"] if e else scope
            sql = """SELECT * FROM state_slot WHERE scope=? AND since <= ?
                     AND (until IS NULL OR until > ?)"""
            args = [scope_id, as_of, as_of]
            if session_id:
                sql += " AND session_id=?"
                args.append(session_id)
            sql += " ORDER BY slot_key"
            rows = conn.execute(sql, args).fetchall()
        return {
            "scope": e["name"] if e else scope,
            "as_of": as_of,
            "slots": [{"slot": r["slot_key"], "value": r["value_text"], "since": r["since"]} for r in rows],
        }

    def episodes(self, session_id: str = "default", limit: int = 50, offset: int = 0,
                 include_archived: bool = False) -> dict:
        """记忆清单：前端要展示"她记得哪些事"时用这个，一次拿全。"""
        where = "(?='*' OR session_id=?)"
        if not include_archived:
            where += " AND archived=0"
        with DB.reader(self.cfg.db_path) as conn:
            total = conn.execute(f"SELECT COUNT(*) c FROM episode WHERE {where}", (session_id, session_id)).fetchone()["c"]
            rows = conn.execute(
                f"""SELECT id, role, text, t_event, t_ingest, source, tier, state, origin, memory_type,
                           author, salience, emotion, access_count, archived, muted,
                           folded_into, superseded_by, correction_of, cooldown_until
                    FROM episode WHERE {where}
                    ORDER BY t_event DESC LIMIT ? OFFSET ?""",
                (session_id, session_id, limit, offset),
            ).fetchall()
        return {
            "total": total,
            "items": [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "role": r["role"],
                    "t_event": r["t_event"],
                    "source": r["source"],
                    "tier": r["tier"],
                    "state": r["state"] or IT.compat_state(r["tier"], r["archived"], r["muted"]),
                    "origin": r["origin"] or IT.ORIGIN_INGEST,
                    "memory_type": r["memory_type"],
                    "author": r["author"],
                    "salience": round(r["salience"], 3),
                    "emotion": round(r["emotion"] or 0, 3),
                    "access_count": r["access_count"],
                    "archived": bool(r["archived"]),
                    "muted": bool(r["muted"]),
                    "folded_into": r["folded_into"],
                    "superseded_by": r["superseded_by"],
                    "correction_of": r["correction_of"],
                    "in_cooldown": bool(r["cooldown_until"] and r["cooldown_until"] > time.time()),
                }
                for r in rows
            ],
        }

    # ── 生命周期（状态机对外入口，供 API/测试用）──────────────
    def item_state(self, item_id: str) -> str:
        """读当前状态；旧库未回填时按 tier/archived/muted 兼容投影。"""
        try:
            return TR.state_of(item_id, cfg=self.cfg)
        except Exception:
            with DB.reader(self.cfg.db_path) as conn:
                r = conn.execute(
                    "SELECT state, tier, archived, muted FROM episode WHERE id=?", (item_id,)
                ).fetchone()
            if r is None:
                return ""
            return r["state"] or IT.compat_state(r["tier"], r["archived"], r["muted"])

    def transition(self, item_id: str, action: str, **kwargs) -> dict:
        """状态转移的唯一门面（API/工具/测试都从这里进）。"""
        kwargs.setdefault("cfg", self.cfg)
        kwargs.setdefault("actor", IT.ACTOR_SYSTEM)
        tr = TR.transition(item_id, action, **kwargs)
        return tr.to_dict()

    def state_log(self, item_id: str, limit: int = 50) -> list[dict]:
        return TR.state_log(item_id, cfg=self.cfg, limit=limit)

    def themes(self, session_id: str | None = None, limit: int = 50) -> list[dict]:
        sid = session_id or "*"
        with DB.reader(self.cfg.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM hyperedge WHERE (?='*' OR session_id=?)
                   ORDER BY member_count DESC LIMIT ?""",
                (sid, sid, limit),
            ).fetchall()
            out = []
            for r in rows:
                members = conn.execute(
                    """SELECT member_type, member_id FROM hyperedge_member
                       WHERE edge_id=? LIMIT 20""",
                    (r["id"],),
                ).fetchall()
                out.append(
                    {
                        "id": r["id"],
                        "label": r["label"],
                        "kind": r["kind"],
                        "member_count": r["member_count"],
                        "t_from": r["t_from"],
                        "t_to": r["t_to"],
                        "summary": r["summary"],
                        "members": [{"type": m["member_type"], "id": m["member_id"]} for m in members],
                    }
                )
        return out

    # ── 特化接口：人格 / 升格 / 订正 / 删除 ──────────────
    def persona_list(self, session_id: str = "default") -> list[dict]:
        return PER.list_items(self.cfg, session_id)

    def persona_add(
        self,
        text: str,
        session_id: str = "default",
        section: str = "free",
        author: str = PR.A_SELF,
        protected: bool = False,
        priority: int = 100,
    ) -> dict:
        pid = PER.add(self.cfg, session_id, text, section, author, protected, priority)
        return {"id": pid, "section": section, "author": author}

    def persona_compile(self, session_id: str = "default", budget_chars: int | None = None) -> dict:
        return PER.compile_persona(self.cfg, session_id, budget_chars)

    def persona_revoke(self, persona_id: str) -> dict:
        return {"revoked": PER.revoke(self.cfg, persona_id)}

    def seed(self, session_id: str = "default", text: str | None = None) -> dict:
        PER.ensure_seed(self.cfg, session_id, text)
        return {"seeded": True, "session_id": session_id}

    def promote(self, item_id: str, target_tier: str = PR.TIER_RESIDENT) -> dict:
        """升格闸门：能不能进常驻，不由分数决定。

        判定收口 `gate.decide()`（内部保留 `PR.can_promote`），落库/留痕收口
        `pipeline.transition()` —— 不再直接 UPDATE episode.tier。
        """
        action = {
            PR.TIER_RESIDENT: "promote_resident",
            PR.TIER_IMPORTANT: "mark_important",
            PR.TIER_RETRIEVAL: "demote",
        }.get(target_tier)
        if action is None:
            return {"ok": False, "reason": f"未知层级 {target_tier}"}

        tr = TR.transition(
            item_id,
            action,
            target=target_tier,
            reason="Memory.promote",
            actor=IT.ACTOR_SYSTEM,
            cfg=self.cfg,
        )
        if tr.ok:
            return {
                "ok": True,
                "reason": tr.reason,
                "from": tr.from_state,
                "to": tr.to_state,
                "state": tr.to_state,
            }
        return {
            "ok": False,
            "reason": tr.reason,
            "from": tr.from_state,
            "to": target_tier,
        }

    def confirm(self, item_id: str) -> dict:
        with DB.tx(self.cfg.db_path) as conn:
            conn.execute("UPDATE episode SET confirmed=1 WHERE id=?", (item_id,))
        return {"confirmed": True, "id": item_id}

    def correct(self, old_id: str, new_text: str, author: str = PR.A_SELF) -> dict:
        """订正：旧条目保留，新条目指回旧条目。原文永不覆盖。

        新条经 `pipeline.submit()` 落库（唯一写入点），并继承旧条状态；
        派生事实与状态槽同步失效仍在本方法内完成。
        """
        now = time.time()
        with DB.reader(self.cfg.db_path) as conn:
            r = conn.execute("SELECT * FROM episode WHERE id=?", (old_id,)).fetchone()
        if r is None:
            return {"ok": False, "reason": "原条目不存在"}

        old = IT.MemoryItem.from_row(r)
        new_id = "ep_corr_" + IDS._digest(old_id, new_text, int(now))  # noqa: SLF001
        actor = {
            PR.A_SELF: IT.ACTOR_MODEL,
            PR.A_TEACHER: IT.ACTOR_USER,
            PR.A_SYSTEM: IT.ACTOR_SYSTEM,
        }.get(author, IT.ACTOR_SYSTEM)
        item = IT.MemoryItem(
            id=new_id,
            session_id=old.session_id,
            text=(new_text or "").strip(),
            origin=IT.ORIGIN_INGEST,
            actor=actor,
            memory_type=old.memory_type,
            state=old.state,
            salience=min(1.0, old.salience + 0.1),
            emotion=old.emotion or 0.0,
            t_event=now,
            t_ingest=now,
            role="correction",
            source=PR.S_REFLECTION,
            author=author,
            meta={"is_correction": True},
        )
        sub = TR.submit(item, project=False, cfg=self.cfg, embedder=self.embedder, llm=self.llm_hot)
        if not sub.ok:
            return {"ok": False, "reason": sub.reason}
        if sub.duplicate:
            return {"ok": True, "new_id": sub.item_id, "old_id": old_id, "duplicate": True}

        with DB.tx(self.cfg.db_path) as conn:
            conn.execute("UPDATE episode SET superseded_by=? WHERE id=?", (sub.item_id, old_id))
            # 订正必须落到事实层，否则 prompt 里会出现"当前状态 A"+"我其实住在 B"的矛盾
            conn.execute(
                "UPDATE fact SET status='invalidated', t_invalid=?, t_valid_to=? WHERE source_episode=? AND status='active'",
                (now, now, old_id),
            )
            conn.execute(
                "UPDATE state_slot SET until=? WHERE source_fact IN (SELECT id FROM fact WHERE source_episode=?) AND until IS NULL",
                (now, old_id),
            )
        TR.transition(
            old_id, "revise", target=sub.item_id, reason="订正",
            actor=actor, cfg=self.cfg,
        )
        return {
            "ok": True,
            "new_id": sub.item_id,
            "old_id": old_id,
            "note": "旧条目保留但不再浮现；派生事实与状态已同步失效",
        }

    def remove(self, item_id: str, author: str = PR.A_TEACHER, reason: str = "manual") -> dict:
        """删除：写墓碑，重建索引也不会复活。状态转移走 pipeline（attempts archive）。"""
        now = time.time()
        with DB.tx(self.cfg.db_path) as conn:
            r = conn.execute("SELECT chunk_key, session_id FROM episode WHERE id=?", (item_id,)).fetchone()
            if r is None:
                return {"ok": False, "reason": "条目不存在"}
            conn.execute(
                "INSERT OR REPLACE INTO tombstone(chunk_key,line_ids,session_id,deleted_at,reason,author) VALUES(?,?,?,?,?,?)",
                (r["chunk_key"] or item_id, f'["{item_id}"]', r["session_id"], now, reason, author),
            )
        tr = TR.transition(
            item_id, "archive", reason=f"manual remove: {reason}",
            actor=IT.ACTOR_USER if author == PR.A_TEACHER else IT.ACTOR_SYSTEM,
            cfg=self.cfg,
        )
        if not tr.ok:
            # 状态机拒绝（如已归档）不应让删除失败：墓碑已是删除真值
            log.debug("remove: transition archive failed: %s", tr.reason)
        return {"ok": True, "tombstoned": True, "id": item_id, "state": tr.to_state}

    def resident_list(self, session_id: str = "default") -> list[dict]:
        with DB.reader(self.cfg.db_path) as conn:
            rows = conn.execute(
                """SELECT id, text, source, tier, memory_type, author, salience, t_event
                   FROM episode WHERE session_id=? AND tier='resident' AND muted=0 AND archived=0
                   ORDER BY salience DESC LIMIT ?""",
                (session_id, self.cfg.policy.resident_max_entries),
            ).fetchall()
        return [dict(r) for r in rows]


    # ── 主动记忆：她自己写 / 改 / 升格 ────────────────────
    def notebook_write(
        self,
        what: str,
        session_id: str = "default",
        noticed: str = "",
        means: str = "",
        next_step: str = "",
        topic: str = "",
    ) -> dict:
        return NB.remember(
            self.cfg, self.embedder, session_id,
            what=what, noticed=noticed, means=means, next_step=next_step, topic=topic,
        )

    def notebook_list(self, session_id: str = "default", tier: str | None = None, limit: int = 50) -> list[dict]:
        return NB.list_notes(self.cfg, session_id, tier, limit)

    def notebook_revise(self, note_id: str, new_text: str, reason: str = "") -> dict:
        return NB.revise(self.cfg, note_id, new_text, reason)

    def notebook_promote(self, note_id: str, target: str = "persona", section: str = "free",
                         session_id: str = "") -> dict:
        return NB.promote(self.cfg, note_id, target, section, session_id)

    def notebook_revert(self, promotion_id: str) -> dict:
        return NB.revert(self.cfg, promotion_id)

    def notebook_review(self, session_id: str = "default") -> dict:
        return NB.review(self.cfg, self.llm_deep, session_id)

    # ── 做梦 ──────────────────────────────────────────────
    def dream(self, session_id: str = "default", force: bool = False) -> dict:
        """做一次梦：重放 → 关联 → 抽象 → 脱敏 → 成梦。"""
        ok, why = DM.should_dream(self.cfg, session_id)
        if not ok and not force:
            return {"skipped": True, "reason": why}
        out = DM.dream(self.cfg, self.llm_deep, self.embedder, session_id)
        if force:
            out["forced"] = True
        return out

    def dreams(self, session_id: str = "default", limit: int = 20) -> list[dict]:
        return DM.list_dreams(self.cfg, session_id, limit)

    def dream_candidates(self, session_id: str = "default") -> list[dict]:
        """升格候选：在多次梦里反复出现的判断。"""
        return DM.candidates_for_promotion(self.cfg, session_id)

    # ── 锦囊 ──────────────────────────────────────────────
    def inbox(self, session_id: str = "default") -> dict:
        return NB.inbox_status(self.cfg, session_id)

    def tick(self, session_id: str = "default") -> dict:
        """维护节拍：该整理就整理，该做梦就做梦。

        harness 层的关键——时间驱动，不靠她自己想起来。
        接入方可以每轮调用（很便宜，只是几个 COUNT），也可以由外部定时器调。
        """
        out: dict = {"session_id": session_id, "did": []}

        st = NB.inbox_status(self.cfg, session_id)
        out["inbox"] = st
        if st["should_review"]:
            rep = NB.review(self.cfg, self.llm_deep, session_id)
            out["review"] = rep
            out["did"].append("review")

        ok, why = DM.should_dream(self.cfg, session_id)
        out["dream_check"] = {"ok": ok, "reason": why}
        if ok:
            d = DM.dream(self.cfg, self.llm_deep, self.embedder, session_id)
            out["dream"] = d
            out["did"].append("dream")

        return out

    # ── 上下文编译（热重载入口）──────────────────────────
    def build_context(self, query: str, session_id: str = "default",
                      budget: int | None = None, as_of: float | None = None) -> dict:
        return CTX.build_context(self.cfg, self.embedder, query, session_id, budget, as_of)

    # ── 会话生命周期 ──────────────────────────────────────
    def close_session(self, session_id: str = "default", now: float | None = None) -> dict:
        """结束一次会话，做三件事：峰终结算、沉淀、遗忘。

        为什么要显式"结束"：峰终效应（peak-end rule）说的是人对一段经历的记忆
        由最高点与结尾决定。系统不标出这两个位置，就永远算不出该强化哪条。

        这一步同时是 Letta sleep-time compute 的轻量版——把整理工作挪到会话
        之间做，而不是每轮都做。
        """
        now = now or time.time()
        out: dict = {"session_id": session_id, "scanned": 0, "peak": None, "tail": 0, "touched": 0}

        with DB.reader(self.cfg.db_path) as conn:
            rows = conn.execute(
                """SELECT id, text, emotion, salience, t_event FROM episode
                   WHERE session_id=? AND archived=0 AND folded_into IS NULL
                   ORDER BY t_event DESC LIMIT 300""",
                (session_id,),
            ).fetchall()
        if not rows:
            out["note"] = "会话为空"
            return out

        out["scanned"] = len(rows)
        scored = [(r["id"], r["emotion"] or SC.emotion_intensity(r["text"] or "")) for r in rows]
        peak_id, peak_emo = max(scored, key=lambda kv: kv[1])
        tail_ids = {r["id"] for r in rows[:3]}
        out["peak"] = {"id": peak_id, "emotion": round(peak_emo, 3)}
        out["tail"] = len(tail_ids)

        with DB.tx(self.cfg.db_path) as conn:
            for eid, emo in scored:
                bonus = 0.0
                if eid == peak_id:
                    bonus += 0.25
                if eid in tail_ids:
                    bonus += 0.15
                if bonus <= 0:
                    if emo > 0:
                        conn.execute("UPDATE episode SET emotion=? WHERE id=?", (min(1.0, emo), eid))
                    continue
                conn.execute(
                    "UPDATE episode SET emotion=?, salience=MIN(1.0, salience + ?) WHERE id=?",
                    (min(1.0, emo + bonus * 0.5), bonus, eid),
                )
                out["touched"] += 1

        # 沉淀：重建活文档、归纳认知、跑遗忘
        try:
            out["pages"] = PAG.rebuild_pages(self.cfg, self.llm_deep, session_id, now).get("count", 0)
        except Exception as e:
            log.debug("close: pages failed %s", e)
        try:
            out["notebook_review"] = NB.review(self.cfg, self.llm_deep, session_id, now)
        except Exception as e:
            log.debug("close: notebook review failed %s", e)
        try:
            out["observations"] = PAG.derive_observations(self.cfg, self.llm_deep, session_id, now).get("made", 0)
        except Exception as e:
            log.debug("close: observations failed %s", e)
        if self.cfg.auto_forget:
            try:
                out["forget"] = self.forget(session_id)
            except Exception as e:
                log.debug("close: forget failed %s", e)
        return out

    # ── 沉淀层：常驻活文档 + 后台归纳 ─────────────────────
    def rebuild_pages(self, session_id: str = "default") -> dict:
        return PAG.rebuild_pages(self.cfg, self.llm_deep, session_id)

    def pages(self, session_id: str = "default") -> list[dict]:
        return PAG.list_pages(self.cfg, session_id)

    def resident_block(self, session_id: str = "default", budget: int | None = None) -> dict:
        return PAG.compile_resident(self.cfg, session_id, budget)

    def derive_observations(self, session_id: str = "default") -> dict:
        return PAG.derive_observations(self.cfg, self.llm_deep, session_id)

    def observations(self, session_id: str = "default") -> list[dict]:
        return PAG.list_observations(self.cfg, session_id)

    # ── 维护 ──────────────────────────────────────────────
    def forget(self, session_id: str | None = None) -> dict:
        return FG.run(self.cfg, self.llm_deep, session_id, embedder=self.embedder).to_dict()

    def audit(self, session_id: str | None = None) -> dict:
        return AUD.audit(self.cfg, session_id)

    def stats(self) -> dict:
        with DB.reader(self.cfg.db_path) as conn:
            def c(t: str, where: str = "") -> int:
                sql = f"SELECT COUNT(*) c FROM {t}"
                if where:
                    sql += f" WHERE {where}"
                return int(conn.execute(sql).fetchone()["c"])

            return {
                "episode": c("episode"),
                "episode_active": c("episode", "archived=0"),
                "fact": c("fact"),
                "fact_active": c("fact", "status='active'"),
                "entity": c("entity"),
                "state_open": c("state_slot", "until IS NULL"),
                "theme": c("hyperedge"),
                "db_path": self.cfg.db_path,
                "embed": self.embedder.health(probe=False),
                "embed_model": getattr(self.embedder, "model_id", "") or self.cfg.embed_model_id,
                "embed_dim": int(getattr(self.embedder, "dim", self.cfg.embed_dim) or 0),
                "vector_mode": self.embedder.vector_mode(),
                "pending_backfill": self.pending_backfill(),
                "llm_hot": {"provider": self.llm_hot.provider, "model": self.llm_hot.model, "calls": self.llm_hot.calls, "failures": self.llm_hot.failures},
                "llm_deep": {"provider": self.llm_deep.provider, "model": self.llm_deep.model, "calls": self.llm_deep.calls, "failures": self.llm_deep.failures},
                "degraded": self.embedder.degraded,
            }

    # ── 向量身份与 NULL 回算（C4）────────────────────────
    def vector_mode(self) -> str:
        """semantic / degraded / none（/readyz 与 stats 共用）。"""
        return self.embedder.vector_mode() if self.embedder is not None else "none"

    def pending_backfill(self, session_id: str | None = None) -> int:
        """还差多少条 NULL 向量（episode + fact）。"""
        with DB.reader(self.cfg.db_path) as conn:
            n = 0
            if session_id:
                n += int(
                    conn.execute(
                        "SELECT COUNT(*) c FROM episode WHERE embedding IS NULL AND session_id=?",
                        (session_id,),
                    ).fetchone()["c"]
                )
                n += int(
                    conn.execute(
                        "SELECT COUNT(*) c FROM fact WHERE embedding IS NULL AND session_id=?",
                        (session_id,),
                    ).fetchone()["c"]
                )
            else:
                n += int(conn.execute("SELECT COUNT(*) c FROM episode WHERE embedding IS NULL").fetchone()["c"])
                n += int(conn.execute("SELECT COUNT(*) c FROM fact WHERE embedding IS NULL").fetchone()["c"])
        return n

    def backfill_embeddings(
        self,
        *,
        limit: int = 500,
        session_id: str | None = None,
        now: float | None = None,
    ) -> dict:
        """把降级期写下的 NULL 向量补齐（job kind=embed_backfill 的执行体）。

        - 只读一批 NULL 行 → 批量算向量 → **一个事务** executemany 回写；
        - 回写同时记录向量身份（embed_model/embed_dim）；
        - 语义模式不可用（hash/none/断路）时直接跳过，不写伪向量。
        """
        emb = self.embedder
        mode = emb.vector_mode() if emb is not None else "none"
        base: dict = {
            "scanned": 0,
            "filled": 0,
            "filled_episode": 0,
            "filled_fact": 0,
            "remaining": 0,
            "embed_model": getattr(emb, "model_id", "") if emb is not None else "",
            "embed_dim": int(getattr(emb, "dim", 0) or 0) if emb is not None else 0,
            "vector_mode": mode,
            "skipped": False,
            "reason": "",
        }
        if emb is None or mode != "semantic" or getattr(emb, "degraded", False):
            base["skipped"] = True
            base["reason"] = f"embedder 不可用（vector_mode={mode}），不写伪向量"
            base["remaining"] = self.pending_backfill(session_id)
            return base

        limit = max(1, min(int(limit), 5000))
        with DB.reader(self.cfg.db_path) as conn:
            if session_id:
                episodes = conn.execute(
                    """SELECT id, text FROM episode
                       WHERE embedding IS NULL AND session_id=? AND text IS NOT NULL AND text!=''
                       ORDER BY t_ingest DESC LIMIT ?""",
                    (session_id, limit),
                ).fetchall()
                facts = conn.execute(
                    """SELECT id, predicate, object_text FROM fact
                       WHERE embedding IS NULL AND session_id=?
                       ORDER BY t_ingest DESC LIMIT ?""",
                    (session_id, limit),
                ).fetchall()
            else:
                episodes = conn.execute(
                    """SELECT id, text FROM episode
                       WHERE embedding IS NULL AND text IS NOT NULL AND text!=''
                       ORDER BY t_ingest DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
                facts = conn.execute(
                    """SELECT id, predicate, object_text FROM fact
                       WHERE embedding IS NULL ORDER BY t_ingest DESC LIMIT ?""",
                    (limit,),
                ).fetchall()

        model_id = str(getattr(emb, "model_id", "") or "").strip() or None
        dim = int(getattr(emb, "dim", 0) or 0) or None
        ep_updates: list[tuple] = []
        fact_updates: list[tuple] = []
        for r in episodes:
            if emb.degraded:
                break
            blob = emb.embed_blob(r["text"])
            if emb.degraded or not blob:
                break
            ep_updates.append((blob, model_id, dim, r["id"]))
        for r in facts:
            if emb.degraded:
                break
            text = f"{r['predicate']} {r['object_text']}".strip()
            blob = emb.embed_blob(text)
            if emb.degraded or not blob:
                break
            fact_updates.append((blob, model_id, dim, r["id"]))

        if ep_updates or fact_updates:
            with DB.tx(self.cfg.db_path) as conn:
                if ep_updates:
                    conn.executemany(
                        "UPDATE episode SET embedding=?, embed_model=?, embed_dim=? WHERE id=?",
                        ep_updates,
                    )
                if fact_updates:
                    conn.executemany(
                        "UPDATE fact SET embedding=?, embed_model=?, embed_dim=? WHERE id=?",
                        fact_updates,
                    )

        base["scanned"] = len(episodes) + len(facts)
        base["filled_episode"] = len(ep_updates)
        base["filled_fact"] = len(fact_updates)
        base["filled"] = len(ep_updates) + len(fact_updates)
        if emb.degraded:
            base["skipped"] = True
            base["reason"] = emb.degraded_reason or "embedding 降级，本轮提前停止"
        base["remaining"] = self.pending_backfill(session_id)
        return base

    def sessions(self) -> list[dict]:
        with DB.reader(self.cfg.db_path) as conn:
            rows = conn.execute(
                """SELECT s.*, (SELECT COUNT(*) FROM episode e WHERE e.session_id=s.id) AS ep_count
                   FROM session s ORDER BY s.last_active DESC LIMIT 50"""
            ).fetchall()
        return [dict(r) for r in rows]

    # ── 守护进程辅助：空闲会话判定（C2）──────────────────
    def idle_sessions(
        self,
        idle_timeout_s: float | None = None,
        *,
        now: float | None = None,
        max_age_s: float | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """返回“空闲到该 settle、但还不算远古”的会话。

        判定：`last_active <= now - idle_timeout`，且 `last_active >= now - max_age`
        （后者避免把几个月前的历史会话全部翻出来重结算）。
        `idle_timeout_s` 缺省读 `ALAYA_IDLE_TIMEOUT`（默认 1800 秒）；
        `max_age_s` 缺省读 `ALAYA_SETTLE_MAX_AGE`（默认 86400 秒）。
        """
        now = time.time() if now is None else float(now)
        timeout = _idle_env_float("ALAYA_IDLE_TIMEOUT", 1800.0) if idle_timeout_s is None else float(idle_timeout_s)
        max_age = _idle_env_float("ALAYA_SETTLE_MAX_AGE", 86400.0) if max_age_s is None else float(max_age_s)
        if timeout <= 0:
            return []
        params: list = [now - timeout]
        sql = """SELECT id, last_active, created_at FROM session
                 WHERE last_active IS NOT NULL AND last_active <= ?"""
        if max_age and max_age > 0:
            sql += " AND last_active >= ?"
            params.append(now - max_age)
        sql += " ORDER BY last_active ASC LIMIT ?"
        params.append(max(1, int(limit)))
        with DB.reader(self.cfg.db_path) as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            {
                "session_id": r["id"],
                "last_active": float(r["last_active"]),
                "idle_s": round(now - float(r["last_active"]), 1),
                "idle_timeout_s": timeout,
            }
            for r in rows
        ]


def _idle_env_float(key: str, default: float) -> float:
    try:
        raw = os.environ.get(key)
        return float(raw) if raw not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)
