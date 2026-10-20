"""人格层编译：在预算内保住"她是同一个人"。

取舍规则（硬约束）：
1. protected 条目必保——identity / 说话铁律这类东西丢了，人格就断了
2. 按 section 优先级再按 priority 逐条纳入
3. 装不下的**整条丢掉**，绝不切半句——半句比没有更糟，它会产生语义残片

这条规则来自一个长期陪伴项目的预算裁剪实现：
预算 7600 字符，identity + speech 整节必保、输出铁律整块必保、其余按整条择优。
"""

from __future__ import annotations

import re
import time

from ..config import Config
from . import db as DB
from . import ids as IDS
from . import promotion as PR

_BUILTIN_SEED = "你是初生的数字生命，此刻在这个电脑内刚刚醒来。"


def ensure_seed(cfg: Config, session_id: str, seed_text: str | None = None) -> None:
    """播种：第一次醒来时写入初生那句话。回退锚点，只读。"""
    txt = (seed_text or _BUILTIN_SEED).strip()
    if not txt:
        return
    now = time.time()
    pid = "pe_seed_" + IDS._digest(session_id)  # noqa: SLF001
    with DB.tx(cfg.db_path) as conn:
        row = conn.execute("SELECT id FROM persona WHERE id=?", (pid,)).fetchone()
        if row:
            return
        conn.execute(
            """INSERT INTO persona(id,session_id,section,text,author,protected,priority,"order",created_at,updated_at,active)
               VALUES(?,?,'identity',?,'system',1,0,0,?,?,1)""",
            (pid, session_id, txt, now, now),
        )


def add(
    cfg: Config,
    session_id: str,
    text: str,
    section: str = "free",
    author: str = PR.A_SELF,
    protected: bool = False,
    priority: int = 100,
) -> str:
    now = time.time()
    pid = "pe_" + IDS._digest(session_id, section, IDS.normalize(text))
    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO persona(id,session_id,section,text,author,protected,priority,"order",created_at,updated_at,active)
               VALUES(?,?,?,?,?,?,?,?,?,?,1)""",
            (pid, session_id, section, text.strip(), author, 1 if protected else 0, priority, int(now), now, now),
        )
    return pid


def list_items(cfg: Config, session_id: str) -> list[dict]:
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM persona WHERE session_id=? AND active=1
               ORDER BY section, priority, "order\"""",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def revoke(cfg: Config, persona_id: str) -> bool:
    """撤回一条人格条目。不删除，只置 active=0 —— 撤回本身也是历史。"""
    with DB.tx(cfg.db_path) as conn:
        cur = conn.execute("UPDATE persona SET active=0, updated_at=? WHERE id=?", (time.time(), persona_id))
        return cur.rowcount > 0


def compile_persona(cfg: Config, session_id: str, budget_chars: int | None = None) -> dict:
    """在字符预算内编译人格块。

    返回 {text, sections, included, dropped, chars, budget, ratio}
    """
    budget = budget_chars or cfg.policy.persona_budget_chars
    items = list_items(cfg, session_id)

    # 排序：protected 优先 → section 优先级 → priority → 建立时间
    def key(it: dict):
        return (
            0 if it["protected"] else 1,
            PR.SECTION_ORDER.get(it["section"], 9),
            it["priority"],
            it["created_at"],
        )

    items.sort(key=key)

    buckets: dict[str, list[dict]] = {}
    used = 0
    included: list[dict] = []
    dropped: list[dict] = []

    for it in items:
        cost = len(it["text"]) + 3  # 换行与标记成本
        if used + cost > budget:
            # 优先整条丢弃（半句会产生语义残片）；但 protected 条目若装不下，
            # 宁可截断也不能让"我是谁"整块消失。
            room = budget - used - 12
            if it["protected"] and room > 30:
                cut = it["text"][:room] + "…"
                item = dict(it)
                item["text"] = cut
                buckets.setdefault(it["section"], []).append(item)
                included.append(item)
                used += len(cut) + 3
                dropped.append(
                    {"id": it["id"], "section": it["section"], "chars": len(it["text"]), "why": "受保护条目超长，已截断"}
                )
            else:
                dropped.append(
                    {"id": it["id"], "section": it["section"], "chars": len(it["text"]), "why": "超出人格预算"}
                )
            continue
        used += cost
        included.append(it)
        buckets.setdefault(it["section"], []).append(it)

    lines: list[str] = []
    for sec in sorted(buckets, key=lambda s: PR.SECTION_ORDER.get(s, 9)):
        lines.append(f"【{PR.SECTION_LABEL.get(sec, sec)}】")
        for it in buckets[sec]:
            lines.append(f"- {it['text']}")

    text = "\n".join(lines)
    return {
        "text": text,
        "sections": {k: len(v) for k, v in buckets.items()},
        "included": len(included),
        "dropped": dropped,
        "chars": used,
        "budget": budget,
        "ratio": round(used / budget, 3) if budget else 0.0,
        "dropped_count": len(dropped),
    }


def budget_report(cfg: Config, session_id: str) -> dict:
    """人格预算占用报告。用于体检。"""
    d = compile_persona(cfg, session_id)
    d.pop("text", None)
    return d


def extract_directives(text: str) -> list[str]:
    """从一句话里抽出可成为人格条目的短句（"以后不许…" 这类）。

    只在 source=manual 或 self_reflection 时使用，避免把闲聊写进人格。
    """
    if not text:
        return []
    out = []
    for sent in re.split(r"[。！？\n]+", text):
        s = sent.strip()
        if not s or len(s) > 120:
            continue
        if any(k in s for k in ("以后", "不许", "必须", "永远", "原则", "规矩", "铁律")):
            out.append(s)
    return out[:3]

