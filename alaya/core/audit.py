"""体检：记忆库健康度。回答"它坏在哪"，而不只是"它有多少条"。"""

from __future__ import annotations

import time

from ..config import Config
from . import db as DB
from . import forget as FG
from . import timeparse


def audit(cfg: Config, session_id: str | None = None, now: float | None = None) -> dict:
    now = now or time.time()
    sid = session_id or "*"
    out: dict = {"generated_at": now, "issues": [], "counts": {}}

    with DB.reader(cfg.db_path) as conn:
        q = lambda sql, *a: conn.execute(sql, a).fetchall()  # noqa: E731

        counts = {
            "episode": q("SELECT COUNT(*) c FROM episode WHERE (?='*' OR session_id=?)", sid, sid)[0]["c"],
            "episode_active": q("SELECT COUNT(*) c FROM episode WHERE (?='*' OR session_id=?) AND archived=0", sid, sid)[0]["c"],
            "episode_folded": q("SELECT COUNT(*) c FROM episode WHERE (?='*' OR session_id=?) AND folded_into IS NOT NULL", sid, sid)[0]["c"],
            "entity": q("SELECT COUNT(*) c FROM entity WHERE (?='*' OR session_id=?)", sid, sid)[0]["c"],
            "fact": q("SELECT COUNT(*) c FROM fact WHERE (?='*' OR session_id=?)", sid, sid)[0]["c"],
            "fact_active": q("SELECT COUNT(*) c FROM fact WHERE (?='*' OR session_id=?) AND status='active'", sid, sid)[0]["c"],
            "fact_invalidated": q("SELECT COUNT(*) c FROM fact WHERE (?='*' OR session_id=?) AND status!='active'", sid, sid)[0]["c"],
            "state_open": q("SELECT COUNT(*) c FROM state_slot WHERE (?='*' OR session_id=?) AND until IS NULL", sid, sid)[0]["c"],
            "theme": q("SELECT COUNT(*) c FROM hyperedge WHERE (?='*' OR session_id=?)", sid, sid)[0]["c"],
            "retrieval": q("SELECT COUNT(*) c FROM retrieval_log")[0]["c"],
        }
        out["counts"] = counts

        issues = out["issues"]

        # 1 同一 (主体,谓词) 有多条 active —— 状态类谓词不该出现
        dup = q(
            """SELECT subject_id, predicate, COUNT(*) c FROM fact
               WHERE (?='*' OR session_id=?) AND status='active'
               GROUP BY subject_id, predicate HAVING c > 1
               ORDER BY c DESC LIMIT 10""",
            sid, sid,
        )
        for r in dup:
            # 只有「状态类」谓词才要求唯一（工作/住处/关系）。
            # 喜欢猫 + 喜欢狗同时有效是完全正常的，不该报冲突。
            if r["predicate"] not in ("工作在", "住在", "关系是"):
                continue
            issues.append({
                "type": "conflicting_facts",
                "severity": "high",
                "detail": f"{r['subject_id']} 的「{r['predicate']}」有 {r['c']} 条同时有效",
                "hint": "状态类谓词应只保留一条，检查消解逻辑或手动作废",
            })

        # 2 只有名字没有画像的实体
        bare = q(
            """SELECT COUNT(*) c FROM entity WHERE (?='*' OR session_id=?)
               AND (attrs IS NULL OR attrs='{}') AND mentions <= 1""",
            sid, sid,
        )[0]["c"]
        if bare:
            issues.append({
                "type": "bare_entities",
                "severity": "low",
                "detail": f"{bare} 个实体只出现过一次且没有属性",
                "hint": "通常是误抽的通名，可考虑过滤或合并",
            })

        # 3 冷条目：长期未召回且重要度低
        cold = 0
        rows = q(
            """SELECT t_event, access_count, salience FROM episode
               WHERE (?='*' OR session_id=?) AND archived=0 AND folded_into IS NULL""",
            sid, sid,
        )
        for r in rows:
            if FG.retention(cfg, r["t_event"], r["access_count"], r["salience"], now) < cfg.forget_fold_r:
                cold += 1
        if cold:
            issues.append({
                "type": "cold_episodes",
                "severity": "medium",
                "detail": f"{cold} 条情节已进入折叠区间但尚未处理",
                "hint": "运行 forget 周期",
            })

        # 4 超边过大且摘要缺失
        bloated = q(
            """SELECT id, label, member_count FROM hyperedge
               WHERE (?='*' OR session_id=?) AND member_count > 30 AND (summary IS NULL OR summary='')
               LIMIT 10""",
            sid, sid,
        )
        for r in bloated:
            issues.append({
                "type": "bloated_theme",
                "severity": "medium",
                "detail": f"主题「{r['label']}」挂了 {r['member_count']} 个成员且没有摘要",
                "hint": "补充 summary，或拆分主题",
            })

        # 5 状态槽指向已失效事实
        orphan = q(
            """SELECT COUNT(*) c FROM state_slot s
               JOIN fact f ON f.id = s.source_fact
               WHERE s.until IS NULL AND f.status != 'active'"""
        )[0]["c"]
        if orphan:
            issues.append({
                "type": "stale_state",
                "severity": "high",
                "detail": f"{orphan} 个当前状态指向已经作废的事实",
                "hint": "状态快照与事实层不一致，需要重建",
            })

        # 6 拒答率过高 —— 说明抽取或召回有问题
        recent = q(
            """SELECT COUNT(*) c, SUM(abstained) a FROM retrieval_log
               WHERE ts > ?""",
            now - 86400 * 7,
        )[0]
        if recent["c"] and recent["c"] >= 10:
            rate = (recent["a"] or 0) / recent["c"]
            if rate > 0.4:
                issues.append({
                    "type": "high_abstain_rate",
                    "severity": "medium",
                    "detail": f"近 7 天拒答率 {rate:.0%}（{recent['a']}/{recent['c']}）",
                    "hint": "可能是抽取没抽到、或阈值过高",
                })

    order = {"high": 0, "medium": 1, "low": 2}
    out["issues"].sort(key=lambda i: order.get(i["severity"], 3))
    out["health"] = _grade(out["issues"], out["counts"])
    return out


def _grade(issues: list[dict], counts: dict) -> dict:
    high = sum(1 for i in issues if i["severity"] == "high")
    med = sum(1 for i in issues if i["severity"] == "medium")
    score = max(0, 100 - high * 22 - med * 8 - (len(issues) - high - med) * 2)
    label = "良好" if score >= 85 else "可用" if score >= 65 else "需要干预"
    return {"score": score, "label": label, "high": high, "medium": med}
