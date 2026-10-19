"""后台任务表（jobs）与 HTTP 幂等键存储。

契约来源：`docs/15-交付蓝图与模块契约.md` §3.5。

设计纪律：
- `job` 表只由 `core/db.py` 的幂等 DDL 创建（不碰 user_version）；本模块只读写。
- job id 是**确定性键** `kind + target + 时间窗`，重复 enqueue 不产生第二条。
- 单写者语义：`claim_due/claim_job` 用 `BEGIN IMMEDIATE` 抢占；
  `lease_owner/leased_until` 用于跨进程租约，`requeue_stale` 收尸。
- `execute_job` 只做 kind → Memory 方法的映射，供 daemon 与 HTTP `/v1/settle` 共用。
- 幂等键（`Idempotency-Key`）存在独立表 `http_idempotency`，24h 内同键返回首次响应；
  该表 DDL 放在本模块（db.py 只允许 jobs 表语句）。

时间单位：全部 epoch 秒；`latency_ms` 之类才用毫秒。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time

from . import db as DB

log = logging.getLogger("alaya.jobs")

# ── job 常量 ──────────────────────────────────────────────────
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_DEAD = "dead"
STATUSES = (STATUS_PENDING, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED, STATUS_DEAD)

KIND_ENRICH = "enrich"
KIND_REVIEW = "review"
KIND_DREAM = "dream"
KIND_SETTLE = "settle"
KIND_FORGET = "forget"
KIND_BACKFILL = "backfill"
KIND_EMBED_BACKFILL = "embed_backfill"
KIND_TICK = "tick"
KINDS = (
    KIND_ENRICH, KIND_REVIEW, KIND_DREAM, KIND_SETTLE, KIND_FORGET,
    KIND_BACKFILL, KIND_EMBED_BACKFILL, KIND_TICK,
)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LEASE_SECONDS = 300.0
DEFAULT_IDLE_TIMEOUT = 1800.0
DEFAULT_IDEMPOTENCY_TTL = 86400.0
DEFAULT_FORGET_INTERVAL = 21600.0   # 6h：daemon 全局遗忘周期


# ── 小工具 ────────────────────────────────────────────────────
def _now(now: float | None = None) -> float:
    return time.time() if now is None else float(now)


def _env_float(key: str, default: float) -> float:
    try:
        raw = os.environ.get(key)
        return float(raw) if raw not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)


def _digest(*parts: object) -> str:
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()


def make_job_id(kind: str, target: str, bucket: int | float) -> str:
    """确定性 job id：kind + target + 时间窗。"""
    return "job_" + _digest(str(kind), str(target), int(bucket))


def _row(r) -> dict:
    """sqlite3.Row → dict（Row 不能直接 dict()）。"""
    if r is None:
        return {}
    return {k: r[k] for k in r.keys()}


def _as_job(row: dict) -> dict:
    row = dict(row or {})
    row["job_id"] = row.get("id", "")
    return row


def idle_timeout_from_env(default: float = DEFAULT_IDLE_TIMEOUT) -> float:
    """ALAYA_IDLE_TIMEOUT（秒）。Contract 里叫 `Config.idle_timeout_s`，本轮不越权改 config.py。"""
    return max(1.0, _env_float("ALAYA_IDLE_TIMEOUT", default))


def idempotency_ttl_from_env(default: float = DEFAULT_IDEMPOTENCY_TTL) -> float:
    return max(1.0, _env_float("ALAYA_IDEMPOTENCY_TTL", default))


def _ensure_db(cfg) -> None:
    """幂等建表（job 由 db.init_db 建；这里补幂等键表）。"""
    DB.init_db(cfg.db_path)


_IDEMPOTENCY_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS http_idempotency (
        id          TEXT PRIMARY KEY,
        scope       TEXT NOT NULL,
        key         TEXT NOT NULL,
        status      TEXT NOT NULL DEFAULT 'running',
        status_code INTEGER,
        response    TEXT,
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL,
        expires_at  REAL NOT NULL,
        UNIQUE(scope, key)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_idem_expires ON http_idempotency(expires_at)",
)

_READY_LOCK = threading.Lock()
_READY_PATHS: set[str] = set()


def _ready(cfg) -> None:
    """每进程每库只做一次建表（job 表在 db.init_db；幂等键表在此补）。

    先建表再登记，失败不会把路径标记为 ready；测试用临时库路径唯一，不会串。
    """
    path = os.path.abspath(str(cfg.db_path))
    with _READY_LOCK:
        if path in _READY_PATHS:
            return
    _ensure_db(cfg)
    with DB.tx(cfg.db_path) as conn:
        for sql in _IDEMPOTENCY_STATEMENTS:
            conn.execute(sql)
    with _READY_LOCK:
        _READY_PATHS.add(path)


def ensure_tables(cfg) -> None:
    _ready(cfg)


# ── 基础读写 ──────────────────────────────────────────────────
def get(cfg, job_id: str) -> dict | None:
    _ready(cfg)
    with DB.reader(cfg.db_path) as conn:
        row = conn.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
    return _as_job(_row(row)) if row is not None else None


def enqueue(
    cfg,
    kind: str,
    target: str,
    next_run: float | None = None,
    *,
    bucket: int | float | None = None,
    now: float | None = None,
    job_id: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict:
    """投递一个 job。同 (kind,target,bucket) 重复投递只留第一条。

    返回保含 `job_id/status/created` 的完整行；`created=False` 表示命中了已有 job。
    """
    _ready(cfg)
    kind = (kind or "").strip().lower()
    if not kind:
        raise ValueError("kind 不能为空")
    target = str(target if target is not None else "")
    now = _now(now)
    run_at = _now(next_run) if next_run is not None else now
    if job_id is None:
        win = int(now // 60) if bucket is None else int(bucket)
        job_id = make_job_id(kind, target, win)
    with DB.tx(cfg.db_path) as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO job
               (id,kind,target,status,attempts,max_attempts,next_run,created_at,updated_at)
               VALUES(?,?,?,?,0,?,?,?,?)""",
            (job_id, kind, target, STATUS_PENDING, int(max_attempts), run_at, now, now),
        )
        created = bool(cur.rowcount)
        row = conn.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
    out = _as_job(_row(row))
    out["created"] = created
    return out


def list_jobs(cfg, *, status: str | None = None, limit: int = 100) -> list[dict]:
    _ready(cfg)
    limit = max(1, min(int(limit), 1000))
    sql = "SELECT * FROM job"
    args: list = []
    if status:
        sql += " WHERE status=?"
        args.append(status)
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    args.append(limit)
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute(sql, tuple(args)).fetchall()
    return [_as_job(_row(r)) for r in rows]


def counts(cfg) -> dict:
    _ready(cfg)
    with DB.reader(cfg.db_path) as conn:
        rows = conn.execute("SELECT status, COUNT(*) c FROM job GROUP BY status").fetchall()
    out = {s: 0 for s in STATUSES}
    for r in rows:
        out[str(r["status"])] = int(r["c"])
    out["total"] = sum(v for k, v in out.items() if k != "total")
    return out


def due_count(cfg, *, now: float | None = None) -> int:
    _ready(cfg)
    now = _now(now)
    with DB.reader(cfg.db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(*) c FROM job
               WHERE status IN (?,?) AND attempts < max_attempts AND next_run <= ?""",
            (STATUS_PENDING, STATUS_FAILED, now),
        ).fetchone()
    return int(row["c"])


def summary(cfg, *, now: float | None = None) -> dict:
    c = counts(cfg)
    c["due"] = due_count(cfg, now=now)
    return c


# ── 抢占 / 租约 ───────────────────────────────────────────────
def _claim_ids(conn, ids: list[str], *, owner: str, now: float, lease_seconds: float) -> list[str]:
    got: list[str] = []
    for jid in ids:
        cur = conn.execute(
            """UPDATE job
               SET status=?, lease_owner=?, leased_until=?, attempts=attempts+1, updated_at=?
               WHERE id=? AND status IN (?,?) AND attempts < max_attempts""",
            (
                STATUS_RUNNING, owner, now + float(lease_seconds), now,
                jid, STATUS_PENDING, STATUS_FAILED,
            ),
        )
        if cur.rowcount:
            got.append(jid)
    return got


def claim_due(
    cfg,
    limit: int = 10,
    *,
    owner: str = "daemon",
    now: float | None = None,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
) -> list[dict]:
    """抢占到点的 pending/failed job（单写者：事务内 UPDATE 抢占）。"""
    _ready(cfg)
    now = _now(now)
    limit = max(1, min(int(limit), 200))
    with DB.tx(cfg.db_path) as conn:
        rows = conn.execute(
            """SELECT id FROM job
               WHERE status IN (?,?) AND attempts < max_attempts AND next_run <= ?
               ORDER BY next_run ASC, created_at ASC LIMIT ?""",
            (STATUS_PENDING, STATUS_FAILED, now, limit),
        ).fetchall()
        ids = [r["id"] for r in rows]
        got = _claim_ids(conn, ids, owner=owner, now=now, lease_seconds=lease_seconds)
        if not got:
            return []
        marks = ",".join("?" * len(got))
        claimed = conn.execute(f"SELECT * FROM job WHERE id IN ({marks})", tuple(got)).fetchall()
    by_id = {r["id"]: _as_job(_row(r)) for r in claimed}
    return [by_id[i] for i in got if i in by_id]


def claim_job(
    cfg,
    job_id: str,
    *,
    owner: str = "http",
    now: float | None = None,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
) -> dict | None:
    """抢占指定 job（HTTP `/v1/settle` 的立即执行线程用）。"""
    _ready(cfg)
    now = _now(now)
    with DB.tx(cfg.db_path) as conn:
        got = _claim_ids(conn, [job_id], owner=owner, now=now, lease_seconds=lease_seconds)
        if not got:
            return None
        row = conn.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
    return _as_job(_row(row)) if row is not None else None


def finish(cfg, job_id: str, *, now: float | None = None) -> dict | None:
    _ready(cfg)
    now = _now(now)
    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            """UPDATE job SET status=?, lease_owner=NULL, leased_until=NULL, last_error=NULL, updated_at=?
               WHERE id=?""",
            (STATUS_DONE, now, job_id),
        )
        row = conn.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
    return _as_job(_row(row)) if row is not None else None


def fail(cfg, job_id: str, error: BaseException | str, *, now: float | None = None) -> dict | None:
    """失败：退避重试直到 max_attempts，超限进 dead。"""
    _ready(cfg)
    now = _now(now)
    msg = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
    msg = (msg or "unknown error")[:2000]
    with DB.tx(cfg.db_path) as conn:
        row = conn.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        attempts = int(row["attempts"] or 0)
        max_attempts = int(row["max_attempts"] or DEFAULT_MAX_ATTEMPTS)
        if attempts >= max_attempts:
            status, next_run = STATUS_DEAD, now
        else:
            status = STATUS_FAILED
            next_run = now + min(float(2 ** attempts), 300.0)
        conn.execute(
            """UPDATE job SET status=?, next_run=?, lease_owner=NULL, leased_until=NULL,
                              last_error=?, updated_at=? WHERE id=?""",
            (status, next_run, msg, now, job_id),
        )
        out = conn.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
    return _as_job(_row(out)) if out is not None else None


def requeue_stale(cfg, *, now: float | None = None) -> dict:
    """收回租约过期的 running job：attempts 未超限回 pending，超限进 dead。"""
    _ready(cfg)
    now = _now(now)
    requeued = dead = 0
    ids: list[str] = []
    with DB.tx(cfg.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM job WHERE status=? AND (leased_until IS NULL OR leased_until < ?)",
            (STATUS_RUNNING, now),
        ).fetchall()
        for r in rows:
            attempts = int(r["attempts"] or 0)
            max_attempts = int(r["max_attempts"] or DEFAULT_MAX_ATTEMPTS)
            if attempts >= max_attempts:
                conn.execute(
                    "UPDATE job SET status=?, lease_owner=NULL, leased_until=NULL, "
                    "last_error=?, updated_at=? WHERE id=?",
                    (STATUS_DEAD, "lease expired（尝试次数已用尽）", now, r["id"]),
                )
                dead += 1
            else:
                conn.execute(
                    """UPDATE job SET status=?, next_run=?, lease_owner=NULL, leased_until=NULL,
                                      last_error=?, updated_at=? WHERE id=?""",
                    (STATUS_PENDING, now, "lease expired，已收回重排", now, r["id"]),
                )
                requeued += 1
            ids.append(r["id"])
    return {"requeued": requeued, "dead": dead, "ids": ids}


# ── 执行器（daemon 与 HTTP 共用）──────────────────────────────
# 这些 kind 执行完顺手检查一次 NULL 向量，并把 embed_backfill 排到下一拍，
# daemon 起来后无需人工触发就能把降级期写下的条目补齐（C4）。
_AUTO_BACKFILL_AFTER = frozenset({KIND_SETTLE, KIND_TICK, KIND_FORGET, KIND_REVIEW})


def _maybe_enqueue_embed_backfill(mem, now: float | None = None) -> dict | None:
    """语义模式 + 有 NULL 向量时，幂等投递一个 embed_backfill job。"""
    try:
        cfg = getattr(mem, "cfg", None)
        emb = getattr(mem, "embedder", None)
        if cfg is None or emb is None:
            return None
        if getattr(emb, "provider", "") in ("hash", "none"):
            return None
        if bool(getattr(emb, "degraded", False)):
            return None
        if hasattr(emb, "vector_mode") and emb.vector_mode() != "semantic":
            return None
        pending = int(mem.pending_backfill()) if hasattr(mem, "pending_backfill") else 0
        if pending <= 0:
            return None
        now = _now(now)
        return enqueue(
            cfg, KIND_EMBED_BACKFILL, "*",
            next_run=now, bucket=int(now // 60), now=now,
        )
    except Exception as e:  # 自动调度失败不影响主 job
        log.debug("auto enqueue embed_backfill failed: %s", e)
        return None


def execute_job(mem, job: dict, *, now: float | None = None) -> dict:
    """按 kind 执行一个已抢占的 job。异常向上抛，由调用方 `fail()` 记录。"""
    kind = str(job.get("kind") or "").strip().lower()
    target = str(job.get("target") or "")
    out: dict = {"kind": kind, "target": target}
    if kind == KIND_SETTLE:
        out["result"] = mem.close_session(target, now=now)
    elif kind == KIND_REVIEW:
        out["result"] = mem.notebook_review(target)
    elif kind == KIND_DREAM:
        out["result"] = mem.dream(target)
    elif kind == KIND_FORGET:
        out["result"] = mem.forget("*" if target in ("", "*") else target)
    elif kind == KIND_TICK:
        out["result"] = mem.tick(target)
    elif kind == KIND_ENRICH:
        from . import pipeline as TR

        res = TR.enrich(
            target,
            cfg=mem.cfg,
            embedder=getattr(mem, "embedder", None),
            llm=getattr(mem, "llm_hot", None),
        )
        out["result"] = res.to_dict() if hasattr(res, "to_dict") else res
    elif kind == KIND_EMBED_BACKFILL:
        sid = "" if target in ("", "*") else target
        out["result"] = mem.backfill_embeddings(session_id=sid or None, now=now)
    elif kind == KIND_BACKFILL:
        out["result"] = {"skipped": True, "reason": "backfill 未实现（后续 W）"}
    else:
        raise ValueError(f"未知 job kind：{kind or '(空)'}")
    if kind in _AUTO_BACKFILL_AFTER:
        auto = _maybe_enqueue_embed_backfill(mem, now)
        if auto is not None:
            out["enqueued"] = {"kind": KIND_EMBED_BACKFILL, "job_id": auto.get("job_id"), "created": auto.get("created")}
    return out


def result_brief(result: dict) -> dict:
    """日志/报告用的结果摘要，避免把整份 result 塞进一行日志。"""
    if not isinstance(result, dict):
        return {"result": str(result)[:200]}
    brief = {"kind": result.get("kind"), "target": result.get("target")}
    inner = result.get("result")
    if isinstance(inner, dict):
        for key in ("skipped", "reason", "scanned", "promoted", "dropped", "made", "count", "did"):
            if key in inner:
                brief[key] = inner[key]
    return brief


# ── HTTP 幂等键 ───────────────────────────────────────────────
def idem_begin(
    cfg,
    scope: str,
    key: str,
    *,
    now: float | None = None,
    ttl: float | None = None,
) -> tuple[str, dict | None]:
    """开始一次幂等请求。

    返回 (状态, 缓存)：
    - ("hit", {"status_code", "body"})：24h 内已有首次结果，直接回放；
    - ("busy", None)：同键请求正在处理（客户端应重试）；
    - ("new", None)：新请求，调用方执行后 `idem_store`；
    - ("invalid", None)：key 不合法。
    """
    key = (key or "").strip()
    if not key or len(key) > 200:
        return "invalid", None
    now = _now(now)
    ttl = idempotency_ttl_from_env() if ttl is None else float(ttl)
    ensure_tables(cfg)
    with DB.tx(cfg.db_path) as conn:
        conn.execute("DELETE FROM http_idempotency WHERE expires_at <= ?", (now,))
        row = conn.execute(
            "SELECT * FROM http_idempotency WHERE scope=? AND key=?", (scope, key)
        ).fetchone()
        if row is not None:
            if str(row["status"]) == "done":
                try:
                    body = json.loads(row["response"] or "{}")
                except (TypeError, ValueError):
                    body = {}
                return "hit", {"status_code": int(row["status_code"] or 200), "body": body}
            return "busy", None
        conn.execute(
            """INSERT INTO http_idempotency
               (id,scope,key,status,status_code,response,created_at,updated_at,expires_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            ("idem_" + _digest(scope, key), scope, key, "running", None, None,
             now, now, now + ttl),
        )
    return "new", None


def idem_store(
    cfg,
    scope: str,
    key: str,
    status_code: int,
    body: dict,
    *,
    now: float | None = None,
) -> None:
    """保存首次响应；同一 key 24h 内回放它。"""
    key = (key or "").strip()
    if not key:
        return
    now = _now(now)
    ttl = idempotency_ttl_from_env()
    ensure_tables(cfg)
    payload = json.dumps(body, ensure_ascii=False)
    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            """UPDATE http_idempotency
               SET status='done', status_code=?, response=?, updated_at=?, expires_at=?
               WHERE scope=? AND key=?""",
            (int(status_code), payload, now, now + ttl, scope, key),
        )


def idem_release(cfg, scope: str, key: str) -> None:
    """请求执行异常时释放占位，允许客户端用同键重试。"""
    key = (key or "").strip()
    if not key:
        return
    ensure_tables(cfg)
    with DB.tx(cfg.db_path) as conn:
        conn.execute(
            "DELETE FROM http_idempotency WHERE scope=? AND key=? AND status='running'",
            (scope, key),
        )


__all__ = [
    "KINDS", "STATUSES",
    "KIND_ENRICH", "KIND_REVIEW", "KIND_DREAM", "KIND_SETTLE", "KIND_FORGET",
    "KIND_BACKFILL", "KIND_EMBED_BACKFILL", "KIND_TICK",
    "STATUS_PENDING", "STATUS_RUNNING", "STATUS_DONE", "STATUS_FAILED", "STATUS_DEAD",
    "idle_timeout_from_env", "idempotency_ttl_from_env",
    "make_job_id", "enqueue", "get", "list_jobs", "counts", "due_count", "summary",
    "claim_due", "claim_job", "finish", "fail", "requeue_stale",
    "execute_job", "result_brief", "ensure_tables",
    "idem_begin", "idem_store", "idem_release",
]
