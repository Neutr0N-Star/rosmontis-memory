"""SQLite 连接、事务、迁移。

连接采用 **thread-local 复用**：每次操作新建连接要付 5.9ms（其中 WAL PRAGMA 占
5.2ms），单次检索开 80+ 连接就会到 600ms。复用后同一线程内是微秒级。

WAL 与 synchronous 只在建库时设一次，不放在连接函数里。

schema 版本：v3（A2 §3.3）。迁移纪律：
- 只加列/加表，不改不删不重命名；
- **单事务**：加列 → 回填 → 建表 → 补 backfill log → 写 user_version，一起 COMMIT；
- 失败 ROLLBACK 保持原版本；重复执行安全（user_version 已到位就不再跑）。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("alaya.db")

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"
_schema_cache: str | None = None
_init_lock = threading.Lock()
_local = threading.local()

SCHEMA_VERSION = 3

# 与 alaya.core.gate.GATE_VERSION 保持一致（运行期从 gate 读取，这里是兜底常量）
_FALLBACK_GATE_VERSION = "3.0.0"


def _read_schema() -> str:
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = _SCHEMA_PATH.read_text(encoding="utf-8")
    return _schema_cache


def connect(db_path: str) -> sqlite3.Connection:
    """取当前线程的连接（复用）。首次调用时建连并设基础 PRAGMA。"""
    key = "_conn_" + str(abs(hash(os.path.abspath(db_path))))
    conn = getattr(_local, key, None)
    if conn is not None:
        return conn

    parent = os.path.dirname(os.path.abspath(db_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=15.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=3000")   # 3s 上限，外部持锁时不让请求无限挂起
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-32000")
    setattr(_local, key, conn)
    return conn


def close_thread_conns() -> None:
    """线程退出前清理。测试里也用它避免文件占用。"""
    for name in list(vars(_local)):
        if name.startswith("_conn_"):
            try:
                getattr(_local, name).close()
            except Exception:
                pass
            delattr(_local, name)


# 性能索引：dense 近因窗口按 (session_id, t_ingest DESC) 取数、序号取 MAX(seq)。
# 用 IF NOT EXISTS 幂等补齐，老库升级不改 user_version；失败只告警不阻塞启动。
_PERF_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_ep_sess_ingest ON episode(session_id, t_ingest DESC)",
    "CREATE INDEX IF NOT EXISTS idx_fact_sess_ingest ON fact(session_id, t_ingest DESC)",
    "CREATE INDEX IF NOT EXISTS idx_ep_sess_seq ON episode(session_id, seq DESC)",
)


def _ensure_perf_indexes(conn: sqlite3.Connection) -> None:
    for sql in _PERF_INDEXES:
        try:
            conn.execute(sql)
        except sqlite3.Error as e:
            log.warning("perf index failed (%s): %s", sql, e)


# ── jobs 表（docs/15 §3.5）：后台节拍/重放任务 ──────────────────
# 用 IF NOT EXISTS 幂等补，老库升级不改 user_version；失败只告警不阻塞启动。
# 本文件只新增这一组与 jobs 相关的语句（C2 契约），不参与其它迁移逻辑。
_JOB_TABLE_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS job (
        id           TEXT PRIMARY KEY,
        kind         TEXT NOT NULL,
        target       TEXT NOT NULL,
        status       TEXT NOT NULL,
        attempts     INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 3,
        next_run     REAL NOT NULL,
        lease_owner  TEXT,
        leased_until REAL,
        last_error   TEXT,
        created_at   REAL NOT NULL,
        updated_at   REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_job_due ON job(status, next_run)",
    "CREATE INDEX IF NOT EXISTS idx_job_lease ON job(status, leased_until)",
)


def _ensure_job_table(conn: sqlite3.Connection) -> None:
    for sql in _JOB_TABLE_STATEMENTS:
        try:
            conn.execute(sql)
        except sqlite3.Error as e:
            log.warning("job table init failed: %s", e)


def init_db(db_path: str) -> None:
    with _init_lock:
        conn = connect(db_path)
        # 这两条是库级设置，只在初始化时设一次
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            conn.executescript(_read_schema())
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            version = SCHEMA_VERSION
        if version < SCHEMA_VERSION:
            _migrate(conn, version)
        _ensure_perf_indexes(conn)
        _ensure_job_table(conn)

        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            conn.execute("INSERT INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        elif str(row["value"]) != str(SCHEMA_VERSION):
            # 迁移成功但 meta 停旧版时补齐；坏数据不阻塞启动
            try:
                conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),))
            except sqlite3.Error as e:
                log.warning("meta schema_version update failed: %s", e)


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """逐版本补列。**整个迁移一个事务**，失败回滚保持旧版本。"""
    log.warning("migrating schema from v%s to v%s", from_version, SCHEMA_VERSION)
    now = time.time()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if from_version < 2:
            _migrate_v2(conn)
        if from_version < 3:
            _migrate_v3(conn, now)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        log.exception("migration v%s → v%s failed, rolled back", from_version, SCHEMA_VERSION)
        raise
    log.warning("migration done (v%s)", SCHEMA_VERSION)


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def _add(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    if column not in _cols(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


# ── v1 → v2：episode 特化维度 + 新表 ───────────────────────────
def _migrate_v2(conn: sqlite3.Connection) -> None:
    for col, ddl in (
        ("source", "TEXT NOT NULL DEFAULT 'dialogue'"),
        ("tier", "TEXT NOT NULL DEFAULT 'retrieval'"),
        ("memory_type", "TEXT NOT NULL DEFAULT 'fact'"),
        ("author", "TEXT NOT NULL DEFAULT 'system'"),
        ("confirmed", "INTEGER NOT NULL DEFAULT 0"),
        ("emotion", "REAL NOT NULL DEFAULT 0"),
        ("muted", "INTEGER NOT NULL DEFAULT 0"),
        ("cooldown_until", "REAL"),
        ("frozen_from", "TEXT"),
        ("correction_of", "TEXT"),
        ("superseded_by", "TEXT"),
        ("chunk_key", "TEXT"),
        ("pinned", "INTEGER NOT NULL DEFAULT 0"),
        ("meta_json", "TEXT"),
    ):
        _add(conn, "episode", col, ddl)

    for col, ddl in (
        ("author", "TEXT NOT NULL DEFAULT 'system'"),
        ("memory_type", "TEXT NOT NULL DEFAULT 'fact'"),
        ("correction_of", "TEXT"),
        ("superseded_by", "TEXT"),
    ):
        _add(conn, "fact", col, ddl)

    # 注意：这里不能用 executescript——它会隐式 COMMIT，破坏单事务迁移
    for sql in (
        """CREATE TABLE IF NOT EXISTS persona (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, section TEXT NOT NULL DEFAULT 'free',
            text TEXT NOT NULL, author TEXT NOT NULL DEFAULT 'self', protected INTEGER NOT NULL DEFAULT 0,
            priority INTEGER NOT NULL DEFAULT 100, "order" INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
        )""",
        """CREATE TABLE IF NOT EXISTS tombstone (
            chunk_key TEXT PRIMARY KEY, line_ids TEXT, session_id TEXT,
            deleted_at REAL NOT NULL, reason TEXT, author TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS page (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, key TEXT NOT NULL, title TEXT NOT NULL,
            body TEXT NOT NULL, sources TEXT, built_at REAL NOT NULL,
            stale INTEGER NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 1
        )""",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_page ON page(session_id, key)",
        """CREATE TABLE IF NOT EXISTS observation (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, text TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'pattern', evidence TEXT,
            confidence REAL NOT NULL DEFAULT 0.6, created_at REAL NOT NULL,
            updated_at REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
        )""",
        "CREATE INDEX IF NOT EXISTS idx_obs ON observation(session_id, active)",
        "CREATE INDEX IF NOT EXISTS idx_ep_tier ON episode(session_id, tier, muted)",
        "CREATE INDEX IF NOT EXISTS idx_ep_chunk ON episode(chunk_key)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_slot_open ON state_slot(scope, slot_key) WHERE until IS NULL",
    ):
        conn.execute(sql)


# ── v2 → v3：状态机字段 + 兼容投影 + 审计表（只加）─────────────
def _migrate_v3(conn: sqlite3.Connection, now: float) -> None:
    for col, ddl in (
        ("origin", "TEXT"),
        ("state", "TEXT"),
        ("state_at", "REAL"),
        ("gate_version", "TEXT"),
    ):
        _add(conn, "episode", col, ddl)
    for col, ddl in (
        ("origin", "TEXT"),
        ("archived", "INTEGER NOT NULL DEFAULT 0"),
    ):
        _add(conn, "fact", col, ddl)

    conn.execute(
        """CREATE TABLE IF NOT EXISTS item_state_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id    TEXT NOT NULL,
            from_state TEXT,
            to_state   TEXT NOT NULL,
            action     TEXT NOT NULL,
            target     TEXT,
            reason     TEXT,
            gate_json  TEXT,
            at         REAL NOT NULL,
            actor      TEXT NOT NULL DEFAULT 'system'
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_state ON episode(session_id, state, muted)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_log_item ON item_state_log(item_id, at)")

    _backfill_episode(conn, now)
    _backfill_fact(conn)
    _backfill_state_log(conn, now)


def _backfill_episode(conn: sqlite3.Connection, now: float) -> None:
    """兼容投影 → 状态。容错：tier='dream'、t_event=1e18、异常 session_id 都不特殊处理。"""
    conn.execute(
        """UPDATE episode SET
             origin = CASE
               WHEN source='notebook' THEN 'tool'
               WHEN source='dream' OR role='dream' THEN 'dream'
               WHEN source='summary' OR role='summary' THEN 'summary'
               ELSE 'ingest'
             END,
             state = COALESCE(state, CASE
               WHEN muted=1 THEN 'muted'
               WHEN archived=1 THEN 'archived'
               WHEN tier='resident' THEN 'resident'
               WHEN tier='important' THEN 'important'
               WHEN tier='inbox' THEN 'inbox'
               WHEN tier='dream' THEN 'dream'
               ELSE 'note'
             END),
             state_at = COALESCE(state_at, ?)
           WHERE origin IS NULL OR state IS NULL OR state_at IS NULL""",
        (now,),
    )


def _backfill_fact(conn: sqlite3.Connection) -> None:
    """fact.origin：来源 episode 是主动写入 → tool，其余 ingest（A2 §3.3 原文口径）。"""
    conn.execute(
        """UPDATE fact SET origin = CASE
             WHEN source_episode IS NOT NULL
                  AND source_episode IN (SELECT id FROM episode WHERE origin='tool') THEN 'tool'
             ELSE 'ingest'
           END
         WHERE origin IS NULL"""
    )
    conn.execute("UPDATE fact SET archived=0 WHERE archived IS NULL")


def _gate_version() -> str:
    try:
        from . import gate as GATE

        return GATE.GATE_VERSION
    except Exception:  # pragma: no cover - 循环导入兜底
        return _FALLBACK_GATE_VERSION


def _backfill_state_log(conn: sqlite3.Connection, now: float) -> None:
    """存量每个 episode 补 1 条 backfill 记录；已有日志的不重复写（幂等）。"""
    gate_json = json.dumps({"version": _gate_version(), "action": "backfill"}, ensure_ascii=False)
    conn.execute(
        """INSERT INTO item_state_log(item_id,from_state,to_state,action,target,reason,gate_json,at,actor)
           SELECT e.id, NULL, e.state, 'backfill', NULL, 'schema v3 migration', ?, ?, 'system'
           FROM episode e
           WHERE e.state IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM item_state_log l WHERE l.item_id=e.id)""",
        (gate_json, now),
    )


@contextmanager
def tx(db_path: str):
    """写事务。遇到临时锁冲突自动退避重试（并发写入实测有 3.75% 失败率）。"""
    conn = connect(db_path)
    acquired = False
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            conn.execute("BEGIN IMMEDIATE")
            acquired = True
            break
        except sqlite3.OperationalError as e:
            last_err = e
            msg = str(e).lower()
            if "readonly" in msg or "locked" in msg or "busy" in msg:
                time.sleep(0.015 * (attempt + 1))
                continue
            raise
    if not acquired:
        raise last_err or sqlite3.OperationalError("cannot begin transaction")

    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise


@contextmanager
def reader(db_path: str):
    conn = connect(db_path)
    yield conn


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def rowcount(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
    return int(row["c"]) if row else 0
