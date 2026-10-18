"""SQLite 连接、事务、迁移。

连接采用 **thread-local 复用**：每次操作新建连接要付 5.9ms（其中 WAL PRAGMA 占
5.2ms），单次检索开 80+ 连接就会到 600ms。复用后同一线程内是微秒级。

WAL 与 synchronous 只在建库时设一次，不放在连接函数里。
"""

from __future__ import annotations

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

SCHEMA_VERSION = 2


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
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            conn.execute("INSERT INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """逐版本补列。已有数据不动，只加不删。"""
    log.warning("migrating schema from v%s to v%s", from_version, SCHEMA_VERSION)

    def cols(table: str) -> set[str]:
        try:
            return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        except sqlite3.Error:
            return set()

    def add(table: str, column: str, ddl: str) -> None:
        if column not in cols(table):
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            except sqlite3.Error as e:
                log.warning("alter %s.%s failed: %s", table, column, e)

    # v1 → v2：episode 补特化维度
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
        add("episode", col, ddl)

    for col, ddl in (
        ("author", "TEXT NOT NULL DEFAULT 'system'"),
        ("memory_type", "TEXT NOT NULL DEFAULT 'fact'"),
        ("correction_of", "TEXT"),
        ("superseded_by", "TEXT"),
    ):
        add("fact", col, ddl)

    # 新表（IF NOT EXISTS，重复执行安全）
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS persona (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, section TEXT NOT NULL DEFAULT 'free',
            text TEXT NOT NULL, author TEXT NOT NULL DEFAULT 'self', protected INTEGER NOT NULL DEFAULT 0,
            priority INTEGER NOT NULL DEFAULT 100, "order" INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS tombstone (
            chunk_key TEXT PRIMARY KEY, line_ids TEXT, session_id TEXT,
            deleted_at REAL NOT NULL, reason TEXT, author TEXT
        );
        CREATE TABLE IF NOT EXISTS page (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, key TEXT NOT NULL, title TEXT NOT NULL,
            body TEXT NOT NULL, sources TEXT, built_at REAL NOT NULL,
            stale INTEGER NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 1
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_page ON page(session_id, key);
        CREATE TABLE IF NOT EXISTS observation (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, text TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'pattern', evidence TEXT,
            confidence REAL NOT NULL DEFAULT 0.6, created_at REAL NOT NULL,
            updated_at REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_obs ON observation(session_id, active);
        CREATE INDEX IF NOT EXISTS idx_ep_tier ON episode(session_id, tier, muted);
        CREATE INDEX IF NOT EXISTS idx_ep_chunk ON episode(chunk_key);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_slot_open ON state_slot(scope, slot_key) WHERE until IS NULL;
        """
    )
    log.warning("migration done")


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
