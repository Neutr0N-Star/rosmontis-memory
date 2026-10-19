-- 藏识 Alaya schema v3（特化版：面向长期陪伴 AI 的人格记忆）
-- 时间统一 epoch 秒（REAL）。物理删除只发生在 tombstone 明确记录之后。
-- v3 只加列/加表（A2 §3.3）：episode 的 origin/state/state_at/gate_version、
-- fact 的 origin/archived、item_state_log；旧列保留为兼容投影，不删不改。

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session (
    id           TEXT PRIMARY KEY,
    agent_id     TEXT NOT NULL DEFAULT 'default',
    user_id      TEXT NOT NULL DEFAULT 'default',
    title        TEXT,
    created_at   REAL NOT NULL,
    last_active  REAL NOT NULL
);

-- ── 人格层：她是谁。受人格预算约束，整条取舍 ──────────────────
-- section: identity 身份 / speech 说话方式 / rule 铁律 / trait 性格 / taste 品味 / free 自由区
CREATE TABLE IF NOT EXISTS persona (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    section     TEXT NOT NULL DEFAULT 'free',
    text        TEXT NOT NULL,
    author      TEXT NOT NULL DEFAULT 'self',   -- self / teacher / system
    protected   INTEGER NOT NULL DEFAULT 0,     -- 1 = 必保，永不被预算裁掉
    priority    INTEGER NOT NULL DEFAULT 100,   -- 数字小的先保留
    "order"     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1      -- 0 = 已撤回
);
CREATE INDEX IF NOT EXISTS idx_persona_sec ON persona(session_id, section, active);

-- ── 事件层：原文不可变 ────────────────────────────────────────
-- source: dialogue 对话 / self_reflection 自省 / seed 种子 / summary 压缩 / manual 手记
-- tier:   resident 常驻 / retrieval 浮现 / important 重要（仅 retrieval 合法）
-- type:   fact 事实 / emotion 情绪 / lesson 教训 / principle 原则 / preference 偏好 / tool_behavior 工具行为
CREATE TABLE IF NOT EXISTS episode (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    seq            INTEGER NOT NULL DEFAULT 0,
    role           TEXT NOT NULL,
    text           TEXT NOT NULL,               -- 原话；占位符 %player% %self% 不替换
    t_event        REAL NOT NULL,
    t_ingest       REAL NOT NULL,
    source         TEXT NOT NULL DEFAULT 'dialogue',
    tier           TEXT NOT NULL DEFAULT 'retrieval',
    memory_type    TEXT NOT NULL DEFAULT 'fact',
    author         TEXT NOT NULL DEFAULT 'system',  -- self / teacher / system
    confirmed      INTEGER NOT NULL DEFAULT 0,      -- 1 = 本人/主人确认过
    salience       REAL NOT NULL DEFAULT 0.5,
    emotion        REAL NOT NULL DEFAULT 0.0,
    access_count   INTEGER NOT NULL DEFAULT 0,
    last_access    REAL,
    embedding      BLOB,
    folded_into    TEXT,
    archived       INTEGER NOT NULL DEFAULT 0,
    pinned         INTEGER NOT NULL DEFAULT 0,
    muted          INTEGER NOT NULL DEFAULT 0,
    cooldown_until REAL,
    frozen_from    TEXT,                        -- 冻结副本来源
    correction_of  TEXT,                        -- 订正了哪条（本条目是新的说法）
    superseded_by  TEXT,                        -- 被哪条订正（本条目是旧的说法，仍保留）
    chunk_key      TEXT,                        -- 幂等键：sha1(sorted line ids)
    meta_json      TEXT,
    -- v3：生命周期真值（唯一）。tier/archived/muted 是它的兼容投影，由 pipeline 双写。
    -- origin: ingest 被动 / tool 模型主动 / system 整流 / dream / summary
    -- state:  inbox / note / resident / important / archived / muted / dream
    origin         TEXT,
    state          TEXT,
    state_at       REAL,                        -- 最近一次状态转移时间
    gate_version   TEXT                         -- 最近一次过闸的 gate 版本
);
CREATE INDEX IF NOT EXISTS idx_ep_state        ON episode(session_id, state, muted);
CREATE INDEX IF NOT EXISTS idx_ep_session_time ON episode(session_id, t_event);
CREATE INDEX IF NOT EXISTS idx_ep_tier         ON episode(session_id, tier, muted);
CREATE INDEX IF NOT EXISTS idx_ep_cool         ON episode(cooldown_until);
CREATE INDEX IF NOT EXISTS idx_ep_type         ON episode(memory_type);
CREATE INDEX IF NOT EXISTS idx_ep_chunk        ON episode(chunk_key);
CREATE INDEX IF NOT EXISTS idx_ep_corr         ON episode(correction_of);

CREATE VIRTUAL TABLE IF NOT EXISTS episode_fts USING fts5(
    episode_id UNINDEXED,
    tokens,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- ── 实体层 ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entity (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    norm_name   TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'concept',
    aliases     TEXT,
    attrs       TEXT,
    salience    REAL NOT NULL DEFAULT 0.5,
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL,
    mentions    INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_norm ON entity(session_id, norm_name);
CREATE INDEX IF NOT EXISTS idx_entity_type ON entity(session_id, type);

-- ── 事实层：双时间 + 作者 ────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact (
    id                TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    subject_id        TEXT NOT NULL,
    predicate         TEXT NOT NULL,
    object_text       TEXT NOT NULL,
    object_entity_id  TEXT,
    polarity          INTEGER NOT NULL DEFAULT 1,
    confidence        REAL NOT NULL DEFAULT 0.7,
    author            TEXT NOT NULL DEFAULT 'system',
    memory_type       TEXT NOT NULL DEFAULT 'fact',
    t_valid_from      REAL NOT NULL,
    t_valid_to        REAL,
    t_ingest          REAL NOT NULL,
    t_invalid         REAL,
    status            TEXT NOT NULL DEFAULT 'active',
    source_episode    TEXT,
    salience          REAL NOT NULL DEFAULT 0.5,
    embedding         BLOB,
    access_count      INTEGER NOT NULL DEFAULT 0,
    last_access       REAL,
    correction_of     TEXT,
    superseded_by     TEXT,
    -- v3：来源与归档（事实永久、只换状态；C3 归档阈值更低，由 forget 执行）
    origin            TEXT,
    archived          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_fact_sp     ON fact(session_id, subject_id, predicate, status);
CREATE INDEX IF NOT EXISTS idx_fact_valid  ON fact(t_valid_from, t_valid_to);
CREATE INDEX IF NOT EXISTS idx_fact_status ON fact(status);
CREATE INDEX IF NOT EXISTS idx_fact_obj    ON fact(object_entity_id);

CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(
    fact_id UNINDEXED,
    tokens,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- ── 主题层：超边（n 元关系）────────────────────────────────────
CREATE TABLE IF NOT EXISTS hyperedge (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'theme',
    label        TEXT NOT NULL,
    summary      TEXT,
    salience     REAL NOT NULL DEFAULT 0.5,
    t_from       REAL NOT NULL,
    t_to         REAL,
    created_at   REAL NOT NULL,
    member_count INTEGER NOT NULL DEFAULT 0,
    embedding    BLOB
);
CREATE INDEX IF NOT EXISTS idx_edge_session ON hyperedge(session_id, kind);

CREATE TABLE IF NOT EXISTS hyperedge_member (
    edge_id     TEXT NOT NULL,
    member_type TEXT NOT NULL,
    member_id   TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    added_at    REAL NOT NULL,
    PRIMARY KEY (edge_id, member_type, member_id)
);
CREATE INDEX IF NOT EXISTS idx_member_lookup ON hyperedge_member(member_type, member_id);

-- ── 状态层 ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS state_slot (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    scope       TEXT NOT NULL,
    slot_key    TEXT NOT NULL,
    value_text  TEXT NOT NULL,
    since       REAL NOT NULL,
    until       REAL,
    source_fact TEXT,
    confidence  REAL NOT NULL DEFAULT 0.7
);
CREATE INDEX IF NOT EXISTS idx_slot_scope ON state_slot(session_id, scope, slot_key);
CREATE INDEX IF NOT EXISTS idx_slot_open  ON state_slot(scope, slot_key, until);
-- 不变式：同一 (scope, slot_key) 在任意时刻最多一条未关闭的槽
CREATE UNIQUE INDEX IF NOT EXISTS uq_slot_open ON state_slot(scope, slot_key) WHERE until IS NULL;

-- ── 墓碑：删除必须不可复活 ──────────────────────────────────
CREATE TABLE IF NOT EXISTS tombstone (
    chunk_key   TEXT PRIMARY KEY,
    line_ids    TEXT,                -- JSON array
    session_id  TEXT,
    deleted_at  REAL NOT NULL,
    reason      TEXT,
    author      TEXT                 -- 谁删的
);

-- ── 检索审计 ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retrieval_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    session_id    TEXT,
    query         TEXT NOT NULL,
    plan_json     TEXT,
    selected_json TEXT,
    excluded_json TEXT,
    latency_ms    REAL,
    abstained     INTEGER NOT NULL DEFAULT 0,
    budget_used   INTEGER,
    budget_total  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_rlog_ts ON retrieval_log(ts);

-- ── 生命周期审计（v3，唯一审计，吸收 notebook_promotion）────────
-- 每次状态转移/promotion 留痕；迁移时存量每条补一条 backfill 记录。
CREATE TABLE IF NOT EXISTS item_state_log (
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
);
CREATE INDEX IF NOT EXISTS idx_log_item ON item_state_log(item_id, at);
