# Rosmontis 运维手册

覆盖：备份/恢复、v2→v4 迁移、降级矩阵、日志与指标、常见故障、数据清理。
库接入示例见 [quickstart-library.md](quickstart-library.md)；完整参数与配置表见 [api-reference.md](api-reference.md)。

> 纪律：`rosmontis.db` 只许只读，任何写操作先在**副本**上演练；迁移前必须 checkpoint + 备份。
> 本文命令以 Windows PowerShell 为主，macOS/Linux 对应命令在括号里给出。

## 1. 运行形态与前置

| 形态 | 命令 | 依赖 |
|---|---|---|
| 库 | `import rosmontis` / `rosmontis.open(...)` | Python 3.10+；可选 numpy/jieba |
| CLI | `python run.py ingest/recall/stats/audit/forget/entities/state/timeline` | 同上 |
| 服务 | `python run.py serve --port 8770` | fastapi、uvicorn（`run.py:71-78`） |
| 演示 | `python run.py demo --reset`（会重建 `demo.db`，别对生产库跑） | 同上 |
| 评测 | `python bench/run_eval.py` / `python bench/pareto.py` | 见 §6.4 |

安装（仓库已带 `pyproject.toml`）：`pip install -e .`（仅库）；`pip install -e ".[serve]"`（HTTP 服务）；`pip install -e ".[dev]"`（评测/开发）。

- SQLite 为 **WAL** 模式（`rosmontis/schema.sql:6-8`、`rosmontis/core/db.py:81-82`）；磁盘上会有 `rosmontis.db-wal`、`rosmontis.db-shm`。
- 单机单进程是当前假设；连接用 thread-local 复用，写事务 `BEGIN IMMEDIATE`，锁冲突退避 2 次、`busy_timeout=3000ms`（`rosmontis/core/db.py:45-63,294-323`）。不要对同一个库起两个服务进程。
- 首启会做环境检查（Python ≥3.10、SQLite FTS5，`run.py:30-42`）。

## 2. 配置生效方式

- 环境变量 > `.env` > 代码默认（`.env` 只 `setdefault`，`rosmontis/config.py:10-24`）。
- 完整环境变量表见 [api-reference.md](api-reference.md) §8。
- 修改配置后必须重建 `Memory`/`Harness`；服务形态的 `POST /v1/settings` 会写 `.env` 并重建（`rosmontis/service/app.py:95-112,487-506`）。
- 两个容易忽略的开关：
  - `ROSMONTIS_INBOX_RETRIEVABLE=1`：允许 `inbox` 速记参与召回（默认关闭，C2=B）。
  - `ROSMONTIS_FIELD=0`：关闭语义场，只保留 dense/sparse/graph（`rosmontis/core/field.py:129-132`）。

## 3. 备份与恢复

### 3.1 备份（一致性优先）

```powershell
# 0) 先停掉服务/所有写入进程
# 1) 检查点：把 -wal 里的数据并回主库（必须在复制前做）
python -c "import sqlite3; c=sqlite3.connect('rosmontis.db'); print(c.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()); c.close()"

# 2) 用 SQLite backup API 复制（比直接 copy 文件更稳）
python -c "import sqlite3; s=sqlite3.connect('rosmontis.db'); d=sqlite3.connect('rosmontis-backup-20261019.db'); s.backup(d); d.close(); s.close()"

# 3) 记录指纹与结构版本
Get-FileHash rosmontis-backup-20261019.db -Algorithm SHA256
python -c "import sqlite3; c=sqlite3.connect('rosmontis-backup-20261019.db'); print(c.execute('PRAGMA user_version').fetchone()[0]); print(c.execute('PRAGMA integrity_check').fetchone()[0]); c.close()"
```

（macOS/Linux：`sqlite3 rosmontis.db "PRAGMA wal_checkpoint(TRUNCATE);"`、`cp`/`sha256sum`，backup API 的 Python 命令相同。）

- **不要**在未 checkpoint、未停写的情况下直接 `copy rosmontis.db`：WAL 里已提交的数据可能不在主文件。
- 备份后检查 `integrity_check` 返回 `ok`、`user_version` 符合预期（当前真库见 §4.0）。
- 备份保留策略由使用方定；迁移前后各留一份（建议保留到验收后）。

### 3.2 恢复

```powershell
# 0) 停服务
# 1) 把备份复制回正式路径
Copy-Item rosmontis-backup-20261019.db rosmontis.db -Force
# 2) 确认没有进程占用后，清掉旧的 WAL/SHM（否则会和旧主库混用）
Remove-Item rosmontis.db-wal, rosmontis.db-shm -ErrorAction SilentlyContinue
# 3) 起服务前做完整性检查
python -c "import sqlite3; c=sqlite3.connect('rosmontis.db'); print(c.execute('PRAGMA user_version').fetchone()[0], c.execute('PRAGMA integrity_check').fetchone()[0]); c.close()"
```

自检：`GET /health` 返回 ok；`GET /v1/stats` 的 `episode/fact/entity` 与备份时记录一致；`GET /v1/audit` 无 high 问题。
恢复旧版本备份时，代码版本要一起回退（schema v2 备份 + v4 代码会触发迁移，不是「恢复原状」）。

## 4. v2 → v4 迁移

### 4.0 当前状态（只读核对）

本文写作时，仓库内两个库仍是 v2：

```text
rosmontis.db  user_version=2  episode=182
demo.db   user_version=2  episode=9
```

（只读 `mode=ro` 查询；未迁移、未写入。）首次用 `Memory()`/`rosmontis.open()` 打开它们时，`DB.init_db` 会自动迁移到 v4（v2→v3→v4 两步，`rosmontis/core/db.py:77-122`）。

### 4.1 迁移做了什么

- 只加列/加表，不删不改（`rosmontis/schema.sql:1-4`、`rosmontis/core/db.py:197-231`）：
  - `episode` 加 `origin / state / state_at / gate_version`；
  - `fact` 加 `origin / archived`；
  - 新建 `item_state_log`；
  - 回填 `state/origin`（`db.py:234-269`），每个 episode 补一条 backfill 日志（`db.py:281-291`）。
- 单事务：`BEGIN IMMEDIATE` → 加列 → 回填 → 建表 → 补日志 → 写 `user_version=3` → COMMIT；失败 ROLLBACK 保持 v2（`db.py:103-122`）。
- 幂等：只在 `user_version < 3` 时跑（`db.py:84-90`）。
- `meta.schema_version` 会同步补成 3（`db.py:92-100`）。

### 4.2 迁移步骤（真库）

```powershell
# 1) 停写 + 备份（见 §3.1），记录 sha256 与 user_version
# 2) 在副本上演练：复制备份到临时路径，用 Memory 打开它（不要指向 rosmontis.db）
python -c "import sys; sys.path.insert(0,'.'); from rosmontis import Config; from rosmontis.core.memory import Memory; c=Config(db_path='rosmontis-dryrun.db'); Memory(c); print('migrated')"
python -c "import sqlite3; c=sqlite3.connect('rosmontis-dryrun.db'); print(c.execute('PRAGMA user_version').fetchone()[0], c.execute('PRAGMA integrity_check').fetchone()[0]); c.close()"
# 3) 副本通过后，对真库执行一次：任何会构造 Memory 的命令都会迁移
python run.py stats
# 4) 验收
python -c "import sqlite3; c=sqlite3.connect('rosmontis.db'); print(c.execute('PRAGMA user_version').fetchone()[0], c.execute('PRAGMA integrity_check').fetchone()[0]); c.close()"
python run.py audit
```

`tests/test_items.py:606-659` 与 `tests/test_pipeline_closure.py` 已覆盖：真库只读副本迁移、行数不变、`item_state_log` 条数=episode 条数、`tier='dream'` 映射、`t_event=1e18`/异常 session_id 容错、二次幂等、注入异常回滚。

### 4.3 回滚

- 首选：停服，用迁移前备份恢复（§3.2）。
- 只回代码不回库也是允许的：v2 代码不读 v3 新列，`state` 可能陈旧；再升级需按 `origin IS NULL / state_at` 重跑回填（见 `docs/15-交付蓝图与模块契约.md` §3.6）。
- **不做 DROP COLUMN**（SQLite 版本依赖 + 数据风险）；迁移只加不删。
- 迁移失败时日志会出现 `migration v2 → v4 failed, rolled back`，库仍是 v2；先看异常，再重试，不要手工改 `user_version`。

### 4.4 迁移后的检查表

- `PRAGMA user_version = 3`、`meta.schema_version = 3`。
- `episode` 行数与迁移前一致；`item_state_log` 条数 = episode 条数（每行至少一条 backfill，`db.py:281-291`）。
- `source='dream'` 的行是 `state='dream'`、`origin='dream'`。
- `PRAGMA integrity_check` = `ok`。
- 现有测试全绿：`python -m unittest discover -s tests -p "test_*.py"`（184 项）。

## 5. 降级矩阵

| 缺失/故障 | 系统行为 | 如何观察 | 代码 |
|---|---|---|---|
| Ollama embedding 不可用 | 退化为确定性 hash 向量，`embedder.degraded=True`；写入不落伪向量（存 NULL） | `/v1/stats` 的 `embed.degraded/reachable`；`degraded` 也会在 `wake` 返回 | `rosmontis/core/embed.py:54-75`、`rosmontis/core/write.py:97-99`、`rosmontis/core/notebook.py:87-93` |
| Ollama LLM 不可用 | 抽取退回规则兜底（`rule_min`）；对话仍可继续 | `/v1/stats` 的 `llm_hot.failures`；`ingest` 的 `source` | `rosmontis/core/extract.py:149-170`、`rosmontis/core/llm.py:96-100` |
| 模型没有工具调用能力 | `make_llm_callable` 静默降级为纯文本，`tool_calls=[]`；五工具不触发 | `turn` 返回的 `tools_called=[]` | `rosmontis/harness.py:398-448` |
| OpenAI 无 key/401/超时 | 请求失败 → 返回 None → Harness 抛「模型没有返回内容」→ HTTP 503 `llm_unavailable` | `llm.last_error`（当前只落日志，不在 stats 里） | `rosmontis/core/llm.py:102-134`、`rosmontis/harness.py:640-643`、`app.py:467-469` |
| numpy 缺失 | 纯 Python 向量运算，功能不变、速度下降 | `rosmontis.core.vec.HAS_NUMPY` | `rosmontis/core/vec.py:12-18` |
| jieba 缺失 | 字符二元组分词，召回率略降 | `rosmontis.core.text._HAS_JIEBA` | `rosmontis/core/text.py:10-15,59-69` |
| FastAPI/uvicorn 缺失 | `run.py serve` 直接提示安装并退出，**没有** fallback HTTP server | 终端提示 | `run.py:71-78` |
| 语义场关闭（`ROSMONTIS_FIELD=0`） | 少一路 field 召回，dense/sparse/graph 仍工作 | `retrieval_log.plan_json.field_enabled=false` | `rosmontis/core/field.py:129-132`、`rosmontis/core/recall.py:355-359` |
| inbox 可见性 | 默认 `inbox` 不参与召回；`ROSMONTIS_INBOX_RETRIEVABLE=1` 打开 | `retrieval_log.plan_json.inbox_retrievable` | `rosmontis/config.py:179-180`、`rosmontis/core/recall.py:92-103,176-177` |

原则：降级必须显式可见；`/v1/stats` 与 `plan_json` 是排查入口（`rosmontis/core/db.py` 注释、`docs/03-技术方案与契约.md:299-309`）。

## 6. 日志、指标、审计

### 6.1 日志

- `Memory()` 初始化时 `logging.basicConfig(level=cfg.log_level)`（`rosmontis/core/memory.py:36-40`）；默认 `INFO`，用 `ROSMONTIS_LOG=DEBUG` 打开调试。
- 日志 logger 命名：`rosmontis.*`（`rosmontis.memory`、`rosmontis.write`、`rosmontis.recall`、`rosmontis.harness`、`rosmontis.service`、`rosmontis.db` 等）。
- 常见关键字：
  - `migrating schema from v2 to v3` / `migration done` / `migration ... failed, rolled back`（`db.py:105-122`）
  - `embed degraded -> hash`（`embed.py:63-67`）
  - `tool <name> failed`（`tools.py:680-683`）、`llm call failed`（`harness.py:629-631`）
  - `forget: scanned=...`（`forget.py:221`）

### 6.2 指标与审计端点

| 入口 | 返回要点 |
|---|---|
| `GET /health` | `{ok, ts}` 旧探针（`app.py:674-677`） |
| `GET /healthz` | liveness：进程与事件循环可响应（`app.py:679-682`） |
| `GET /readyz` | readiness：只读检查库/schema/降级；失败返回 503（`app.py:684-717`） |
| `GET /v1/stats` | `episode/fact/entity/state_open/theme/db_path/embed/llm_hot/llm_deep/degraded` + `config`（`rosmontis/core/memory.py:835-856`、`app.py:650-654`） |
| `GET /v1/audit` | `issues[]`（high/medium/low）+ `health{score,label}`（`rosmontis/core/audit.py`） |
| `GET /v1/config` | 当前配置（`openai_key` 不输出，`config.py:198-202`） |
| `GET /v1/settings` | 模型配置 + `openai_key_set`（`app.py:468-485`） |

没有 Prometheus/外部指标后端；要接监控，从 `/v1/stats`、`/v1/audit` 与下面三张审计表自采。

### 6.3 审计表

| 表 | 内容 | 排查用途 |
|---|---|---|
| `retrieval_log` | 每次召回：query、plan_json、selected、excluded、latency、abstained | 查「为什么没召回/为什么拒答」；`plan_json` 含 `filtered_by_state/inbox_retrievable/field_enabled` |
| `item_state_log` | 每次状态迁移/promotion：from/to/action/reason/gate_json/actor | 查「这条记忆谁改的、为什么」 |
| `dream_log` | 每次做梦：replay/bind/abstract/decoupled/text/evidence | 查「梦为什么没做/做了什么」 |

示例：

```powershell
python -c "import sqlite3; c=sqlite3.connect('rosmontis.db'); c.row_factory=sqlite3.Row; [print(dict(r)) for r in c.execute('SELECT ts,query,abstained,latency_ms,plan_json FROM retrieval_log ORDER BY id DESC LIMIT 5')]; c.close()"
```

### 6.4 评测

```bash
PYTHONIOENCODING=utf-8 python bench/run_eval.py                 # 质量评测，输出 JSON
PYTHONIOENCODING=utf-8 python bench/run_eval.py --field off     # 关场对照
PYTHONIOENCODING=utf-8 python bench/pareto.py                   # 配置网格 → bench/report.md
```

- 评测库建在系统临时目录，跑完删除，**不碰 `rosmontis.db`**（`bench/run_eval.py` 文件头）。
- 缺 Ollama/网络会自动回退「脚本化金标 + 规则抽取 + hash 向量」，并写进 JSON 的 `meta.*_backend`。
- `bench/report.md` 由脚本生成，禁止手填数字。
- 设计目标与判据：`docs/15-交付蓝图与模块契约.md` §2；当前结论：`bench/report.md`。

## 7. 常见故障

| 现象 | 可能原因 | 排查/处理 |
|---|---|---|
| `python run.py serve` 报缺 fastapi/uvicorn | 没装服务依赖 | `pip install fastapi uvicorn`（`run.py:71-78`） |
| 首轮对话特别慢（数秒~数十秒） | Ollama 冷启动/模型加载；或默认 ollama 不可用后走超时 | 预热模型；离线用 `embed_provider=hash` + `llm_hot=rule` |
| 503 `llm_unavailable` | 模型返回空（密钥错、断网、上游拒绝） | 查日志里 `llm chat failed` / `last_error`；`/v1/settings` 核对 base/key/model |
| 422 `invalid_argument` / `missing_session` | body 缺字段、空文本、`session_id="*"` | `*` 是内部保留，HTTP 会 400/422（`app.py:55-60`） |
| 409 `rejected` | gate 拒绝（升格频率/幅度/来源、笔记写失败） | 看返回 `error.message`；`item_state_log` 里有 `gate_json.checks` |
| 404 `not_found` | item/entity/job id 不存在 | 确认 id；job 只存在当前 Handle 内存 |
| `database is locked` | 多进程写、长事务、外部占用 | 停掉第二个进程；`busy_timeout=3s`；必要时 checkpoint |
| 迁移失败 | 列/表冲突、磁盘、外部锁 | 看 `migration ... rolled back`；库仍是 v2；修复后重跑，不要手改 `user_version` |
| 写入成功但立刻召回不到 | 异步 enrich 还没跑完；或条目是 `inbox`（默认不可检索） | 等 `wait_enrich`/后台池；查 `item_state`；需要时开 `ROSMONTIS_INBOX_RETRIEVABLE=1` |
| `context` 返回空 / `abstained=true` | 没有相关记忆；hash 模式词面不重合；阈值未标定 | 用原文关键词查询；接 bge-m3；按语料标定 `theta_abstain`（`config.py:65`） |
| `turn` 的 `tools_called=[]` | 模型不支持工具调用/没配 key | 换支持 function calling 的模型；查 `make_llm_callable` 降级日志 |
| `retrieval_log` / WAL 持续变大 | 每次召回都记日志；WAL 未 checkpoint | 定期 checkpoint；按 `ts` 归档/清理 `retrieval_log`（运维自定策略） |
| Windows 控制台中文乱码 | 控制台编码 | `$env:PYTHONIOENCODING="utf-8"` |

## 8. 数据清理与重置

- 单条处理：`rosmontis_skip`（静音/不删）→ `item_state= muted`；`POST /v1/remove`（写墓碑+归档）；`POST /v1/items/{id}/transition` 可做 `archive/restore`。
- 整角色：`POST /v1/wipe?session_id=...`，**不可逆**；清理 `_WIPE_TABLES` 中的表并额外按 episode 清 `item_state_log`（`rosmontis/core/memory.py:339-362`）。
  - 注意：当前 `wipe` 不清 `dream_log`，也不清旧的 `notebook_promotion` 兼容表（不在 `_WIPE_TABLES` 里，`memory.py:333-337`）；需要彻底清空时按业务自行处理。
- 整库重置：停服 → 把 `rosmontis.db`、`rosmontis.db-wal`、`rosmontis.db-shm` 一起改名/移走 → 重新初始化；**不要**只删主库。
- 演示库：`python run.py demo --reset` 会重建 `demo.db`；不要把它指向 `rosmontis.db`。

## 9. 上线/变更演练清单

1. 备份 + checkpoint + 记录 `sha256`/`user_version`。
2. 在副本上跑迁移与 `integrity_check`；跑 `python -m unittest discover -s tests -p "test_*.py"`（184 项）。
3. 记录基线：`/v1/stats`、`/v1/audit`、20 条固定查询的 `retrieval_log`（当前 `Memory.recall` 会写库，基线只用副本 + 冻结时钟）。
4. 变更后复核：`/health`、`/v1/stats`、`/v1/audit`、固定查询命中率/延迟。
5. 保留迁移前/后两份备份直到验收完成；回滚预案见 §4.3。









