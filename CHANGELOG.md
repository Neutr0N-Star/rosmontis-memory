# 变更日志

本项目遵循语义化版本；`api_version`（对外接口版本）与包版本分开管理，破坏性接口改动才升。

## [未发布]

### Changed

- **性能默认值变化**：`numpy` 提为默认依赖（`dependencies = ["numpy>=1.24"]`），向量余弦与批量运算默认走 numpy；`numpy` extra 保留为兼容壳，`pip install -e ".[numpy]"` 继续可用。纯 Python 兜底仍在，未装 numpy 也能运行。
- **备份改为在线备份 + 自检**：`rosmontis backup` 与迁移前备份改用 SQLite 在线备份 API，并核对核心表行数；并发读者在场时不再产出缺数据的 `.bak`，自检不过就报错退出（不再是 `rc=0` 假成功）。
- **未实现的 embedding 通道启动即失败**：`ROSMONTIS_EMBED=local` / `openai` 或拼错的值不再静默降级为 `hash`，而是抛 `embed_provider_unavailable`；`context()` 新增 `vector_mode` / `degraded` / `degraded_reason` 三个字段。

## [2.0.0] - 2026-10-20

**Rosmontis 的首个公开发布版本**，包含此前本地迭代的全部修复。`api_version` 为 `"1"`：HTTP `/v1/*` 端点、五个工具名与错误信封，从这一版起就是对外契约。

### Fixed · 数据正确性

- **订正不再失效**：`state_slot.source_fact` 此前从未被写入，导致 `correct()` 里"关掉旧状态槽"的 UPDATE 恒命中空集，`audit` 的 `stale_state` 也因此永久报 0（安全网结构性失明）。现在写入时会记下槽由哪条事实推出，两条链路都接上了。
- **补录历史不再作废"当前真理"**：此前乱序写入会把当前槽的 `until` 设成一个更早的时刻（负区间），并把历史状态顶成当前状态。现在 `t_event` 早于当前槽起点时只插入一段已闭合的历史区间 `[t_event, cur.since)`，当前槽原样保留；`_insert_slot` 另有一道负区间兜底。
- **`as_of` 时态成立**：此前回到过去检索时，状态块照样标「当前状态」，事实也按"现在是否有效"硬过滤。现在状态块标题随时态切换（「当时的·可能已变化」），事实有效性改为区间判定 `t_valid_from <= as_of < t_valid_to`，冷却判定改用真实时钟，同一 `as_of` 的检索结果可复现。
- **换模型后可恢复**：`embed_backfill` 与 `pending_backfill` 此前只认 `embedding IS NULL`，换模型后旧身份向量既不重算也不计数，`/readyz` 全绿而库里一半数据在语义检索中失明。现在判据包含"身份不符"，并拆分上报 `null_vectors` / `stale_identity`；维度不符的条目也计入 `identity_skipped`，不再提前 `continue` 掉计数。

### Fixed · 信任与可观测

- **备份不再静默产出空文件**：此前 `wal_checkpoint(TRUNCATE)` 返回 `busy` 时只 `except` 不看返回值，并发读时复制出的 `.bak` 可能是坏库，而命令 `rc=0` 还打印 sha256。现在改用 SQLite 在线 backup API，备份后自检三张核心表的行数，失败即抛 `BackupError` / 非零退出。迁移前备份走同一实现——**没备份成功就不许迁移**。
- **未知 embedding provider 启动即失败**：此前 `ROSMONTIS_EMBED=local` / `openai` / 拼写错误都会静默变成 `hash`（无语义），且降级时写路径不落向量，用户只会觉得"她记不住事"。现在抛 `embed_provider_unavailable`（`retryable=False`），错误消息带原值与可用清单。
- **降级不再只有 `/readyz` 一个出口**：`Handle.context()` 新增 `vector_mode` / `degraded` / `degraded_reason` 三个字段（只增不改），库形态使用者也能察觉。
- **`wipe()` 失败点名**：此前逐表 `except Exception: pass`，某表删失败仍返回成功。现在返回 `failed_tables`。**顺带修掉一个真 bug：`hyperedge_member` 从来没被 wipe 过。**
- **锁冲突不再是裸异常**：`tx()` 重试耗尽后抛 `RosmontisError("db_locked", retryable=True)`，宿主可按 `retryable` 处理。

### Fixed · 其他

- `INSERT OR IGNORE` 命中时不再假报 `added`：返回 `{"action": "EXISTS"}`，且不再重复插 `fact_fts`。
- 文本截断按子句边界切，不再硬切字符（`text[:60]` 会切出"…却"这种半句）。
- 做梦不再把源 episode 的 `emotion` 打 0.6 折写回（此前可高频触发、不可恢复，且会污染情绪加权的 A/B 结论）。
- `_entity_seeds` 去掉 `instr` 子串扫，改 `norm_name IN` + 前缀匹配。20k 实体实测 2.06ms → 0.01ms，结果集逐条一致。

### Changed

- **numpy 提为默认依赖**：此前 `dependencies = []`，默认安装走纯 Python 余弦，2000 episode 的 `context()` P50 约 1.4s（`cosine_batch` 占 95%）。`[embed]` extra 保留但**当前无推理侧消费者**。
- `RosmontisError` 移到 `rosmontis/errors.py`（`core` 不能 import `api`，且 `db.tx` 也要抛同一类型）。`rosmontis.api.RosmontisError` 等既有导入路径保持不变。
- `rosmontis/__init__.py` 导出面不变。

### Known limitations（本版未修，已知且记录在案）

- **状态槽**已修负区间与"补录顶替当前真理"，但**事实层（`fact.t_valid_to`）尚未同步**：补录历史事件时仍可能写出 `t_valid_to < t_valid_from`。状态层已安全，事实层的时间轴仍是下一轮的事。
- **`field.py` 的历史事实仍按 `status` 硬过滤**：`dense` / `sparse` 两路已改区间判定，语义场这一路还是"现在无效就看不见"，查历史可能少一路候选。
- **注入预算三套口径打架**：`max(80, budget×depth)` 会放大、常驻块额外累加、硬闸从尾部整行丢弃。后果是小预算时注入为空、大预算时先放大再硬砍。README 的"绝不超预算"在公共 API 层成立，"小预算也能注点东西"不成立。
- **工具层可绕过闸门**：`_skip` 在闸门拒绝后仍强制写库，已归档条目可被"静音复活"、事实被静音即不可逆作废。
- **`settle` 不幂等**：重复调用每次叠加 salience，反复调用可把条目推到近 1.0。
- **遗忘/做梦以 `t_event` 而非 `t_ingest` 为时间基准**：带真实历史时间戳导入的会话可能首次写入即被判冷归档。
- **异步 enrich 只有内存队列**：进程重启后未抽取的 episode 永久停在"只有原文"态，无补偿任务。
- **语义场点火强度两套量纲**：`routes` 分支给排名分、`_vec_seeds` 给校准余弦，按绝对量设的阈值在默认路径下失效。
- **`local` / `openai` 向量通道未实现**：原计划随 v1.1 交付，延至 v1.2。模型下载器已可用，但下载的模型暂无推理侧消费者。

## [1.1.0] - 2026-10-20

首个面向公开分发的版本：加了默认安全、模型资产与向量可靠性。

### Added

- **本地访问 token**：首次 `rosmontis serve` 生成并打印，写入 `~/.rosmontis/token`；除 `/healthz`、`/readyz`、`/docs`、`/openapi.json` 外，全部 HTTP 面要求 `Authorization: Bearer`。
- **绑定安全**：`--host` 非回环时必须显式提供 `--token` / `ROSMONTIS_TOKEN`，否则拒绝启动；容器镜像入口自动生成并打印 token。
- **破坏性接口确认**：`/v1/wipe`、`/v1/settings` 需要 `X-Confirm: yes` 或 `?confirm=yes`。
- **模型资产管理**：`rosmontis models list/install/where/verify/remove`；断点续传、3 次重试、SHA-256 校验、镜像可换、失败保留旧模型。模型不进 wheel。
- **向量可靠性**：断路器（远端失败后不再发起请求、指数退避、恢复自动切回）、向量身份（schema v4 记录 `embed_model`/`embed_dim`，检索只比同身份）、`embed_backfill` 补齐降级期空向量。`/readyz` 输出 `vector_mode`、`embed_model`、`pending_backfill`、断路器状态。
- **迁移守卫**：高于当前支持的库版本拒绝打开；迁移前自动备份；`rosmontis.open(..., migrate="auto"|"never"|"dry_run")`。
- **`ROSMONTIS_HOME`**：`~/.rosmontis` 统一承载 db、模型、token、`.env`；开发态仍兼容仓库根 `.env`。
- 发布工程：`py.typed`、sdist + wheel 构建与 `twine check` 进入 CI、契约测试（版本号一致、公共面冻结、openapi 路由齐全）。

### Changed

- **无模型不再写脏记忆**：没有真实模型输出时不写 assistant 行，库层抛 `llm_unavailable`（`retryable=True`），HTTP 返回 503；流式发 `error` 事件后终止，不落半截回复。
- CORS 默认不再放行 `Origin: null`；`file://` 场景需显式配置 `ROSMONTIS_CORS_ORIGINS=null`。
- 包元数据补 MIT 许可证分类、Beta 状态、Python 3.13；作者改为项目维护者。
- 服务根路径 `/` 只返回服务信息 JSON，`/debug` 与静态页面入口移除。

### Removed

- 仓库内前端目录、内部施工稿、一次性对照脚本与演示二进制库；这些不再随发布物分发。

### Security

- 本版本起，把服务暴露到公网需要自行加反向代理与网关鉴权；默认配置只服务本机。

## [1.0.0] - 2026-10-19

首个可交付版本：本地记忆后端（库 + HTTP + 守护进程），无前端。

### Added

- **统一记忆对象与生命周期状态机**：`episode` 升为唯一 item，新增 `origin/state/state_at/gate_version`、`item_state_log`；schema v3 单事务幂等迁移，失败回滚保持旧版本。
- **主被动一体**：主动笔记、做梦产物、折叠摘要全部走唯一写入管线（`INSERT INTO episode` 只在 `core/pipeline.py`），`origin ∈ {ingest, tool, system, dream, summary}`。
- **语义场检索（第四路）**：三路点火（向量/FTS/实体）+ 稀疏扩散（凸组合，有界收敛）+ ACT-R 激活 + 三空间衰减；`ROSMONTIS_FIELD=0` 可关。
- **公共 API v1**：`rosmontis.open()` / `Handle`（`wake / context / turn / observe / settle / settle_now / close`），`context` 为纯函数（不写库、不超预算）。
- **HTTP 接口**：`/v1/turn`、**`/v1/turn/stream`（SSE：meta→tool→delta→done）**、`/v1/context`、`/v1/observe`、`/v1/session/*`、`/v1/settle`、`/v1/jobs`、`/v1/state|timeline|episodes`，以及 `/healthz`、`/readyz`。
- **守护进程 `rosmontis daemon`**（含 `--once`）：60s 心跳，自动 review / dream / settle / forget，按 `ROSMONTIS_IDLE_TIMEOUT`（默认 1800 秒）对空闲会话结算；SIGINT/SIGTERM 优雅收尾。
- **任务与幂等**：`job` 表（租约/退避/死信/重放）；写接口支持 `Idempotency-Key`（24h 内同键回放且只写一条）。
- **分发与工程化**：`pyproject.toml`、`rosmontis` 命令、`python -m rosmontis`、Dockerfile + compose、`openapi.json` 契约快照、GitHub Actions CI、MIT LICENSE、`docs/RELEASING.md`。
- **文档**：README（接入优先）、`docs/14–17`、`docs/guides/{quickstart-library,api-reference,operations}.md`、本变更日志。
- **测试与基准**：124 项单元测试全绿；`bench/` 可复现质量基准与帕累托前沿（200 段合成对话 / 135 题）。

### Changed

- `tier`/`source` 降为**兼容投影**，`state` 为生命周期真值；召回按状态过滤（`archived/muted` 恒不可见，`inbox` 默认不可检索，可用 `ROSMONTIS_INBOX_RETRIEVABLE` 打开）。
- 错误信封统一为 `{ok, data, error{code, retryable, message}}`，5xx 默认 `retryable=true`。
- CORS 默认只放行本机来源与 `file://`，可用 `ROSMONTIS_CORS_ORIGINS` 覆盖。

### Fixed

- 异步 enrich 完成后重算 `salience`，同步/异步入口结果一致（差 0）。
- `/v1/context` 不再偷偷写 `retrieval_log`、不再超出调用方预算。
- 向量扫描：补 `(session_id, t_ingest DESC)` 等索引、去掉 `(?='*' OR col=?)` 反模式（20k 库 dense 682ms → 472ms）。
- **`forget` 批量化**：100k 库 16.15s → 1.05s（SQL 条数 32,161 → 1,383）。
- fact 侧 FTS 静默失败、sparse 隐藏条目被挤空等旧缺陷。

### Known limitations

- 不含前端；模型自备（Ollama / OpenAI 兼容 / `rule` 降级）。
- 默认只绑 `127.0.0.1`、无鉴权、单机单进程；多用户/公网部署等后续版本。
- 无 ANN/量化索引：十万条以内够用，再大需外挂索引。
- 基准集为合成数据，参数（拒答阈值、salience、衰减）尚未用真实语料标定。
- Windows 控制台建议 `set PYTHONUTF8=1`，否则中文输出可能乱码。
