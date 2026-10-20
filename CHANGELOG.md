# 变更日志

本项目遵循语义化版本；`api_version`（对外接口版本）与包版本分开管理，破坏性接口改动才升。

## [1.1.0] - 2026-10-20

首个面向公开分发的版本：加了默认安全、模型资产与向量可靠性。

### Added

- **本地访问 token**：首次 `alaya serve` 生成并打印，写入 `~/.alaya/token`；除 `/healthz`、`/readyz`、`/docs`、`/openapi.json` 外，全部 HTTP 面要求 `Authorization: Bearer`。
- **绑定安全**：`--host` 非回环时必须显式提供 `--token` / `ALAYA_TOKEN`，否则拒绝启动；容器镜像入口自动生成并打印 token。
- **破坏性接口确认**：`/v1/wipe`、`/v1/settings` 需要 `X-Confirm: yes` 或 `?confirm=yes`。
- **模型资产管理**：`alaya models list/install/where/verify/remove`；断点续传、3 次重试、SHA-256 校验、镜像可换、失败保留旧模型。模型不进 wheel。
- **向量可靠性**：断路器（远端失败后不再发起请求、指数退避、恢复自动切回）、向量身份（schema v4 记录 `embed_model`/`embed_dim`，检索只比同身份）、`embed_backfill` 补齐降级期空向量。`/readyz` 输出 `vector_mode`、`embed_model`、`pending_backfill`、断路器状态。
- **迁移守卫**：高于当前支持的库版本拒绝打开；迁移前自动备份；`alaya.open(..., migrate="auto"|"never"|"dry_run")`。
- **`ALAYA_HOME`**：`~/.alaya` 统一承载 db、模型、token、`.env`；开发态仍兼容仓库根 `.env`。
- 发布工程：`py.typed`、sdist + wheel 构建与 `twine check` 进入 CI、契约测试（版本号一致、公共面冻结、openapi 路由齐全）。

### Changed

- **无模型不再写脏记忆**：没有真实模型输出时不写 assistant 行，库层抛 `llm_unavailable`（`retryable=True`），HTTP 返回 503；流式发 `error` 事件后终止，不落半截回复。
- CORS 默认不再放行 `Origin: null`；`file://` 场景需显式配置 `ALAYA_CORS_ORIGINS=null`。
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
- **语义场检索（第四路）**：三路点火（向量/FTS/实体）+ 稀疏扩散（凸组合，有界收敛）+ ACT-R 激活 + 三空间衰减；`ALAYA_FIELD=0` 可关。
- **公共 API v1**：`alaya.open()` / `Handle`（`wake / context / turn / observe / settle / settle_now / close`），`context` 为纯函数（不写库、不超预算）。
- **HTTP 接口**：`/v1/turn`、**`/v1/turn/stream`（SSE：meta→tool→delta→done）**、`/v1/context`、`/v1/observe`、`/v1/session/*`、`/v1/settle`、`/v1/jobs`、`/v1/state|timeline|episodes`，以及 `/healthz`、`/readyz`。
- **守护进程 `alaya daemon`**（含 `--once`）：60s 心跳，自动 review / dream / settle / forget，按 `ALAYA_IDLE_TIMEOUT`（默认 1800 秒）对空闲会话结算；SIGINT/SIGTERM 优雅收尾。
- **任务与幂等**：`job` 表（租约/退避/死信/重放）；写接口支持 `Idempotency-Key`（24h 内同键回放且只写一条）。
- **分发与工程化**：`pyproject.toml`、`alaya` 命令、`python -m alaya`、Dockerfile + compose、`openapi.json` 契约快照、GitHub Actions CI、MIT LICENSE、`docs/RELEASING.md`。
- **文档**：README（接入优先）、`docs/14–17`、`docs/guides/{quickstart-library,api-reference,operations}.md`、本变更日志。
- **测试与基准**：124 项单元测试全绿；`bench/` 可复现质量基准与帕累托前沿（200 段合成对话 / 135 题）。

### Changed

- `tier`/`source` 降为**兼容投影**，`state` 为生命周期真值；召回按状态过滤（`archived/muted` 恒不可见，`inbox` 默认不可检索，可用 `ALAYA_INBOX_RETRIEVABLE` 打开）。
- 错误信封统一为 `{ok, data, error{code, retryable, message}}`，5xx 默认 `retryable=true`。
- CORS 默认只放行本机来源与 `file://`，可用 `ALAYA_CORS_ORIGINS` 覆盖。

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
