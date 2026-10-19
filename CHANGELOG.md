# 变更日志

本项目遵循语义化版本；`api_version`（对外接口版本）与包版本分开管理，破坏性接口改动才升。

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
