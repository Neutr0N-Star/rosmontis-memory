# Rosmontis API 参考

版本：包 `1.1.0`、公共 API `1`（`rosmontis/__init__.py:10-11`、`rosmontis/api.py:55`）、schema v4（`rosmontis/core/db.py` 的 `SCHEMA_VERSION`）。
本文以当前工作树代码为准。服务形态（HTTP）见 §6；运维流程见 [operations.md](operations.md)。

## 0. 三层入口

| 入口 | 用途 | 代码 |
|---|---|---|
| `rosmontis.open()` → `Handle` | 宿主接入的稳定公共 API v1（推荐） | `rosmontis/api.py:437-484` |
| `Memory` | 进程内门面，高级/兼容用法 | `rosmontis/core/memory.py:35` |
| `Harness` | 我们替宿主管对话循环 | `rosmontis/harness.py:451` |
| `rosmontis.service.app:app` | FastAPI 服务 | `rosmontis/service/app.py:236` |

统一约定：
- 时间一律 epoch 秒（REAL，`rosmontis/schema.sql:2`）。
- id 是确定性哈希/前缀：`ep_`（episode）、`fa_`（fact）、`en_`（entity）、`sl_`（state_slot）、`he_`（hyperedge）、`pe_`（persona）、`ep_tool_`（工具写入）。见 `rosmontis/core/ids.py`、`rosmontis/tools.py:326`。
- 写入 episode 的唯一入口是 `core/pipeline.submit`（`rosmontis/core/pipeline.py:203`）；状态迁移唯一入口是 `pipeline.transition`（`pipeline.py:358`）；升格判定唯一入口是 `gate.decide`（`rosmontis/core/gate.py:168`）。
- `state` 是生命周期真值，`tier / archived / muted` 是兼容投影（`rosmontis/core/item.py:1-10,43-52`）。

## 1. 公共 API：`rosmontis.open`

```python
import rosmontis
h = rosmontis.open("my_memory.db", {"embed_provider": "hash", "llm_hot": "rule"})
```

签名（`rosmontis/api.py:437`）：

```python
rosmontis.open(db_path: str | Config, config: dict | None = None) -> Handle
```

- `db_path`：SQLite 文件路径；父目录不存在会自动创建（`rosmontis/core/db.py:52-54`）。
- 也接受 `Config` 实例作为第一个参数（此时忽略 `config`，`rosmontis/api.py:452-455`）。
- `config` 可覆盖任意 `Config` 字段；另接受 Handle 级选项：

| Handle 选项 | 默认 | 作用 |
|---|---|---|
| `llm_callable` / `llm` | `None` | 模型回调 `(messages, tools) -> dict`；不传则 `turn` 只回「（没有接模型）」但仍记录用户输入 |
| `system_prompt` | 内置一句 | 拼进 system prompt 的本体提示（`rosmontis/harness.py:474`） |
| `auto_tick` | `False` | 每轮是否在后台跑 `tick()`；公共 API 默认交给外部调度器（`rosmontis/api.py:112-120`） |
| `memory_budget` | 配置默认 | 每轮检索 token 预算 |
| `max_tool_rounds` | `3` | 工具循环上限（`rosmontis/harness.py:43,477`） |
| `observe_window` | `300.0` | `observe` 同文本幂等窗口（秒，`rosmontis/api.py:111,219-233`） |

未知键会被忽略；`config` 不是 dict 时抛 `RosmontisError("invalid_argument")`（`rosmontis/api.py:459-467`）。
打开失败抛 `open_failed`（retryable，`rosmontis/api.py:483-484`）。

`Handle` 还暴露只读属性 `cfg`、`mem`、`harness`、`closed`（`rosmontis/api.py:123-137`），以及上下文管理器 `with`（`rosmontis/api.py:391-396`）。

## 2. Handle 方法

### 2.1 `wake(space, conversation)`

- 返回：`{ok, space, conversation, session_id, persona_text, persona_chars, persona_budget, persona_items, resident_text, resident_tokens, resident_pages, inbox, degraded}`（`rosmontis/api.py:140-153`、`rosmontis/harness.py:482-506`）。
- 副作用：写入种子人格（`INSERT OR IGNORE`，幂等，`rosmontis/core/persona.py:25-40`）；读人格/常驻/inbox。
- 幂等：重复调用不会重复播种。

### 2.2 `context(space, conversation, query, budget=None)`

- 返回：`{ok, space, conversation, session_id, query, system_prefix, memory_text, tokens, memory_tokens, persona_tokens, abstained, budget, parts, latency_ms, as_of, pure}`（`rosmontis/api.py:155-185`）。
- `system_prefix` 是人格文本；`memory_text` 是 `<memory>...</memory>` 检索块；`parts` 里分别是 persona / memory 明细。
- **纯函数**：不写 `episode`、不写 `retrieval_log`；`memory_tokens` 被硬闸压到 `budget` 内（按整行丢弃，绝不截半句，`rosmontis/harness.py:104-126,159`）。
- `budget <= 0` 或非整数抛 `invalid_argument`（`rosmontis/api.py:159-165`）。

### 2.3 `turn(space, conversation, text, *, history=None, extra_system=None)`

- 返回：`TurnResult.to_dict()` + `ok/space/conversation/session_id`（`rosmontis/api.py:187-208`）：
  `reply, memory_used{persona_chars, memory_items, memory_tokens, depth, abstained, tokens}, tools_called[{tool,args,ok,reason,id,state,tool_call_id}], abstained, latency_ms, rounds, usage, finish_reason`。
- 副作用：
  - `ingest_async(user)` 写一条 episode（`rosmontis/harness.py:543`）；
  - 若配置了 `auto_tick`，后台跑 `tick()`（`rosmontis/harness.py:546-547`）；
  - 组装上下文（`retrieval_log` 会写，`rosmontis/harness.py:550`）；
  - 工具循环最多 `max_tool_rounds` 轮，`dispatch` 结果回灌（`rosmontis/harness.py:616-695`）；
  - 回复非空时 `ingest_async(assistant)` 再写一条（`rosmontis/harness.py:557-558`）。
- 空文本抛 `invalid_argument`；模型返回空内容抛 `llm_unavailable(retryable=True)`（`rosmontis/api.py:199-207`、`rosmontis/harness.py:640-643`）。
- 未先 `wake` 会自动 `wake`（`rosmontis/api.py:201,403-408`）。

### 2.4 `observe(space, conversation, user_text, assistant_text)`

- 返回：`{ok, space, conversation, session_id, queued, idempotent, written[], user?, assistant?}`；`user/assistant` 是 `ingest_async` 的结果（`rosmontis/api.py:210-250`）。
- 副作用：对非空文本各写一条 episode（异步抽取）；不生成回答。
- 幂等：同一 `(space, conversation, user_text, assistant_text)` 在 `observe_window` 内第二次调用返回 `idempotent=True`，不再写（`rosmontis/api.py:219-233`）。
- 空输入抛 `invalid_argument`（`rosmontis/api.py:216-217`）。

### 2.5 `settle(space, conversation)` / `job(job_id)` / `wait(job_id, timeout=60)`

- `settle` 返回：`{ok, space, conversation, session_id, job_id, queued, deduplicated}`（`rosmontis/api.py:252-296`）。
- 后台线程执行 `Harness.sleep`：`close_session` → 条件 `notebook_review` → `dream`（`rosmontis/api.py:298-311`、`rosmontis/harness.py:698-726`）。同 session 进行中的任务去重（`deduplicated=True`）。
- `job(job_id)` 返回 `{ok, job_id, kind, state, space, conversation, session_id, created_at, finished_at, result, error}`（`rosmontis/api.py:314-320,421-434`）；`state` 为 `running / done / failed`。
- `wait` 轮询到结束；超时抛 `timeout(retryable=True)`（`rosmontis/api.py:322-334`）。
- job 只存在当前 Handle 的内存里，`close()` 后不再可查（`rosmontis/api.py:107,357-389`）。

### 2.6 `tool(space, conversation, name, args)`

执行一次工具调用，等价于 `tools.dispatch`（`rosmontis/api.py:337-341`）。返回结构见 §3。

### 2.7 `item_state(item_id)` / `transition(...)`

- `item_state` 返回 `{ok, id, kind, session_id, state, tier, source, archived, muted, superseded}`（episode）或 `{..., status, archived, superseded}`（fact）；不存在抛 `not_found`（`rosmontis/api.py:343-349`、`rosmontis/tools.py:495-525`）。
- `transition(item_id, action, *, target="", reason="", actor="system")` 返回六键 `{ok, id, state, from, to, reason}`（`rosmontis/api.py:351-354`、`rosmontis/tools.py:528-640`）。
- 常用 action：`promote`（`target=persona/page/resident/important`）、`skip/mute`、`archive/remove`、`restore/unmute`（`rosmontis/tools.py:235-252,528-640`）。是否允许由 gate 判定。

### 2.8 `close()`

- 可重复调用；等后台 settle 与 enrich 收尾（`rosmontis/api.py:357-389`）。关闭后再调用其他方法抛 `closed`（`rosmontis/api.py:399-401`）。

## 3. 五个工具

工具 schema 在 `rosmontis/tools.py:53-150`；dispatcher 在 `rosmontis/tools.py:644-683`。
写类工具统一返回六键：`{ok, id, state, from, to, reason}`（`rosmontis/tools.py:157-174`）；被 gate 拒绝时 `ok=false`、`state` 停在旧态、`reason` 给模型看。

| 工具 | 必填参数 | 可选参数 | 成功返回要点 | 主要副作用 |
|---|---|---|---|---|
| `rosmontis_remember` | `what` | `noticed / means / next / topic` | `id, state, from, to` | 新建 episode + FTS + `item_state_log`；后台 enrich 投影 entity/fact/state/edge |
| `rosmontis_recall` | `query` | `budget`（默认 600） | `{ok, found, prompt, count}` 或 `{ok, found:false, reason}` | 写 `retrieval_log`；命中条目 `access_count+1`、进冷却 |
| `rosmontis_promote` | `note_id` | `section`（identity/rule/speech/trait/taste/free） | `id`（persona id）, `state, from, to` | 写 `persona` + `item_state_log`；episode 状态不变 |
| `rosmontis_revise` | `note_id, new_text` | `reason` | `id`（新条目 id）, `state` | 新订正 episode、旧条 `superseded_by`、`item_state_log` 留痕 |
| `rosmontis_skip` | `item_id` | `reason` | `state=muted` | episode：`state=muted`（不删）；fact：`status=invalidated, archived=1` |

逐项说明：

### 3.1 `rosmontis_remember`
- 只有 `what` → `state=inbox`（默认不可检索）；有 `noticed/means/next` 任一 → `state=note`（`rosmontis/core/gate.py:85-112`、`rosmontis/tools.py:305-361`）。
- 写入走 `pipeline.submit(origin='tool', actor='model')`，并 `project=True` 排队异步 enrich（`rosmontis/tools.py:316-345`、`rosmontis/core/pipeline.py:297-299`）。
- **不做文本去重**：工具层每次生成 `ep_tool_<uuid>` id（`rosmontis/tools.py:325-340`）；要幂等请用 `Handle.observe`。
- 失败原因示例：`what 不能为空`、gate 拒绝原因（`rosmontis/tools.py:306-308`）。

### 3.2 `rosmontis_recall`
- 调 `Memory.recall`，读取三/四路召回 + 预算打包（`rosmontis/tools.py:655-667`）。
- 注意它有写副作用：`retrieval_log` 落库（`rosmontis/core/recall.py:1026-1046`）、命中 episode/fact 的 `access_count/last_access/cooldown_until` 更新（`rosmontis/core/memory.py:173-208`）。
- `abstained=true` 时返回 `{ok:true, found:false, reason}`，不是错误。

### 3.3 `rosmontis_promote`
- 走 `notebook.promote(target='persona')` → `gate.decide('promote_persona')`（`rosmontis/tools.py:364-401`、`rosmontis/core/notebook.py:190-256`）。
- 四道闸：PR 硬限制（工具行为/情绪等）+ 频率 ≤2 次/会话 + 幅度 ≤200 字 + 来源必须是 `source='notebook'` 的笔记（`rosmontis/core/gate.py:272-316`）。
- 成功时写一条 `persona`，并把 promotion 记进 `item_state_log`；**原笔记的 state 不迁移**（gate 返回 `to == from`，`rosmontis/core/gate.py:316`）。
- 拒绝示例：`单次会话最多改 2 次人设（已用 2 次）。人格不该每天推翻`。

### 3.4 `rosmontis_revise`
- 只允许 `source='notebook'` 的条目（`rosmontis/core/notebook.py:150-151`）。
- 新条目继承旧 state，写 `correction_of` 元信息并 `superseded_by` 旧条；`item_state_log` 记 `revise`（`rosmontis/core/notebook.py:143-186`）。
- 与 `Memory.correct` 不同：后者会失效派生 fact/state（`rosmontis/core/memory.py:566-628`）。

### 3.5 `rosmontis_skip`
- episode：走 `pipeline.transition('skip')`，`state=muted`，同时写入长冷却；不删除（`rosmontis/tools.py:423-444`、`rosmontis/core/gate.py:254-255`）。
- fact：没有 episode 状态机，直接 `status='invalidated', t_invalid=now`；schema ≥ v3 的库有 `fact.archived` 列时同时置 `archived=1`，`t_valid_to` 保持 NULL（`rosmontis/tools.py:447-491`）。
- `item_state(fact)` 会把这种状态显示为 `muted`（`rosmontis/tools.py:256-274`）。

## 4. 四钩子

宿主自己拥有对话循环时，按四个事件调用 Handle（契约见 `docs/17-接入形态与接口方案.md` §3）：

| 钩子 | 调用 | 返回 | 纪律 |
|---|---|---|---|
| `session_start` | `h.wake(space, conversation)` | 人格/常驻/inbox/degraded | 幂等（种子 `INSERT OR IGNORE`） |
| `pre_prompt` | `h.context(space, conversation, query, budget)` | `system_prefix / memory_text / tokens / abstained` | **纯函数**：不写库；`memory_tokens <= budget` |
| `post_turn` | `h.observe(space, conversation, user_text, assistant_text)` | `{queued, written, ...}` | 立即返回；同文本窗口内幂等 |
| `session_end` | `h.settle(space, conversation)` | `{job_id}` | **立即返回**；后台结算/整理/做梦；可选 `h.wait(job_id)` |

最小宿主循环：

```python
w = h.wake(space, conversation)
ctx = h.context(space, conversation, query, budget=600)
messages = [
    {"role": "system", "content": ctx["system_prefix"] + "\n\n" + ctx["memory_text"]},
    {"role": "user", "content": query},
]
reply = your_model(messages)
h.observe(space, conversation, query, reply)
job = h.settle(space, conversation)
# 进程退出前：h.wait(job["job_id"], timeout=60) 然后 h.close()
```

若把循环交给 Harness，则用 `h.turn(...)` 替代 `context + 调模型 + observe`；工具循环与 `usage/finish_reason` 由 Harness 处理（`rosmontis/harness.py:521-576`）。

## 5. 错误码

### 5.1 公共 API（`RosmontisError`）

```python
try:
    h.context("u1", "c1", "q", budget=0)
except rosmontis.RosmontisError as e:
    print(e.code, e.message, e.retryable)
```

`RosmontisError.as_dict()` 返回 `{"ok": False, "data": None, "error": {"code","message","retryable"}}`（`rosmontis/api.py:59-73`）。

| code | 触发 | retryable |
|---|---|---|
| `invalid_argument` | `space` 空/为 `*`、`text` 空、`budget` 非正、config 非 dict | false |
| `llm_unavailable` | 模型没有返回内容 | true |
| `not_found` | `item_state` / `job` / `wait` 找不到 id | false |
| `timeout` | `wait` 超时 | true |
| `closed` | Handle 已关闭 | false |
| `open_failed` | 打开记忆空间失败 | true |

### 5.2 HTTP 服务

统一信封：`{"ok": bool, "data": ..., "error": null | {"code","message","retryable"}}`（`rosmontis/service/app.py:264-312`）。
注意：`fail()` 构造的错误不带 `retryable`（`app.py:311-312`），异常处理器与 `ok()` 的拒绝分支带（`app.py:264-290,299-308`）。

| HTTP | code | 场景 |
|---|---|---|
| 200 | — | 正常 |
| 400 | `empty_text` / `http_400` | 空文本；`session_id=*`（`app.py:55-60,317-318`） |
| 404 | `not_found` | 实体/条目不存在 |
| 409 | `rejected` | 业务层返回 `ok:false`（gate 拒绝等，`app.py:293-308`） |
| 422 | `invalid_argument` / `empty_text` / `missing_session` | 参数校验；空输入；缺 session（`app.py:264-272,459-466`） |
| 500 | `ingest_failed` / `batch_failed` / `recall_failed` / `turn_failed` / `transition_failed` / `onboard_failed` / `internal` | 未处理异常 |
| 503 | `llm_unavailable` | 模型没给出回复（`app.py:467-469,547-548`） |

## 6. HTTP 端点（服务形态）

启动：`python run.py serve --port 8770`（`run.py:71-87`）；OpenAPI：`http://127.0.0.1:8770/docs`。

| 方法 | 路径 | 关键参数 | 副作用 / 说明 |
|---|---|---|---|
| POST | `/v1/episodes` | body：`text, session_id, role, t_event` | 异步写入一条 episode，返回 `queued`（`app.py:315-323`） |
| POST | `/v1/episodes/batch` | body：EpisodeIn 数组 | 逐条同步写入（`app.py:325-330`） |
| POST | `/v1/recall` | `query, session_id, as_of, budget, exclude_ids` | 写 `retrieval_log`，命中条目计访问/冷却（`app.py:333-347`） |
| GET | `/v1/episodes` | `session_id, limit, offset, include_archived` | 列表（含 `tier/state/origin`，`app.py:349-356`） |
| GET | `/v1/entities` | `session_id, type, limit` | 实体列表 |
| GET | `/v1/entities/{id}` | `as_of` | 实体详情+事实+状态 |
| GET | `/v1/facts` | `session_id, status（默认 active）, limit` | 默认只返回 active；`status=*` 才含失效（`app.py:373-379`） |
| GET | `/v1/state/{scope}` | `session_id, as_of` | 当前/历史状态快照 |
| GET | `/v1/timeline/{entity_id}` | — | 事实变更史 |
| GET | `/v1/themes` | `session_id, limit` | 超边主题 |
| GET | `/v1/sessions` | — | 会话列表 |
| POST | `/v1/sessions/{sid}/close` | — | 峰终结算+沉淀+遗忘（`app.py:393-396`） |
| GET | `/v1/items/{item_id}/state` | — | 状态机查询（`app.py:403-409`） |
| POST | `/v1/items/{item_id}/transition` | body：`action,target,reason,actor` | 显式状态迁移（`app.py:411-424`） |
| GET | `/v1/notebook` | `session_id, tier, limit` | 笔记列表 |
| POST | `/v1/notebook` | body：`what,noticed,means,next_step,topic,session_id` | 写笔记（`app.py:431-437`） |
| POST | `/v1/notebook/promote` | body：`note_id,target,section,session_id` | 升格人设/页面 |
| POST | `/v1/notebook/revise` | body：`note_id,new_text,reason` | 订正 |
| POST | `/v1/notebook/revert` | query：`promotion_id` | 回滚升格 |
| POST | `/v1/notebook/review` | query：`session_id` | 整理速记 |
| POST | `/v1/turn` | body：`text,session_id` | 一轮对话（`app.py:456-472`） |
| POST | `/v1/session/wake` | query：`session_id` | 开会话 |
| POST | `/v1/session/sleep` | query：`session_id` | 关会话：峰终+整理+做梦 |
| POST | `/v1/context` | body：`query,session_id,budget,as_of` | **纯函数**：不写 `retrieval_log`、不写 `episode`，`memory_tokens` 硬压到 `budget` 内（与库 `Handle.context` 同一实现，走 `build_context(..., pure=True)`） |
| POST | `/v1/tick` | query：`session_id` | 维护节拍 |
| POST | `/v1/dream` | query：`session_id, force` | 做一次梦 |
| GET | `/v1/dreams` | `session_id, limit` | 梦记录 |
| GET | `/v1/dreams/candidates` | `session_id` | 升格候选 |
| GET | `/v1/inbox` | `session_id` | 锦囊状态 |
| POST | `/v1/forget/run` | query：`session_id` | 跑遗忘 |
| GET | `/v1/pages` | `session_id` | 活文档 |
| POST | `/v1/pages/rebuild` | query：`session_id` | 重建活文档 |
| GET | `/v1/resident` | `session_id` | 常驻块编译结果 |
| GET | `/v1/observations` | `session_id` | 后台归纳 |
| POST | `/v1/observations/derive` | query：`session_id` | 触发归纳 |
| GET | `/v1/persona` | `session_id` | 人格条目 |
| POST | `/v1/persona` | body：`text,session_id,section,author,protected,priority` | 写人格 |
| GET | `/v1/persona/compile` | `session_id` | 按预算编译人格 |
| POST | `/v1/promote` | query：`item_id, target_tier` | 升格（`app.py:634-636`） |
| POST | `/v1/correct` | body：`old_id,new_text,author` | 订正并失效派生 fact/state |
| POST | `/v1/remove` | query：`item_id,author,reason` | 写墓碑并归档 |
| POST | `/v1/onboard` | body：`session_id,name,desc,seed` | 首次见面由她开口（`app.py:522-555`） |
| POST | `/v1/wipe` | query：`session_id` | **不可逆**删除一个角色数据 |
| GET | `/v1/settings` | — | 读模型配置（密钥只回 `openai_key_set`） |
| POST | `/v1/settings` | body：模型配置字段 | 写 `.env` 并重建运行时（`app.py:501-520`） |
| GET | `/v1/config` | — | 当前配置（不含密钥） |
| GET | `/v1/stats` | — | 统计 + 降级状态 + config |
| GET | `/v1/audit` | `session_id` | 体检（问题清单+健康分） |
| GET | `/health` | — | 旧探针：进程活着（`app.py:674-677`） |
| GET | `/healthz` | — | liveness：事件循环可响应（`app.py:679-682`） |
| GET | `/readyz` | — | readiness：只读检查库、schema 版本、降级状态；失败 503（`app.py:684-717`） |
| GET | `/` / `/debug` | — | 静态页面（若 `web/` 存在） |

## 7. 状态模型（排障必看）

### 7.1 origin（谁发起）
`ingest`（被动对话）/ `tool`（模型工具）/ `system`（系统整流）/ `dream` / `summary`（`rosmontis/core/item.py:19-24`）。
旧库回填规则见 `rosmontis/core/db.py:234-256`。

### 7.2 state（7 态）
`inbox / note / resident / important / archived / muted / dream`（`rosmontis/core/item.py:33-41`）。

| state | 默认可召回？ | 说明 |
|---|---|---|
| `note` | ✅ | 被动 ingest 默认态 |
| `resident` | ✅ | 常驻层 |
| `important` | ✅ | 重要档（打分 ×1.18，`rosmontis/core/score.py`） |
| `inbox` | ❌（`inbox_retrievable=False` 时） | 只有 `what` 的主动速记 |
| `archived` / `muted` | ❌ | 恒不可见 |
| `dream` | ❌ | 由 `source='dream'` 排除 |

可见性判据：`item.recall_visible()` + `recall._state_visible`（`rosmontis/core/item.py:94-96`、`rosmontis/core/recall.py:92-103`）；开关 `ROSMONTIS_INBOX_RETRIEVABLE`（`rosmontis/config.py:180`）。

### 7.3 兼容投影
`state → (tier, archived, muted)`：见 `rosmontis/core/item.py:43-52`。因此看到 episode 的 `tier` 不要直接当状态用；`state` 才是真值。

### 7.4 升格动作
gate 支持的动作与允许起点见 `rosmontis/core/gate.py:30-77`；例如 `promote_resident` 只能从 `note/important` 起，`promote_persona/page` 还要求来源是 `notebook`（`gate.py:272-316`）。
每次迁移写一条 `item_state_log`（`rosmontis/schema.sql:220-234`）。

## 8. 配置项全表（`rosmontis/config.py`）

所有配置都可由环境变量覆盖（`.env` 只在进程启动时 `setdefault`，`rosmontis/config.py:10-24`）。
`rosmontis.open(db_path, config_dict)` 里的键名就是下表「字段」一列。

注意：`weights / budget / policy` 是三组 dataclass，`rosmontis.open` 不会把嵌套 dict 自动展开（`rosmontis/api.py:462-467`）。要改嵌套字段，直接构造 `Config` 再作为第一个参数传入：

```python
from rosmontis import Config, open
cfg = Config(db_path="my_memory.db")
cfg.weights.theta_abstain = 0.22
h = open(cfg)
```

### 8.1 顶层 `Config`

| 字段 | 环境变量 | 默认 | 说明 / 消费方 |
|---|---|---|---|
| `db_path` | `ROSMONTIS_DB` | `rosmontis.db` | 库文件 |
| `embed_provider` | `ROSMONTIS_EMBED` | `ollama` | `ollama` / `hash`（`rosmontis/core/embed.py`） |
| `embed_model` | `ROSMONTIS_EMBED_MODEL` | `bge-m3:latest` | Ollama embedding 模型 |
| `embed_dim` | `ROSMONTIS_EMBED_DIM` | `1024` | 向量维度；旧向量维度不符会被跳过 |
| `ollama_url` | `ROSMONTIS_OLLAMA` | `http://127.0.0.1:11434` | Ollama 地址 |
| `embed_timeout` | `ROSMONTIS_EMBED_TIMEOUT` | `8.0` | embedding 超时（秒） |
| `embed_batch` | `ROSMONTIS_EMBED_BATCH` | `8` | **当前无消费方** |
| `llm_hot` | `ROSMONTIS_LLM_HOT` | `ollama` | 抽取/热路径 provider：`ollama/openai/rule` |
| `llm_deep` | `ROSMONTIS_LLM_DEEP` | `ollama` | 深路径 provider |
| `llm_hot_model` | `ROSMONTIS_LLM_HOT_MODEL` | `minicpm5-2b:q8-64k` | 热路径模型 |
| `llm_deep_model` | `ROSMONTIS_LLM_DEEP_MODEL` | `minicpm5-2b:q8-64k` | 深路径模型 |
| `llm_timeout` | `ROSMONTIS_LLM_TIMEOUT` | `45.0` | LLM 超时（秒） |
| `llm_temperature` | `ROSMONTIS_LLM_TEMP` | `0.1` | 温度 |
| `openai_base` | `ROSMONTIS_OPENAI_BASE` | `""` | OpenAI 兼容 base URL |
| `openai_key` | `ROSMONTIS_OPENAI_KEY` | `""` | 密钥（`as_dict()` 不输出） |
| `openai_model` | `ROSMONTIS_OPENAI_MODEL` | `deepseek-chat` | OpenAI 兼容模型名 |
| `recall_k_dense` | `ROSMONTIS_K_DENSE` | `40` | 向量召回路 top-k（会被反射深度覆盖） |
| `recall_k_sparse` | `ROSMONTIS_K_SPARSE` | `40` | FTS 召回路 top-k |
| `recall_k_graph` | `ROSMONTIS_K_GRAPH` | `30` | 图召回路 top-k |
| `budget_tokens` | `ROSMONTIS_BUDGET` | `1400` | 默认检索预算（token） |
| `mmr_lambda` | `ROSMONTIS_MMR` | `0.72` | **当前无消费方**（打包用 `budget.redundancy_lambda`） |
| `graph_hops` | `ROSMONTIS_HOPS` | `2` | **当前无消费方**（扩散跳数用 `REFLEX_DEPTH[*].hops`） |
| `inbox_retrievable` | `ROSMONTIS_INBOX_RETRIEVABLE` | `false` | 是否允许 `inbox` 参与召回（C2） |
| `forget_archive_r` | `ROSMONTIS_ARCHIVE_R` | `0.08` | 归档保留率阈值 |
| `forget_fold_r` | `ROSMONTIS_FOLD_R` | `0.25` | 折叠区间阈值（`audit` 使用） |
| `forget_protect_salience` | `ROSMONTIS_PROTECT_sal` | `0.8` | 超过此 salience 不归档 |
| `fold_min_members` | `ROSMONTIS_FOLD_MIN` | `4` | 折叠所需超边成员数 |
| `decay_base_strength` | `ROSMONTIS_DECAY_BASE` | `86400*21` | 衰减基础强度（秒） |
| `use_llm_extract` | `ROSMONTIS_LLM_EXTRACT` | `true` | 是否用 LLM 抽取 |
| `use_llm_resolve` | `ROSMONTIS_LLM_RESOLVE` | `false` | **当前无消费方** |
| `auto_forget` | `ROSMONTIS_AUTO_FORGET` | `true` | 写后/关会话是否跑遗忘 |
| `log_level` | `ROSMONTIS_LOG` | `INFO` | 日志级别 |

### 8.2 `Config.weights`（`ScoringWeights`，`config.py:50-65`）

`rel=0.34`、`time=0.14`、`salience=0.20`、`graph=0.14`、`emotion=0.18`、`polarity_penalty=0.35`、`scope_penalty=0.20`、`access_boost=0.08`、`theta_high=0.46`、`theta_low=0.24`、`theta_abstain=0.18`。
其中 `theta_high/theta_low` 当前无消费方；`theta_abstain` 用于拒答（`rosmontis/core/recall.py:289`）。

### 8.3 `Config.policy`（`RecallPolicy`，`config.py:68-103`）

| 字段 | 默认 | 说明 / 消费方 |
|---|---|---|
| `persona_budget_chars` | `7600` | 人格字符预算（`rosmontis/core/persona.py:85`） |
| `resident_budget_chars` | `1500` | **当前无消费方**（常驻块实际按 token） |
| `resident_max_entries` | `20` | 仅 `Memory.resident_list` 使用（该函数当前无调用者） |
| `resident_max_tokens` | `420` | 常驻活文档 token 上限（`rosmontis/core/pages.py:86`） |
| `cooldown_normal_days` | `3.0` | 普通条目召回后冷却 |
| `cooldown_important_hours` | `24.0` | important 条目冷却 |
| `stale_tau_days` | `14.0` | 反近因时间常数 |
| `w_fresh` / `w_stale` | `0.55` / `1.00` | 反近因权重（`rosmontis/core/score.py`） |
| `s_min` / `s_min_important` | `0.20` / `0.14` | **当前无消费方** |
| `important_top_k` | `2` | **当前无消费方** |
| `inbox_max_items` | `30` | 速记积压上限（`rosmontis/core/notebook.py:468`） |
| `inbox_max_age_hours` | `24.0` | 最老速记年龄 |
| `review_interval_hours` | `12.0` | 两次整理最小间隔 |
| `dream_interval_hours` | `6.0` | **当前无消费方**（dream.py 硬编码 `MIN_INTERVAL_HOURS=6`） |
| `dream_min_items` | `5` | **当前无消费方**（dream.py 硬编码 `MIN_NEW_ITEMS=5`） |
| `reflex_default` | `context` | **当前无消费方**（深度由问题形态决定） |

### 8.4 `Config.budget`（`BudgetPlan`，`config.py:107-140`）

`state=0.15`、`fact=0.40`、`episode=0.35`、`theme=0.10`、`quality_first=true`、`floor_facts=6`、`floor_states=true`、`fidelity_l3=1.00`、`fidelity_l2=0.86`、`fidelity_l1=0.72`、`fidelity_l0=0.46`、`pool_multiplier=2.4`、`compress_at=1.0`、`prune_at=1.0`、`redundancy_lambda=0.28`。
其中 `quality_first/compress_at/prune_at` 当前无消费方；`floor_facts`、`pool_multiplier`、`redundancy_lambda`、`fidelity_*` 用于打包（`rosmontis/core/recall.py:882-996`）。

### 8.5 不在 `Config` 里的开关

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `ROSMONTIS_FIELD` | 开 | `0/false/off/no` 关闭语义场（`rosmontis/core/field.py:129-132`） |
| `PYTHONIOENCODING` | — | Windows 下跑脚本/测试建议设为 `utf-8` |

> 调参前先确认字段有消费方；上表标「当前无消费方」的字段改动不会影响运行行为。
> 配置修改后需要重建 `Memory`/`Harness`（HTTP 的 `/v1/settings` 会自动重建，`app.py:95-112`）。





