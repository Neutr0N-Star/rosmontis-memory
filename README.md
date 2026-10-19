# 藏识 Alaya

长期陪伴 AI 的**记忆后端**。给你自己的前端 / agent 提供「记得住、会遗忘、会成长」的记忆能力。

> **边界**：本仓库只有记忆引擎与接口，**不含前端**。任何 UI（Web / 桌面 / 移动）都用 HTTP 接口接进来。
> 接口方案见 [docs/17-接入形态与接口方案.md](docs/17-接入形态与接口方案.md)。

```
pip install -e .                 # 只装核心（Python 标准库即可跑）
pip install -e ".[serve,dev]"    # 需要 HTTP 服务 / 评测时再装依赖
alaya serve --port 8770          # 起本地后端（默认只绑 127.0.0.1）
```

---

## 它解决什么

普通记忆系统只回答「哪段文字最像」。长期陪伴要回答四个：

| 问题 | 普通方案 | Alaya |
|---|---|---|
| 他住哪 | 找最像的片段，可能翻出三年前那条 | 状态快照，只留当前 |
| 他以前住哪 | 做不到 | 指定时刻的状态快照（as-of） |
| 他喜欢猫吗（中间变过） | 新旧两条一起返回、自相矛盾 | 旧条失效但留痕，只答当前 |
| 上次说过什么 | top-k 硬截断 | 预算内连续填充，分数可分辨 |

三个设计约束：

1. **记忆是分层的**：Episode（原文，不可变）→ Entity / Theme → Fact（双时间）→ State（当前快照）。任何上层结论都能回到原文。
2. **注入是有预算的**：人格块、常驻块、检索块预算互相独立，绝不超预算，宁可少注入也不截半句。
3. **主被动一体**：用户说的话（被动抽取）与模型主动写的笔记（主动记忆）走**同一条写入管线、同一个状态机**，只用一个 `origin` 字段区分来源。

---

## 三种用法

### ① 当库用（Python）

```python
import alaya

h = alaya.open("alaya.db")          # 打开/新建记忆库（首次会建表）
h.wake("u1", "chat")                # 会话开始：载入人格与常驻块

ctx = h.context("u1", "chat", "我最近搬到哪了？", budget=1200)   # 纯函数：不写库、不超预算
system = ctx["system_prefix"] + "\n\n" + ctx["memory_text"]     # 直接拼进你模型的 system prompt（memory_text 可能为空）
print(ctx["tokens"], ctx["abstained"])                          # 注入 token / 是否拒答

r = h.turn("u1", "chat", "我搬到上海了。", history=[])           # 一轮：记录 → 注入 → 生成 → 回写
print(r["reply"])

h.settle("u1", "chat")              # 打烊：峰终结算 + 整理 + 可能做梦（立即返回 job_id）
h.close()
```

完整可运行示例：`python examples/hooks_demo.py`（不需要网络、不需要模型）。
接入文档：[docs/guides/quickstart-library.md](docs/guides/quickstart-library.md)

### ② 当本地服务用（任意语言 / 任意前端）

```bash
alaya serve --port 8770
# 打开 http://127.0.0.1:8770/docs 看 OpenAPI 文档
```

```bash
curl -X POST http://127.0.0.1:8770/v1/turn \
  -H "Content-Type: application/json" \
  -d '{"session_id":"u1","text":"我今天有点累"}'
```

需要逐字输出时用 SSE：`POST /v1/turn/stream`（事件序列 `meta → (tool)* → delta* → done`，`fetch` 读流即可）：

```js
const r = await fetch("http://127.0.0.1:8770/v1/turn/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ session_id: "u1", text: "我今天有点累" }),
});
const reader = r.body.getReader();      // 逐块解析 "event: delta\ndata: {...}\n\n"
```

Docker 一条命令：

```bash
docker compose up --build      # 只绑 127.0.0.1:8770，数据在 ./data
```

前端只需要三类调用：**说话**（`/v1/turn` 或 `/v1/turn/stream` 流式）、**看**（`/v1/state`、`/v1/episodes`）、**打烊**（`/v1/session/sleep`）。
禁止直连数据库——换前端不该改后端。

### ③ 当后端嵌进别人的 agent

用 `/v1/context`（纯函数，不写库）取注入块，用 `/v1/observe` 回写对话；生命周期自行掌控。
接口清单与版本承诺：[docs/17-接入形态与接口方案.md](docs/17-接入形态与接口方案.md)

---

## 配置

全部可用环境变量覆盖（前缀 `ALAYA_`），完整表见 [docs/guides/api-reference.md](docs/guides/api-reference.md)。

| 变量 | 默认 | 说明 |
|---|---|---|
| `ALAYA_DB` | `alaya.db` | 记忆库文件路径 |
| `ALAYA_EMBED` | `ollama` | `ollama` / `hash`（离线确定性降级） |
| `ALAYA_LLM_HOT` | `ollama` | 抽取与对话用的模型通道：`ollama` / `openai` / `rule` |
| `ALAYA_OPENAI_BASE` / `_KEY` / `_MODEL` | — | 走 OpenAI 兼容接口时填 |
| `ALAYA_DENSE_WINDOW` | `20000` | 向量检索近因窗口，调小更快但会牺牲久远记忆 |
| `ALAYA_INBOX_RETRIEVABLE` | `false` | 速记层是否允许被检索（默认否） |

离线可跑：`ALAYA_EMBED=hash ALAYA_LLM_HOT=rule`（质量降级，仅演示）。

### 语义向量（模型作为可下载资产）

模型**不打包进 wheel**，按需下载到本地（默认 `~/.alaya/models/`，可用 `ALAYA_MODEL_DIR` 指到已有目录）：

```bash
alaya models list                          # 可用模型与安装状态
alaya models install bge-small-zh-v1.5     # 断点续传 + 重试 + SHA-256 校验；--mirror 可换镜像
alaya models where bge-small-zh-v1.5       # 打印 onnx / tokenizer / 目录
alaya models verify bge-small-zh-v1.5      # 校验文件完整性
```

> 本地 ONNX 推理通道（`ALAYA_EMBED=local`，需 `pip install -e ".[embed]"`）与 OpenAI 兼容 embedding 通道（`ALAYA_EMBED=openai`）随 v1.1 落地；当前 `ollama` 与 `hash` 两条通道可用。
> `/readyz` 会报告 `vector_mode`（`semantic` / `degraded` / `none`）与待回算条目数。

---

## 运行与运维

```bash
alaya serve --port 8770      # 本地服务（默认只绑 127.0.0.1）
alaya daemon                 # 后台守护：自动整理/做梦/结算/遗忘（--once 只跑一拍）
alaya backup                 # 备份（先 WAL checkpoint 再拷，运行中也安全）
python run.py demo --reset   # 三个月对话演示（不需要模型）
python -m unittest discover -s tests -p "test_*.py"   # 全部测试（124 项）
python bench/run_eval.py     # 质量基准（200 段合成对话 / 135 题）
python bench/pareto.py       # 配置网格 → 帕累托前沿 → bench/report.md
```

- 健康检查：`/healthz`（存活）、`/readyz`（库可读、schema 版本、降级状态）。
- 任务与节拍：`GET /v1/jobs` 看积压；`POST /v1/settle` 立即返回 job_id；空闲超过 `ALAYA_IDLE_TIMEOUT`（默认 1800 秒）时 daemon 自动结算。
- 幂等：写接口（`/v1/episodes`、`/v1/turn`、`/v1/observe`、`/v1/settle`）支持 `Idempotency-Key`，24h 内同键回放且只写一条。
- 备份：库里是 WAL 模式，**不要直接复制文件**；用 `alaya backup`（先 checkpoint 再复制并输出 sha256），恢复见 [docs/guides/operations.md](docs/guides/operations.md)。
- 升级旧库：首次打开会自动迁移格式；数据可弃时直接换个新库文件即可。

---

## 接口与版本

- v1 冻结面：`/v1/*` 端点、五个工具（`alaya_remember/recall/promote/revise/skip`）、错误信封 `{ok,data,error{code,retryable,message}}`。
- 破坏性改动升 `/v2`；库内部签名（`pipeline.*`、`field.*`）不承诺稳定。
- OpenAPI 契约：`/docs`、`/openapi.json`，仓库内另存快照 `openapi.json`。

---

## 文档

- 接入：[quickstart（库）](docs/guides/quickstart-library.md) ｜ [API 参考](docs/guides/api-reference.md) ｜ [运维手册](docs/guides/operations.md) ｜ [接入形态方案](docs/17-接入形态与接口方案.md)
- 设计：[整合方案](docs/14-主被动记忆原生整合方案.md) ｜ [模块契约](docs/15-交付蓝图与模块契约.md) ｜ [优化清单](docs/16-优化清单.md)
- 历史（评审/迭代记录，非当前契约）：`docs/00-13`

---

## 许可

MIT，见 [LICENSE](LICENSE)。
