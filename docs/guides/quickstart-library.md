# Alaya 快速开始（库形态）

目标：不联网、不装模型，从零到「写下第一条记忆 → 下一轮能取回 → 正常打烊」。
全流程只需要 Python 3.10+ 和带 FTS5 的 SQLite；服务形态（HTTP / 可视化）另见 [operations.md](operations.md)。

> 对应代码：`alaya/api.py`（公共 API v1）、`alaya/harness.py`（Harness）、`examples/hooks_demo.py`（四钩子示例）。
> 本文示例已按当前 API 实跑验证；验证方法见文末「自检」。

## 0. 拿到代码并接入 import 路径

```bash
git clone <repo-url> alaya-memory
cd alaya-memory
```

当前工作树已带 `pyproject.toml`（`name="alaya-memory"`，`requires-python=">=3.10"`，核心依赖为空）。推荐本地 path 安装：

```powershell
# Windows PowerShell（在仓库根目录）
pip install -e .
python -c "import alaya; print(alaya.__version__, alaya.api_version)"
```

```bash
# macOS / Linux
pip install -e .
python -c "import alaya; print(alaya.__version__, alaya.api_version)"
```

按需装服务/评测依赖：`pip install -e ".[serve]"`（fastapi/uvicorn/pydantic/jieba）、`pip install -e ".[dev]"`。
不想安装时，用 path 注入也可以：

```powershell
$env:PYTHONPATH = (Resolve-Path .).Path
python -c "import alaya; print(alaya.__version__, alaya.api_version)"
```

```bash
export PYTHONPATH="$PWD"
python -c "import alaya; print(alaya.__version__, alaya.api_version)"
```

预期输出 `1.0.0 1`（包版本 / 公共 API 版本，见 `alaya/__init__.py:10-11`）。

也可以在脚本开头直接加路径：

```python
import sys
sys.path.insert(0, r"D:\MyHub\06-Projects\alaya-memory")
```

依赖：核心库只用标准库；`numpy`、`jieba` 是可选的加速/分词增强（`alaya/core/vec.py:12-18`、`alaya/core/text.py:10-15`）；`fastapi`、`uvicorn` 只在起 HTTP 服务时需要。

## 1. 三行接入

```python
import alaya

h = alaya.open("my_memory.db", {"embed_provider": "hash", "llm_hot": "rule", "llm_deep": "rule"})
h.wake("u1", "c1")
```

- `alaya.open(db_path, config)` 创建/打开一个记忆空间，返回 `Handle`（`alaya/api.py:437`）。
- `config` 是 `Config` 字段组成的 dict；`embed_provider="hash"` + `llm_hot="rule"` 表示完全离线、确定性运行。
- 默认值是本地 Ollama（`alaya/config.py:151-163`）；没有 Ollama 时第一次检索会降级成哈希向量，见 [operations.md](operations.md) 的降级矩阵。
- v1 里 `space` 就是底层 `session_id`：同一 `space` 的不同 `conversation` 共享记忆；`conversation` 原样带回，并参与 `observe` 幂等键（`alaya/api.py:26-32`）。

## 2. 写下第一条记忆

### 2.1 宿主自己管对话循环（四钩子形态）

```python
import os
import tempfile

import alaya

work = tempfile.mkdtemp(prefix="alaya_quickstart_")
db = os.path.join(work, "memory.db")

# 离线配置：hash 向量 + 规则抽取
h = alaya.open(db, {
    "embed_provider": "hash",
    "llm_hot": "rule",
    "llm_deep": "rule",
    "log_level": "ERROR",
})

# session_start
w = h.wake("u1", "c1")
print("wake:", w["session_id"], "persona_items=", w["persona_items"], "degraded=", w["degraded"])

# post_turn：把一来一回写进记忆（不生成回答）
h.observe("u1", "c1", "我叫小林，在杭州做设计。", "记住了。")
h.observe("u1", "c1", "我最近在准备搬家。", "慢慢来。")

# pre_prompt：取本轮可注入的上下文（纯函数，不写库）
ctx = h.context("u1", "c1", "搬家", budget=400)
print("memory_tokens:", ctx["memory_tokens"], "abstained:", ctx["abstained"])
print(ctx["memory_text"])

# session_end：立即返回 job_id，后台打烊
job = h.settle("u1", "c1")
done = h.wait(job["job_id"], timeout=60)
print("settle:", done["state"], done["result"].get("steps"))

h.close()

# 清理临时库
import shutil
shutil.rmtree(work, ignore_errors=True)
```

预期输出（hash + rule 模式下检索是词面级，`搬家` 能命中刚才那句）：

```text
memory_tokens: 7 abstained: False
<memory>
【想起来的片段】
- 我最近在准备搬家。
</memory>
settle: done ['峰终结算']
```

- `observe` 每个非空文本各写一条 episode，返回 `queued=True`；同一 `(space, conversation, user_text, assistant_text)` 在 `observe_window`（默认 300 秒）内重复调用返回 `idempotent=True`（`alaya/api.py:210-250`）。
- `context` 是纯函数：不写 `episode` / `retrieval_log`，且 `memory_tokens <= budget`（`alaya/harness.py:104-126,159`）。
- `settle` 不阻塞：返回 `job_id`，后台线程跑 `close_session` + `notebook_review` + `dream`（`alaya/api.py:252-311`、`alaya/harness.py:698-726`）。
- `wake` 返回 `persona_text / persona_chars / persona_budget / persona_items / resident_text / resident_tokens / resident_pages / inbox / degraded`（`alaya/harness.py:482-506`）。

### 2.2 把循环交给 Harness（模型自己调工具）

`alaya_remember` 等五个工具只有在模型返回 `tool_calls` 时才会触发。下面用一个假模型跑通工具链：

```python
import os
import tempfile

import alaya


def fake_llm(messages, tools=None):
    # 第一次：调工具写笔记；第二次（工具结果已回灌）：正常收尾
    if not [m for m in messages if m.get("role") == "tool"]:
        return {"content": "", "tool_calls": [{
            "name": "alaya_remember",
            "arguments": {
                "what": "他最近在准备搬家",
                "noticed": "提到房东时语气很轻",
                "means": "可能不想搬，只是没得选",
                "next": "先问需不需要帮忙",
                "topic": "生活",
            },
        }]}
    return {"content": "我记下了。", "tool_calls": []}


work = tempfile.mkdtemp(prefix="alaya_turn_")
db = os.path.join(work, "memory.db")
h = alaya.open(db, {
    "embed_provider": "hash", "llm_hot": "rule", "llm_deep": "rule",
    "llm_callable": fake_llm, "memory_budget": 400, "log_level": "ERROR",
})

h.wake("u1", "c1")
r = h.turn("u1", "c1", "我最近在准备搬家")
print(r["reply"])
print([(t["tool"], t["ok"], t["state"]) for t in r["tools_called"]])

ctx = h.context("u1", "c1", "搬家")
print(ctx["memory_tokens"], ctx["memory_text"])

h.settle("u1", "c1")
h.close()
import shutil
shutil.rmtree(work, ignore_errors=True)
```

预期：`[('alaya_remember', True, 'note')]`，随后 `context` 同时命中工具笔记和用户原话。
`turn` 的返回值还带 `usage`、`finish_reason`、`rounds`、`memory_used`（`alaya/harness.py:560-576`）。

## 3. 换真实模型

公共 API 只认一个回调：`llm_callable(messages, tools) -> dict`。返回值里可以带 `content`、`tool_calls`、`usage`、`finish_reason`；`tool_calls` 既接受 OpenAI 原生嵌套 `function.arguments`（JSON 串），也接受扁平的 `{"name","arguments": {...}}`（`alaya/harness.py:12-21,212-265`）。

```python
import alaya
from alaya import Config
from alaya.core.llm import LLM
from alaya.harness import make_llm_callable

cfg = Config(
    embed_provider="ollama",                 # 需要本机 Ollama + bge-m3
    llm_hot="openai",                       # OpenAI 兼容通道
    openai_base="https://api.deepseek.com/v1",
    openai_key="sk-...",
    openai_model="deepseek-chat",
)
llm = LLM(
    provider=cfg.llm_hot,
    model=cfg.llm_hot_model,
    ollama_url=cfg.ollama_url,
    timeout=cfg.llm_timeout,
    temperature=cfg.llm_temperature,
    openai_base=cfg.openai_base,
    openai_key=cfg.openai_key,
    openai_model=cfg.openai_model,
)
h = alaya.open("my_memory.db", {
    "embed_provider": "ollama",
    "llm_hot": "openai",
    "llm_deep": "openai",
    "openai_base": cfg.openai_base,
    "openai_key": cfg.openai_key,
    "openai_model": cfg.openai_model,
    "llm_callable": make_llm_callable(llm),
})
```

- `make_llm_callable(llm)` 会优先走原生工具协议（OpenAI 兼容 / Ollama）；上游没有工具能力、没密钥或超时时**安静降级**为纯文本通道（`tool_calls=[]`），不抛异常（`alaya/harness.py:398-448`）。
- 因此：模型不支持 function calling 时，五个工具不会触发，但对话与被动记忆仍正常。
- 离线但没有 bge-m3 时，`hash` 向量只适合跑通链路与测试，不适合评估语义召回。
- 不传 `llm_callable` 时，`turn` 只会返回「（没有接模型）」，但仍会记录用户那句话（`alaya/harness.py:619-620`、`543`）。

## 4. 关闭与清理

- `h.close()` 可重复调用；会等后台 settle 与 enrich 收尾（`alaya/api.py:357-389`）。
- 推荐用上下文管理器：

```python
with alaya.open("my_memory.db", {"embed_provider": "hash", "llm_hot": "rule"}) as h:
    h.wake("u1", "c1")
    h.observe("u1", "c1", "你好", "你好呀。")
    h.settle("u1", "c1")
```

- SQLite 是 WAL 模式，磁盘上可能出现 `my_memory.db-wal` / `my_memory.db-shm`。**不要只删主库文件**；停掉进程后三个一起清，或按 [operations.md](operations.md) 的备份/恢复流程处理。
- 生产库第一次被 `alaya.open()` 打开时会自动执行 schema 迁移（v2 → v3，`alaya/core/db.py:77-122`）。已有 `alaya.db` 时先备份，见 [operations.md](operations.md)。

## 5. 常见错误

| 异常 / 返回 | 含义 | 处理 |
|---|---|---|
| `AlayaError.code == "invalid_argument"` | `space` 为空/为 `*`、`text` 为空、`budget <= 0` | 检查入参（`alaya/api.py:157-165,199-200,411-418`） |
| `llm_unavailable`（`retryable=True`） | 模型没有返回内容 | 检查模型服务/密钥；重试或换模型（`alaya/api.py:206-207`） |
| `not_found` | `item_state` / `job` / `wait` 查不到 id | 确认 id 来源；`job` 只在当前 Handle 生命期内有效 |
| `timeout`（`retryable=True`） | `wait` 超时 | 增加 timeout 或改用 `job()` 轮询 |
| `closed` | Handle 已关闭 | 重新 `alaya.open()` |
| `context` 返回 `abstained=True`、`memory_text=""` | 没有相关记忆，或 hash 模式词面不重合 | 换更贴近原文的查询；接 bge-m3 后再评估语义召回 |

工具被闸门拒绝时不会抛异常，而是返回 `{"ok": False, "reason": "..."}`，模型可读（`alaya/tools.py:157-174`）。

## 6. 自检

```bash
# 1) 四钩子示例（临时库，不联网）
python examples/hooks_demo.py

# 2) 全量测试（6 个文件共 93 项；全部用临时库，不写 alaya.db）
python -m unittest discover -s tests -p "test_*.py"
```

预期：示例最后打印「四个钩子全部跑通 ✓」；测试输出 `Ran 93 tests ... OK`。
如果 PowerShell 把日志里的迁移告警当成 stderr 而显示 `[exit code: 1]`，以 `Ran 93 tests ... OK` 为准（`tests/test_items.py` 里的注入失败用例会主动打印一次 `migration ... rolled back`）。

## 7. 下一步

- 接口细节（每个参数/返回/副作用）：[api-reference.md](api-reference.md)
- 备份恢复、v2→v3 迁移、降级、日志、排障：[operations.md](operations.md)
- Harness 设计：`docs/12-harness设计.md`；主被动整合口径：`docs/14-主被动记忆原生整合方案.md`
