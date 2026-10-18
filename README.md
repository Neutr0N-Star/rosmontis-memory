# 藏识 Alaya

长期陪伴 AI 的记忆器官。它换掉的不是向量库，是记忆的**存在形式**——从"一堆等距的文本片段"换成"一个有状态、有时间、会遗忘、会改主意的世界模型"。

```
python run.py demo        # 先看它怎么工作（30 秒）
python run.py serve       # 起服务 + 可视化界面 http://127.0.0.1:8770
```

---

## 它解决什么

普通记忆系统只回答一个问题：**哪段文字最像**。

长期陪伴要回答的是四个：

| 问题 | 普通方案 | 藏识 |
|---|---|---|
| 他住哪 | 找最像的片段，可能翻出三年前那条 | 状态快照，只留当前 |
| 他以前住哪 | 做不到 | 时间旅行，指定时刻快照 |
| 他喜欢猫吗（中间变过） | 新旧两条一起返回，自相矛盾 | 旧条失效但留痕，只答当前 |
| 上次说过什么 | top-k 硬截断，要么全中要么全不中 | 预算内连续填充，分数可分辨 |

## 三个关键设计

**一、记忆分五层，上层是下层的投影**

```
State   当前世界快照          ← 读的时候永远先看这层
  ↑
Fact    结构化断言 + 双时间    ← 从 Episode 抽取
  ↑
Entity  人物/地点/物件
  ↑
Theme   主题超边（n 元关系）
  ↑
Episode 原始事件（不可变）      ← 一切可回溯到这里
```

上面任何一层出错，都能顺着链回到原文核对。

**二、注入分三档，互相不抢预算**

| 档 | 什么时候 | 成本 |
|---|---|---|
| 常驻活文档 | 每轮直接读，不检索不调 LLM | ~70 token |
| 状态快照 | 每轮注入，永不裁剪 | ~40 token |
| 检索浮现 | 按问题召回，有冷却 | 按需 |

三者预算独立。人格预算（7600 字符）、常驻预算（1500 字符）、检索预算（token 数）互相挪用不了——因为"她忘了自己是谁"比"这次没想起某句话"严重得多。

**三、记忆是场，不是条目**

借自两个 2026 年的开源方案，都落在离散图上实现，不引入 PDE 求解器：

- **ACT-R 激活**（Ori Mnemos）：`A = ln(Σ t_j^-0.5) + B`。同样被访问 5 次，昨天访问的和一年前访问的，激活完全不同。
- **图扩散**（FieldMem 的 `D∇²φ` 离散版）：命中节点沿超边把激活传给邻居。这是"想起 A 就顺带想起 B"的工程版——向量检索只在查询那一刻建立相似，扩散走的是**已经存在的结构**。
- **三空间衰减率**（Ori）：人格 0.1x、知识 1.0x、操作状态 3.0x。人格几乎不衰减，对话情节烧得快。
- **四级反射深度**（NeuralMemory）：`instant / context / habit / deep`，按问题形态伸缩检索规模。问"我住哪"只花 360 token，"总结所有经历"给到 1280。

**四、主动记忆：她自己写**

被动记忆回答「她怎么想起来」，主动记忆回答「她自己决定记什么」。

笔记本条目有四段：`what / noticed / means / next`。**只有 what 的条目不许进笔记层**——这是防流水账的硬闸。

分层按「将来什么时候用得上」分：速记（用完清）/ 笔记（会衰减）/ 沉淀（只重写）/ 人设（0.1x 衰减）。

改人设受四道闸：频率（单次会话 ≤2 次）、幅度（≤200 字）、来源（必须来自笔记）、留痕（可回滚）。

**五、四段时间尺度**

收集与整理必须分开，否则两件事都做不好：

| 尺度 | 对应 | 做什么 |
|---|---|---|
| 秒级 | 锦囊 inbox | 只收集，零加工 |
| 小时级 | 到点整理 | 合并、提炼成笔记 |
| 天级 | 做梦 | 重放、关联、抽象、**脱敏** |
| 多日级 | 成长 | 反复出现的判断改写成人格 |

**做梦是这个闭环里最关键的一环**：情绪强度降、语义不变，这是"接受"的工程实现，不是把事压成一句话。

**六、token 是约束，不是目标**

优化顺序是硬的：

1. **按质量选**：先按分数决定要哪些，不比性价比
2. **压表达**：同一条记忆有 L0 指针 / L1 三元组 / L2 一句话 / L3 原文四级，超预算就从低分的开始降级
3. **才砍内容**：降级后仍超预算，从低分删，但保性能地板（状态全保 + Top6 事实）

绝不会出现"为了省 token 换一条更便宜但更差的记忆"。

## 像人一样的四个机制

| 人的特征 | 实现 |
|---|---|
| 记得要点不记得字 | Fact 层存断言，原文只作溯源 |
| 情绪事件记得特别牢 | 情感强度独立打分 + 峰终效应加成 |
| 久远的事偶尔会想起 | **反近因**：久没被想起的权重更高（不是越新越重要） |
| 会重构、会改主意 | 折叠成摘要；订正保留原文，旧说法失效但可查 |

## 接入：四行代码

Harness 是包在模型外面的一圈流程，把记忆接进对话循环。

```python
from alaya import Config
from alaya.harness import Harness

h = Harness(Config(), my_llm)        # my_llm(messages, tools) -> {content, tool_calls}
h.wake("u1")

while True:
    r = h.turn(input(), "u1")
    print(r["reply"])

h.sleep("u1")
```

三个生命周期用最朴素的说法：

| 方法 | 做什么 |
|---|---|
| `wake` | 播种、载入人格 |
| `turn` | 一轮：记录 → 取上下文 → 调模型 → 执行工具 |
| `sleep` | 峰终结算、整理锦囊、够条件就做梦 |

不绑定任何 SDK——OpenAI、Ollama、DeepSeek 或一个假模型都能塞进来。
完整演示（不需要模型）：`python demo/harness_demo.py`

### 给模型的五个工具

| 工具 | 什么时候用 |
|---|---|
| `alaya_remember` | 这件事值得留给以后的自己 |
| `alaya_recall` | 主动查，不是被动注入 |
| `alaya_promote` | 一条判断反复想到，该变成「我就是这样」 |
| `alaya_revise` | 以前记错了 |
| `alaya_skip` | 这件事不想再提了 |

工具被拒时会返回明确原因（比如「单次会话最多改 2 次人设」），模型能据此调整。

## 界面

`python run.py serve` 之后打开 http://127.0.0.1:8770/ —— 这是正经的对话应用：

```
┌──────────────────────────────┐
│ 藏识        3,200 token  [记忆] [打烊] │
├──────────────────────────────┤
│  你：我今天有点累              │
│  她：那就别撑着。              │
│      50 token · 3 条记忆 · 3.3s│
├──────────────────────────────┤
│ [说点什么…]              [说] │
└──────────────────────────────┘
```

- **记忆**抽屉：当前状态 / 实体 / 事实 / 笔记 / 记忆条目，只读
- **打烊**：峰终结算 + 沉淀 + 可能做梦
- 每轮底下显示这一轮注入了多少 token、几条记忆、耗时

调试用的检查台在 `/debug`。

## 快速开始

```bash
# 环境：Python 3.10+，SQLite 需带 FTS5（标准发行版自带）
pip install fastapi uvicorn jieba     # 只服务层需要

# 1 看演示（不需要模型）
python run.py demo --reset

# 2 起服务 + 界面
python run.py serve --port 8770

# 3 命令行
python run.py recall "我住哪"
python run.py audit
python run.py ingest "我搬到上海了"
```

默认走本地 Ollama（`bge-m3` 做向量、`minicpm5-2b` 做抽取）。没有 Ollama 也能跑，会自动降级并在 `/v1/stats` 里标出来。

```bash
# 换云端
set ALAYA_LLM_HOT=openai
set ALAYA_OPENAI_BASE=https://api.deepseek.com/v1
set ALAYA_OPENAI_KEY=sk-...
set ALAYA_OPENAI_MODEL=deepseek-chat

# 完全离线
set ALAYA_EMBED=hash
set ALAYA_LLM_HOT=rule
```

## 写入是异步的

本地 2B 模型抽一条要 2-10 秒，同步等待会把对话卡死。所以：

```
POST /v1/episodes  →  原文立刻落库（~50ms），返回
                    →  抽取在后台线程跑，补上事实与状态
```

原文先存保证不丢。抽取失败不影响对话，下次还能补。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/episodes` | 写入一句话（**异步**，返回 `queued`） |
| GET | `/v1/episodes` | 记忆清单（分页 / session 过滤 / 含归档） |
| POST | `/v1/recall` | 检索，返回证据包 + 打分明细 + 实际注入的上下文 |
| GET | `/v1/state/{scope}` | 当前状态快照，带 `as_of` 可时间旅行 |
| GET | `/v1/timeline/{entity_id}` | 事实变更史 |
| GET | `/v1/entities` | 实体列表 |
| GET | `/v1/facts` | 事实列表（含已失效） |
| GET | `/v1/pages` | 常驻活文档 |
| POST | `/v1/pages/rebuild` | 重建活文档 |
| GET | `/v1/persona` | 人格条目 |
| GET | `/v1/persona/compile` | 按预算编译人格块 |
| POST | `/v1/promote` | 升格（能不能进常驻，不由分数决定） |
| POST | `/v1/correct` | 订正（旧条保留，新条指回旧条） |
| POST | `/v1/remove` | 删除（写墓碑，不可复活） |
| POST | `/v1/turn` | **一轮对话**：记录 → 组上下文 → 调模型 → 写回 |
| POST | `/v1/session/wake` | 开会话：播种、载入人格 |
| POST | `/v1/session/sleep` | 关会话：峰终 + 沉淀 + 做梦 |
| POST | `/v1/context` | 一次拿全：人格 + 常驻 + 检索（热重载入口） |
| GET | `/v1/notebook` | 笔记列表（可按 tier 过滤） |
| POST | `/v1/notebook` | 写一条笔记（what/noticed/means/next） |
| POST | `/v1/notebook/promote` | 升格到人设或沉淀 |
| POST | `/v1/notebook/revise` | 订正（原文保留） |
| POST | `/v1/notebook/revert` | 回滚一次升格 |
| POST | `/v1/notebook/review` | 整理速记层 |
| POST | `/v1/tick` | **维护节拍**：该整理就整理，该做梦就做梦 |
| POST | `/v1/dream` | 做一次梦（重放→关联→抽象→脱敏→成梦） |
| GET | `/v1/dreams` | 梦的记录 |
| GET | `/v1/dreams/candidates` | 升格候选（多次梦里反复出现的判断） |
| GET | `/v1/inbox` | 锦囊状态：积压多少、该不该整理 |
| POST | `/v1/sessions/{sid}/close` | **结束会话**：峰终结算 + 沉淀 + 遗忘 |
| POST | `/v1/forget/run` | 跑遗忘周期 |
| GET | `/v1/audit` | 体检 |
| GET | `/v1/stats` | 统计 + 降级状态 |
| POST | `/v1/observations/derive` | 后台归纳 |
| GET | `/v1/persona/compile` | 按预算编译人格块 |
| GET | `/docs` | OpenAPI 文档 |

统一响应：`{"ok": bool, "data": {...}, "error": null | {"code","message"}}`

## 目录

```
alaya/
├─ config.py          配置（全可用环境变量覆盖）
├─ schema.sql         数据模型
├─ core/
│  ├─ memory.py       门面：库形态的对外 API
│  ├─ write.py        写入管线（状态转移）
│  ├─ extract.py      抽取（LLM 为主，规则兜底）
│  ├─ recall.py       四阶段检索 + 预算打包
│  ├─ score.py        打分、情感、token 估算
│  ├─ pages.py        常驻活文档 + 后台归纳
│  ├─ persona.py      人格预算编译
│  ├─ promotion.py    升格闸门
│  ├─ forget.py       遗忘（降权/折叠/归档）
│  ├─ audit.py        体检
│  └─ vec.py text.py timeparse.py ids.py db.py embed.py llm.py
├─ service/app.py     FastAPI 薄适配层
├─ cli.py             命令行
└─ demo_runner.py     三个月对话演示
web/index.html        可视化界面（单文件）
tests/test_all.py     15 项机制测试
docs/                 需求、蓝图、契约、评审、测试报告
```

## 测试

```bash
python tests/test_all.py
```

15 项，覆盖幂等、状态转移、极性、时间旅行、预算、拒答、冷却、墓碑、人格预算。

## 已知限制

- 向量检索是暴力扫描（numpy）。十万条以内毫秒级，再大需要外挂 ANN。
- 关系抽取依赖 LLM。用 `rule` 模式时只认三个最基础的模式，覆盖率很低——这是故意的，错抽比漏抽代价大。
- 常驻活文档需要手动或定时重建，没有自动触发。
- 单机单进程。多进程写同一库未压测。
- 会话需要显式关闭（`POST /v1/sessions/{sid}/close`）才会做峰终结算。不关就没有峰终。
- 跨会话共享记忆没做：默认按 session 隔离，要共享得显式传 `session_id="*"`。
- 拒答阈值（0.18）是初始值，需要按真实语料标定。

## 性能（2026-10-18 复测）

| 操作 | 修复前 | 现在 |
|---|---|---|
| 单次检索（200 条） | 625ms | **1ms** |
| 写入接收 | 16.1s（同步阻塞） | 50ms（异步） |

关键改动是连接池 + WAL PRAGMA 只设一次 + `ensure_tables` 移出读路径。

## 设计文档

- [痛点调研](docs/00-痛点调研.md) — 长期陪伴 AI 的记忆到底难在哪
- [产品蓝图](docs/02-产品蓝图.md) — 概念模型与能力清单
- [架构契约](docs/03-技术方案与契约.md) — 数据模型、算法、接口
- [评审记录](docs/04-评审记录.md) — 三道评审门各自打回了什么
- [测试报告](docs/05-测试报告.md)
