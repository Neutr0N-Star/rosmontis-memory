# C1 · forget 批量化报告（16s 级维护作业 → ~1s）

> 工作线：性能收口｜任务：`task-16`｜实现者：pipeline-builder
> 依据：`docs/_wip/B1-性能审计.md` O11（forget 的 N+1 与逐条状态机）、O12（内联衰减）
> 结论：**100k 档 forget p50 从 2,979 ms（本机同 harness 改造前）降到 1,048 ms，-64.8%；SQL 条数 32,161 → 1,383（-95.7%）**。100k ≤1.5 s 目标达成。语义由「同 fixture 双库、同 now、新旧实现逐项对照」证明（新增 9 项测试全绿），98 项既有测试全绿。

---

## 1 改了什么

| 文件 | 改动 |
|---|---|
| `alaya/core/pipeline.py` | **只加批量入口** `transition_many(item_ids, action, *, reason, actor, cfg)`：一次分批读 episode → 每行仍走同一 `gate.decide` → 一个事务内「按目标态分组的批量 `UPDATE episode`」+「分块多行 `INSERT INTO item_state_log`」。既有 `submit/enrich/transition` 签名与语义未改；非批量安全动作自动退回逐条 `transition`。另加内部 `_chunks/_load_episode_rows`。 |
| `alaya/core/forget.py` | `run_decay_only` 归档改 `transition_many`；`run()` 的冷条目归档改 `transition_many`（逐条 `BEGIN IMMEDIATE` → 1 个事务）；超边成员从「每条冷条目一次查询」改为**一次批量 IN 查询**（`_hyperedge_hits`，5000 条 id 约 6 条 SQL）；多边归属的 O(n²) `next(...)` 改为 `row_by_id` O(1) 查表；折叠回写的逐成员 UPDATE 改为按组 `UPDATE … WHERE id IN` + 一次 `executemany` 写超边成员。折叠摘要仍逐组走 `pipeline.submit`（每条摘要仍是一个合法 item，带 create 日志）。 |
| `tests/test_forget_batch.py` | 新增 9 项（详见 §3） |
| `docs/_wip/C1-forget批量化报告.md` | 本报告 |

未改：schema、`db.py`、`gate.py`、`item.py`、`memory.py`、其它 core 文件；不碰 `web/`；未写 `alaya.db`；未 commit、未装依赖、未联网。

---

## 2 性能实测（bench/_profile，100k 为真测非外推）

环境：Windows / Python 3.12.10 / SQLite 3.49.1；合成库 `bench/_profile/gen_db.py 100k`（episode=100000、fact=40000、entity=15000、hyperedge=5000、709.6 MB，生成 7.47 s）；测量脚本 `bench/_profile/bench_write.py`（执行 ingest/enrich/close_session/dream 后测 forget，与 B1 同 harness）。

| 档 | 改造前 p50（4 次） | 改造后 p50（4 次） | 本机提速 | B1 记录改造前 → 提速 |
|---|---|---|---|---|
| 20k | **2,206 ms**（1989 / 2117 / 2295 / 2499） | **845 ms**（752 / 802 / 887 / 904） | **2.61×** | 2,835 ms → 3.36× |
| 100k | **2,979 ms**（2792 / 2933 / 3025 / 3039） | **1,048 ms**（985 / 1038 / 1058 / 1062） | **2.84×** | 16,154 ms → 15.4× |

SQL 条数（同一次 forget 调用内 business SQL，FTS 内部语句不计）：

| 档 | 改造前 | 改造后 | 降幅 |
|---|---|---|---|
| 20k | 34,060（B1 记录，同 harness） | **2,583** | **-92.4%** |
| 100k | 32,161（B1 记录，同 harness） | **1,383** | **-95.7%** |

- 100k 目标 ≤1.5 s **达成（1.048 s）**；20k 的「≥8×」分支未达成（2.61×），但验收的 100k 分支已满足。
- 为什么 100k 不比 20k 慢多少：`run()` 本身有 `LIMIT 5000` + 折叠上限，两档 scanned/archived 都是 5000；100k 的折叠组更少（674 vs 1,273 members），所以固定成本近似、100k 反而略高一点仅来自大库页缓存。
- 附带收益：`run_decay_only`（O12 热路径内联衰减）**100k 18.1/9.1 ms、20k 16.5/6.5 ms**（B1 改造前：100k 839/853 ms、20k 393~537 ms）。
- 一致性抽检（100k 跑完的库副本）：`archived=1` 5,148 条 = `state='archived'` 5,148 条，双向 mismatch 0；archive 日志 5,148 条、重复 0；摘要 135 条；`item_state_log` 总 5,289 行。

> 注：B1 的 100k 16.15 s 来自其测量环境/缓存状态；本轮在生成好的同一 100k 合成库上复测改造前为 2.98 s。报告同时给「本机前后对照」与「B1 记录前值 → 本轮后值」两组数字，避免拿环境差异当提速。

---

## 3 语义等价怎么证明

`tests/test_forget_batch.py` 的做法：**同一个 fixture 库复制成两份、用同一个 `now`**，一份跑新 `forget.run()`，一份跑测试内逐字保留的 `_legacy_forget()`（改造前的逐条 `TR.transition` + 每条冷条目一次超边查询 + O(n²) `next` 去重）。逐项断言：

1. **归档集合等价**：报告 `scanned/archived/folded/protected/folded_into`、`archived=1` 的 id 集合、全表 `state/tier/archived/muted/folded_into` 指纹、摘要行（id/text/origin/state/tier/role/source/salience）全部逐条相等。
2. **审计等价**：`item_state_log` 的 `(item_id, action, from_state, to_state, target, reason, gate_json, actor)` 指纹完全相等；每条被归档条目恰好 1 条 `archive` 日志；每条摘要恰好 1 条 `create` 日志。
3. **保护语义**：`tier='resident'` 与 `pinned=1` 由扫描 SQL 排除、高 salience 只计 protected；三者都保持原 state、`archived=0`、无 archive 日志。
4. **折叠语义**：只折叠成员数 ≥ `fold_min_members` 的超边组（fixture 3 组 ×5 人生成 3 条摘要）；多边归属仍归「成员数最多的边」且与旧实现同一条摘要；小组（3 人）与无超边条目只归档不折叠；摘要仍 `role=source='summary'`、`state='note'`、`salience=0.55`、回挂超边权重 1.2。
5. **幂等**：连跑两次，第二次的 `scanned/archived/folded/protected`、摘要数、全表指纹均与旧实现第二次一致；任何 item 的 archive 日志 ≤1。
6. **墓碑不受影响**：forget 后 tombstone 行原样保留，episode 不被物理删除。
7. **N+1 断言**：用 `sqlite3` trace 统计 business SQL —— 新实现 <120 条，旧实现 >250 条，且 `old/new > 4`（fixture 141 条冷条目；实际新实现条数是常数级）。
8. **批量入口与逐条等价**：`pipeline.transition_many` 与逐个 `transition` 在同一镜像库上比较返回结果 `(ok,item_id,from,to,reason)`、episode 指纹与日志指纹，全部相等（含缺失 id、已归档幂等、skipped 行等混合输入）。
9. **报告结构不变**：`ForgetReport.to_dict()` 仍恰好是 `{scanned, archived, folded, folded_into, protected, errors}`；`run_decay_only` 也走批量且 resident/pinned 不动。

---

## 4 残留 N+1 与剩余成本

- **已消除**：每条冷条目的超边查询（旧 5,000 条 SQL → 新 ~6 条 IN 查询）；逐条 `BEGIN IMMEDIATE + UPDATE + INSERT log`（旧 ~10,000 条 → 新 1 次批量 UPDATE + ~10 次分块多行 INSERT）；多边去重的 O(n²) 查找。
- **残留（不是 N+1）**：折叠摘要仍逐组一次 `pipeline.submit`（100k 实测 135 组、20k 255 组；每组约 10 条 SQL = 1 条摘要 episode + FTS + 1 条 create 日志 + 幂等/会话/取号）。这是「每个折叠组必须产出一个合法 item」的语义成本，不是按冷条目数增长。
- 若要把 100k 再压到 <1 s，可加 `pipeline.submit_many` 把摘要也批量写（收益约 0.2–0.3 s，风险是唯一写入点与状态机审计的复杂度），本轮未做，因为 1.5 s 目标已达成。
- 批量事务的失败面变大：整批 ROLLBACK（`_maybe_forget` 已捕获、下一轮重试），不再单条跳过；tests 用「整批同 now 对照」覆盖了正常路径，异常路径靠幂等重跑。

---

## 5 复现命令

```powershell
# 生成 100k 合成库（7.5s）
python -X utf8 bench/_profile/gen_db.py 100k

# 改造后测量（写路径 + forget + 内联衰减；p50/n_sql 见 bench/_profile/out/write_*.json）
python -X utf8 bench/_profile/bench_write.py 20k 100k

# 语义对照 + 全量回归
$env:PYTHONIOENCODING='utf-8'
python tests/test_forget_batch.py        # Ran 9 tests … OK
python tests/test_all.py                 # Ran 15 tests … OK
python tests/test_items.py               # Ran 22 tests … OK
python tests/test_pipeline_closure.py    # Ran 8 tests … OK
python tests/test_field.py               # Ran 20 tests … OK
python tests/test_api.py                 # Ran 20 tests … OK
python tests/test_stream.py              # Ran 5 tests … OK
python tests/test_visibility.py          # Ran 8 tests … OK
```

> `bench/_profile/` 在 `.gitignore` 内（脚手架与合成库不入库），本轮测量产物未进 git。

---

## 6 需配合

- `forget` 移出请求热路径（O12 的 job/守护进程化）属于 jobs 线；本轮已把单次成本从秒级降到 ~1 s，且内联衰减降到 ~10 ms，剩余收益主要靠「不在请求里跑」。
- 真库 `alaya.db` 全程未写（仍 `user_version=2`、182 episode、mtime 未变）；待 Lead 统一 commit。
