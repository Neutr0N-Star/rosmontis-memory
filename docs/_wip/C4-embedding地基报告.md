# C4 · embedding 地基报告（断路器 / 向量身份 v4 / NULL 回算）

> 工作线：embedding 地基｜任务：`task-19`｜实现者：pipeline-builder
> 依据：`docs/18-接入调研与宿主契约.md` §5-B1（B1a 无断路 / B1b 降级不可逆 / B1c 无向量身份）
> 结论：三处地基已修。断路后连续调用**只探测 1 次**；schema v4 在真库只读副本上幂等迁移（182 episode / 28 fact 行数不变、旧向量身份 NULL 仍可检索）；`embed_backfill` job 实测一次补 **12/12 条** NULL 向量并写入身份；`/readyz`（经 `memory.stats()`）可见 `vector_mode / embed_model / pending_backfill`。既有 132 项测试 + 新增 15 项全绿。不引依赖、未碰 `alaya.db`。

---

## 1 改了什么

| 文件 | 改动 |
|---|---|
| `alaya/core/embed.py` | 断路器（首次探测短超时 2s → 失败置 `_available=False` → 退避窗口内不打远程 → 指数退避 30s→300s 封顶重探 → 恢复自动切回）；`_probe_done` 真正启用；`vector_mode()/identity()/circuit_state()`；`health(probe=False)` 不联网；`hash` 视为 degraded、`none` 为无向量模式；`provider_registry` 扩展点空壳（只登记内置 ollama/hash/none，不实现 ONNX） |
| `alaya/config.py` | 新增 `embed_model_id`（默认回落 `embed_model`）→ 写入 `episode/fact.embed_model` 的稳定身份 |
| `alaya/schema.sql` | v4：episode/fact 各加 `embed_model TEXT`、`embed_dim INTEGER`（只加列） |
| `alaya/core/db.py` | `SCHEMA_VERSION=4`；`_migrate_v4()` 并入既有**单事务**迁移（v2→v4 一次 COMMIT；失败 ROLLBACK 保持旧版本；重复执行幂等） |
| `alaya/core/write.py` | 写入记录身份：episode 在 `pipeline.submit` 后补一次身份 UPDATE（其契约无身份字段）；fact 的 INSERT 直接带两列；降级期 fact 不再写哈希伪向量（留 NULL 给回算）；新增 `embedding_identity()/stamp_episode_embedding()` |
| `alaya/core/notebook.py` | 主动笔记写入记录身份；降级期不落伪向量（先算后判 degraded） |
| `alaya/core/recall.py` | `_dense` 与 `_refine_cosine` 只比同身份向量：`embed_model`/`embed_dim` 不匹配即跳过；两列 NULL 的旧数据视为"未知身份"，仍参与检索不报错；`plan_json.identity_skipped` 可审计 |
| `alaya/core/memory.py` | Embedder 传入 `model_id`；`stats()` 增加 `vector_mode / embed_model / embed_dim / pending_backfill`，`health(probe=False)` 不联网；新增 `pending_backfill()/backfill_embeddings()/vector_mode()` |
| `alaya/core/jobs.py` | 新增 `KIND_EMBED_BACKFILL="embed_backfill"` 并接入 `execute_job`；settle/tick/forget/review 执行完自动幂等投递下一拍的 embed_backfill（daemon 起来即自动补） |
| `tests/test_embed_resilience.py` | 新增 15 项（见 §6） |
| `tests/test_items.py` | 3 处 `user_version == 3` 断言改为 `DB.SCHEMA_VERSION`（v4 升级的必然结果；断言语义不变） |
| `docs/_wip/C4-embedding地基报告.md` | 本报告 |

未改：`pipeline.py / gate.py / item.py / service/app.py / field.py / web/`；未 commit、未装依赖、未联网；`alaya.db` 只以 `mode=ro` 打开。

---

## 2 断路器（B1a）

### 2.1 行为

```
embed(text)
  ├─ cache 命中 → 直接返回（语义向量缓存；降级期的哈希向量不进缓存）
  ├─ provider=ollama:
  │    若 circuit open 且未到 open_until → 直接 hash fallback（remote_attempts 不增长）
  │    否则远程调用：
  │      首次探测 → timeout=min(2s, cfg)
  │      失败 → _available=False，fail_streak+1，open_until=now+min(300, 30·2^(n-1))
  │              degraded=True（不再把哈希向量写库，见下）
  │      成功 → _available=True，streak=0，degraded=False，清空降级期缓存
  └─ provider=none → 返回 []（无向量模式）；hash → 确定性哈希（仅测试/兜底，degraded）
```

- 退避序列：30s → 60s → 120s → 240s → **300s 封顶**；窗口内调用零远程。
- `_probe_done` 现在表示"本轮是否已探测过"，用于状态可见性；`circuit_state()` 暴露 `state/fail_streak/next_probe_in/probe_done/remote_attempts/probe_attempts/fallback_count`。
- `health(probe=False)` 只看断路器（`stats/readyz` 用，不产生网络等待）；`health(probe=True)` 在 open 窗口内也不联网。
- 降级期 `vector_mode=degraded`；恢复后自动 `semantic`，无需重启。

### 2.2 实测证据

- 降级期**连续写 10 条**（每条约 1 次 embedding 调用）：`remote_calls == 1`（首次探测），其余全部短路 fallback —— 对应测试 `test_consecutive_failures_probe_only_once`。
- 模拟半死远端（每次远程 sleep 0.2s）调用 20 次：总耗时 < 1s（只有第 1 次付了等待）—— `test_no_8s_waits_after_open`。
- 退避/恢复/缓存刷新：`test_exponential_backoff_then_recovery`、`test_recovery_clears_degraded_cache`、`test_health_probe_false_never_networks`。

---

## 3 向量身份 v4（B1c）

- schema v4 只加列：`episode.embed_model/embed_dim`、`fact.embed_model/embed_dim`；`user_version=4`。
- 迁移在原有单事务里完成（`_migrate(2)` 一次跑 v2→v3→v4）：加列 → v3 回填 → 写 v4 列 → `user_version=4` → COMMIT；任何一步失败整体 ROLLBACK 保持旧版本。
- 写入记录：`config.embed_model_id`（默认=模型名）→ `episode/fact.embed_model`，`embed_dim=cfg.embed_dim`；fact 在 INSERT 时同时写。
- 检索：`recall._dense` / `_refine_cosine` 只对「身份相同或身份未知(NULL)」的向量算余弦；明确不同 `embed_model`/`embed_dim` 的行跳过并计入 `plan_json.identity_skipped`，不报错。
- 旧数据：两列 NULL = 未知身份，**仍可用**。

**实测（真库只读副本）**

| 项 | 结果 |
|---|---|
| `user_version` | 2 → **4** |
| 行数 | episode 182、fact 28（不变） |
| `integrity_check` | ok |
| 旧行身份 | 182 + 28 条全部 `embed_model/embed_dim = NULL` |
| 旧向量可用性 | 用与旧向量同向的 query 调 `recall._dense` → 40 条命中、无异常（NULL 身份放行） |
| 幂等 | 第二次 `init_db` 无变化、无报错 |
| 失败回滚 | monkeypatch `_migrate_v4` 抛错 → `user_version` 仍 3、v4 列不存在、数据不变 |

另外用合成 v3 库验证了 v3→v4 的加列 + 幂等，见 `TestVectorIdentityMigration`。

---

## 4 NULL 向量回算（B1b）

- 新 job kind：`KIND_EMBED_BACKFILL = "embed_backfill"`，已进 `KINDS` 与 `execute_job`。
- 执行体 `Memory.backfill_embeddings(limit=500, session_id=None)`：
  1. 一批查 `embedding IS NULL` 的 episode/fact（一条 SELECT，不用逐条 N+1）；
  2. 批量算向量（失败/降级立即停，不写伪向量）；
  3. **一个事务** `executemany` 回写 `embedding + embed_model + embed_dim`；
  4. 返回 `scanned/filled/filled_episode/filled_fact/remaining/embed_model/embed_dim/skipped/reason`。
- 自动补：daemon 的 settle/tick/forget/review job 执行完，`execute_job` 会检查"语义模式 + pending>0"，幂等投递 `embed_backfill`（60s bucket 去重），下一拍自动补齐。没有 daemon 时也可手工 `jobs.enqueue(cfg, jobs.KIND_EMBED_BACKFILL, "*")`。
- 降级期写入不落伪向量：episode 由原有 `degraded` 判定；本轮补上 fact（旧实现会用 hash 向量写 fact，导致"永远不会 NULL、永不回算"）。

**实测证据**（离线桩，模拟降级 → 恢复）

```
降级期写入：8 条 episode（含 2 条带 fact 的抽取）
  → episode NULL 10 条、fact NULL 2 条；vector_mode=degraded；remote_calls=1
恢复后跑一次 backfill：
  {"scanned":12,"filled":12,"filled_episode":10,"filled_fact":2,
   "remaining":0,"skipped":false,"embed_model":"bge-m3:latest","embed_dim":8}
回写身份：episode/fact 的 embed_model=bge-m3:latest、embed_dim=8
再降级写 3 条 → tick job 执行后自动 enqueue {"kind":"embed_backfill","created":true}
```

---

## 5 `/readyz` 与 `memory.stats()`

`/readyz` 的 `checks.providers.stats` 直接摊平自 `memory.stats()`，新增字段（实测输出）：

```json
{
  "vector_mode": "degraded",
  "embed_model": "bge-m3:latest",
  "embed_dim": 1024,
  "pending_backfill": 0,
  "degraded": true,
  "embed": {
    "provider": "hash",
    "model_id": "bge-m3:latest",
    "vector_mode": "degraded",
    "circuit": {"state": "closed", "fail_streak": 0, "next_probe_in": 0.0,
                "probe_done": false, "remote_attempts": 0, "fallback_count": 0}
  }
}
```

`vector_mode` 语义：
- `semantic`：provider=ollama 且断路器未处于降级（默认 bge-m3 正常）；
- `degraded`：ollama 失败/断路、或 `provider=hash`（测试兜底，非语义向量）；
- `none`：`provider=none`（无向量模式，写入留 NULL）。

`health(probe=False)` 保证 readyz/stats 不会为了探活卡 2–8 秒。

---

## 6 测试

新增 `tests/test_embed_resilience.py` 15 项：

1. 连续失败只探测 1 次（20 次调用 → remote_attempts=1）；
2. 断路后 20 次调用总耗时 <1s（半死远端）；
3. 指数退避 30→60→120→240→300 封顶 + 半开重探 + 恢复自动切回；
4. 恢复后清掉降级期缓存、同一句话重新用语义向量；
5. `health(probe=False/True)` 在 open 窗口内零网络；
6. v3→v4 加列、旧数据 NULL、幂等、integrity；
7. v4 迁移失败 ROLLBACK 保持 v3；
8. 真库只读副本 v2→v4（行数不变、旧身份 NULL、integrity）；
9. 写入记录身份（episode + notebook）；
10. `_dense` 同身份过滤：同模型命中、NULL 身份命中、异模型排除、`identity_skipped=1`；
11. 降级期写入留 NULL（episode/fact 都不写伪向量）；
12. 恢复后 backfill 补齐 + 身份正确 + pending 归零；
13. stats 含 `vector_mode/embed_model/pending_backfill/embed_dim`；
14. hash provider → `vector_mode=degraded`；
15. `execute_job(embed_backfill)` 可执行、tick job 会自动排 backfill。

回归：现有全部测试文件（`test_all / test_api / test_daemon / test_field / test_forget_batch / test_items / test_models / test_pipeline_closure / test_stream / test_visibility`）共 **132 项全绿**，加新 15 项共 147 项全绿（判定以 `Ran N tests … OK` 为准）。

> `tests/test_items.py` 有 3 处硬编码 `user_version == 3`，v4 后必然失败；已改为 `DB.SCHEMA_VERSION`（断言仍是"迁移到当前版本"，未放宽任何语义）。

---

## 7 残留与风险

1. **`field.py` 的向量点火仍只按维度校验**（该文件属语义场线、不在本轮独占清单）：换模型且维度相同时，field 点火仍可能混入异身份向量。`recall._dense` 已修；建议 WS-B 在 field `_vec_seeds` 复用 `recall._identity_ok` / 同款 SQL 过滤。
2. **pipeline 契约没有身份字段**：episode 的身份靠 `write/notebook` 在 submit 后补一次 UPDATE（每条写入 +1 SQL，亚毫秒）。若后续给 `pipeline.submit(item)` 的 MemoryItem 加 `embed_model/embed_dim` 字段，可并入 INSERT（本轮约定不改 pipeline.py）。
3. **backfill 每轮有 limit（每表 500）**：大积压需要多轮 daemon 节拍（自动 60s bucket 重投）；可在 jobs 层加剩余量驱动的连续批，但当前非阻塞。
4. **旧数据身份永久 NULL**：允许（未知身份仍可用）；若确认旧向量都是同一模型，可后续做一次"身份认领"迁移，把 NULL 回填为已知模型，但需要有证据，本轮不做。
5. **首探短超时 2s**：极慢但健康的本地模型首次可能被误判一次，30s 后以完整超时重探可恢复；`probe_timeout` 可在构造时调整。
6. **自动补的触发依赖 daemon 的 settle/tick/forget/review job**：只跑 HTTP 请求、不跑 daemon 的宿主需要自己投一次 `embed_backfill`（API/CLI 已可 enqueue）。
7. `provider_registry` 只是扩展点空壳：内置仍只有 ollama/hash/none，**未实现 ONNX/local**（按指示由模型安装器线另起）。
8. hash provider 现在从构造起即 `degraded=True`，且 notebook/fact 不再写哈希伪向量（与 episode 口径统一）。这是行为变化，已用回归测试确认不影响既有功能。

---

## 8 复现

```powershell
$env:PYTHONIOENCODING='utf-8'
python tests/test_embed_resilience.py   # Ran 15 tests … OK
python tests/test_all.py                # Ran 15 tests … OK
python tests/test_items.py              # Ran 22 tests … OK
# …其余 test_*.py 同样全绿

# schema v4 副本迁移（真库只读）
python - <<'PY'  # 见 tests/test_embed_resilience.py::test_real_db_copy_to_v4
PY
```
