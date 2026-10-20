# 隐私说明

Rosmontis 存的是**对话记忆**，属于私密数据。这份文档说明数据放在哪、去哪、怎么删。

## 数据不出本机

- 记忆库是本地 SQLite 单文件（默认 `rosmontis.db`，或 `ROSMONTIS_DB` 指定）。
- 模型推理：用你的 Ollama / 自备 OpenAI 兼容通道 / 本地 ONNX 模型。**本项目不运营任何服务端，也不上传数据。**
- 无遥测、无统计上报、无自动更新检查。

## 本地文件清单

| 路径 | 内容 |
|---|---|
| `rosmontis.db`（+ `-wal`/`-shm`） | 全部记忆、状态、审计日志 |
| `~/.rosmontis/token` | 本地接口访问 token |
| `~/.rosmontis/models/` | 下载的嵌入模型（可用 `ROSMONTIS_MODEL_DIR` 改） |
| `.env` 或环境变量 | 你的模型通道密钥（本项目不读取也不上传） |

## 会离开本机的数据

只有你自己配置的模型通道：当 `ROSMONTIS_LLM_HOT` / `ROSMONTIS_EMBED` 指向云端（OpenAI 兼容）时，抽取、对话与向量化会把对应文本发给该服务商。用本地 Ollama 或本地 ONNX 模型则不出机。

## 删除数据

- 单个会话：`POST /v1/wipe?session_id=...`（不可逆；需 `X-Confirm: yes`）。
- 单条记忆：`POST /v1/remove`（写墓碑，默认不复活）。
- 彻底删除：停止服务后直接删除 `rosmontis.db` 及其 `-wal`/`-shm` 文件；备份文件（`*.bak`）需一并删除。
- 模型与 token：删除 `~/.rosmontis/` 即可。

## 给部署者的提醒

- 不要把服务绑定到 `0.0.0.0` 后暴露公网。
- 共享机器上运行时，`rosmontis.db` 与 token 文件的文件权限应限制为当前用户可读。
- 日志默认不含对话正文；开启调试级别可能打印检索明细，注意脱敏。
