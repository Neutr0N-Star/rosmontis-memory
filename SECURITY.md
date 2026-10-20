# 安全策略

## 报告漏洞

请不要开公开 issue。通过仓库的私有安全渠道（GitHub Security Advisories）或直接联系维护者，附上复现步骤与影响范围。48 小时内给首次回复。

## 默认安全模型

Alaya 是**本地优先**的记忆后端：

- 默认只绑定 `127.0.0.1`，不对局域网或公网开放。
- 首次启动生成访问 token（`~/.alaya/token`），除 `/healthz`、`/readyz` 外所有 `/v1/*` 要求 `Authorization: Bearer <token>`。
- 绑定非回环地址时，必须显式提供 token（`--token` 或 `ALAYA_TOKEN`），否则拒绝启动。
- CORS 默认只放行本机开发来源；`file://` 来源需自行加入 `ALAYA_CORS_ORIGINS`。
- `/v1/wipe`、`/v1/settings` 需要额外的确认头 `X-Confirm: yes`。

## 你的责任

把服务暴露到公网 = 把你的对话记忆暴露到公网。若确需远程访问，放在反向代理与 TLS 之后，并自行加网关级鉴权。

## 不覆盖范围

- 宿主机被攻破、token 文件被读取。
- 接入方自己的模型通道（Ollama / OpenAI 兼容端点）的安全与合规。
- 模型资产的供应链（下载源与校验值由使用方自行决定是否固定）。
