#!/bin/sh
# 容器入口：非回环绑定时服务要求显式 token，因此这里保证一定有一个。
# 优先用 ROSMONTIS_TOKEN；否则复用/生成 ~/.rosmontis/token，并打印一次。
set -e

: "${ROSMONTIS_HOME:=/data}"
export ROSMONTIS_HOME
mkdir -p "$ROSMONTIS_HOME"

if [ -z "${ROSMONTIS_TOKEN:-}" ]; then
  TOKEN_FILE="$ROSMONTIS_HOME/token"
  if [ -f "$TOKEN_FILE" ]; then
    ROSMONTIS_TOKEN="$(cat "$TOKEN_FILE")"
  else
    ROSMONTIS_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    printf '%s' "$ROSMONTIS_TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE" 2>/dev/null || true
    echo "[rosmontis] 已生成访问 token（写入 $TOKEN_FILE）：$ROSMONTIS_TOKEN"
  fi
  export ROSMONTIS_TOKEN
fi

exec rosmontis serve --host 0.0.0.0 --port "${ROSMONTIS_PORT:-8770}"
