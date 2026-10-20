#!/bin/sh
# 容器入口：非回环绑定时服务要求显式 token，因此这里保证一定有一个。
# 优先用 ALAYA_TOKEN；否则复用/生成 ~/.alaya/token，并打印一次。
set -e

: "${ALAYA_HOME:=/data}"
export ALAYA_HOME
mkdir -p "$ALAYA_HOME"

if [ -z "${ALAYA_TOKEN:-}" ]; then
  TOKEN_FILE="$ALAYA_HOME/token"
  if [ -f "$TOKEN_FILE" ]; then
    ALAYA_TOKEN="$(cat "$TOKEN_FILE")"
  else
    ALAYA_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    printf '%s' "$ALAYA_TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE" 2>/dev/null || true
    echo "[alaya] 已生成访问 token（写入 $TOKEN_FILE）：$ALAYA_TOKEN"
  fi
  export ALAYA_TOKEN
fi

exec alaya serve --host 0.0.0.0 --port "${ALAYA_PORT:-8770}"
