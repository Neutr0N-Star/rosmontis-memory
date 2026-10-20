# Rosmontis 本地服务镜像
# 构建：docker build -t rosmontis-memory .
# 运行：docker run -p 127.0.0.1:8770:8770 -v rosmontis-data:/data rosmontis-memory
#      首次启动会把访问 token 写入卷（/data/token）并打印到日志；客户端带
#      Authorization: Bearer <token> 访问 /v1/*。
FROM python:3.12-slim

WORKDIR /app

# 只装核心 + 服务依赖；numpy/jieba 由 serve extra 带
COPY pyproject.toml README.md LICENSE ./
COPY rosmontis ./rosmontis
RUN pip install --no-cache-dir ".[serve]"

# 模型与记忆都放卷里（ROSMONTIS_HOME 派生出 db / models / token）
ENV ROSMONTIS_HOME=/data \
    ROSMONTIS_EMBED=hash \
    ROSMONTIS_LLM_HOT=rule \
    PYTHONIOENCODING=utf-8

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/data"]
EXPOSE 8770

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8770/healthz', timeout=3).status==200 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
