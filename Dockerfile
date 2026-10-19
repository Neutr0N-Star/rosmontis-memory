# 藏识 Alaya 本地服务镜像
# 构建：docker build -t alaya-memory .
# 运行：docker run -p 127.0.0.1:8770:8770 -v alaya-data:/data alaya-memory
FROM python:3.12-slim

WORKDIR /app

# 只装核心 + 服务依赖；numpy/jieba 由 serve extra 带
COPY pyproject.toml README.md LICENSE ./
COPY alaya ./alaya
RUN pip install --no-cache-dir ".[serve]"

# 默认走离线降级（hash 向量 + 规则抽取），接自己的模型请在 compose 里覆盖环境变量
ENV ALAYA_DB=/data/alaya.db \
    ALAYA_EMBED=hash \
    ALAYA_LLM_HOT=rule \
    PYTHONIOENCODING=utf-8

VOLUME ["/data"]
EXPOSE 8770

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8770/healthz', timeout=3).status==200 else 1)"

CMD ["alaya", "serve", "--host", "0.0.0.0", "--port", "8770"]
