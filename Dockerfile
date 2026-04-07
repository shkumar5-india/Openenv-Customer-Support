FROM python:3.10-slim
LABEL maintainer="openenv-community"
LABEL description="OpenEnv Customer Support Environment"
LABEL version="1.0.0"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY server/ ./server/
COPY env/ ./env/
COPY data/ ./data/
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY uv.lock .
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
