FROM python:3.10-slim
LABEL maintainer="openenv-community"
LABEL description="OpenEnv Customer Support Environment"
LABEL version="1.0.0"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY server.py .
COPY inference.py .
COPY openenv.yaml .
COPY env/__init__.py env/__init__.py
COPY env/environment.py env/environment.py
COPY env/models.py env/models.py
COPY env/tasks.py env/tasks.py
COPY env/graders.py env/graders.py
COPY data/support_tickets.csv data/support_tickets.csv
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1
CMD ["uvicorn", "server:app","--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
