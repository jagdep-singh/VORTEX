FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/data \
    XDG_CONFIG_HOME=/data/.config \
    XDG_DATA_HOME=/data/.local/share

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY agent /app/agent
COPY client /app/client
COPY config /app/config
COPY context /app/context
COPY docker /app/docker
COPY hooks /app/hooks
COPY prompts /app/prompts
COPY safety /app/safety
COPY tools /app/tools
COPY ui /app/ui
COPY utils /app/utils
COPY main.py /app/main.py

RUN pip install --no-cache-dir --no-build-isolation .

RUN mkdir -p /workspace /data/.config /data/.local/share \
    && chmod +x /app/docker/entrypoint.sh \
    && chmod -R 777 /workspace /data

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/entrypoint.sh"]
