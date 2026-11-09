# syntax=docker/dockerfile:1
# ==============================================================================
# Stage 1: Builder - install dependencies via uv and pre-bake quantized ONNX BGE
# ==============================================================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install . && \
    mkdir -p "${FASTEMBED_CACHE_PATH}" && \
    /opt/venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/fastembed_cache')"

# ==============================================================================
# Stage 2: Runtime - minimal offline-ready image with pre-baked ONNX weights
# ==============================================================================
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    AGENT_SESHAT_HOME=/data/.agent-seshat \
    AGENT_SESHAT_HOST=0.0.0.0 \
    AGENT_SESHAT_PORT=8765

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/fastembed_cache /opt/fastembed_cache
COPY sample_skills /app/sample_skills

RUN mkdir -p /data/.agent-seshat/skills /data/.agent-seshat/cache

EXPOSE 8765

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request, sys; r = urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2); sys.exit(0 if r.status == 200 else 1)"

ENTRYPOINT ["sesh"]
CMD ["serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8765"]
