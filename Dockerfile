FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/app \
  PATH="/app/.venv/bin:$PATH" \
  UV_COMPILE_BYTECODE=1 \
  UV_HTTP_TIMEOUT=300 \
  UV_REQUEST_TIMEOUT=300 \
  VIRTUAL_ENV=/app/.venv
WORKDIR /app
RUN pip install --no-cache-dir "uv>=0.5,<1.0"
COPY pyproject.toml uv.lock ./
RUN uv venv /app/.venv && \
  uv sync --frozen --no-install-project || \
  (sleep 5  && uv sync --frozen --no-install-project) || \
  (sleep 15 && uv sync --frozen --no-install-project)
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); print('OK: tiktoken cached')"
COPY . .
RUN mkdir -p /app/data /app/exports /app/web /app/hooks && chmod 777 /app/data
EXPOSE 8082
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8082/health')" || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]