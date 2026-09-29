FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /srv
ENV UV_LINK_MODE=copy HF_HOME=/srv/.hf PATH="/srv/.venv/bin:$PATH" PYTHONUNBUFFERED=1 ANONYMIZED_TELEMETRY=False
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
COPY eval ./eval
COPY static ./static
COPY data/raw ./data/raw
RUN python -m app.ingest && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"
RUN chmod -R a+rwX /srv/data
ENV HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]