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
# Models are baked in and the hub goes offline below, so pick the reranker at build time:
#   docker build --build-arg RERANK_MODEL=<model> .
ARG RERANK_MODEL=BAAI/bge-reranker-base
ENV RERANK_MODEL=${RERANK_MODEL}
RUN python -m app.ingest && python -c "from sentence_transformers import CrossEncoder; from app import config; CrossEncoder(config.RERANK_MODEL)"
RUN chmod -R a+rwX /srv/data
ENV HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]