# From prototype to production

| Area | Prototype (this repo) | Production |
|---|---|---|
| Ingestion | Script over local PDFs | Async queue (Celery/SQS) workers, idempotent by content hash, OCR fallback, versioned documents, re-index without downtime |
| Vector store | Chroma on disk | Qdrant/Weaviate/pgvector as a managed service, native hybrid search, metadata filters (course, tenant, ACL) |
| BM25 | In-memory rank_bm25 | Engine-native sparse (Qdrant sparse vectors / OpenSearch) |
| Reranker | CPU cross-encoder | GPU or hosted rerank API, batch + timeout with fallback to hybrid |
| Cache | In-process LRU | Redis: exact-match cache + semantic cache + embedding cache, TTL, invalidate on re-ingest |
| LLM | Single blocking call | Streaming (SSE), retries/backoff, timeouts, provider fallback, prompt caching, model routing (cheap for easy queries) |
| Guardrails | Score threshold + prompt | Plus citation verification (claim vs passage check), prompt-injection filtering on documents, PII redaction, input length limits |
| Eval | Recall/MRR script | 50-100+ golden set, faithfulness/citation accuracy via LLM judge, run in CI as a regression gate, sampled online eval on real traffic |
| Observability | Log lines | Langfuse/OpenTelemetry traces per stage, dashboards for p50/p95 latency, cost per query, refusal rate, thumbs-down rate, alerts |
| Security | None | Auth (JWT/API keys), per-tenant isolation, rate limiting, secrets manager, CORS |
| Deploy | docker compose | Container on ECS/Cloud Run/K8s, autoscaling, health/readiness probes, Terraform, CI/CD with staging + canary |
| Data loop | feedback.jsonl | Thumbs-down review queue -> new golden cases -> retrieval/prompt fixes |
