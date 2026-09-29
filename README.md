---
title: CourseMate
emoji: 📚
sdk: docker
app_port: 7860
---

# CourseMate: citation-grounded course Q&A (RAG)

Hybrid retrieval (dense + BM25, RRF) -> cross-encoder rerank -> LLM answer with `[source, p.X]` citations and refusal when the material lacks the answer.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your API key
# put 1-2 PDFs in data/raw/ (OpenStax / NCERT chapter)
python -m app.ingest
uvicorn app.main:app --reload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What is ...?"}'
```

## Evaluate
Copy `eval/golden.example.json` to `eval/golden.json`, write 20-30 questions (5-10 unanswerable), then:
```bash
python -m eval.run_eval eval/golden.json
```
Paste the table below. Tune `REFUSE_THRESHOLD` in `.env` using the unanswerable set.

## Results
(paste table here)

## Next steps
See PRODUCTION.md.
