import logging
import os
import time

from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.retrieval import Retriever
from app.generate import answer


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

log = logging.getLogger("coursemate")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "20"))

CACHE_MAX = 500

retriever: Retriever | None = None

cache: OrderedDict[str, dict] = OrderedDict()

hits: dict[str, deque] = defaultdict(deque)


# --------------------------------------------------
# Application lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever

    retriever = Retriever()

    # Warm up reranker
    _ = retriever.reranker

    yield


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="CourseMate",
    lifespan=lifespan
)


# --------------------------------------------------
# CORS configuration
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request models
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    source: str | None = None
    k: int = Field(default=5, ge=1, le=8)


class Feedback(BaseModel):
    question: str = Field(max_length=500)
    answer: str = Field(max_length=5000)
    thumbs_up: bool


# --------------------------------------------------
# Rate limiting
# --------------------------------------------------

def rate_limit(request: Request):
    fwd = request.headers.get("x-forwarded-for")

    ip = (
        fwd.split(",")[0].strip()
        if fwd
        else request.client.host
    )

    now = time.time()

    q = hits[ip]

    while q and now - q[0] > 60:
        q.popleft()

    if len(q) >= RATE_PER_MIN:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute."
        )

    q.append(now)


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "chunks": len(retriever.chunks)
    }


# --------------------------------------------------
# Sources endpoint
# --------------------------------------------------

@app.get("/sources")
def sources():
    return sorted({
        c["source"]
        for c in retriever.chunks
    })


# --------------------------------------------------
# Ask endpoint
# --------------------------------------------------

@app.post("/ask")
def ask(req: AskRequest, request: Request):

    key = f"{req.question.strip().lower()}|{req.source}|{req.k}"

    if key in cache:
        return {
            **cache[key],
            "cached": True
        }

    rate_limit(request)

    t0 = time.perf_counter()

    where = (
        {"source": req.source}
        if req.source
        else None
    )

    passages = retriever.search(
        req.question,
        "rerank",
        k=req.k,
        where=where
    )

    t1 = time.perf_counter()

    try:
        text, usage = answer(
            req.question,
            passages
        )

    except Exception:
        log.exception("LLM call failed")

        raise HTTPException(
            status_code=502,
            detail="The language model is temporarily unavailable. Try again shortly."
        )

    t2 = time.perf_counter()

    out = {
        "answer": text,

        "citations": [
            {
                "source": p["source"],
                "page": p["page"],
                "score": p["score"]
            }
            for p in passages
        ],

        "latency_ms": {
            "retrieval": round((t1 - t0) * 1000),
            "generation": round((t2 - t1) * 1000)
        },

        "tokens": usage,
        "cached": False
    }

    log.info(
        "ask q=%r retrieval_ms=%s gen_ms=%s tokens=%s",
        req.question,
        out["latency_ms"]["retrieval"],
        out["latency_ms"]["generation"],
        usage
    )

    cache[key] = out

    if len(cache) > CACHE_MAX:
        cache.popitem(last=False)

    return out


# --------------------------------------------------
# Feedback endpoint
# --------------------------------------------------

@app.post("/feedback")
def feedback(fb: Feedback, request: Request):

    rate_limit(request)

    with open(
        "data/feedback.jsonl",
        "a",
        encoding="utf-8"
    ) as f:
        f.write(fb.model_dump_json() + "\n")

    return {
        "ok": True
    }


# --------------------------------------------------
# Serve frontend
# --------------------------------------------------

app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static"
)