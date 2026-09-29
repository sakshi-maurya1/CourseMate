import json, math, re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from app import config
from app.fusion import rrf


def tokenize(t: str) -> list[str]:
    return re.findall(r"\w+", t.lower())


class Retriever:
    def __init__(self):
        with open(config.CHUNKS_PATH) as f:
            self.chunks = [json.loads(l) for l in f]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.col = chromadb.PersistentClient(path=config.CHROMA_DIR).get_collection("course")
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(config.RERANK_MODEL)
        return self._reranker

    def dense(self, q: str, n: int, where: dict | None = None) -> list[str]:
        emb = self.embedder.encode(config.QUERY_PREFIX + q, normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=[emb], n_results=n, where=where)
        return res["ids"][0]

    def sparse(self, q: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(q))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.chunks[i]["id"] for i in top]

    def search(self, q: str, mode: str = "rerank", k: int = 5, fetch: int = 20,
               where: dict | None = None) -> list[dict]:
        """mode: dense | hybrid | rerank"""
        if mode == "dense":
            ids = self.dense(q, k, where)
            return [dict(self.by_id[i], score=None) for i in ids]
        fused = [i for i, _ in rrf([self.dense(q, fetch, where), self.sparse(q, fetch)])][:fetch]
        if where:
            fused = [i for i in fused if self.by_id[i]["source"] == where["source"]]
        if not fused:
            return []
        if mode == "hybrid":
            return [dict(self.by_id[i], score=None) for i in fused[:k]]
        pairs = [(q, self.by_id[i]["text"]) for i in fused]
        raw = self.reranker.predict(pairs)
        ranked = sorted(zip(fused, raw), key=lambda x: x[1], reverse=True)[:k]
        return [dict(self.by_id[i], score=1 / (1 + math.exp(-float(s)))) for i, s in ranked]
