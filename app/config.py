import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
RAW_DIR = "data/raw"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
REFUSE_THRESHOLD = float(os.getenv("REFUSE_THRESHOLD", "0.30"))  # sigmoid(rerank score)
CHUNK_WORDS = 300
CHUNK_OVERLAP = 50
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "50"))  # less native text than this -> page is scanned
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_FIGURES = os.getenv("OCR_FIGURES", "1") == "1"      # OCR labels inside diagrams