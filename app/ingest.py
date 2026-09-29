"""Usage: python -m app.ingest   (reads PDFs from data/raw)"""
import json, pathlib
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from app import config, ocr
from app.chunking import chunk_words


def load_pages(pdf_path):
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < config.OCR_MIN_CHARS:              # scanned page
            text, conf = ocr.ocr_page(page)
            if text.strip():
                yield i, text, True, conf
            continue
        if config.OCR_FIGURES:                              # diagram labels
            fig = ocr.figure_text(doc, page)
            if fig:
                text += f"\n[Figure text: {fig}]"
        yield i, text, False, 100.0


def main():
    chunks = []
    for pdf in sorted(pathlib.Path(config.RAW_DIR).glob("*.pdf")):
        for page_no, text, was_ocr, conf in load_pages(pdf):
            for j, c in enumerate(chunk_words(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP)):
                chunks.append({"id": f"{pdf.stem}-p{page_no}-c{j}", "text": c,
                               "source": pdf.stem, "page": page_no,
                               "ocr": was_ocr, "ocr_conf": round(conf, 1)})
    if not chunks:
        raise SystemExit("No PDFs found in data/raw")

    pathlib.Path(config.CHUNKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    model = SentenceTransformer(config.EMBED_MODEL)
    embs = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                        show_progress_bar=True, batch_size=32)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection("course")
    except Exception:
        pass
    col = client.create_collection("course", metadata={"hnsw:space": "cosine"})
    B = 500
    for i in range(0, len(chunks), B):
        part = chunks[i:i + B]
        col.add(ids=[c["id"] for c in part], embeddings=embs[i:i + B].tolist(),
                documents=[c["text"] for c in part],
                metadatas=[{"source": c["source"], "page": c["page"], "ocr": c["ocr"]} for c in part])
    n_ocr = sum(c["ocr"] for c in chunks)
    low = sum(c["ocr"] and c["ocr_conf"] < 60 for c in chunks)
    print(f"Ingested {len(chunks)} chunks ({n_ocr} from OCR, {low} low-confidence)")


if __name__ == "__main__":
    main()