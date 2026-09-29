from app.chunking import chunk_words
from app.fusion import rrf


def test_rrf_prefers_docs_in_both_lists():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]])
    assert [i for i, _ in fused][:2] == ["a", "c"]


def test_chunk_overlap_and_coverage():
    text = " ".join(str(i) for i in range(1000))
    chunks = chunk_words(text, size=300, overlap=50)
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]
    assert chunks[-1].split()[-1] == "999"
