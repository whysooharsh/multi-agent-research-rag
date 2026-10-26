from rag.retrieval import rrf
from scripts.build_corpus import CHUNK_WORDS, chunk_text


def doc(i):
    return {"id": i, "title": "t", "url": "u", "text": "x", "score": 0}


def test_rrf_hand_computed():
    fused = rrf([[doc(1), doc(2)], [doc(2), doc(3)]], k=60, top=3)
    assert [d["id"] for d in fused] == [2, 1, 3]
    assert abs(fused[0]["score"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(fused[1]["score"] - 1 / 61) < 1e-12


def test_rrf_top_limit():
    assert len(rrf([[doc(i) for i in range(20)]], top=5)) == 5


def test_chunks_respect_size_and_overlap():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(200)]
    chunks = chunk_text(" ".join(sentences))
    assert len(chunks) > 1
    assert all(len(c.split()) <= CHUNK_WORDS + 8 for c in chunks)
    last_sentence = chunks[0].split(". ")[-1].rstrip(".")
    assert last_sentence in chunks[1]
