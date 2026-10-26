from functools import lru_cache

import psycopg
import requests
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

from rag import config


@lru_cache(maxsize=1)
def _embedder() -> TextEmbedding:
    return TextEmbedding(config.EMBED_MODEL)


def embed_queries(queries: list[str]) -> list[list[float]]:
    prefixed = [config.QUERY_PREFIX + q for q in queries]
    return [v.tolist() for v in _embedder().embed(prefixed)]


def embed_documents(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _embedder().embed(texts)]


def bm25_search(query: str, k: int = config.FINAL_K, mode: str = "bm25") -> list[dict]:
    r = requests.get(
        f"{config.GO_SEARCH_URL}/search",
        params={"q": query, "k": min(k, 50), "mode": mode},
        timeout=10,
    )
    r.raise_for_status()
    return [
        {
            "id": d["id"],
            "title": d["title"],
            "url": d["url"],
            "text": d["text"],
            "score": d["score"],
        }
        for d in r.json()["results"]
    ]


def vector_search(query: str, k: int = config.FINAL_K) -> list[dict]:
    vec = embed_queries([query])[0]
    with psycopg.connect(config.DATABASE_URL) as conn:
        register_vector(conn)
        rows = conn.execute(
            "SELECT id, title, url, text, 1 - (embedding <=> %s::vector) AS score "
            "FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            (vec, vec, k),
        ).fetchall()
    return [
        {"id": i, "title": t, "url": u, "text": x, "score": float(s)}
        for i, t, u, x, s in rows
    ]


def rrf(
    rankings: list[list[dict]], k: int = config.RRF_K, top: int = config.FINAL_K
) -> list[dict]:
    scores: dict[int, float] = {}
    docs: dict[int, dict] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            scores[d["id"]] = scores.get(d["id"], 0.0) + 1.0 / (k + rank)
            docs.setdefault(d["id"], d)
    best = sorted(scores, key=lambda i: scores[i], reverse=True)[:top]
    return [{**docs[i], "score": scores[i]} for i in best]


def hybrid_search(query: str, k: int = config.FINAL_K) -> list[dict]:
    bm = bm25_search(query, config.FETCH_K)
    vec = vector_search(query, config.FETCH_K)
    return rrf([bm, vec], top=k)
