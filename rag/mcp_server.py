from mcp.server.fastmcp import FastMCP

from rag import retrieval

mcp = FastMCP("research-search")


def _trim(results: list[dict]) -> list[dict]:
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "url": r["url"],
            "text": r["text"],
            "score": round(r["score"], 5),
        }
        for r in results
    ]


@mcp.tool()
def bm25_search(query: str, k: int = 10) -> list[dict]:
    """Keyword search (BM25) over Wikipedia chunks about the history of computing.

    Best for exact terms: names, dates, product or model names, rare words.
    Weak at paraphrases.

    Args:
        query: Keywords or a short question.
        k: Number of chunks to return (1-50).

    Returns:
        List of {id, title, url, text, score}, best first. `id` is the citation id.
    """
    return _trim(retrieval.bm25_search(query, max(1, min(k, 50))))


@mcp.tool()
def vector_search(query: str, k: int = 10) -> list[dict]:
    """Semantic (embedding) search over Wikipedia chunks about the history of computing.

    Best for meaning and paraphrase, when the question's wording may differ from the text.
    Weak at exact rare names or numbers.

    Args:
        query: A natural-language question or statement.
        k: Number of chunks to return (1-50).

    Returns:
        List of {id, title, url, text, score}, best first. `id` is the citation id.
    """
    return _trim(retrieval.vector_search(query, max(1, min(k, 50))))


@mcp.tool()
def hybrid_search(query: str, k: int = 10) -> list[dict]:
    """Hybrid search: BM25 and vector results fused with Reciprocal Rank Fusion.

    Default choice. Combines exact-term matching with semantic matching.

    Args:
        query: A natural-language question or keywords.
        k: Number of chunks to return (1-50).

    Returns:
        List of {id, title, url, text, score}, best first. `id` is the citation id.
        `score` is the RRF score (only meaningful for ordering).
    """
    return _trim(retrieval.hybrid_search(query, max(1, min(k, 50))))


if __name__ == "__main__":
    mcp.run(transport="stdio")
