import argparse
import json

import psycopg
from pgvector.psycopg import register_vector

from rag import config
from rag.retrieval import embed_documents

BATCH = 64

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    title     TEXT NOT NULL,
    url       TEXT NOT NULL,
    text      TEXT NOT NULL,
    embedding vector(384) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

UPSERT = """
INSERT INTO chunks (id, title, url, text, embedding) VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title, url = EXCLUDED.url,
    text = EXCLUDED.text, embedding = EXCLUDED.embedding
"""


def read_batches():
    batch = []
    with config.CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) == BATCH:
                yield batch
                batch = []
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="empty the chunks table first")
    args = ap.parse_args()
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        conn.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )  # must exist before register_vector
        register_vector(conn)
        conn.execute(SCHEMA)
        if args.reset:
            conn.execute("TRUNCATE chunks")
        done = 0
        for batch in read_batches():
            vecs = embed_documents(
                [c["text"] for c in batch]
            )
            rows = [
                (c["id"], c["title"], c["url"], c["text"], v)
                for c, v in zip(batch, vecs)
            ]
            with conn.cursor() as cur:
                cur.executemany(UPSERT, rows)
            done += len(batch)
            print(f"  upserted {done}", flush=True)
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"done. chunks in table: {n}")


if __name__ == "__main__":
    main()
