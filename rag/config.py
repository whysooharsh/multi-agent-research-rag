import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", ROOT / "data" / "corpus.jsonl"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")
GO_SEARCH_URL = os.getenv("GO_SEARCH_URL", "http://localhost:8000")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
# bge models want this prefix on QUERIES only, never on documents.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

FETCH_K = 40
FINAL_K = 10
RRF_K = 60

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
MAX_CRITIC_LOOPS = 2
