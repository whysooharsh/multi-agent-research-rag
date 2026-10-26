import argparse
import asyncio
import json
import random

from rag import config, retrieval
from pydantic import BaseModel, Field

from rag.llm import get_llm

QFILE = config.ROOT / "data" / "retrieval_questions.json"
RESULTS = config.ROOT / "data" / "retrieval_results.json"
TOP = 10

PROMPT = (
    "Below are {n} numbered passages from Wikipedia. For EACH passage write ONE question that "
    "the passage answers.\n"
    "Rules: paraphrase. Do NOT copy distinctive words or phrases, names, or numbers from a "
    "passage when you can describe them differently; a reader should need to understand the "
    "meaning, not match keywords. Each question must make sense without seeing the passage.\n\n"
    "{passages}"
)
BATCH = 10


class Questions(BaseModel):
    questions: list[str] = Field(description="Exactly one question per passage, in passage order.")


async def generate(n: int) -> list[dict]:
    chunks = [json.loads(line) for line in config.CORPUS_PATH.open(encoding="utf-8")]
    sample = random.Random(0).sample(chunks, min(n, len(chunks)))
    llm = get_llm(0.3).with_structured_output(Questions)
    out = []
    for i in range(0, len(sample), BATCH):
        batch = sample[i : i + BATCH]
        passages = "\n\n".join(f"Passage {j + 1}:\n{c['text']}" for j, c in enumerate(batch))
        res = await llm.ainvoke(PROMPT.format(n=len(batch), passages=passages))
        if len(res.questions) != len(batch):
            raise RuntimeError(f"expected {len(batch)} questions, got {len(res.questions)}")
        out += [
            {"gold_id": c["id"], "title": c["title"], "question": q.strip()}
            for c, q in zip(batch, res.questions)
        ]
        print(f"  generated {len(out)}/{len(sample)}", flush=True)
    return out


def score(search, questions: list[dict]) -> dict:
    hits5, rr = 0, 0.0
    for q in questions:
        ids = [r["id"] for r in search(q["question"], TOP)]
        if q["gold_id"] in ids[:5]:
            hits5 += 1
        if q["gold_id"] in ids:
            rr += 1.0 / (ids.index(q["gold_id"]) + 1)
    n = len(questions)
    return {"hit@5": hits5 / n, "mrr@10": rr / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    if QFILE.exists() and not args.regen:
        questions = json.loads(QFILE.read_text())
    else:
        questions = asyncio.run(generate(args.n))
        QFILE.write_text(json.dumps(questions, indent=1))
    print(f"{len(questions)} synthetic questions")

    modes = {
        "bm25": retrieval.bm25_search,
        "vector": retrieval.vector_search,
        "hybrid": retrieval.hybrid_search,
    }
    results = {name: score(fn, questions) for name, fn in modes.items()}
    RESULTS.write_text(json.dumps(results, indent=1))

    print(f"\n{'mode':<8} {'hit@5':>7} {'MRR@10':>8}")
    for name, r in results.items():
        print(f"{name:<8} {r['hit@5']:>7.3f} {r['mrr@10']:>8.3f}")


if __name__ == "__main__":
    main()
