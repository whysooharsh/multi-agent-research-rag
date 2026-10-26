import json
import random

from rag import config, retrieval

QFILE = config.ROOT / "data" / "retrieval_questions.json"
OUT = config.ROOT / "data" / "retrieval_ranks.json"


def rank_of(fn, question: str, gold: int):
    ids = [r["id"] for r in fn(question, 10)]
    return ids.index(gold) + 1 if gold in ids else None


def bootstrap_ci(diffs: list[float], n_boot: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    qs = json.loads(QFILE.read_text())
    modes = {
        "bm25": retrieval.bm25_search,
        "vector": retrieval.vector_search,
        "hybrid": retrieval.hybrid_search,
    }
    ranks = {
        m: [rank_of(f, q["question"], q["gold_id"]) for q in qs]
        for m, f in modes.items()
    }
    OUT.write_text(json.dumps(ranks))

    hit5 = lambda r: [1 if v and v <= 5 else 0 for v in r]
    rr = lambda r: [1 / v if v else 0.0 for v in r]
    n = len(qs)
    for m, r in ranks.items():
        print(
            f"{m:<7} hit@5={sum(hit5(r)) / n:.3f} mrr@10={sum(rr(r)) / n:.3f} not_in_top10={sum(v is None for v in r)}"
        )

    h, b = hit5(ranks["hybrid"]), hit5(ranks["bm25"])
    print(
        f"hybrid-only hits={sum(x and not y for x, y in zip(h, b))} "
        f"bm25-only hits={sum(y and not x for x, y in zip(h, b))} both={sum(x and y for x, y in zip(h, b))}"
    )
    lo, hi = bootstrap_ci([x - y for x, y in zip(h, b)])
    print(f"hit@5 hybrid-bm25 95% CI: [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = bootstrap_ci(
        [x - y for x, y in zip(rr(ranks["hybrid"]), rr(ranks["bm25"]))]
    )
    print(f"MRR@10 hybrid-bm25 95% CI: [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
