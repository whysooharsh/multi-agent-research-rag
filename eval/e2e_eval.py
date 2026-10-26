import argparse
import asyncio
import json
import random
import re
from collections import defaultdict

from rag import config, mcp_client
from rag.agent import Critique, build_graph
from rag.llm import get_llm, text_of

QFILE = config.ROOT / "data" / "e2e_questions.json"
RESULTS = config.ROOT / "data" / "e2e_results.json"


async def generate(n: int) -> list[str]:
    by_title = defaultdict(list)
    for line in config.CORPUS_PATH.open(encoding="utf-8"):
        c = json.loads(line)
        by_title[c["title"]].append(c)
    titles = [t for t, cs in by_title.items() if len(cs) >= 2]
    rng = random.Random(1)
    llm = get_llm(0.3)
    questions = []
    for t in rng.sample(titles, n):
        a, b = rng.sample(by_title[t], 2)
        msg = await llm.ainvoke(
            "Write ONE research question that needs information from BOTH passages below to "
            "answer well. Don't copy distinctive phrases. It must make sense on its own. "
            f"Output only the question.\n\nPassage 1:\n{a['text']}\n\nPassage 2:\n{b['text']}"
        )
        questions.append(text_of(msg).strip())
    return questions


async def judge(answer: str, chunks: list[dict]) -> float:
    by_id = {c["id"]: c for c in chunks}
    cited = sorted({int(i) for i in re.findall(r"\[(\d+)\]", answer)} & by_id.keys())
    ctx = "\n\n".join(f"[{i}] {by_id[i]['text']}" for i in cited)
    llm = get_llm()
    res = await llm.with_structured_output(Critique).ainvoke(
        "Judge this answer. List each factual claim; mark it supported only if the chunk(s) it "
        f"cites explicitly state it. Uncited claims are unsupported.\n\nChunks:\n{ctx}\n\nAnswer:\n{answer}"
    )
    if not res.claims:
        return 0.0
    ok = sum(
        c.supported and bool(c.cited_ids) and all(i in by_id for i in c.cited_ids)
        for c in res.claims
    )
    return ok / len(res.claims)


async def run_mode(graph, questions: list[str], tool: str) -> list[dict]:
    rows = []
    for i, q in enumerate(questions):
        try:
            cfg = {"configurable": {"retrieval_tool": tool}, "recursion_limit": 25}
            st = await graph.ainvoke(
                {"question": q, "retrieved": [], "critique": {}, "draft": ""}, cfg
            )
            g = await judge(st["draft"], st["chunks"])
            rows.append(
                {
                    "question": q,
                    "judge_groundedness": g,
                    "critic_groundedness": st["groundedness"],
                    "drafts": st["loops"],
                }
            )
        except Exception as e:
            rows.append({"question": q, "error": f"{type(e).__name__}: {e}"[:200]})
        print(f"  [{tool}] {i + 1}/{len(questions)}", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    return {
        "answered": n,
        "errors": len(rows) - n,
        "mean_judge_groundedness": sum(r["judge_groundedness"] for r in ok) / n
        if n
        else None,
        "mean_critic_groundedness": sum(r["critic_groundedness"] for r in ok) / n
        if n
        else None,
        "mean_drafts": sum(r["drafts"] for r in ok) / n if n else None,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    if QFILE.exists() and not args.regen:
        questions = json.loads(QFILE.read_text())
    else:
        questions = await generate(args.n)
        QFILE.write_text(json.dumps(questions, indent=1))

    await mcp_client.start()  # must happen in the main task, not inside parallel nodes
    graph = build_graph()
    out = {}
    try:
        for name, tool in [("hybrid", "hybrid_search"), ("bm25", "bm25_search")]:
            rows = await run_mode(graph, questions, tool)
            out[name] = {"summary": summarize(rows), "rows": rows}
    finally:
        await mcp_client.stop()
    RESULTS.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v["summary"] for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
