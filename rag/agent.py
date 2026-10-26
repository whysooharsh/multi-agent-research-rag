import operator
import re
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from rag import config, mcp_client
from rag.llm import get_llm, text_of

MAX_CONTEXT_CHUNKS = 12


class State(TypedDict):
    question: str
    sub_queries: list[str]
    retrieved: Annotated[list[dict], operator.add]
    chunks: list[dict]
    draft: str
    critique: dict
    loops: int
    groundedness: float


class Plan(BaseModel):
    sub_queries: list[str] = Field(
        description="2-4 self-contained search queries that together cover the question.",
        min_length=1,
        max_length=4,
    )


class ClaimCheck(BaseModel):
    claim: str = Field(description="One factual claim from the answer.")
    cited_ids: list[int] = Field(description="Chunk ids cited for this claim.")
    supported: bool = Field(
        description="True only if the cited chunks explicitly support the claim."
    )
    reason: str = Field(description="One short sentence.")


class Critique(BaseModel):
    claims: list[ClaimCheck]


def _llm():
    return get_llm()


def _format_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['id']}] ({c['title']})\n{c['text']}" for c in chunks)


async def planner(state: State) -> dict:
    plan = (
        await _llm()
        .with_structured_output(Plan)
        .ainvoke(
            "Break this research question into 2-4 independent search queries for a Wikipedia "
            "corpus about the history of computing. Each query must make sense on its own.\n\n"
            f"Question: {state['question']}"
        )
    )
    return {"sub_queries": plan.sub_queries[:4], "loops": 0}


def fan_out(state: State, config: RunnableConfig):
    tool = config.get("configurable", {}).get("retrieval_tool", "hybrid_search")
    return [
        Send("retriever", {"sub_query": q, "tool": tool}) for q in state["sub_queries"]
    ]


async def retriever(payload: dict) -> dict:
    results = await mcp_client.call_tool(
        payload["tool"], query=payload["sub_query"], k=8
    )
    return {"retrieved": results}


async def collect(state: State) -> dict:
    hits: dict[int, int] = {}
    first: dict[int, dict] = {}
    for c in state["retrieved"]:
        hits[c["id"]] = hits.get(c["id"], 0) + 1
        first.setdefault(c["id"], c)
    order = sorted(
        first, key=lambda i: -hits[i]
    )
    return {"chunks": [first[i] for i in order[:MAX_CONTEXT_CHUNKS]]}


async def synthesizer(state: State) -> dict:
    prompt = (
        "Answer the question using ONLY the sources below. After every sentence containing a "
        "factual claim, cite the supporting chunk ids like [12] or [12][40]. Do not cite ids "
        "that are not listed. If the sources do not contain the answer, say so plainly.\n\n"
        f"Sources:\n{_format_chunks(state['chunks'])}\n\nQuestion: {state['question']}"
    )
    bad = [c for c in state.get("critique", {}).get("claims", []) if not c["supported"]]
    if bad and state.get("draft"):
        feedback = "\n".join(f'- "{c["claim"]}": {c["reason"]}' for c in bad)
        prompt += (
            f"\n\nYour previous draft:\n{state['draft']}\n\nA reviewer found these claims "
            f"unsupported by their citations:\n{feedback}\n"
            "Rewrite the answer: remove or fix those claims, and keep only what the sources support."
        )
    msg = await _llm().ainvoke(prompt)
    return {
        "draft": text_of(msg),
        "loops": state.get("loops", 0) + 1,
    }


CITE = re.compile(r"\[(\d+)\]")


async def critic(state: State) -> dict:
    by_id = {c["id"]: c for c in state["chunks"]}
    cited = sorted({int(i) for i in CITE.findall(state["draft"])} & by_id.keys())
    result = (
        await _llm()
        .with_structured_output(Critique)
        .ainvoke(
            "You are a strict fact-checker. Split the answer into individual factual claims. For each "
            "claim, look ONLY at the chunks it cites and decide if they explicitly support it. "
            "A claim with no citation, or a citation to a chunk that does not say it, is unsupported.\n\n"
            f"Cited chunks:\n{_format_chunks([by_id[i] for i in cited])}\n\nAnswer:\n{state['draft']}"
        )
    )
    claims = []
    for c in result.claims:
        d = c.model_dump()
        if not d["cited_ids"] or any(i not in by_id for i in d["cited_ids"]):
            d["supported"] = False
            d["reason"] = d["reason"] or "missing or invalid citation"
        claims.append(d)
    score = sum(c["supported"] for c in claims) / len(claims) if claims else 0.0
    return {"critique": {"claims": claims}, "groundedness": score}


def after_critic(state: State):
    all_ok = all(c["supported"] for c in state["critique"]["claims"])
    if all_ok or state["loops"] > config.MAX_CRITIC_LOOPS:
        return END
    return "synthesizer"


def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("retriever", retriever)
    g.add_node("collect", collect)
    g.add_node("synthesizer", synthesizer)
    g.add_node("critic", critic)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["retriever"])
    g.add_edge("retriever", "collect")
    g.add_edge("collect", "synthesizer")
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", after_critic, ["synthesizer", END])
    return g.compile(checkpointer=checkpointer)
