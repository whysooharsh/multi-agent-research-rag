import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field

from rag import config, mcp_client
from rag.agent import build_graph
from sse_starlette.sse import EventSourceResponse

RUN_TIMEOUT_S = 240


@asynccontextmanager
async def lifespan(app: FastAPI):
    await mcp_client.start()
    async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as saver:
        await saver.setup()
        app.state.graph = build_graph(saver)
        yield
    await mcp_client.stop()


app = FastAPI(title="Research Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    thread_id: str | None = None
    retrieval: Literal["hybrid", "bm25", "vector"] = "hybrid"


def _summarize(node: str, update: dict) -> dict:
    if node == "planner":
        return {"sub_queries": update["sub_queries"]}
    if node == "retriever":
        return {"chunks_found": len(update["retrieved"])}
    if node == "collect":
        return {"unique_chunks": len(update["chunks"])}
    if node == "synthesizer":
        return {"draft_number": update["loops"]}
    if node == "critic":
        bad = [c["claim"] for c in update["critique"]["claims"] if not c["supported"]]
        return {"groundedness": update["groundedness"], "unsupported_claims": bad}
    return {}


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


async def _run_stream(graph, inputs, cfg, thread_id: str):
    yield _sse("start", {"thread_id": thread_id})
    try:
        async with asyncio.timeout(RUN_TIMEOUT_S):
            async for chunk in graph.astream(inputs, cfg, stream_mode="updates"):
                for node, update in chunk.items():
                    if update:
                        yield _sse("node", {"node": node, **_summarize(node, update)})
            state = (await graph.aget_state(cfg)).values
    except TimeoutError:
        yield _sse(
            "error",
            {"message": "run timed out; resume with POST /runs/{thread_id}/resume"},
        )
        return
    except (
        Exception
    ) as e:
        yield _sse("error", {"message": f"{type(e).__name__}: {e}"[:500]})
        return

    cited = {int(i) for i in re.findall(r"\[(\d+)\]", state.get("draft", ""))}
    sources = [
        {"id": c["id"], "title": c["title"], "url": c["url"]}
        for c in state.get("chunks", [])
        if c["id"] in cited
    ]
    yield _sse(
        "answer",
        {
            "answer": state.get("draft", ""),
            "groundedness": state.get("groundedness"),
            "drafts": state.get("loops"),
            "sources": sources,
        },
    )


@app.post("/ask")
async def ask(req: AskRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    cfg = {
        "configurable": {
            "thread_id": thread_id,
            "retrieval_tool": f"{req.retrieval}_search",
        },
        "recursion_limit": 25,
    }
    inputs = {"question": req.question, "retrieved": [], "critique": {}, "draft": ""}
    return EventSourceResponse(_run_stream(app.state.graph, inputs, cfg, thread_id))


@app.get("/runs/{thread_id}")
async def run_status(thread_id: str):
    snap = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.created_at:
        raise HTTPException(404, "unknown thread_id")
    v = snap.values
    return {
        "thread_id": thread_id,
        "status": "running_or_interrupted" if snap.next else "finished",
        "next": list(snap.next),
        "question": v.get("question"),
        "drafts": v.get("loops"),
        "groundedness": v.get("groundedness"),
        "answer": v.get("draft"),
    }


@app.post("/runs/{thread_id}/resume")
async def resume(thread_id: str):
    cfg = {
        "configurable": {"thread_id": thread_id, "retrieval_tool": "hybrid_search"},
        "recursion_limit": 25,
    }
    snap = await app.state.graph.aget_state(cfg)
    if not snap.created_at:
        raise HTTPException(404, "unknown thread_id")
    if not snap.next:
        raise HTTPException(409, "run already finished")
    return EventSourceResponse(_run_stream(app.state.graph, None, cfg, thread_id))


@app.get("/health")
async def health():
    return {"ok": True}
