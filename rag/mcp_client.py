import json
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_stack: AsyncExitStack | None = None
_session: ClientSession | None = None


async def start() -> None:
    global _stack, _session
    if _session is not None:
        return
    _stack = AsyncExitStack()
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "rag.mcp_server"]
    )
    read, write = await _stack.enter_async_context(stdio_client(params))
    _session = await _stack.enter_async_context(ClientSession(read, write))
    await _session.initialize()


async def stop() -> None:
    global _stack, _session
    if _stack is not None:
        await _stack.aclose()
    _stack, _session = None, None


async def call_tool(name: str, **args) -> list[dict]:
    if _session is None:
        raise RuntimeError("call `await mcp_client.start()` from the main task before running the graph")
    res = await _session.call_tool(name, args)
    if res.isError:
        raise RuntimeError(f"MCP tool {name} failed: {res.content}")
    return [json.loads(c.text) for c in res.content if c.type == "text"]
