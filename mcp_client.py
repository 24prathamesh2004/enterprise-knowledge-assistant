"""Client for the custom Enterprise Service Desk MCP server.

The finalize node calls `call_service_desk` on every run: it always writes an
audit record via the `log_interaction` MCP tool, and — when the answer is low
confidence or the input was rejected — opens a ticket via `create_ticket`.
All calls go over stdio through langchain-mcp-adapters.
"""
from __future__ import annotations

import asyncio
import json

from config import MCP_SERVER_SCRIPT


def _parse(out) -> dict:
    """Normalize an MCP tool result (often a JSON string) to a dict."""
    if isinstance(out, dict):
        return out
    if isinstance(out, str):
        try:
            return json.loads(out)
        except (ValueError, TypeError):
            return {"raw": out}
    return {"raw": str(out)}


async def _acall(question, answer, scores, escalate, escalate_summary) -> dict:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "service_desk": {
                "command": "python",
                "args": [str(MCP_SERVER_SCRIPT)],
                "transport": "stdio",
            }
        }
    )
    tools = {t.name: t for t in await client.get_tools()}

    result = {"escalated": False, "ticket_id": None, "mcp_log": None}
    if "log_interaction" in tools:
        out = await tools["log_interaction"].ainvoke(
            {"question": question, "answer": answer, "scores": scores or {}}
        )
        result["mcp_log"] = _parse(out)
    if escalate and "create_ticket" in tools:
        out = await tools["create_ticket"].ainvoke(
            {"summary": escalate_summary, "category": "knowledge-gap", "priority": "high"}
        )
        result["escalated"] = True
        result["ticket_id"] = _parse(out).get("ticket_id")
    return result


def call_service_desk(question, answer, scores=None, escalate=False, escalate_summary="") -> dict:
    """Sync wrapper around the async MCP calls (safe under a running event loop)."""
    import nest_asyncio

    nest_asyncio.apply()
    coro = _acall(question, answer, scores, escalate, escalate_summary)
    try:
        return asyncio.run(coro)
    except RuntimeError:  # already inside a running loop (Flask / LangGraph)
        return asyncio.get_event_loop().run_until_complete(coro)
