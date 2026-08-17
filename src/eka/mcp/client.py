"""Client for the custom "Enterprise Service Desk" MCP server.

The finalize node calls :func:`call_service_desk` every run to write an audit
record via the ``log_interaction`` MCP tool, and — when the answer is low
confidence or the input was rejected — to open a ticket via ``create_ticket``.
All calls go through langchain-mcp-adapters over stdio.
"""
from __future__ import annotations

import asyncio
import json

from ..config import MCP_SERVER_SCRIPT


def _server_config() -> dict:
    return {
        "service_desk": {
            "command": "python",
            "args": [str(MCP_SERVER_SCRIPT)],
            "transport": "stdio",
        }
    }


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

    client = MultiServerMCPClient(_server_config())
    tools = await client.get_tools()
    by_name = {t.name: t for t in tools}

    result: dict = {"escalated": False, "ticket_id": None, "mcp_log": None}
    if "log_interaction" in by_name:
        out = await by_name["log_interaction"].ainvoke(
            {"question": question, "answer": answer, "scores": scores or {}}
        )
        result["mcp_log"] = _parse(out)
    if escalate and "create_ticket" in by_name:
        out = await by_name["create_ticket"].ainvoke(
            {"summary": escalate_summary, "category": "knowledge-gap", "priority": "high"}
        )
        parsed = _parse(out)
        result["escalated"] = True
        result["ticket_id"] = parsed.get("ticket_id")
    return result


def call_service_desk(
    question: str,
    answer: str,
    scores: dict | None = None,
    escalate: bool = False,
    escalate_summary: str = "",
) -> dict:
    """Sync wrapper around the async MCP calls (safe under a running loop)."""
    import nest_asyncio

    nest_asyncio.apply()
    coro = _acall(question, answer, scores, escalate, escalate_summary)
    try:
        return asyncio.run(coro)
    except RuntimeError:
        return asyncio.get_event_loop().run_until_complete(coro)
