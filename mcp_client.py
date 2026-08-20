"""MCP client for the Enterprise Service Desk."""

import asyncio
import json
import nest_asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from config import MCP_SERVER_SCRIPT

nest_asyncio.apply()


def parse_mcp_result(result) -> dict:
    """Convert MCP tool result to dict."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (ValueError, TypeError):
            return {"raw": result}
    return {"raw": str(result)}


async def call_mcp_tools(question: str, answer: str, scores: dict, escalate: bool, summary: str) -> dict:
    """Call MCP service desk tools."""
    client = MultiServerMCPClient({
        "service_desk": {
            "command": "python",
            "args": [str(MCP_SERVER_SCRIPT)],
            "transport": "stdio",
        }
    })
    
    tools = {t.name: t for t in await client.get_tools()}
    result = {"escalated": False, "ticket_id": None, "mcp_log": None}
    
    # Always log the interaction
    if "log_interaction" in tools:
        output = await tools["log_interaction"].ainvoke({
            "question": question,
            "answer": answer,
            "scores": scores or {},
        })
        result["mcp_log"] = parse_mcp_result(output)
    
    # Escalate if needed
    if escalate and "create_ticket" in tools:
        output = await tools["create_ticket"].ainvoke({
            "summary": summary,
            "category": "knowledge-gap",
            "priority": "high",
        })
        ticket_data = parse_mcp_result(output)
        result["escalated"] = True
        result["ticket_id"] = ticket_data.get("ticket_id")
    
    return result


def call_service_desk(question: str, answer: str, scores: dict = None,
                     escalate: bool = False, escalate_summary: str = "") -> dict:
    """Sync wrapper for MCP service desk calls."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            call_mcp_tools(question, answer, scores, escalate, escalate_summary)
        )
    finally:
        loop.close()


def as_list(result) -> list:
    """MCP list tools hand back JSON text; turn it into a list."""
    if isinstance(result, list):
        return result
    if isinstance(result, str):
        try:
            data = json.loads(result)
            return data if isinstance(data, list) else []
        except (ValueError, TypeError):
            return []
    return []


async def read_service_desk_stats() -> dict:
    """Count open tickets and logged interactions via the MCP read tools."""
    client = MultiServerMCPClient({
        "service_desk": {
            "command": "python",
            "args": [str(MCP_SERVER_SCRIPT)],
            "transport": "stdio",
        }
    })

    tools = {t.name: t for t in await client.get_tools()}
    open_tickets = interactions = 0

    if "get_recent_tickets" in tools:
        rows = await tools["get_recent_tickets"].ainvoke({"limit": 100, "status": "open"})
        open_tickets = len(as_list(rows))
    if "get_recent_interactions" in tools:
        rows = await tools["get_recent_interactions"].ainvoke({"limit": 100})
        interactions = len(as_list(rows))

    return {"open_tickets": open_tickets, "interactions": interactions}


def service_desk_stats() -> dict:
    """Sync wrapper for the /stats command."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(read_service_desk_stats())
    finally:
        loop.close()
