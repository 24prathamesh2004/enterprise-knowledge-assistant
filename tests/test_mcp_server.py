"""Custom MCP service-desk server tests against a temp SQLite DB."""
import importlib

import pytest


@pytest.fixture()
def server(tmp_path, monkeypatch):
    mod = importlib.import_module("mcp_server.server")
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "test_desk.db")
    mod._init_db()
    return mod


def _fn(tool):
    # FastMCP may return the bare function or a wrapper exposing `.fn`.
    return getattr(tool, "fn", tool)


def test_create_and_get_ticket(server):
    created = _fn(server.create_ticket)(
        summary="Cannot find leave carry-over rule", category="knowledge-gap", priority="high"
    )
    assert created["ticket_id"].startswith("TICK-")
    assert created["status"] == "open"

    fetched = _fn(server.get_ticket)(created["ticket_id"])
    assert fetched["summary"] == "Cannot find leave carry-over rule"
    assert fetched["priority"] == "high"


def test_list_tickets(server):
    _fn(server.create_ticket)(summary="one")
    _fn(server.create_ticket)(summary="two")
    open_tickets = _fn(server.list_tickets)(status="open")
    assert len(open_tickets) == 2


def test_get_missing_ticket(server):
    res = _fn(server.get_ticket)("TICK-NOPE")
    assert "error" in res


def test_log_interaction_writes_audit(server):
    res = _fn(server.log_interaction)(
        question="How many leave days?", answer="22 days.", scores={"faithfulness": 0.9}
    )
    assert res["audit_id"].startswith("AUD-")
