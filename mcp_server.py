"""Custom MCP server — Northwind "Enterprise Service Desk".

Exposes two tools over stdio, backed by a local SQLite database:
  - log_interaction: audit every assistant answer (called on every run)
  - create_ticket:   escalate a low-confidence / rejected query to a human

Runs standalone (`python mcp_server.py`) and imports nothing from the project so
it works when the client spawns it as a subprocess.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).resolve().parent / "service_desk.db"
mcp = FastMCP("service-desk")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY, summary TEXT NOT NULL, category TEXT,
                priority TEXT, status TEXT, created_at TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS audit (
                audit_id TEXT PRIMARY KEY, question TEXT, answer TEXT,
                scores TEXT, created_at TEXT
            )"""
        )


@mcp.tool()
def create_ticket(summary: str, category: str = "general", priority: str = "normal") -> dict:
    """Open a support ticket for a human agent. Returns the new ticket id + status."""
    ticket_id = "TICK-" + uuid.uuid4().hex[:8].upper()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?)",
            (ticket_id, summary, category, priority, "open", _now()),
        )
    return {"ticket_id": ticket_id, "status": "open"}


@mcp.tool()
def log_interaction(question: str, answer: str, scores: dict | None = None) -> dict:
    """Append an audit record of one assistant interaction. Returns the audit id."""
    audit_id = "AUD-" + uuid.uuid4().hex[:8].upper()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
            (audit_id, question, answer, json.dumps(scores or {}), _now()),
        )
    return {"audit_id": audit_id}


_init_db()

if __name__ == "__main__":
    mcp.run()
