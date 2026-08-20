"""MCP server for Enterprise Service Desk - provides audit and ticketing tools."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).resolve().parent / "service_desk.db"
mcp = FastMCP("service-desk")


def get_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database tables if they don't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                category TEXT,
                priority TEXT,
                status TEXT,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit (
                audit_id TEXT PRIMARY KEY,
                question TEXT,
                answer TEXT,
                scores TEXT,
                created_at TEXT
            )
        """)


def get_timestamp() -> str:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
def create_ticket(summary: str, category: str = "general", priority: str = "normal") -> dict:
    """Create a support ticket for human review."""
    ticket_id = "TICK-" + uuid.uuid4().hex[:8].upper()
    
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?)",
            (ticket_id, summary, category, priority, "open", get_timestamp()),
        )
    
    return {"ticket_id": ticket_id, "status": "open"}


@mcp.tool()
def log_interaction(question: str, answer: str, scores: dict = None) -> dict:
    """Log an assistant interaction for audit trail."""
    audit_id = "AUD-" + uuid.uuid4().hex[:8].upper()
    
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
            (audit_id, question, answer, json.dumps(scores or {}), get_timestamp()),
        )
    
    return {"audit_id": audit_id}


@mcp.tool()
def get_recent_tickets(limit: int = 10, status: str = "open") -> list[dict]:
    """Retrieve recent tickets for monitoring."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM tickets WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    
    return [dict(row) for row in rows]


@mcp.tool()
def get_recent_interactions(limit: int = 5) -> list[dict]:
    """Get recent interactions for conversation history."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT question, answer, scores, created_at FROM audit "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    
    return [dict(row) for row in rows]


if __name__ == "__main__":
    init_database()
    mcp.run()
