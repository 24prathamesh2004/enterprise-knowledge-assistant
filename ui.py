"""FastAPI UI for the Enterprise Knowledge Assistant."""

import math
import traceback
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from graph import build_graph
from mcp_client import service_desk_stats

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="LangGraph-based assistant with RAG, RAGAS evaluation, and MCP integration",
    version="1.0.0",
)

graph_instance = None


def get_graph():
    """Lazy load graph on first request."""
    global graph_instance
    if graph_instance is None:
        graph_instance = build_graph()
    return graph_instance


def json_safe(obj):
    """Replace NaN with None for valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class ExportRequest(BaseModel):
    messages: list
    session_title: str
    session_date: str


@app.post("/export-pdf")
async def export_pdf(request: ExportRequest):
    """Generate PDF export of chat session."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name

        doc = SimpleDocTemplate(tmp_path, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], spaceAfter=30)
        user_style = ParagraphStyle('User', parent=styles['Normal'], leftIndent=20, rightIndent=0, textColor='blue')
        assistant_style = ParagraphStyle('Assistant', parent=styles['Normal'], leftIndent=0, rightIndent=20, textColor='black')

        story = [
            Paragraph("Enterprise Knowledge Assistant - Chat Export", title_style),
            Paragraph(f"<b>Session:</b> {request.session_title}", styles['Normal']),
            Paragraph(f"<b>Date:</b> {request.session_date}", styles['Normal']),
            Paragraph(f"<b>Messages:</b> {len(request.messages)}", styles['Normal']),
            Spacer(1, 20),
        ]

        for message in request.messages:
            sender = "You" if message['type'] == 'user' else "Assistant"
            story.append(Paragraph(f"<b>{sender} ({message['timestamp']}):</b>", styles['Heading3']))

            style = user_style if message['type'] == 'user' else assistant_style
            story.append(Paragraph(message['content'], style))

            if message.get('metrics'):
                metrics_text = ""
                if message['metrics'].get('faithfulness'):
                    metrics_text += f"Faithfulness: {message['metrics']['faithfulness']:.2f} "
                if message['metrics'].get('answer_relevancy'):
                    metrics_text += f"Relevancy: {message['metrics']['answer_relevancy']:.2f}"
                if metrics_text:
                    story.append(Paragraph(f"<i>Metrics: {metrics_text}</i>", styles['Normal']))

            if message.get('escalation'):
                story.append(Paragraph(f"<i>Escalated - Ticket: {message['escalation']}</i>", styles['Normal']))

            story.append(Spacer(1, 15))

        doc.build(story)

        return FileResponse(
            tmp_path,
            media_type='application/pdf',
            filename=f"chat_export_{request.session_date.replace('/', '-')}.pdf",
            background=BackgroundTask(os.unlink, tmp_path),
        )

    except ImportError:
        return JSONResponse(
            status_code=501,
            content={"error": "PDF generation not available. Install reportlab: pip install reportlab"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"PDF generation failed: {str(e)}"}
        )


class AskResponse(BaseModel):
    answer: str
    evaluation_summary: str
    scores: dict
    escalated: bool
    ticket_id: Optional[str]
    trace: list[str]


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the UI page."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))


@app.get("/stats")
async def stats():
    """Live service-desk counts for the /stats command."""
    try:
        return service_desk_stats()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask", response_model=AskResponse)
async def ask(request: QuestionRequest):
    """Process a question through the assistant."""
    try:
        question = request.question.strip()
        graph = get_graph()
        result = graph.invoke({"question": question, "session_id": request.session_id})
        
        return json_safe({
            "answer": result.get("answer", ""),
            "evaluation_summary": result.get("evaluation_summary", ""),
            "scores": result.get("ragas_scores", {}),
            "escalated": result.get("escalated", False),
            "ticket_id": result.get("ticket_id"),
            "trace": [e.get("node") for e in result.get("trace", [])],
        })
    except Exception as e:
        error_detail = f"{type(e).__name__}: {str(e)}"
        print(f"Error in /ask endpoint: {error_detail}")
        print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "answer": "An error occurred while processing your request.",
                "evaluation_summary": f"Error: {error_detail}",
                "scores": {},
                "escalated": False,
                "ticket_id": None,
                "trace": ["error"],
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
