# Enterprise Knowledge Assistant — Implementation Plan

> Paste-ready for docs/PLAN.md. In plan mode this document lives at the plan path; on approval, the first implementation commit copies it to docs/PLAN.md.

## Context

Why this exists: L3 Agentic AI submission — an Enterprise Knowledge Assistant on LangGraph that Retrieves → Responds → Evaluates (RAGAS) and actively uses an MCP server, with end-to-end observability. Employees waste time hunting facts across disconnected systems; this assistant answers from a governed knowledge base, self-scores answer quality, and escalates to a ticketing system when it cannot answer confidently.

Grading weight (design to it): LangGraph 30% · RAG 25% · RAGAS 20% · MCP 15% · Docs 10%.

Starting point: repo is an empty scaffold (assignment PDF + git only).

Confirmed decisions (clarifying questions resolved):
- LLM: Anthropic Claude for generation and the RAGAS judge.
- Embeddings: local HuggingFace sentence-transformers — Anthropic has no embeddings API, so eval + vector store share one local model. Net: one cloud key (ANTHROPIC_API_KEY).
- Knowledge source: synthetic enterprise policy set + an authored golden eval set.
- MCP: a custom Python MCP server ("Enterprise Service Desk": tickets + audit log).
- Scope: FULL — conditional retry edge (bonus) + context precision/recall (bonus) + guardrails (innovation) + minimal HTML/CSS UI.

Outcome: a runnable, well-documented project maximizing every rubric line + bonus, with a guardrails narrative, and no gratuitous complexity.

### Your architecture questions, answered
- Multi-agent? Yes in the rubric's sense: three specialized nodes (Retriever, Responder, Evaluator) are the required "agents". It is a structured node graph, not autonomous negotiating agents — exactly what the PDF asks for ("keep the flow linear").
- Supervisor agent? No. A supervisor/router contradicts "keep the flow linear" and adds complexity for zero rubric credit. The only branch we add is the PDF-blessed bonus conditional edge (retry retrieval on low scores) — a plain edge function, no supervisor.
- Guardrails (innovation)? Yes, and wired into the graph: an input guard (empty/oversized/prompt-injection/out-of-scope) plus an output groundedness guard that, together with RAGAS scores, drives the retry and an MCP escalation (auto-file a ticket when confidence is low). High marks, low complexity.

## 1. Tech stack decision & version compatibility

Pin every version in requirements.txt. I could not fetch live PyPI versions this session (network classifier was temporarily down), so treat the numbers below as known-good baselines and run `pip index versions <pkg>` to confirm the latest compatible pins before locking.

| Concern | Choice | Why (local/demo scale) |
|---|---|---|
| Orchestration | LangGraph (pin latest 0.3+/1.x) + langchain-core | Required by the PDF; StateGraph gives explicit nodes/edges + built-in tracing. |
| Generation LLM | Anthropic Claude via langchain-anthropic (ChatAnthropic) — claude-haiku-4-5 default, configurable to claude-sonnet-5 | Fast/cheap, strong instruction-following → reliable grounded answers and clean RAGAS judge JSON. |
| RAGAS judge LLM | Same Claude (Sonnet for stricter judging, configurable) | One provider/key; Claude JSON reliability avoids the parse failures small local models cause. |
| Embeddings | HuggingFace sentence-transformers (BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2) via langchain-huggingface | Anthropic has NO embeddings API; local model = free, offline, one less key. Used by BOTH the vector store and RAGAS answer-relevancy. |
| Vector store | Chroma (persistent) via langchain-chroma | Zero external service, local persistence, metadata filtering, easy to inspect for screenshots. FAISS is the fallback. |
| Evaluation | RAGAS (pin 0.2.x; class-based metrics API) | Required; supports the 2 mandatory + 2 bonus metrics with Langchain wrappers. |
| MCP | Official mcp Python SDK (FastMCP) for the custom server + langchain-mcp-adapters (MultiServerMCPClient) client | Custom server = top MCP marks; adapters expose MCP tools to the node as LangChain tools. |
| Observability | LangSmith env tracing + a manual structured node logger | Belt-and-suspenders: guaranteed log screenshot even with no LangSmith key. |
| UI | Flask + one HTML template + one CSS file | "Simple HTML & CSS" per your ask; Flask is a tiny dependency. |
| Tests | pytest (+ a FakeLLM/FakeEmbeddings for deterministic runs) | Standard; enables offline smoke tests. |

Compatibility flags to watch:
- Anthropic ≠ embeddings: must pair Claude with HF (or Voyage) embeddings — handled above.
- RAGAS constrains langchain-core: install ragas first, then add langchain-anthropic/-huggingface/-chroma compatible with that core; pin the resolved set.
- asyncio on Windows: RAGAS and MCP (stdio) are async. Use nest_asyncio; run the graph off the Flask request thread (worker thread) or via asyncio.run to avoid "event loop already running".
- MCP stdio subprocess: the client spawns the server with `python mcp_server/server.py`; ensure python is on PATH. langchain-mcp-adapters tools are async → call inside the node's loop.
- HF first run downloads model weights (~90–130MB) → note offline caveat; pick a small model. Use the CPU-only torch wheel to keep install light.
- Pydantic v2 across the stack (LangGraph/LangChain/RAGAS/FastMCP) — avoid any v1 pins.

## 2. LangGraph state schema

Single shared `TypedDict` (LangGraph-idiomatic; a reducer only where we append). In `src/eka/graph/state.py`:

```python
from typing import TypedDict, Annotated, Optional
import operator

class GraphState(TypedDict, total=False):
    question: str                  # INPUT
    reference: Optional[str]        # golden-set ground truth; enables bonus metrics
    is_valid: bool                 # input guard verdict
    guardrail_reason: Optional[str]
    top_k: int                     # retrieval breadth (bumped on retry)
    retrieved_contexts: list[str]  # chunk texts (RAGAS "contexts")
    retrieved_meta: list[dict]     # {source, score} per chunk
    answer: str                    # generated answer
    ragas_scores: dict             # {faithfulness, answer_relevancy, ...}
    evaluation_summary: str        # human-readable interpretation
    retry_count: int               # bounds the conditional retry
    escalated: bool                # MCP ticket filed?
    ticket_id: Optional[str]
    mcp_log: Optional[dict]        # MCP audit result
    error: Optional[str]
    trace: Annotated[list[dict], operator.add]  # manual node log (append reducer)
```

Read/write matrix:
| Node | Reads | Writes |
|---|---|---|
| guard_input | question | is_valid, guardrail_reason, top_k, retry_count |
| retrieve | question, top_k | retrieved_contexts, retrieved_meta |
| generate | question, retrieved_contexts | answer |
| evaluate | question, answer, retrieved_contexts, reference | ragas_scores, evaluation_summary |
| decide (edge) | ragas_scores, retry_count | (routes only) |
| finalize (MCP) | question, answer, ragas_scores, is_valid | escalated, ticket_id, mcp_log |
| (all) | — | trace (append) |

## 3. Node-by-node spec (graph flow)

Flow: `guard_input → retrieve → generate → evaluate → [decide] → {retrieve (retry) | finalize} → END`. The core rubric spine is retrieve → generate → evaluate; guard/finalize/decide are the innovation + bonus.

**guard_input** (innovation)
- Purpose: reject empty/oversized input, obvious prompt-injection ("ignore previous instructions"), and clearly out-of-scope asks; initialize top_k and retry_count.
- In: question. Out: is_valid, guardrail_reason, top_k=4, retry_count=0.
- External calls: none (pure rules + light heuristics). Error handling: on invalid, set is_valid=False + reason; graph short-circuits to finalize with a safe refusal.
- Logs: verdict + reason.

**retrieve** (Retriever Agent — required)
- Purpose: embed the query, similarity-search Chroma for top_k chunks.
- In: question, top_k. Out: retrieved_contexts (texts), retrieved_meta ({source, score}).
- External: HF embeddings + Chroma. Error handling: empty result → flag so generate produces the "insufficient context" refusal; exceptions captured to state.error.
- Logs: query, top_k, #chunks, top source + score.

**generate** (Response Agent — required)
- Purpose: Claude answers strictly from retrieved_contexts (grounded prompt: cite sources; if context is insufficient, say so — do not invent).
- In: question, retrieved_contexts. Out: answer.
- External: ChatAnthropic. Error handling: API error → retry w/ backoff (tenacity), then state.error + safe message.
- Logs: prompt token estimate, answer preview.

**evaluate** (Evaluator Agent — required)
- Purpose: score the answer with RAGAS and write an interpretation.
- In: question, answer, retrieved_contexts, reference. Out: ragas_scores {faithfulness, answer_relevancy [+ context_precision, context_recall if reference]}, evaluation_summary.
- External: RAGAS with Claude judge + HF embeddings. Error handling: wrap in try/except + nest_asyncio; on failure record NaN + note (never crash the graph).
- Logs: each metric score + the interpretation string.

**decide** (bonus conditional edge, in graph/edges.py)
- Purpose: retry retrieval once when quality is low.
- Logic: if (faithfulness < 0.70 or answer_relevancy < 0.70) and retry_count < 1 → return "retrieve" after retry_count += 1 and top_k = 8 (broaden); else → "finalize".
- Logs: decision + which threshold tripped.

**finalize** (MCP consumer — required "active use" every run)
- Purpose: persist an audit record and escalate when needed.
- In: question, answer, ragas_scores, is_valid. Out: mcp_log, escalated, ticket_id.
- External: custom MCP server via MultiServerMCPClient — ALWAYS calls `log_interaction(question, answer, scores)`; if is_valid is False OR scores stayed low after retry → also calls `create_ticket(...)` to escalate to a human, capturing ticket_id.
- Error handling: MCP connect/tool errors → state.error, but still return the answer to the user.
- Logs: tool(s) called, ticket_id / audit id.

Why this shape wins: the three required agents are the unmistakable spine; the conditional edge is the exact bonus the PDF names; guardrails + MCP escalation form one coherent "answer confidently or escalate" story that shows real MCP use and product judgment.

## 4. RAG pipeline design

Document source (synthetic, authored by us in data/knowledge_base/): ~10 realistic enterprise policy docs in Markdown — e.g. leave-policy, remote-work, expense-reimbursement, IT-security, code-of-conduct, onboarding, benefits, travel, data-privacy, procurement. Each ~1–2 pages with headed sections. Self-contained, no licensing, easy to write golden Q/A against.

Golden eval set (data/eval/golden_set.json): ~10–12 items {question, reference (ground-truth answer), source} spanning the docs — powers bonus context precision/recall and the aggregate eval report.

Chunking (src/eka/rag/splitter.py): RecursiveCharacterTextSplitter, chunk_size=800, chunk_overlap=120 (~15%), separators ["\n## ", "\n### ", "\n\n", "\n", " "]. Reasoning: policy docs are section-structured; splitting on headings first keeps a policy clause intact; 800 chars balances retrieval precision vs. giving Claude enough context; 15% overlap avoids clause truncation at boundaries.

Embeddings: BAAI/bge-small-en-v1.5 (384-dim, fast, strong on short passages) via langchain-huggingface; normalized embeddings for cosine.

Vector store (src/eka/rag/vectorstore.py): Chroma persistent at .chroma/, one collection "enterprise_kb"; each record = {id, embedding, document(text), metadata{source, section, chunk_id}}. Built offline by scripts/build_index.py so the graph never re-indexes at query time.

Retrieval params: top_k=4 default (bumped to 8 on the bonus retry); cosine similarity; optional relevance floor ~0.2 — if the best score is below the floor, mark context insufficient so generate refuses instead of hallucinating (feeds the groundedness guard).

## 5. RAGAS integration design

In src/eka/evaluation/ragas_eval.py. Wrap models once: `LangchainLLMWrapper(ChatAnthropic(...))` and `LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(...))`. Build a single-sample `EvaluationDataset` per query and call `evaluate(...)`.

Metrics and their inputs:
| Metric | PDF | Needs | Notes |
|---|---|---|---|
| Faithfulness | mandatory | question, answer, contexts | LLM judge; claims grounded in contexts. No reference → runs every query. |
| Answer Relevancy | mandatory | question, answer, (embeddings) | LLM + embeddings; runs every query. |
| Context Precision | bonus | question, contexts, reference | LLMContextPrecisionWithReference; needs golden reference. |
| Context Recall | bonus | question, contexts, reference | LLMContextRecall; needs golden reference. |

Runtime rule: faithfulness + answer_relevancy always; add the two context metrics only when state.reference exists (golden-set / eval mode) so arbitrary user queries never fail for lack of a reference.

Retry threshold (bonus edge): faithfulness < 0.70 OR answer_relevancy < 0.70 triggers one retry with top_k 4→8; retry_count caps at 1 to prevent loops.

Interpretation text (interpret_scores()): map each score to plain language — ≥0.8 "strong", 0.6–0.8 "acceptable, minor gaps", <0.6 "weak — answer may be unsupported/off-topic" — plus one summary line ("Faithful and on-topic; safe to trust." vs "Low faithfulness — likely unsupported claims; escalated."). Shown in the UI + written to the MCP audit log.

Aggregate eval (scripts/run_eval.py): runs the whole golden set through the graph, prints a per-question + mean table for all four metrics → the RAGAS screenshot + README analysis.

## 6. MCP integration design (custom Python server)

Server (mcp_server/server.py) — FastMCP "Enterprise Service Desk", stdio transport, backed by a local SQLite file (service_desk.db) so results persist and are screenshot-able. Tools exposed:
- `create_ticket(summary, category, priority="normal") -> {ticket_id, status}` — files a support ticket for a human.
- `get_ticket(ticket_id) -> {...}` and `list_tickets(status="open") -> [...]` — inspect tickets.
- `log_interaction(question, answer, scores) -> {audit_id}` — appends an audit row (query, answer, RAGAS scores, timestamp).

Client + node: the finalize node uses langchain-mcp-adapters `MultiServerMCPClient({"service_desk": {command: "python", args: ["mcp_server/server.py"], transport: "stdio"}})`, loads the tools, and:
- ALWAYS calls `log_interaction` → satisfies "actively used by a node during execution" on every run.
- Conditionally calls `create_ticket` when the input guard rejected the query, or RAGAS stayed low after the retry (the assistant escalates what it can't answer).

Use case (genuine, not decorative): unstructured RAG answers policy questions; the MCP server is the enterprise system-of-record the assistant writes back to — auditing every interaction and opening a human ticket when confidence is low. This mirrors the PDF's ticketing/CRM framing and makes MCP load-bearing.

Flags: server runs as a stdio subprocess (needs python on PATH); adapter tools are async (call within the node's event loop / asyncio.run under nest_asyncio); keep tool schemas small and typed.

## 7. Observability design

Two layers so the execution path is always visible:
- LangSmith tracing: set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY (+ project name). LangGraph auto-traces every node/edge → one trace shows all nodes firing in order. Screenshot-friendly.
- Manual structured logger (src/eka/observability/logging.py): a `@log_node` wrapper (or explicit log() calls) recording each node's name, key inputs, key outputs, and duration into state.trace and to stdout. Guarantees the "node-by-node log" screenshot even with no LangSmith account.

app.py prints the compiled graph (`graph.get_graph().draw_mermaid()` / ASCII) at startup so the flow is self-documenting.

## 8. Repo structure (modular)

```
enterprise-knowledge-assistant/
├─ README.md  requirements.txt  .env.example  .gitignore
├─ docs/            PLAN.md, architecture.md, screenshots/
├─ data/            knowledge_base/ (policy .md)   eval/golden_set.json
├─ mcp_server/      server.py (custom FastMCP)     service_desk.db
├─ src/eka/
│  ├─ config.py           # model ids, paths, thresholds, top_k
│  ├─ rag/                # loader.py, splitter.py, vectorstore.py, retriever.py
│  ├─ graph/              # state.py, nodes.py, edges.py, build.py
│  ├─ evaluation/         # ragas_eval.py
│  ├─ mcp/                # client.py (MultiServerMCPClient wiring)
│  ├─ guardrails/         # guards.py (input + groundedness)
│  ├─ observability/      # logging.py
│  └─ app.py              # CLI: run one question through the graph
├─ ui/              server.py (Flask), templates/index.html, static/style.css
├─ scripts/         build_index.py, run_eval.py
└─ tests/           test_*.py
```

## 9. Build sequence → commits (runnable after each)

1. `chore: project structure, pinned requirements, config, .env.example, .gitignore`
2. `feat(data): synthetic policy knowledge base + golden eval set`
3. `feat(rag): document loader + recursive chunking (+ tests)`
4. `feat(rag): Chroma vector store build + build_index script`
5. `feat(rag): similarity retriever with top_k (+ test)`
6. `feat(graph): state schema + retrieve & generate nodes + linear graph + CLI`  ← end-to-end answer, no eval yet
7. `feat(eval): RAGAS faithfulness + answer relevancy + interpretation; wire Evaluator → Retrieve→Respond→Evaluate→End`
8. `feat(mcp): custom FastMCP service-desk server + client + finalize node (log_interaction each run)`
9. `feat(guardrails): input guard + groundedness refusal (innovation)`
10. `feat(graph): bonus conditional retry edge on low RAGAS scores`
11. `feat(eval): bonus context precision + recall via golden set + run_eval aggregate`
12. `feat(mcp): auto create_ticket escalation on low-confidence / guard-blocked answers`
13. `feat(obs): structured node logging + LangSmith env wiring + mermaid print`
14. `feat(ui): minimal Flask + HTML/CSS front-end calling the graph`
15. `test: component tests + full-graph smoke test`
16. `docs: README (all required sections) + architecture diagram`
17. `chore: strip secrets, verify .gitignore excludes .env/venv/.chroma, finalize pins`

Ordering keeps a runnable artifact from commit 6 onward; each later commit adds one isolated capability.

## 10. Risk list (mapped to grading weight)

1. LangGraph clarity (30%): risk = flow doesn't read as the three named agents. Mitigate = nodes literally named retrieve/generate/evaluate as the spine, mermaid diagram in README + trace screenshot, conditional edge clearly labeled.
2. RAG quality (25%): risk = weak retrieval → low faithfulness cascades. Mitigate = sensible chunking, persistent Chroma, grounded prompt, show retrieved contexts + sources, relevance floor → refuse rather than hallucinate.
3. RAGAS (20%): risk = eval errors, scores not shown, or no interpretation. Mitigate = robust async-safe wrapper (nest_asyncio, try/except, Claude judge for clean JSON), always display scores + plain-language interpretation, guarantee both mandatory metrics.
4. MCP (15%): risk = "configured but not actively used". Mitigate = log_interaction fires every run inside finalize and appears in the trace/DB; create_ticket escalation shows real behavior.
5. Docs (10%) + reproducibility: risk = missing README sections/screenshots or a broken `pip install`. Mitigate = README template covering every required subsection, screenshot checklist, pinned + verified requirements, clean-clone smoke run before submit.

## 11. Test plan

Unit (pytest, deterministic via FakeLLM/FakeEmbeddings where possible):
- loader: loads all N docs; splitter: chunk count > docs, respects size/overlap invariants.
- vectorstore/retriever: build then query returns k chunks; a known policy question retrieves its source doc.
- nodes: each node given a minimal state returns the expected keys; guard rejects empty/injection and passes valid input.
- edges: decide() returns "retrieve" when a score < 0.70 and retry_count < 1, else "finalize".
- ragas_eval: interpret_scores maps ranges → correct labels (pure function, no LLM).
- mcp: server tools create/get/list a ticket and write an audit row against a temp SQLite DB.

Full-graph smoke test: build + invoke with a golden question; assert answer non-empty; ragas_scores has faithfulness & answer_relevancy as floats in [0,1]; retrieved_contexts non-empty; an MCP audit row was written; state.trace shows nodes in order guard→retrieve→generate→evaluate→finalize; no unhandled exception. Live-LLM assertions gated behind an env flag; default uses fakes for offline runs.

## 12. Deliverables checklist (maps to every PDF submission item)

Source code
- [ ] Complete, runnable source (this repo)
- [ ] No venv/ committed · [ ] No .env committed · [ ] No keys/tokens/secrets (.env.example only) · [ ] .gitignore excludes .env, venv, .chroma, *.db

Execution evidence (3–5 screenshots in docs/screenshots/)
- [ ] App startup (CLI banner or UI loaded)
- [ ] Graph execution trace node-by-node (LangSmith trace or the structured log)
- [ ] RAGAS evaluation results (per-query + run_eval aggregate)
- [ ] Final output (UI answer with scores + interpretation)
- [ ] MCP evidence (audit row / created ticket in service_desk.db)

README sections
- [ ] Architecture overview + diagram
- [ ] Setup (env, dependencies, config, ANTHROPIC_API_KEY)
- [ ] Execution steps + sample input / expected output
- [ ] RAG design (source, chunking, embedding model, vector DB)
- [ ] LangGraph design (nodes, responsibilities, flow)
- [ ] MCP integration (server, tools, use case)
- [ ] RAGAS evaluation (metrics, results, brief analysis)

Submission
- [ ] Send source + README + screenshots to sharad.rajore@zensar.com

## Verification (end-to-end)
1. Fresh clone → `pip install -r requirements.txt` → set ANTHROPIC_API_KEY in .env.
2. `python scripts/build_index.py` → confirms .chroma/ populated (prints chunk count).
3. `python -m src.eka.app --question "How many paid leave days do I get?"` → prints trace, answer, RAGAS scores + interpretation, and an MCP audit id.
4. `python scripts/run_eval.py` → aggregate RAGAS table over the golden set (grab the screenshot).
5. `python ui/server.py` → open the page, ask a question, see answer + scores.
6. `pytest` → unit + smoke tests green.
7. Confirm service_desk.db has audit rows (and a ticket after an out-of-scope / unanswerable question).

No open questions remain — document source (synthetic policy set) and MCP (custom Python server) are locked from the clarifying round, so sections 4 and 6 are final. Ready to implement on approval.
