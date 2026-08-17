# Enterprise Knowledge Assistant

An L3 Agentic AI project: a LangGraph assistant that answers enterprise policy
questions from a governed knowledge base, self-scores every answer with **RAGAS**,
and actively uses a **custom MCP server** to audit interactions and escalate
low-confidence answers to a human via a support ticket.

The flow is a linear three-agent spine — **Retrieve → Respond → Evaluate** — with
one bonus conditional retry edge and input/groundedness **guardrails** wired in.

```
guard_input ─▶ retrieve ─▶ generate ─▶ evaluate ─┬─(scores OK)─▶ finalize ─▶ END
     │                         ▲                  │
     │(invalid input)          └── prepare_retry ─┘ (low RAGAS scores, once)
     └───────────────────────────────────────────▶ finalize (safe refusal + audit)
```

## Architecture overview

| Node | Role | Reads | Writes | External |
|---|---|---|---|---|
| `guard_input` | Input guardrail (innovation) | question | is_valid, guardrail_reason, top_k, retry_count | none |
| `retrieve` | **Retriever agent** | question, top_k | retrieved_contexts, retrieved_meta | HF embeddings + Chroma |
| `generate` | **Response agent** | question, contexts | answer | Claude (grounded prompt) |
| `evaluate` | **Evaluator agent** | question, answer, contexts, reference | ragas_scores, evaluation_summary | RAGAS + Claude judge |
| `decide` (edge) | Bonus conditional retry | ragas_scores, retry_count | routes only | none |
| `prepare_retry` | Broaden retrieval | retry_count, top_k | retry_count+1, top_k=8 | none |
| `finalize` | **MCP consumer** | question, answer, scores, is_valid | escalated, ticket_id, mcp_log | custom MCP server |

State is a single shared `TypedDict` (`src/eka/graph/state.py`); the `trace` field
uses an `operator.add` reducer so every node appends a structured log entry.

- **Guardrails (innovation):** `guard_input` rejects empty/oversized input and
  obvious prompt-injection; `generate` refuses instead of hallucinating when
  retrieval is below the relevance floor. Together they drive the "answer
  confidently or escalate" story: a rejected or low-confidence query gets an
  audit record **and** a human ticket via MCP.
- **No supervisor agent.** The assignment asks to "keep the flow linear," so the
  only branch is the PDF-blessed bonus retry edge — a plain routing function, no
  supervisor/router.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then edit .env
# set ANTHROPIC_API_KEY=sk-ant-...
```

The first run downloads the local embedding model `BAAI/bge-small-en-v1.5`
(~90–130 MB) from HuggingFace; after that it runs offline. Generation and the
RAGAS judge use Anthropic Claude, so **one** cloud key is needed
(`ANTHROPIC_API_KEY`). Anthropic has no embeddings API, which is why embeddings
are local and shared by both the vector store and RAGAS answer-relevancy.

## Execution

```bash
# 1. Build the vector index (offline, one time) — persists to .chroma/
python scripts/build_index.py

# 2. Ask one question via the CLI (prints the graph, trace, answer, scores, MCP audit id)
python -m src.eka.app -q "How many paid leave days do I get per year?"

# 3. Evaluate the whole golden set and print an aggregate RAGAS table
python scripts/run_eval.py

# 4. Run the web UI (http://127.0.0.1:5000)
python ui/server.py

# 5. Tests (offline; models/RAG/MCP are faked in the smoke test)
pytest -q
```

### Sample input / expected output

Input: `How many paid leave days do I get per year?`

Expected (abridged):

```
=== Answer ===
Full-time employees are entitled to 22 days of paid annual leave per calendar
year, accrued at 1.83 days per completed month of service. (leave-policy.md)

=== RAGAS evaluation ===
- faithfulness: 0.95 — strong
- answer_relevancy: 0.91 — strong
Summary: Faithful and on-topic; the answer is well grounded in the sources.

=== MCP service desk ===
audit: {'audit_id': 'AUD-1A2B3C4D'}
```

An out-of-scope or unanswerable question instead returns a safe refusal and is
escalated: `escalated -> ticket TICK-XXXXXXXX`.

## RAG design

- **Source:** 10 synthetic enterprise policy documents in `data/knowledge_base/`
  (leave, remote work, expenses, IT security, code of conduct, onboarding,
  benefits, travel, data privacy, procurement) for a fictional company,
  "Northwind Technologies". Self-contained, no licensing concerns.
- **Chunking:** `RecursiveCharacterTextSplitter`, `chunk_size=800`,
  `chunk_overlap=120` (~15%), heading-aware separators
  (`\n## `, `\n### `, `\n\n`, `\n`, ` `) so a policy clause stays intact.
- **Embeddings:** `BAAI/bge-small-en-v1.5` (384-dim, normalized) via
  `langchain-huggingface` — local, free, shared with RAGAS.
- **Vector store:** Chroma (persistent, `.chroma/`), collection `enterprise_kb`,
  built offline by `scripts/build_index.py`.
- **Retrieval:** cosine similarity, `top_k=4` (broadened to `8` on the bonus
  retry). A relevance floor (`0.20`) marks weak retrieval as "insufficient
  context" so the generator refuses rather than hallucinates.

## LangGraph design

Three required agents form the spine (`retrieve`, `generate`, `evaluate`), plus
`guard_input` (innovation) and `finalize` (MCP). The bonus conditional edge
`decide_after_eval` retries retrieval once (broader `top_k`) when
`faithfulness < 0.70` or `answer_relevancy < 0.70`, capped by `retry_count` to
prevent loops. See the table above for the per-node read/write contract. The CLI
prints the compiled graph (ASCII/mermaid) at startup so the flow is
self-documenting.

## MCP integration

- **Server:** `mcp_server/server.py` — a custom `FastMCP` "Enterprise Service
  Desk" over stdio, backed by SQLite (`service_desk.db`). Tools:
  `create_ticket`, `get_ticket`, `list_tickets`, `log_interaction`.
- **Client:** `src/eka/mcp/client.py` uses `langchain-mcp-adapters`
  `MultiServerMCPClient` to spawn the server and load its tools.
- **Active use every run:** the `finalize` node **always** calls
  `log_interaction` (writing an audit row), and **conditionally** calls
  `create_ticket` when the input was rejected or RAGAS stayed low after the
  retry. The MCP server is the enterprise system-of-record the assistant writes
  back to — auditing every interaction and opening a human ticket when
  confidence is low.

## RAGAS evaluation

- **Mandatory:** Faithfulness, Answer Relevancy — run on every query.
- **Bonus:** Context Precision, Context Recall — run only when a golden-set
  reference is supplied (so arbitrary user queries never fail for lack of a
  reference).
- **Judge:** Claude (`claude-sonnet-5`); embeddings reuse the local HF model.
- **Interpretation:** `interpret_scores()` maps each score to plain language
  (≥0.8 strong · 0.6–0.8 acceptable · <0.6 weak) plus a one-line verdict, shown
  in the CLI and UI and written to the MCP audit log.
- **Aggregate:** `scripts/run_eval.py` runs the golden set and prints per-question
  and mean scores for all four metrics.

Evaluation is async-safe (`nest_asyncio`) and defensive: on any failure it
records `NaN` and never crashes the graph.

## Observability

Two layers: a manual structured logger (`src/eka/observability/logging.py`,
`@log_node`) that records each node's name, key outputs, and duration into
`state["trace"]` and stdout (guaranteed node-by-node log with no external
account); plus optional LangSmith tracing via env vars
(`LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`).

## Repository layout

```
data/knowledge_base/   synthetic policy .md docs
data/eval/             golden_set.json
mcp_server/            custom FastMCP service-desk server (SQLite)
src/eka/               config, rag/, graph/, evaluation/, mcp/, guardrails/, observability/, app.py
ui/                    Flask server + HTML/CSS/JS (local, unauthenticated)
scripts/               build_index.py, run_eval.py
tests/                 component + full-graph smoke tests
docs/                  PLAN.md, architecture.md, screenshots/
```

## Security notes

- The Flask UI binds to `127.0.0.1` and has **no authentication** — it is a local
  demo only. Do not expose it to a network without adding access controls.
- No secrets are committed: `.env.example` only. `.gitignore` excludes `.env`,
  virtualenvs, `.chroma/`, and `*.db`. All data and PII are fictional.

