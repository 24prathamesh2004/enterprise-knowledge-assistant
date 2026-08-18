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
| `generate` | **Response agent** | question, contexts | answer | Groq LLM (grounded prompt) |
| `evaluate` | **Evaluator agent** | question, answer, contexts, reference | ragas_scores, evaluation_summary | RAGAS + Groq judge |
| `decide_after_eval` (edge) | Bonus conditional retry | ragas_scores, retry_count | routes only | none |
| `prepare_retry` | Broaden retrieval | retry_count, top_k | retry_count+1, top_k=8 | none |
| `finalize` | **MCP consumer** | question, answer, scores, is_valid | escalated, ticket_id, mcp_log | custom MCP server |

State is a single shared `TypedDict` (`GraphState` in `graph.py`); the `trace`
field uses an `operator.add` reducer so every node appends a structured log entry
(its name, inputs, outputs, and duration).

- **Guardrails (innovation):** `guard_input` rejects empty/oversized input and
  obvious prompt-injection; `generate` refuses instead of hallucinating when
  retrieval is below the relevance floor. Together they drive the "answer
  confidently or escalate" story: a rejected or low-confidence query gets an
  audit record **and** a human ticket via MCP.
- **No supervisor agent.** The assignment asks to "keep the flow linear," so the
  only branch is the bonus retry edge — a plain routing function, no
  supervisor/router.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then edit .env and set GROQ_API_KEY=gsk_...
```

Generation and the RAGAS judge both run on **Groq** (OpenAI GPT-OSS models), so
**one** cloud key is needed (`GROQ_API_KEY`). Groq has no embeddings API, so
embeddings are local (`BAAI/bge-small-en-v1.5`, ~90–130 MB, downloaded from
HuggingFace on first run, then offline) and shared by both the vector store and
RAGAS answer-relevancy.

| Purpose | Model | Override env var |
|---|---|---|
| Generation (Response agent) | `openai/gpt-oss-20b` | `EKA_GEN_MODEL` |
| RAGAS judge (Evaluator agent) | `openai/gpt-oss-120b` | `EKA_JUDGE_MODEL` |
| Embeddings | `BAAI/bge-small-en-v1.5` | `EKA_EMBED_MODEL` |

## Execution

Everything runs from the repo root through a single entry point, `app.py`.

```bash
# 1. Build the vector index (one time) — persists to .chroma/
python app.py --build-index

# 2. Ask one question (prints the graph, trace, answer, scores, MCP audit id)
python app.py --ask "How many paid leave days do I get per year?"

# 3. Evaluate the whole golden set and print an aggregate RAGAS table
python app.py --eval

# 4. Run the web UI (http://127.0.0.1:5000)
python ui.py

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

- **Source:** 10 synthetic enterprise policy documents in `knowledge_base/`
  (leave, remote work, expenses, IT security, code of conduct, onboarding,
  benefits, travel, data privacy, procurement) for a fictional company,
  "Northwind Technologies". Self-contained, no licensing concerns.
- **Chunking:** `RecursiveCharacterTextSplitter`, `chunk_size=800`,
  `chunk_overlap=120` (~15%), heading-aware separators
  (`\n## `, `\n### `, `\n\n`, `\n`, ` `) so a policy clause stays intact.
- **Embeddings:** `BAAI/bge-small-en-v1.5` (384-dim, normalized) via
  `langchain-huggingface` — local, free, shared with RAGAS.
- **Vector store:** Chroma (persistent, `.chroma/`), collection `enterprise_kb`,
  built offline by `python app.py --build-index`.
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
self-documenting. All state, nodes, edges, and graph assembly live in `graph.py`.

## MCP integration

- **Server:** `mcp_server.py` — a custom `FastMCP` "Enterprise Service Desk" over
  stdio, backed by SQLite (`service_desk.db`). Tools: `log_interaction` (append
  an audit row) and `create_ticket` (open a human ticket).
- **Client:** `mcp_client.py` uses `langchain-mcp-adapters` `MultiServerMCPClient`
  to spawn the server as a subprocess and load its tools.
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
- **Judge:** Groq `openai/gpt-oss-120b`; embeddings reuse the local HF model.
- **Interpretation:** `interpret_scores()` maps each score to plain language
  (≥0.8 strong · 0.6–0.8 acceptable · <0.6 weak) plus a one-line verdict, shown
  in the CLI and UI and written to the MCP audit log.
- **Aggregate:** `python app.py --eval` runs the golden set and prints per-question
  and mean scores for all four metrics.

Evaluation is async-safe (`nest_asyncio`) and defensive: on any failure it
records `NaN` and never crashes the graph.

## Observability

Two layers: a manual structured logger (`observability.py`, `@log_node`) that
records each node's name, key inputs, key outputs, and duration into
`state["trace"]` and stdout (guaranteed node-by-node log with no external
account); plus optional LangSmith tracing via env vars
(`LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`).

## Repository layout

```
app.py             single CLI entry point (--build-index | --ask | --eval)
config.py          models, paths, thresholds, top_k, Groq LLM factory
rag.py             loader + chunking + Chroma vector store + retriever
graph.py           GraphState + all nodes + edges + graph assembly
guards.py          input validation + context-sufficiency guardrails
ragas_eval.py      RAGAS scoring + plain-language interpretation
mcp_client.py      MCP client wiring (MultiServerMCPClient)
mcp_server.py      custom FastMCP service-desk server (SQLite)
observability.py   @log_node structured logger
ui.py              Flask server (local, unauthenticated) + templates/index.html
knowledge_base/    synthetic policy .md docs
golden_set.json    golden eval set (question + reference + source)
tests/test_core.py component + full-graph smoke tests
docs/screenshots/  execution evidence
```

## Security notes

- The Flask UI binds to `127.0.0.1` and has **no authentication** — it is a local
  demo only. Do not expose it to a network without adding access controls.
- No secrets are committed: `.env.example` only. `.gitignore` excludes `.env`,
  virtualenvs, `.chroma/`, and `*.db`. All data and PII are fictional.
- Strip `.env` before submitting the project.

