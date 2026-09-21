# GraphRAG — Emirates NBD Strategic Report

A retrieval-augmented chatbot over a bank's strategic report that answers from **two** retrieval paths: dense vector search over hierarchically chunked text, and N-hop traversal of a Neo4j knowledge graph of typed entity relationships — so questions like *"who owns Emirates Islamic?"* are answered from a `SUBSIDIARY_OF` edge rather than from whichever paragraph happens to share the most vocabulary with the question.

---

## Architecture

```text
                          INGESTION
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   PDF report                                                     │
  │       │                                                          │
  │       ▼                                                          │
  │   Docling ── layout model · reading order · tables to markdown   │
  │       │                                                          │
  │       ▼                                                          │
  │   Parent chunks  (~1000 tokens)  ────────────────┐               │
  │       │                                          │               │
  │       ├── split ──▶ Child chunks (~200 tokens)   │               │
  │       │                     │                    │               │
  │       │                     ▼                    │               │
  │       │              all-MiniLM-L6-v2            │               │
  │       │              (local, CPU)                │               │
  │       │                     │                    │               │
  │       ▼                     │                    ▼               │
  │   LLM extraction            │              parents.json          │
  │   (structured output)       │                                    │
  │       │                     │                                    │
  └───────┼─────────────────────┼────────────────────────────────────┘
          │                     │
          │                     ▼
          │              ┌─────────────┐
          │              │    FAISS    │  child vectors
          │              └─────────────┘
          │                     ▲
          ▼                     │
    ┌──────────────┐            │        emirates_nbd_graph.cypher
    │    Neo4j     │◀───────────┼──────── 
    │ typed edges  │            │
    │ + provenance │            │
    └──────────────┘            │
          │                     │
──────────┼─────────────────────┼────────────────────────────────────
          │      QUERY  (LangGraph state machine)
          │                     │
                    ┌───────────────────┐
                    │     question      │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │       route       │   intent classifier
                    └─────────┬─────────┘
                 ┌────────────┼────────────┐
          vector │            │ hybrid     │ graph
                 │      (runs both,        │
                 │       in parallel)      │
                 ▼            ▼            ▼
        ┌──────────────┐          ┌──────────────────┐
        │ vector_search│          │   graph_search   │
        │ top-k child  │          │  seed entities,  │
        │ chunks, then │          │  N-hop traversal │
        │ expand to    │          │                  │
        │ parents      │          │                  │
        └──────┬───────┘          └────────┬─────────┘
               │                           │
               └─────────────┬─────────────┘
                             ▼
                    ┌───────────────────┐
                    │       fuse        │  Reciprocal Rank Fusion
                    │                   │  + cross-encoder rerank
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │     generate      │
                    └─────────┬─────────┘
                              │  Server-Sent Events
                              ▼
                    ┌───────────────────┐
                    │   Web chat UI     │  served by FastAPI at /
                    └───────────────────┘
```

**Reading it.** Two details carry the design. First, FAISS matches a *child*
chunk but the *parent* is what gets returned — that is the small-to-big
expansion. Second, both retrieval arms produce the same kind of result, so
fusion has something comparable to merge; the graph reaches the vector store to
resolve its triples back to passages.

## Key features

- **Hierarchical parent–child chunking.** ~200-token children are embedded for precise matching; the ~1000-token parent they were split from is what reaches the LLM. Children are cut *from* parent text, so the link is exact rather than heuristic.
- **Hybrid retrieval with an intent router.** A LangGraph node classifies each question as `vector`, `graph`, or `hybrid` and returns a *list of node names* — LangGraph turns that into a parallel fan-out, so hybrid runs both arms concurrently.
- **Reciprocal Rank Fusion + cross-encoder reranking**, applied only on the hybrid path, where two incomparable rankings (cosine similarity vs. hop distance) actually need merging.
- **Graph can be built without an LLM, automatically.** A hand-authored `.cypher` file is loaded into Neo4j on first startup whenever the graph is empty, and the `:Entity` label is projected onto it so retrieval can traverse it. A fresh clone therefore has a working graph with zero LLM calls and no manual step.
- **Cross-provider LLM failover.** Requesty → OpenRouter → OpenAI, whichever have keys. Survives `402 Payment Required` and `429 rate limit` at the provider level, which per-provider model fallback cannot.
- **Idempotent ingestion.** Chunk ids are content-addressed and FAISS duplicates are deleted before re-adding, so re-running updates in place instead of duplicating or erroring.
- **Page-linked citations.** Markers render inline as `1 p.12` and link into the source PDF at `#page=12`.
- **Graceful degradation.** Neo4j down → vector-only. Reranker unavailable → RRF order kept. Router LLM fails → keyword heuristic. Docling missing → `pypdf` fallback. The API starts even when every backing service is down, so `/health` can report which one.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Developed on 3.13 |
| Neo4j | 5.x | Aura, or Community via Docker. No APOC required. |
| Docker | optional | Only for running Neo4j locally or the containerised app |
| Node.js | not required | The web client has no build step |
| RAM | ~4 GB free | PyTorch + embedding model + cross-encoder |
| Disk | ~3 GB | `torch` dominates; models are cached after first run |

An API key for **at least one** of: Requesty, OpenRouter, or OpenAI. Embeddings and reranking run locally and need no key.

---

## Project structure

```
.
├── app/
│   ├── core/                  # Cross-cutting foundations
│   │   ├── config.py          #   Typed settings; the only module reading env vars
│   │   ├── schemas.py         #   Pydantic models shared by API, pipeline and UI
│   │   └── metadata.py        #   Document/chunk metadata: pages, sizes, signals
│   ├── ingestion/             # Getting documents in
│   │   ├── loader.py          #   Docling parsing, with pypdf/pdfplumber fallback
│   │   ├── chunker.py         #   Parent-child hierarchical splitting
│   │   ├── extractor.py       #   LLM structured output -> entities/relationships
│   │   └── pipeline.py        #   Stage orchestration + progress reporting
│   ├── storage/               # Persistence
│   │   ├── vector_store.py    #   FAISS over children + JSON parent store
│   │   ├── graph_store.py     #   Neo4j writes, traversal, subgraph queries
│   │   └── cypher_loader.py   #   Loads a .cypher file; shared by CLI and startup
│   ├── retrieval/             # Finding the right context
│   │   └── retriever.py       #   Intent router, graph traversal, RRF + reranking
│   ├── chat/                  # Producing the answer
│   │   ├── llm.py             #   Model + embedding factories, provider failover
│   │   ├── prompts.py         #   Extraction and answer prompts
│   │   └── rag_graph.py       #   The LangGraph state machine
│   └── api/
│       └── main.py            # FastAPI app, all routes, serves the web client
├── web/
│   ├── index.html         # Chat UI markup
│   ├── styles.css         # Styles; no framework
│   └── app.js             # SSE streaming, citations, graph rendering
├── scripts/
│   ├── ingest_cli.py      # Ingest a PDF (parse, chunk, embed, extract)
│   └── load_cypher.py     # Load a .cypher file and project :Entity onto it
├── tests/
│   ├── conftest.py        # Fixtures; auto-skip when services are unavailable
│   ├── unit/              # Pure logic — no network, database or LLM
│   ├── integration/       # Requires a populated index and Neo4j
│   ├── evaluation/        # RAG quality metrics (opt-in)
│   └── fixtures/          # Labelled evaluation questions
├── Company/               # Source PDFs
├── data/                  # Generated: FAISS index, parents.json (gitignored)
├── emirates_nbd_graph.cypher   # Hand-authored knowledge graph
├── docker-compose.yml     # Optional local Neo4j; optional containerised app
├── Dockerfile             # Single image: API + web client
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Setup

### 1. Clone and install

```bash
git clone <repository-url> xpin-assign
cd xpin-assign
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux use `source .venv/bin/activate`.

### 2. Configure

```bash
copy .env.example .env
```

Edit `.env` and set at minimum a Neo4j URI/password and one LLM API key. See the [configuration reference](#configuration-reference).

### 3. Start Neo4j

With Neo4j Aura, skip this step and point `NEO4J_URI` at the Aura endpoint. To run Neo4j locally:

```bash
docker compose --profile local-db up -d neo4j
```

Then set `NEO4J_URI=bolt://localhost:7687` in `.env`.

### 4. Load the knowledge graph — optional

The app loads `emirates_nbd_graph.cypher` into Neo4j automatically on first
startup, whenever it finds the graph empty. Skip to step 5 unless you want to
run it by hand.

To load it explicitly, preview the parse, or reset and reload:

```bash
python scripts/load_cypher.py emirates_nbd_graph.cypher
python scripts/load_cypher.py emirates_nbd_graph.cypher --dry-run
python scripts/load_cypher.py emirates_nbd_graph.cypher --wipe
```

The automatic load only ever runs at zero entities, so it cannot overwrite an
existing graph. Disable it with `GRAPH_BOOTSTRAP=false`.

### 5. Ingest the PDF (builds the vector index)

```bash
python scripts/ingest_cli.py --reset
```

Useful variants:

```bash
python scripts/ingest_cli.py --max-pages 20          # quick smoke test
python scripts/ingest_cli.py --no-graph              # vectors only, no LLM calls
python scripts/ingest_cli.py --path Company/other.pdf --doc-id other
```

`--no-graph` skips LLM extraction entirely. Combined with step 4, the whole system runs with **zero LLM calls at ingestion time**.

### 6. Run

```bash
uvicorn app.api.main:app --port 8000
```

Open **http://localhost:8000**. API docs are at `/docs`.

### Docker alternative

```bash
docker compose --profile app up -d --build
```

---

## Configuration reference

| Variable | Purpose | Example |
|---|---|---|
| `NEO4J_URI` | Bolt endpoint | `neo4j+ssc://xxxx.databases.neo4j.io` |
| `NEO4J_USERNAME` | Neo4j user | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `<password>` |
| `NEO4J_DATABASE` | Database name | `neo4j` |
| `LLM_PROVIDER` | Primary provider | `requesty` \| `openrouter` \| `openai` \| `custom` |
| `LLM_FALLBACK_PROVIDERS` | Failover order; keyless providers skipped | `openrouter,openai` |
| `LLM_MODEL` | Model id, must support tool calling | `nvidia/nemotron-3-super-120b-a12b` |
| `LLM_MAX_TOKENS` | Cap on generated tokens | `1000` |
| `LLM_TEMPERATURE` | Sampling temperature | `0.0` |
| `REQUESTY_API_KEY` | Requesty key | `rqsty-...` |
| `OPENROUTER_API_KEY` | OpenRouter key | `sk-or-v1-...` |
| `OPENAI_API_KEY` | OpenAI key | `sk-...` |
| `LLM_API_KEY` / `LLM_BASE_URL` | Override for any OpenAI-compatible endpoint | `http://localhost:11434/v1` |
| `EMBEDDING_MODEL` | Local embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `RERANKER_MODEL` | Local cross-encoder | `BAAI/bge-reranker-base` |
| `USE_RERANKER` | Disable to skip reranking | `true` |
| `PARENT_CHUNK_TOKENS` | Context block size | `1000` |
| `CHILD_CHUNK_TOKENS` | Embedded unit size | `200` |
| `VECTOR_TOP_K` | Child chunks retrieved | `12` |
| `PARENT_TOP_N` | Parent blocks reaching the prompt | `5` |
| `GRAPH_MAX_HOPS` | Traversal depth (clamped to 3) | `2` |
| `ROUTING_MODE` | `llm` \| `heuristic` (no extra call) \| `hybrid` (force both) | `llm` |
| `RRF_K` | RRF damping constant | `60` |
| `RRF_VECTOR_WEIGHT` / `RRF_GRAPH_WEIGHT` | Retriever influence | `1.0` / `0.8` |
| `EXTRACTION_WORKERS` | Concurrent extraction calls | `8` |
| `EXTRACTION_RPM` | Request pacing; `0` disables | `30` |
| `DOCS_DIR` | Source-document directory; also the allow-list for citation PDF links | `Company Docs` |
| `DEFAULT_PDF_PATH` | Document ingested by default | `Company Docs/strategic_report_2025_emirates_nbd.pdf` |
| `GRAPH_BOOTSTRAP` | Load the bundled `.cypher` on startup when the graph is empty | `true` |
| `GRAPH_BOOTSTRAP_FILE` | File used by that bootstrap | `emirates_nbd_graph.cypher` |

> `.env` takes precedence over OS environment variables. This is deliberate: `docker compose` exports these names into the shell, which otherwise silently shadows edits to `.env`. Inside a container no `.env` is present, so injected variables apply as normal.

---

## Example usage

| Question | Route | Why |
|---|---|---|
| Who owns Emirates Islamic? | `graph` | Pure ownership relation; answered from `SUBSIDIARY_OF` |
| Which awards did the group win? | `graph` | `AWARDED_TO` edges enumerate directly |
| What profit did Emirates NBD report? | `hybrid` | Named entity + a figure that lives in the text |
| What stake was acquired in India? | `hybrid` | Graph supplies the link, text supplies the terms |
| Summarise the strategic priorities | `vector` | Narrative; traversal adds noise |
| Explain the approach to sustainability | `vector` | No relation to traverse from |
| What is Tesla's share price? | any | Not in the document — the system declines |

Streaming via curl:

```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Who owns Emirates Islamic?"}'
```

Event order: `status` → `citations` → `triples` → `subgraph` → `token`×N → `done`.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/chat` | Streamed answer (SSE) |
| `POST` | `/api/v1/chat/sync` | Same, single JSON response |
| `POST` | `/api/v1/ingest` | Queue ingestion of a server-side PDF |
| `POST` | `/api/v1/ingest/upload` | Queue ingestion of an uploaded PDF |
| `GET` | `/api/v1/ingest/{job_id}` | Poll ingestion progress |
| `GET` | `/api/v1/graph/subgraph` | Node/edge JSON for visualisation |
| `GET` | `/api/v1/graph/entities` | Entity search |
| `GET` | `/api/v1/graph/stats` | Index and graph counters |
| `GET` | `/api/v1/document/{doc_id}` | Source PDF, for citation links |
| `GET` | `/health` | Per-component status |

---

## Testing

```bash
pytest tests/unit -q                        # always runs; no services needed
pytest tests/integration -q                 # auto-skips if stores are empty
pytest tests/evaluation -m evaluation -sv   # RAG quality report; calls the LLM
```

The evaluation layer scores retrieval (hit rate, recall@k, MRR, context precision), groundedness (is the answer traceable to retrieved text, are the numbers real, do `[n]` markers resolve), and system behaviour (routing accuracy, graph contribution, expansion ratio). Metrics are deterministic and use no LLM judge.

---

## Roadmap

- Entity resolution across documents (alias clustering beyond case folding)
- Incremental re-ingestion driven by `content_hash` rather than full `--reset`
- Cypher generation from natural language, as a third retrieval path
- Conversation memory persisted server-side rather than sent per request

---

## License

MIT
