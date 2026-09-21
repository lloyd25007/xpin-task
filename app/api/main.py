"""FastAPI application: ingestion, chat streaming and graph inspection.

One module for the whole HTTP surface. It is the thinnest layer in the system --
every route delegates immediately to the pipeline or the stores -- so splitting
it across a package added directories without adding clarity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import AsyncGenerator

from fastapi import (
    APIRouter, BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import PROJECT_ROOT, settings
from app.storage.graph_store import GraphStore, entity_key
from app.ingestion.pipeline import ingest_pdf
from app.core.schemas import (
    ChatRequest, GraphEdge, GraphNode, GraphStats, IngestJob, IngestRequest,
    JobStatus, SubgraphResponse,
)
from app.storage.vector_store import VectorStore


# ═══════════════════════════════════════════════════════════════════════
# Shared singletons
# ═══════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the shared FAISS + parent store instance."""
    logger.info("Initialising vector store")
    return VectorStore()


@lru_cache(maxsize=1)
def get_graph_store() -> GraphStore:
    """Return the shared Neo4j store instance (connection is lazy)."""
    logger.info("Initialising graph store")
    return GraphStore()


@lru_cache(maxsize=1)
def get_rag_graph():
    """Return the compiled LangGraph retrieval pipeline.

    Imported inside the function rather than at module scope to keep the import
    graph acyclic: the pipeline imports the stores, which this module also
    provides.
    """
    from app.chat.rag_graph import build_rag_graph

    logger.info("Compiling RAG graph")
    return build_rag_graph(get_vector_store(), get_graph_store())


def reset_caches() -> None:
    """Drop the cached singletons so the next request rebuilds them.

    Called after an ingest that used ``reset``: the FAISS index on disk has been
    replaced, and a store still holding the old in-memory index would keep
    serving stale vectors.
    """
    get_vector_store.cache_clear()
    get_rag_graph.cache_clear()
    logger.info("Cleared cached stores; they will be rebuilt on next use")

# ═══════════════════════════════════════════════════════════════════════
# POST /api/v1/ingest -- asynchronous document ingestion
# ═══════════════════════════════════════════════════════════════════════

ingest_router = APIRouter(prefix="/api/v1", tags=["ingestion"])


# --------------------------------------------------------------------------- #
# Job persistence
# --------------------------------------------------------------------------- #

def _job_path(job_id: str) -> Path:
    """Filesystem location of one job's status document."""
    return settings.jobs_dir / f"{job_id}.json"


def _save_job(job: IngestJob) -> None:
    """Write a job's status to disk.

    Called on every progress tick, so it must stay cheap and must never raise:
    a failed status write should not abort the ingestion it is reporting on.
    """
    try:
        _job_path(job.job_id).write_text(job.model_dump_json(indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not persist job %s: %s", job.job_id, exc)


def _load_job(job_id: str) -> IngestJob | None:
    """Read a job's status from disk, or ``None`` if the id is unknown."""
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        return IngestJob.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Corrupt job file %s: %s", job_id, exc)
        return None


# --------------------------------------------------------------------------- #
# Background worker
# --------------------------------------------------------------------------- #

def _run_ingestion(job: IngestJob, request: IngestRequest) -> None:
    """Execute the ingestion pipeline and keep the job document current.

    Runs in FastAPI's background task pool (a worker thread), so it must not
    raise: an unhandled exception there is invisible to the client. Every
    failure is caught and recorded on the job instead.
    """
    job.status = JobStatus.RUNNING
    _save_job(job)

    def _progress(stage: str, fraction: float, counters: dict) -> None:
        """Mirror pipeline progress onto the persisted job document."""
        job.stage = stage
        job.progress = round(min(max(fraction, 0.0), 1.0), 3)
        for key, value in (counters or {}).items():
            if hasattr(job, key):
                setattr(job, key, value)
        _save_job(job)

    try:
        counters = ingest_pdf(
            pdf_path=request.path,
            doc_id=request.doc_id,
            extract_graph=request.extract_graph,
            reset=request.reset,
            max_pages=request.max_pages,
            progress=_progress,
            vector_store=get_vector_store(),
            graph_store=get_graph_store(),
        )
        for key, value in counters.items():
            if hasattr(job, key):
                setattr(job, key, value)
        job.status = JobStatus.COMPLETED
        job.stage = "completed"
        job.progress = 1.0
    except Exception as exc:
        logger.exception("Ingestion job %s failed", job.job_id)
        job.status = JobStatus.FAILED
        job.stage = "failed"
        job.error = str(exc)
    finally:
        job.finished_at = datetime.now(timezone.utc)
        _save_job(job)
        # The on-disk index changed, so cached stores must be rebuilt.
        reset_caches()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@ingest_router.post("/ingest", response_model=IngestJob, status_code=202)
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestJob:
    """Queue ingestion of a PDF that already exists on the server.

    Returns ``202 Accepted`` with a job document. Poll
    ``GET /api/v1/ingest/{job_id}`` to follow progress.

    Raises:
        HTTPException 404: the requested path does not exist.
    """
    source = settings.resolve(request.path or settings.default_pdf_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {source}")

    job = IngestJob(
        job_id=uuid.uuid4().hex[:12],
        doc_id=request.doc_id or source.stem,
        source=str(source),
        status=JobStatus.PENDING,
    )
    _save_job(job)

    # Normalise the path so the worker does not re-resolve it differently.
    request.path = str(source)
    background_tasks.add_task(_run_ingestion, job, request)

    logger.info("Queued ingestion job %s for %s", job.job_id, source.name)
    return job


@ingest_router.post("/ingest/upload", response_model=IngestJob, status_code=202)
async def ingest_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF to ingest"),
    doc_id: str | None = Form(default=None),
    extract_graph: bool = Form(default=True),
    reset: bool = Form(default=False),
    max_pages: int | None = Form(default=None),
) -> IngestJob:
    """Queue ingestion of an uploaded PDF.

    The upload is streamed to ``data/uploads`` first, because the background
    task outlives the request and ``UploadFile``'s temporary handle is closed
    as soon as the response is returned.

    Raises:
        HTTPException 400: the upload is not a PDF.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_dir = settings.data_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / Path(file.filename).name

    try:
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    finally:
        await file.close()

    job = IngestJob(
        job_id=uuid.uuid4().hex[:12],
        doc_id=doc_id or target.stem,
        source=str(target),
        status=JobStatus.PENDING,
    )
    _save_job(job)

    background_tasks.add_task(
        _run_ingestion,
        job,
        IngestRequest(
            path=str(target), doc_id=doc_id, extract_graph=extract_graph,
            reset=reset, max_pages=max_pages,
        ),
    )
    logger.info("Queued upload ingestion job %s for %s", job.job_id, target.name)
    return job


@ingest_router.get("/ingest/{job_id}", response_model=IngestJob)
async def get_job(job_id: str) -> IngestJob:
    """Return the current status of an ingestion job.

    Raises:
        HTTPException 404: unknown job id.
    """
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job


@ingest_router.get("/ingest", response_model=list[IngestJob])
async def list_jobs(limit: int = 20) -> list[IngestJob]:
    """List recent ingestion jobs, newest first."""
    jobs: list[IngestJob] = []
    for path in sorted(settings.jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        job = _load_job(path.stem)
        if job is not None:
            jobs.append(job)
        if len(jobs) >= limit:
            break
    return jobs

# ═══════════════════════════════════════════════════════════════════════
# POST /api/v1/chat -- SSE-streamed grounded answers
# ═══════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)
chat_router = APIRouter(prefix="/api/v1", tags=["chat"])

# Sent periodically so proxies and load balancers do not close an idle stream
# while the retrieval stage is still running.
_KEEPALIVE_COMMENT = ": keep-alive\n\n"


def _sse(event: str, data) -> str:
    """Format one Server-Sent Event frame.

    The SSE wire format is ``event: <name>\\ndata: <payload>\\n\\n``. The payload
    is JSON-encoded on a single line, because a raw newline inside ``data:``
    would be parsed as a field break and split the frame.
    """
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def _event_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Drive the LangGraph pipeline and yield SSE frames as results arrive.

    Uses ``astream_events`` rather than ``ainvoke`` so token deltas surface
    while the graph is still running. Events are filtered by node and tag:

    * ``on_chain_end`` for ``route`` and ``fuse`` -- gives the routing decision
      and the citations before generation begins.
    * ``on_chat_model_stream`` tagged ``final_answer`` -- the answer tokens.
      The tag matters: the router also calls an LLM, and its tokens must not
      leak into the answer.

    Any exception is reported as an ``error`` frame rather than a dropped
    connection, so the client can show something useful.
    """
    graph = get_rag_graph()
    graph_store = get_graph_store()

    initial_state = {
        "question": request.question,
        "history": request.history,
        "top_k": request.top_k,
        "max_hops": request.max_hops,
        "use_graph": request.use_graph,
    }

    answer_parts: list[str] = []
    citations_sent = False
    seeds: list[dict] = []

    try:
        yield _sse("status", {"stage": "retrieving", "message": "Analysing question..."})

        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            name = event.get("name")
            tags = event.get("tags") or []

            # --- Routing decision -------------------------------------
            if kind == "on_chain_end" and name == "route":
                output = (event.get("data") or {}).get("output") or {}
                strategy = output.get("strategy", "hybrid")
                yield _sse(
                    "status",
                    {
                        "stage": "routed",
                        "strategy": strategy,
                        "reason": output.get("routing_reason", ""),
                        "entities": output.get("entities", []),
                        "message": f"Strategy: {strategy}",
                    },
                )

            # --- Graph arm finished: keep seeds for the subgraph ------
            elif kind == "on_chain_end" and name == "graph_search":
                output = (event.get("data") or {}).get("output") or {}
                seeds = output.get("seeds") or []

            # --- Retrieval finished: emit evidence before generation --
            elif kind == "on_chain_end" and name == "fuse" and not citations_sent:
                output = (event.get("data") or {}).get("output") or {}
                citations = output.get("citations") or []
                triples = output.get("used_triples") or []
                citations_sent = True

                yield _sse("citations", [c.model_dump() for c in citations])
                yield _sse("triples", [t.model_dump() for t in triples])

                # The visualiser needs the same subgraph the answer was grounded in.
                subgraph = _build_subgraph(graph_store, seeds, request, triples)
                yield _sse("subgraph", subgraph)
                yield _sse("status", {"stage": "generating", "message": "Writing answer..."})

            # --- Answer tokens ----------------------------------------
            elif kind == "on_chat_model_stream" and "final_answer" in tags:
                chunk = (event.get("data") or {}).get("chunk")
                text = getattr(chunk, "content", "") if chunk is not None else ""
                if isinstance(text, str) and text:
                    answer_parts.append(text)
                    yield _sse("token", text)

        answer = "".join(answer_parts)

        # The "no context" path returns an answer without streaming any tokens,
        # so fall back to the graph's final state for the text.
        if not answer:
            final_state = await graph.ainvoke(initial_state)
            answer = final_state.get("answer", "")
            if answer:
                yield _sse("token", answer)

        yield _sse("done", {"answer": answer})

    except asyncio.CancelledError:
        # Client navigated away mid-stream. Normal, not an error.
        logger.info("Chat stream cancelled by client")
        raise
    except Exception as exc:
        logger.exception("Chat stream failed")
        yield _sse("error", {"message": str(exc)})


def _build_subgraph(graph_store, seeds: list[dict], request: ChatRequest, triples: list) -> dict:
    """Assemble the node/edge payload for the graph inspector.

    Prefers the seeds the graph arm actually resolved. When the router chose the
    vector-only path there are no seeds, so entity names are recovered from the
    triples instead -- and if there are no triples either, the visualiser simply
    receives an empty graph and renders a placeholder.
    """
    seed_ids = [s["id"] for s in seeds]

    if not seed_ids and triples:
        seed_ids = list({entity_key(t.source) for t in triples})[:8]

    if not seed_ids:
        return {"nodes": [], "edges": [], "seeds": []}

    try:
        payload = graph_store.subgraph(
            seed_ids,
            max_hops=request.max_hops or settings.graph_max_hops,
            limit=120,
        )
        payload["seeds"] = seed_ids
        return payload
    except Exception as exc:
        logger.warning("Could not build subgraph: %s", exc)
        return {"nodes": [], "edges": [], "seeds": seed_ids}


@chat_router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Answer a question, streaming the response over SSE.

    Args:
        request: question, optional history, and per-request retrieval overrides.

    Returns:
        ``text/event-stream`` emitting status, citations, triples, subgraph,
        token and done events.
    """
    return StreamingResponse(
        _event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx not to buffer the stream, which would defeat streaming.
            "X-Accel-Buffering": "no",
        },
    )


@chat_router.post("/chat/sync")
async def chat_sync(request: ChatRequest) -> dict:
    """Non-streaming variant returning the whole result in one JSON response.

    Useful for scripted evaluation and for clients that cannot consume SSE.
    """
    graph = get_rag_graph()
    state = await graph.ainvoke(
        {
            "question": request.question,
            "history": request.history,
            "top_k": request.top_k,
            "max_hops": request.max_hops,
            "use_graph": request.use_graph,
        }
    )
    return {
        "answer": state.get("answer", ""),
        "strategy": state.get("strategy", ""),
        "routing_reason": state.get("routing_reason", ""),
        "citations": [c.model_dump() for c in state.get("citations", [])],
        "triples": [t.model_dump() for t in state.get("used_triples", [])],
    }

# ═══════════════════════════════════════════════════════════════════════
# GET /api/v1/graph/* -- graph inspection
# ═══════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)
graph_router = APIRouter(prefix="/api/v1/graph", tags=["graph"])


@graph_router.get("/subgraph", response_model=SubgraphResponse)
async def get_subgraph(
    q: str | None = Query(default=None, description="Free-text query; seeds are resolved by full-text search."),
    entity: str | None = Query(default=None, description="Exact entity name to centre the subgraph on."),
    hops: int = Query(default=2, ge=1, le=3, description="Traversal depth."),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum edges returned."),
) -> SubgraphResponse:
    """Return a node/edge subgraph for visualisation.

    Seeds can be given three ways, tried in order:

    1. ``entity`` -- an exact name, canonicalised to its node id.
    2. ``q`` -- free text, resolved through the full-text index.
    3. Neither -- returns the most connected entities in the graph, which makes
       the endpoint a useful "show me the graph" default for the UI's first load.

    Args:
        q: free-text seed query.
        entity: exact entity name.
        hops: traversal depth (1-3).
        limit: maximum edges.

    Returns:
        Nodes, edges, and the seed ids the traversal started from.

    Raises:
        HTTPException 503: the graph database is unreachable.
    """
    graph_store = get_graph_store()
    if not graph_store.verify():
        raise HTTPException(status_code=503, detail="Graph database is unreachable")

    # --- Resolve seeds ---------------------------------------------------
    seed_ids: list[str] = []
    if entity:
        key = entity_key(entity)
        # Trust the exact key only if that node exists; otherwise fall back to search.
        found = graph_store.find_entities(entity, limit=3)
        seed_ids = [key] if any(f["id"] == key for f in found) else [f["id"] for f in found]
    elif q:
        seed_ids = [f["id"] for f in graph_store.find_entities(q, limit=8)]
    else:
        seed_ids = [e["id"] for e in _most_connected(graph_store, limit=8)]

    if not seed_ids:
        return SubgraphResponse(nodes=[], edges=[], seeds=[])

    payload = graph_store.subgraph(seed_ids, max_hops=hops, limit=limit)
    return SubgraphResponse(
        nodes=[GraphNode(**n) for n in payload["nodes"]],
        edges=[GraphEdge(**e) for e in payload["edges"]],
        seeds=seed_ids,
    )


def _most_connected(graph_store, limit: int = 8) -> list[dict]:
    """Find the highest-degree entities -- the natural centre of the graph.

    Used when the caller gives no seed, so the visualiser opens on the most
    informative part of the graph rather than an arbitrary corner.
    """
    try:
        return graph_store._run(
            """
            MATCH (e:Entity)-[r]-(:Entity)
            WITH e, count(r) AS degree
            RETURN e.id AS id, e.name AS name, coalesce(e.type,'Entity') AS type, degree
            ORDER BY degree DESC
            LIMIT $limit
            """,
            limit=limit,
        )
    except Exception as exc:
        logger.warning("Could not compute most-connected entities: %s", exc)
        return []


@graph_router.get("/entities")
async def search_entities(
    q: str = Query(..., min_length=1, description="Search text."),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    """Search entities by name or description.

    Backs the entity picker in the Graph Inspector.

    Raises:
        HTTPException 503: the graph database is unreachable.
    """
    graph_store = get_graph_store()
    if not graph_store.verify():
        raise HTTPException(status_code=503, detail="Graph database is unreachable")
    return graph_store.find_entities(q, limit=limit)


@graph_router.get("/entity/{entity_id}/provenance")
async def entity_provenance(entity_id: str, limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """Trace an entity back to the chunks it was extracted from.

    This is the provenance mapping made browsable: given a node, return the
    parent blocks (with their text) and child chunk ids that mention it, so a
    graph fact can always be checked against the source passage.

    Raises:
        HTTPException 503: the graph database is unreachable.
    """
    graph_store = get_graph_store()
    if not graph_store.verify():
        raise HTTPException(status_code=503, detail="Graph database is unreachable")

    chunks = graph_store.chunks_for_entities([entity_id], limit=limit)
    parent_store = get_vector_store().parent_store

    parents = []
    seen: set[str] = set()
    for chunk in chunks:
        # A child chunk points at its parent; a parent chunk is its own anchor.
        parent_id = chunk.get("parent_id") or chunk.get("chunk_id")
        if not parent_id or parent_id in seen:
            continue
        seen.add(parent_id)
        parent = parent_store.get(parent_id)
        if parent is not None:
            parents.append(
                {
                    "parent_id": parent.parent_id,
                    "doc_id": parent.doc_id,
                    "pages": parent.pages,
                    "section": parent.section,
                    "text": parent.text,
                }
            )

    return {"entity_id": entity_id, "chunks": chunks, "parents": parents}


@graph_router.get("/stats", response_model=GraphStats)
async def graph_stats() -> GraphStats:
    """Counters across both stores, for the UI sidebar.

    Never raises on a dead database: an unreachable graph reports zeros so the
    sidebar still renders the vector-side numbers.
    """
    graph_store = get_graph_store()
    vector_store = get_vector_store()

    stats = graph_store.stats() if graph_store.verify() else {
        "nodes": 0, "relationships": 0, "chunks": 0, "documents": [],
    }
    return GraphStats(
        nodes=stats["nodes"],
        relationships=stats["relationships"],
        chunks=stats["chunks"],
        documents=stats["documents"] or vector_store.parent_store.doc_ids,
        vector_count=vector_store.vector_count,
        parent_count=vector_store.parent_count,
    )


@graph_router.get("/schema")
async def graph_schema() -> dict:
    """List the entity types and relationship types present in the graph.

    Shows at a glance what the extraction produced -- the fastest way to judge
    whether a prompt change improved or degraded the graph.
    """
    graph_store = get_graph_store()
    if not graph_store.verify():
        raise HTTPException(status_code=503, detail="Graph database is unreachable")

    try:
        entity_types = graph_store._run(
            "MATCH (e:Entity) RETURN coalesce(e.type,'Entity') AS type, count(*) AS count "
            "ORDER BY count DESC LIMIT 50"
        )
        rel_types = graph_store._run(
            "MATCH ()-[r]->() WHERE type(r) <> 'MENTIONED_IN' AND type(r) <> 'HAS_CHILD' "
            "RETURN type(r) AS type, count(*) AS count ORDER BY count DESC LIMIT 50"
        )
        return {"entity_types": entity_types, "relationship_types": rel_types}
    except Exception as exc:
        logger.error("Schema query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# ═══════════════════════════════════════════════════════════════════════
# Application assembly
# ═══════════════════════════════════════════════════════════════════════

# Configure logging before anything imports a logger, so module-level loggers
# inherit this format rather than the root default.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown.

    Startup deliberately does *not* build the FAISS index or fail on an
    unreachable database. The API must come up so ``/health`` can report what is
    wrong; making startup depend on every backing service would turn a
    degradation into an outage.
    """
    logger.info("Starting GraphRAG API")
    logger.info("  LLM       : %s", settings.openrouter_model)
    logger.info("  Embeddings: %s", settings.embedding_model)
    logger.info("  Neo4j     : %s", settings.neo4j_uri)

    # Ensure constraints and indexes exist, but treat failure as non-fatal.
    try:
        store = get_graph_store()
        if store.verify():
            store.init_schema()
            logger.info("  Graph     : connected, schema ready")
        else:
            logger.warning("  Graph     : UNREACHABLE (vector-only mode)")
    except Exception as exc:
        logger.warning("  Graph     : schema init failed: %s", exc)

    yield

    logger.info("Shutting down; closing Neo4j driver")
    get_graph_store().close()


app = FastAPI(
    title="GraphRAG - Emirates NBD Annual Report",
    description=(
        "Hierarchical parent-child RAG with knowledge-graph fusion.\n\n"
        "- **Ingestion**: Docling parsing, parent/child chunking, FAISS vectors, "
        "LLM entity+relationship extraction into Neo4j.\n"
        "- **Retrieval**: an intent router picks vector, graph, or hybrid; hybrid "
        "fuses both with Reciprocal Rank Fusion and a cross-encoder reranker.\n"
        "- **Generation**: grounded answers streamed over SSE with citations and "
        "graph triples."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# The Streamlit UI is served from a different origin, so it needs CORS.
# Wide-open is acceptable for a local single-user app; tighten for deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(chat_router)
app.include_router(graph_router)


@app.get("/health", tags=["system"])
async def health() -> dict:
    """Report the state of every backing component.

    Always returns 200. A 503 would make the endpoint useless for diagnosing
    *which* component is down, which is the only reason to call it.
    """
    vector_store = get_vector_store()
    graph_store = get_graph_store()

    graph_ok = graph_store.verify()
    vector_count = vector_store.vector_count

    return {
        "status": "ok",
        "components": {
            "vector_store": {
                "ready": vector_count > 0,
                "vectors": vector_count,
                "parents": vector_store.parent_count,
                "documents": vector_store.parent_store.doc_ids,
            },
            "graph_store": {"ready": graph_ok, "uri": settings.neo4j_uri},
            "llm": {"model": settings.openrouter_model, "configured": bool(settings.openrouter_api_key)},
            "embeddings": {"model": settings.embedding_model},
        },
        "config": {
            "parent_chunk_tokens": settings.parent_chunk_tokens,
            "child_chunk_tokens": settings.child_chunk_tokens,
            "routing_mode": settings.routing_mode,
            "reranker": settings.reranker_model if settings.use_reranker else None,
        },
    }


@app.get("/api", tags=["system"])
async def api_index() -> dict:
    """Machine-readable index of the endpoints."""
    return {
        "service": "GraphRAG - Emirates NBD",
        "docs": "/docs",
        "endpoints": {
            "ingest": "POST /api/v1/ingest",
            "ingest_upload": "POST /api/v1/ingest/upload",
            "job_status": "GET /api/v1/ingest/{job_id}",
            "chat_stream": "POST /api/v1/chat",
            "chat_sync": "POST /api/v1/chat/sync",
            "subgraph": "GET /api/v1/graph/subgraph",
            "health": "GET /health",
        },
    }


@app.get("/api/v1/document/{doc_id}", tags=["documents"])
async def document_file(doc_id: str) -> FileResponse:
    """Stream the source PDF a document was ingested from.

    This is what makes a citation verifiable rather than merely plausible: the
    page number on a citation links here with a ``#page=N`` fragment, which
    every browser PDF viewer honours, so one click lands on the page the claim
    came from.

    Only files inside the configured document directories are served, and the
    resolved path is checked against them -- ``doc_id`` arrives from the client
    and must never be able to address an arbitrary file.

    Raises:
        HTTPException 404: no ingested document matches that id.
    """
    allowed_roots = [
        settings.docs_path.resolve(),
        (settings.data_path / "uploads").resolve(),
    ]

    for root in allowed_roots:
        if not root.is_dir():
            continue
        for candidate in root.glob("*.pdf"):
            if candidate.stem != doc_id:
                continue
            resolved = candidate.resolve()
            # Defence in depth: a symlink out of the directory is still a
            # traversal, so verify containment after resolving.
            if not any(resolved.is_relative_to(r) for r in allowed_roots):
                continue
            return FileResponse(
                resolved,
                media_type="application/pdf",
                # inline, so the browser's viewer opens it at the #page anchor
                # instead of downloading the file.
                headers={"Content-Disposition": f'inline; filename="{resolved.name}"'},
            )

    raise HTTPException(status_code=404, detail=f"No source document for '{doc_id}'")


# --------------------------------------------------------------------------- #
# Web client
#
# The chat UI is plain HTML/CSS/JS with no build step, so FastAPI can serve it
# directly. One process and one origin for both the API and the page: no CORS
# in practice, no second container, and the UI cannot drift out of step with
# the endpoints it calls.
#
# Mounted last, because a mount at "/" would otherwise shadow the API routes.
# --------------------------------------------------------------------------- #

_WEB_DIR = PROJECT_ROOT / "web"

if _WEB_DIR.is_dir():
    # Assets are addressed as /static/... by index.html.
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the chat interface."""
        return FileResponse(_WEB_DIR / "index.html")
else:
    logger.warning("web/ not found; serving the API only")

    @app.get("/", include_in_schema=False)
    async def index_missing() -> dict:
        """Fallback when the web client is absent."""
        return {"service": "GraphRAG", "docs": "/docs", "web_client": "not installed"}
