"""
Pydantic models shared by the ingestion pipeline, the FastAPI routes and the UI.

Keeping every wire-format in one module means the Streamlit client and the
FastAPI server can never silently disagree about a field name, and the OpenAPI
docs at /docs describe the real contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Graph extraction — the schema the LLM is forced to fill in.
# =============================================================================

class ExtractedEntity(BaseModel):
    """A single named entity (a future node in Neo4j)."""

    name: str = Field(description="Exact surface form as written in the document, e.g. 'Emirates NBD'.")
    type: str = Field(description="Entity category, e.g. Company, Subsidiary, Person, Product, Segment, Metric, Technology, Location.")
    description: str = Field(default="", description="One short clause about the entity, drawn only from this chunk.")


class ExtractedRelationship(BaseModel):
    """A typed, directed edge between two entities.

    ``type`` becomes the literal Neo4j relationship type, which is why the
    prompt insists on UPPER_SNAKE_CASE verbs such as OWNS or ACQUIRED.
    """

    source: str = Field(description="Name of the subject entity.")
    source_type: str = Field(default="Entity", description="Category of the subject entity.")
    target: str = Field(description="Name of the object entity.")
    target_type: str = Field(default="Entity", description="Category of the object entity.")
    type: str = Field(description="UPPER_SNAKE_CASE verb, e.g. OWNS, ACQUIRED, REPORTED, INVESTS_IN.")
    evidence: str = Field(default="", description="Short verbatim quote from the chunk supporting the edge.")


class GraphExtraction(BaseModel):
    """Container the LLM returns for one parent chunk."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# =============================================================================
# Ingestion API
# =============================================================================

class JobStatus(str, Enum):
    """Lifecycle of an asynchronous ingestion job."""

    PENDING = "pending"     # accepted, not yet picked up
    RUNNING = "running"     # actively parsing / embedding / extracting
    COMPLETED = "completed"
    FAILED = "failed"


class IngestRequest(BaseModel):
    """Body for POST /api/v1/ingest when ingesting a file already on disk."""

    path: str | None = Field(default=None, description="Server-side path to a PDF. Defaults to the configured report.")
    doc_id: str | None = Field(default=None, description="Stable id for the document. Defaults to the filename stem.")
    extract_graph: bool = Field(default=True, description="Set false to build vectors only and skip LLM graph extraction.")
    reset: bool = Field(default=False, description="Wipe the existing index and graph before ingesting.")
    max_pages: int | None = Field(default=None, description="Ingest only the first N pages — useful for smoke tests.")


class IngestJob(BaseModel):
    """Status document for one ingestion run, polled by GET /api/v1/ingest/{job_id}."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    doc_id: str | None = None
    source: str | None = None
    stage: str = Field(default="queued", description="Human-readable current step.")
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    pages: int = 0
    parent_chunks: int = 0
    child_chunks: int = 0
    entities: int = 0
    relationships: int = 0
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


# =============================================================================
# Chat API
# =============================================================================

class ChatRequest(BaseModel):
    """Body for POST /api/v1/chat."""

    question: str = Field(min_length=1, description="The user's natural-language question.")
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior turns as [{'role': 'user'|'assistant', 'content': ...}] for follow-up resolution.",
    )
    top_k: int | None = Field(default=None, description="Override the number of child chunks retrieved.")
    max_hops: int | None = Field(default=None, description="Override graph traversal depth.")
    use_graph: bool = Field(default=True, description="Set false to run plain vector RAG without graph context.")


class Citation(BaseModel):
    """One retrieved parent block, rendered by the UI as an expandable card."""

    parent_id: str
    doc_id: str
    text: str
    pages: list[int] = Field(default_factory=list)
    section: str | None = None
    score: float = Field(default=0.0, description="Best similarity score among the child chunks that matched.")
    matched_child_ids: list[str] = Field(default_factory=list, description="Provenance: which children hit.")
    # --- metadata, so a reader can judge a passage before opening it ---
    source: str = Field(default="", description="Source filename.")
    char_count: int = Field(default=0)
    token_count: int = Field(default=0)
    has_table: bool = Field(default=False, description="Chunk contains a recovered table.")
    has_figures: bool = Field(default=False, description="Chunk contains currency amounts or percentages.")
    ingested_at: str = Field(default="", description="When this chunk was indexed.")
    retrieved_by: list[str] = Field(default_factory=list, description="Which retriever(s) found it: vector, graph.")


class Triple(BaseModel):
    """One graph relationship shown alongside the answer."""

    source: str
    source_type: str = "Entity"
    relation: str
    target: str
    target_type: str = "Entity"
    evidence: str = ""
    parent_ids: list[str] = Field(default_factory=list, description="Provenance back to the vector chunks.")
    hop: int = Field(default=1, description="Traversal distance from the seed entity.")
    pages: list[int] = Field(default_factory=list, description="Pages this relationship was extracted from.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0,
                              description="Derived from how many chunks independently produced this triple.")
    mention_count: int = Field(default=1, description="Independent extractions of this triple.")


class GraphNode(BaseModel):
    """Node in the subgraph payload consumed by the visualiser."""

    id: str
    label: str
    type: str = "Entity"
    description: str = ""


class GraphEdge(BaseModel):
    """Edge in the subgraph payload consumed by the visualiser."""

    source: str
    target: str
    type: str
    evidence: str = ""


class SubgraphResponse(BaseModel):
    """Response of GET /api/v1/graph/subgraph — plain node/edge JSON."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    seeds: list[str] = Field(default_factory=list, description="Entities the traversal started from.")


class ChatEvent(BaseModel):
    """One Server-Sent Event frame.

    The UI switches on ``event``:
      status    -> progress line while retrieval runs
      citations -> the parent chunks (sent before generation starts)
      triples   -> graph relationships used as context
      subgraph  -> node/edge JSON for the graph inspector
      token     -> a streamed fragment of the answer
      done      -> terminal frame carrying the full answer
      error     -> terminal frame carrying a message
    """

    event: Literal["status", "citations", "triples", "subgraph", "token", "done", "error"]
    data: Any = None


class GraphStats(BaseModel):
    """Summary counters for the sidebar / health view."""

    nodes: int = 0
    relationships: int = 0
    chunks: int = 0
    documents: list[str] = Field(default_factory=list)
    vector_count: int = 0
    parent_count: int = 0
