"""
End-to-end ingestion.

Stages, in order::

    load  ->  chunk  ->  embed  ->  extract  ->  write graph  ->  persist

Each stage reports progress through a callback so the async job endpoint can
show where a long ingest has got to. Vectors are written and saved *before*
graph extraction begins: extraction is the slow, failure-prone part, and a run
that dies halfway through it should still leave a working vector index behind
rather than nothing at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from app.core.config import settings
from app.ingestion.chunker import chunk_documents
from app.core.metadata import DocumentMetadata
from app.ingestion.extractor import extract_batch, to_graph_rows
from app.ingestion.loader import load_pdf_docling
from app.storage.graph_store import GraphStore
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Signature of the progress hook: (stage_name, fraction_complete, extra_counters)
ProgressFn = Callable[[str, float, dict], None]


def _noop_progress(stage: str, progress: float, counters: dict) -> None:
    """Default progress sink used when no callback is supplied."""
    logger.info("[%s] %.0f%% %s", stage, progress * 100, counters or "")


def _chunk_rows(parents, children) -> tuple[list[dict], list[dict]]:
    """Build the ``:Chunk`` node rows and their parent-child links.

    Chunk nodes are the provenance anchors in the graph. They store only a short
    preview -- the full text lives in the parent store and FAISS -- so the graph
    stays small while still letting a Cypher query walk from an entity to the
    exact passage it came from.

    Returns:
        ``(chunk_rows, link_rows)`` ready for the GraphStore write methods.
    """
    def _row(chunk, chunk_id: str, kind: str, parent_id: str | None) -> dict:
        """One :Chunk node row, carrying the same metadata as the vector."""
        m = chunk.meta or {}
        return {
            "id": chunk_id,
            "kind": kind,
            "doc_id": chunk.doc_id,
            "source": m.get("source", ""),
            "parent_id": parent_id,
            "pages": chunk.pages,
            "page_start": m.get("page_start"),
            "page_end": m.get("page_end"),
            "section": chunk.section or "",
            "preview": chunk.text[:300],
            "char_count": m.get("char_count", len(chunk.text)),
            "token_count": m.get("token_count", 0),
            "has_table": m.get("has_table", False),
            "has_figures": m.get("has_figures", False),
            "ingested_at": m.get("ingested_at", ""),
            "parser": m.get("parser", ""),
        }

    chunk_rows = [_row(p, p.parent_id, "parent", None) for p in parents]
    chunk_rows += [_row(c, c.child_id, "child", c.parent_id) for c in children]
    link_rows = [{"parent_id": c.parent_id, "child_id": c.child_id} for c in children]
    return chunk_rows, link_rows


def ingest_pdf(
    pdf_path: str | Path | None = None,
    doc_id: str | None = None,
    extract_graph: bool = True,
    reset: bool = False,
    max_pages: int | None = None,
    progress: ProgressFn | None = None,
    vector_store: VectorStore | None = None,
    graph_store: GraphStore | None = None,
) -> dict:
    """Ingest one PDF into the vector index and the knowledge graph.

    Args:
        pdf_path: PDF to ingest; defaults to the configured annual report.
        doc_id: stable document id; defaults to the filename stem.
        extract_graph: set ``False`` to build vectors only. Useful for a fast
            first pass, since extraction dominates the runtime.
        reset: wipe both stores before ingesting. Without it, ingestion is
            additive and idempotent -- content-addressed ids mean re-running
            over the same PDF updates rather than duplicates.
        max_pages: ingest only the first N pages (smoke tests and demos).
        progress: callback ``(stage, fraction, counters)``.
        vector_store / graph_store: injected for reuse by the API; created on
            demand when omitted.

    Returns:
        Counters describing what was written: pages, parent_chunks,
        child_chunks, entities, relationships.

    Raises:
        FileNotFoundError: if the PDF does not exist.
    """
    report = progress or _noop_progress
    pdf_path = settings.resolve(pdf_path or settings.default_pdf_path)
    doc_id = doc_id or Path(pdf_path).stem

    vector_store = vector_store or VectorStore()
    graph_store = graph_store or GraphStore()

    counters = {
        "pages": 0, "parent_chunks": 0, "child_chunks": 0,
        "entities": 0, "relationships": 0,
    }

    # ---------------------------------------------------------------- reset --
    if reset:
        report("resetting stores", 0.0, {})
        vector_store.clear()
        try:
            graph_store.clear()
        except Exception as exc:
            # A missing/unreachable graph must not block a vector-only ingest.
            logger.warning("Could not clear graph: %s", exc)

    # ----------------------------------------------------------------- load --
    report("parsing pdf with docling", 0.02, {})
    # Docling recovers reading order and table structure; it falls back to
    # pypdf internally if unavailable, so this call always returns pages.
    pages = load_pdf_docling(pdf_path, doc_id=doc_id, max_pages=max_pages)
    counters["pages"] = len(pages)
    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path}")
    report("parsed pdf", 0.10, {"pages": len(pages)})

    # ---------------------------------------------------------------- chunk --
    report("chunking", 0.12, {})
    # Built once per ingest and stamped onto every chunk, so a vector and its
    # graph counterpart always describe the same source in the same terms.
    doc_meta = DocumentMetadata.build(
        path=pdf_path,
        doc_id=doc_id,
        page_count=len(pages),
        parser=pages[0].metadata.get("parser", "unknown") if pages else "unknown",
        embedding_model=settings.embedding_model,
        parent_tokens=settings.parent_chunk_tokens,
        child_tokens=settings.child_chunk_tokens,
    )
    parents, children = chunk_documents(pages, doc_meta=doc_meta)
    counters["parent_chunks"] = len(parents)
    counters["child_chunks"] = len(children)
    report("chunked", 0.20, {"parent_chunks": len(parents), "child_chunks": len(children)})

    # ---------------------------------------------------------------- embed --
    # Children only: parents are never embedded, just stored for expansion.
    report("embedding child chunks", 0.22, {})
    vector_store.add_parents(parents)
    vector_store.add_children(children)
    vector_store.save()   # persist now, so a later extraction failure loses nothing
    report("vector index built", 0.45, {"vectors": vector_store.vector_count})

    # ------------------------------------------------------- graph: chunks --
    graph_available = graph_store.verify()
    if not graph_available:
        logger.warning("Neo4j unreachable; vectors are indexed but no graph was written")
        report("graph skipped (database unreachable)", 1.0, counters)
        return counters

    graph_store.init_schema()
    report("writing chunk provenance", 0.48, {})
    chunk_rows, link_rows = _chunk_rows(parents, children)
    graph_store.upsert_chunks(chunk_rows)
    graph_store.link_parent_child_chunks(link_rows)
    report("chunk provenance written", 0.52, {"chunks": len(chunk_rows)})

    if not extract_graph:
        report("done (vectors only)", 1.0, counters)
        return counters

    # -------------------------------------------------------------- extract --
    # The slowest stage by far: one LLM call per parent chunk.
    def _extraction_progress(done: int, total: int) -> None:
        """Map extraction completion onto the 0.52 - 0.90 slice of the bar."""
        fraction = 0.52 + 0.38 * (done / max(total, 1))
        report(f"extracting graph ({done}/{total} chunks)", fraction, {})

    results = extract_batch(parents, progress_callback=_extraction_progress)

    # ---------------------------------------------------- graph: write rows --
    report("writing entities and relationships", 0.90, {})
    all_entities: list[dict] = []
    all_relationships: list[dict] = []
    all_mentions: list[dict] = []

    for parent, extraction in results:
        entity_rows, rel_rows, mention_rows = to_graph_rows(parent, extraction)
        all_entities.extend(entity_rows)
        all_relationships.extend(rel_rows)
        all_mentions.extend(mention_rows)

    # Deduplicate before writing: the same entity appears in many chunks, and
    # collapsing here turns thousands of MERGEs into hundreds.
    unique_entities = {e["id"]: e for e in all_entities}
    graph_store.upsert_entities(list(unique_entities.values()))
    written = graph_store.upsert_relationships(all_relationships)

    # Mentions are already unique per (entity, chunk) pair after this dedupe.
    unique_mentions = {(m["entity_id"], m["chunk_id"]): m for m in all_mentions}
    graph_store.link_entities_to_chunks(list(unique_mentions.values()))

    counters["entities"] = len(unique_entities)
    counters["relationships"] = written

    report("done", 1.0, counters)
    logger.info(
        "Ingested %s: %d pages, %d parents, %d children, %d entities, %d relationships",
        doc_id, counters["pages"], counters["parent_chunks"],
        counters["child_chunks"], counters["entities"], counters["relationships"],
    )
    return counters
