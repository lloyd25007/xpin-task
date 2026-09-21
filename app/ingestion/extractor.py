"""
Entity and relationship extraction via LLM structured output.

Extraction runs over **parent** chunks, not children. A 200-token child rarely
contains a complete relationship -- the subject is often a paragraph above the
verb -- whereas a ~1000-token parent usually holds the whole statement. The
resulting triples are then attributed to that parent *and* to all of its child
chunks, which is what gives every edge a provenance trail down to the exact
units the vector index searches.

Reliability matters more than cleverness here: one bad chunk must never abort a
400-page ingest. Every call is retried with backoff, falls back from tool-calling
to JSON mode if the model does not support tools, and finally returns an empty
extraction rather than raising.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.metadata import utc_now
from app.ingestion.chunker import ParentChunk
from app.chat.llm import get_extraction_llm, get_structured_extractor
from app.chat import prompts
from app.core.schemas import ExtractedEntity, ExtractedRelationship, GraphExtraction
from app.storage.graph_store import entity_key, sanitize_rel_type

logger = logging.getLogger(__name__)

# Chunks shorter than this rarely contain an extractable relationship, and
# calling the LLM on them wastes quota on the free tier.
_MIN_CHARS_FOR_EXTRACTION = 200

# Hard cap on text sent per call. Parents are ~1000 tokens, but a pathological
# table-heavy chunk can be far larger after cleaning.
_MAX_CHARS_PER_CALL = 8000


class _ExtractionFailed(Exception):
    """Internal marker so tenacity retries only genuine extraction failures."""


class _RateLimiter:
    """Token-bucket throttle shared by every extraction worker.

    OpenRouter's ``:free`` models allow only a handful of requests per minute.
    Without a throttle, four worker threads burn the whole allowance in seconds
    and then spend the rest of the ingest in retry backoff -- which is slower
    than simply not exceeding the limit in the first place.

    This paces requests to a fixed minimum interval. It is intentionally simple:
    a lock, a timestamp, and a sleep. Requests are serialised only at the moment
    of dispatch, so workers still overlap on the network round trip.
    """

    def __init__(self, requests_per_minute: int) -> None:
        """Set the pace. A non-positive rate disables throttling entirely."""
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        """Block until the next request is allowed to go out."""
        if self._min_interval <= 0:
            return
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


# Built once from configuration and shared by every worker in the process.
_rate_limiter = _RateLimiter(settings.extraction_requests_per_minute)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(_ExtractionFailed),
    reraise=False,
)
def _call_structured(text: str, doc_id: str, page_hint: str) -> GraphExtraction:
    """Ask the LLM for entities and relationships as a validated object.

    Tries native structured output first (the model supports tool calling, so
    the schema is enforced server-side). If that path fails -- some free-tier
    endpoints silently drop tool definitions -- falls back to plain JSON-mode
    prompting and parses the response defensively.

    Args:
        text: the parent chunk text.
        doc_id: document identifier, included in the prompt for entity resolution.
        page_hint: human-readable page range, e.g. " (pages 12-13)".

    Returns:
        A :class:`GraphExtraction`. Empty rather than raising when the model
        returns nothing usable.

    Raises:
        _ExtractionFailed: on transport errors, so tenacity retries.
    """
    # Pace the call so the free-tier rate limit is respected rather than hit.
    _rate_limiter.acquire()

    llm = get_extraction_llm()
    user_prompt = prompts.EXTRACTION_USER.format(
        doc_id=doc_id, page_hint=page_hint, text=text[:_MAX_CHARS_PER_CALL]
    )
    messages = [
        {"role": "system", "content": prompts.EXTRACTION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    # --- Preferred path: schema-enforced structured output -----------------
    # Built through the fallback helper rather than llm.with_structured_output:
    # once a model is wrapped for failover it is a plain runnable and no longer
    # exposes that method, so structuring has to happen per provider first.
    try:
        result = get_structured_extractor(GraphExtraction).invoke(messages)
        if isinstance(result, GraphExtraction):
            return result
        if isinstance(result, dict):
            return GraphExtraction.model_validate(result)
        logger.debug("Structured output returned %s; trying JSON fallback", type(result))
    except Exception as exc:
        # Network/rate-limit errors deserve a retry; schema errors do not, but
        # distinguishing them reliably across providers is not worth the
        # complexity -- the JSON fallback below covers the non-retryable case.
        logger.debug("Structured output failed (%s); trying JSON fallback", exc)

    # --- Fallback path: ask for raw JSON and parse it ----------------------
    try:
        raw = llm.invoke(
            messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Return ONLY a JSON object, no prose and no code fences, with keys "
                        '"entities" (each: name, type, description) and "relationships" '
                        '(each: source, source_type, target, target_type, type, evidence).'
                    ),
                }
            ]
        )
        return _parse_json_extraction(raw.content if hasattr(raw, "content") else str(raw))
    except Exception as exc:
        raise _ExtractionFailed(str(exc)) from exc


def _parse_json_extraction(raw: str) -> GraphExtraction:
    """Parse a possibly-messy JSON response into a GraphExtraction.

    Models wrap JSON in ```json fences, prepend explanations, or emit trailing
    commas. This extracts the outermost JSON object and validates it, returning
    an empty extraction rather than raising when the text is unsalvageable.
    """
    if not isinstance(raw, str):
        raw = str(raw)

    # Strip code fences, then grab the outermost {...} block.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.M)
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        logger.debug("No JSON object found in extraction response")
        return GraphExtraction()

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        # One common repair: trailing commas before a closing brace/bracket.
        try:
            data = json.loads(re.sub(r",(\s*[}\]])", r"\1", match.group(0)))
        except json.JSONDecodeError as exc:
            logger.debug("Could not parse extraction JSON: %s", exc)
            return GraphExtraction()

    entities = [
        ExtractedEntity(
            name=str(e.get("name", "")).strip(),
            type=str(e.get("type", "Entity")).strip() or "Entity",
            description=str(e.get("description", "") or "")[:300],
        )
        for e in data.get("entities", []) or []
        if isinstance(e, dict) and str(e.get("name", "")).strip()
    ]
    relationships = [
        ExtractedRelationship(
            source=str(r.get("source", "")).strip(),
            source_type=str(r.get("source_type", "Entity")).strip() or "Entity",
            target=str(r.get("target", "")).strip(),
            target_type=str(r.get("target_type", "Entity")).strip() or "Entity",
            type=str(r.get("type", "RELATED_TO")).strip() or "RELATED_TO",
            evidence=str(r.get("evidence", "") or "")[:400],
        )
        for r in data.get("relationships", []) or []
        if isinstance(r, dict) and str(r.get("source", "")).strip() and str(r.get("target", "")).strip()
    ]
    return GraphExtraction(entities=entities, relationships=relationships)


def extract_from_parent(parent: ParentChunk) -> GraphExtraction:
    """Extract a graph fragment from one parent chunk.

    Args:
        parent: the context block to read.

    Returns:
        The extraction, or an empty one if the chunk is too short to be worth a
        call or every retry failed. Never raises -- ingestion of the remaining
        chunks must continue regardless.
    """
    if len(parent.text) < _MIN_CHARS_FOR_EXTRACTION:
        return GraphExtraction()

    page_hint = ""
    if parent.pages:
        page_hint = (
            f" (page {parent.pages[0]})"
            if len(parent.pages) == 1
            else f" (pages {parent.pages[0]}-{parent.pages[-1]})"
        )

    try:
        result = _call_structured(parent.text, parent.doc_id, page_hint)
        return result if isinstance(result, GraphExtraction) else GraphExtraction()
    except Exception as exc:
        logger.warning("Extraction failed for parent %s: %s", parent.parent_id, exc)
        return GraphExtraction()


def to_graph_rows(parent: ParentChunk, extraction: GraphExtraction) -> tuple[list[dict], list[dict], list[dict]]:
    """Convert one extraction into rows ready for Neo4j.

    This is where provenance is attached. Every entity is linked to the parent
    chunk it came from *and* to that parent's children, and every relationship
    records both id sets -- so a triple retrieved by traversal can always be
    traced back to the passage that justifies it and to the vectors that would
    have found it.

    Entity names are canonicalised with :func:`entity_key` so "Emirates NBD" and
    "emirates nbd" collapse to one node, and relationship endpoints are dropped
    if either side is missing from the entity list -- a self-consistent fragment
    is worth more than a dangling edge.

    Args:
        parent: the chunk the extraction came from.
        extraction: the LLM's output.

    Returns:
        ``(entity_rows, relationship_rows, mention_rows)`` for the three
        corresponding GraphStore write methods.
    """
    seen_at = utc_now()  # one timestamp for the whole extraction, not per row
    entity_rows: list[dict] = []
    known: dict[str, str] = {}  # canonical key -> display name

    for entity in extraction.entities:
        name = entity.name.strip()
        if not name or len(name) > 200:
            continue
        key = entity_key(name)
        known[key] = name
        entity_rows.append(
            {
                "id": key,
                "name": name,
                "type": (entity.type or "Entity").strip()[:60],
                "description": (entity.description or "").strip()[:300],
                "doc_id": parent.doc_id,
                # Where and when this mention was seen, so the node can record
                # its page footprint and first/last-seen timestamps.
                "page": parent.pages[0] if parent.pages else None,
                "seen_at": seen_at,
            }
        )

    relationship_rows: list[dict] = []
    for rel in extraction.relationships:
        source, target = rel.source.strip(), rel.target.strip()
        if not source or not target or source.lower() == target.lower():
            continue  # self-loops carry no information here

        source_key, target_key = entity_key(source), entity_key(target)

        # The model sometimes references an entity in a relationship without
        # listing it under "entities". Rather than drop the edge, create the
        # missing node from the relationship's own type hints.
        for key, name, type_hint in (
            (source_key, source, rel.source_type),
            (target_key, target, rel.target_type),
        ):
            if key not in known:
                known[key] = name
                entity_rows.append(
                    {
                        "id": key,
                        "name": name,
                        "type": (type_hint or "Entity").strip()[:60],
                        "description": "",
                        "doc_id": parent.doc_id,
                        "page": parent.pages[0] if parent.pages else None,
                        "seen_at": seen_at,
                    }
                )

        relationship_rows.append(
            {
                "source_id": source_key,
                "target_id": target_key,
                "type": sanitize_rel_type(rel.type),
                "evidence": (rel.evidence or "").strip()[:400],
                # Provenance in both directions of the hierarchy.
                "parent_ids": [parent.parent_id],
                "child_ids": parent.child_ids,
                "doc_id": parent.doc_id,
                # Page footprint and lineage. The model id matters because a
                # graph built by two different models is worth telling apart.
                "pages": parent.pages,
                "seen_at": seen_at,
                "model": settings.openrouter_model,
            }
        )

    # Link every entity to the parent chunk and to each of its children.
    mention_rows = [
        {"entity_id": key, "chunk_id": chunk_id}
        for key in known
        for chunk_id in [parent.parent_id, *parent.child_ids]
    ]

    return entity_rows, relationship_rows, mention_rows


def extract_batch(parents: list[ParentChunk], max_workers: int | None = None,
                  progress_callback=None) -> list[tuple[ParentChunk, GraphExtraction]]:
    """Extract from many parent chunks concurrently.

    Extraction is network-bound, so threads (not processes) are the right tool:
    each worker spends its time waiting on OpenRouter. ``max_workers`` is kept
    low by default because free-tier endpoints rate-limit aggressively, and a
    429 storm is slower than a modest amount of parallelism.

    Args:
        parents: chunks to process.
        max_workers: concurrent LLM calls.
        progress_callback: called as ``(done, total)`` after each completion,
            used to drive the ingestion job's progress field.

    Returns:
        ``(parent, extraction)`` pairs. Order is not guaranteed to match the
        input, which does not matter since every result carries its own parent.
    """
    max_workers = max_workers or settings.extraction_workers
    results: list[tuple[ParentChunk, GraphExtraction]] = []
    total = len(parents)
    if total == 0:
        return results

    logger.info("Extracting graph from %d parent chunks (%d workers)", total, max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(extract_from_parent, p): p for p in parents}
        for done, future in enumerate(as_completed(futures), start=1):
            parent = futures[future]
            try:
                results.append((parent, future.result()))
            except Exception as exc:
                logger.warning("Worker failed on parent %s: %s", parent.parent_id, exc)
                results.append((parent, GraphExtraction()))
            if progress_callback:
                progress_callback(done, total)

    entities = sum(len(e.entities) for _, e in results)
    relationships = sum(len(e.relationships) for _, e in results)
    logger.info("Extracted %d entities and %d relationships", entities, relationships)
    return results
