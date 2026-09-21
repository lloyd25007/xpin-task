"""
Metadata attached to everything the ingestion pipeline produces.

One module, because the vector store and the graph store must describe the
same chunk the same way. If a chunk is "page 12-13 of the strategic report,
ingested at T, 840 characters, contains a table", then a FAISS hit and a Neo4j
node both need to say that — otherwise provenance breaks the moment you try to
reconcile a graph fact with the passage it came from.

Three levels, each building on the last:

* :class:`DocumentMetadata` — facts about the source file. Computed once per
  ingest and stamped onto every chunk derived from it.
* **chunk metadata** — position, size and content signals for one chunk.
* **graph metadata** — timestamps and mention counts that only make sense once
  the same entity has been seen in several places.

Why bother: metadata is what makes retrieval *filterable* and results
*auditable*. Without ``page_start`` you cannot scope a query to a section;
without ``ingested_at`` you cannot tell a stale vector from a fresh one; without
``mention_count`` every graph edge looks equally certain, including the one the
model extracted once from a caption.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string.

    A string rather than a datetime because these values are written to JSON,
    to FAISS document metadata and to Neo4j properties — all three want a
    scalar, and a single canonical format avoids three different conversions.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Markdown tables are how Docling emits recovered table structure. A chunk
# containing one is worth flagging: it holds figures tied to row labels, which
# is exactly the content that answers quantitative questions.
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.M)

# Currency amounts and percentages. Their presence marks a chunk as carrying
# hard numbers rather than narrative.
_FIGURE_RE = re.compile(r"(?:AED|USD|EUR|GBP)\s?[\d,.]+|[\d,.]+\s?(?:%|bn|mn|billion|million)", re.I)


@dataclass
class DocumentMetadata:
    """Facts about one source document, stamped onto all of its chunks."""

    doc_id: str
    source_path: str
    source_name: str
    file_size_bytes: int = 0
    content_hash: str = ""
    page_count: int = 0
    parser: str = "docling"
    ingested_at: str = field(default_factory=utc_now)
    embedding_model: str = ""
    parent_chunk_tokens: int = 0
    child_chunk_tokens: int = 0

    @classmethod
    def build(cls, path: str | Path, doc_id: str, page_count: int, parser: str,
              embedding_model: str, parent_tokens: int, child_tokens: int) -> "DocumentMetadata":
        """Derive document metadata from the file on disk.

        The content hash is over size + mtime + name rather than the file bytes:
        hashing 7MB on every ingest costs more than it is worth, and those three
        change together whenever the file is actually replaced.
        """
        p = Path(path)
        try:
            stat = p.stat()
            size, stamp = stat.st_size, int(stat.st_mtime)
        except OSError:
            size, stamp = 0, 0

        digest = hashlib.sha1(f"{p.name}|{size}|{stamp}".encode()).hexdigest()[:12]
        return cls(
            doc_id=doc_id,
            source_path=str(p),
            source_name=p.name,
            file_size_bytes=size,
            content_hash=digest,
            page_count=page_count,
            parser=parser,
            embedding_model=embedding_model,
            parent_chunk_tokens=parent_tokens,
            child_chunk_tokens=child_tokens,
        )

    def as_dict(self) -> dict:
        """Plain dict, for embedding into chunk metadata."""
        return asdict(self)


def content_signals(text: str) -> dict:
    """Describe what *kind* of content a chunk holds.

    These are cheap, deterministic signals — no model involved — that let a
    caller prefer table-bearing chunks for "how much…" questions and prose for
    "why…" questions, and let a reader see at a glance why a chunk was ranked
    where it was.

    Returns:
        ``has_table``, ``has_figures``, ``figure_count``, ``word_count``.
    """
    figures = _FIGURE_RE.findall(text)
    return {
        "has_table": bool(_TABLE_RE.search(text)),
        "has_figures": bool(figures),
        "figure_count": len(figures),
        "word_count": len(text.split()),
    }


def estimate_tokens(text: str) -> int:
    """Token count for a chunk, exact where possible.

    Uses tiktoken so the number matches the budget the chunker was configured
    with, and falls back to the standard 4-characters-per-token approximation
    when the encoding files are unavailable offline.
    """
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def chunk_metadata(*, text: str, doc: DocumentMetadata, pages: list[int],
                   section: str | None, index: int, kind: str) -> dict:
    """Assemble the full metadata record for one chunk.

    Args:
        text: the chunk's text.
        doc: metadata of the document it came from.
        pages: page numbers the chunk spans.
        section: best-effort section heading.
        index: position of the chunk within its parent (child) or document (parent).
        kind: ``"parent"`` or ``"child"``.

    Returns:
        A flat dict of scalars and short lists. Flat and scalar because FAISS
        metadata, JSON and Neo4j properties all reject nested structures.
    """
    return {
        # identity and provenance
        "doc_id": doc.doc_id,
        "source": doc.source_name,
        "kind": kind,
        # position
        "pages": pages,
        "page_start": pages[0] if pages else None,
        "page_end": pages[-1] if pages else None,
        "section": section,
        "index": index,
        # size
        "char_count": len(text),
        "token_count": estimate_tokens(text),
        # what the chunk contains
        **content_signals(text),
        # lineage — lets you spot vectors built by an older config or model
        "ingested_at": doc.ingested_at,
        "parser": doc.parser,
        "embedding_model": doc.embedding_model,
        "content_hash": doc.content_hash,
    }
