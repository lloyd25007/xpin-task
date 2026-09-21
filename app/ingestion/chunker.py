"""
Hierarchical ("small-to-big") parent-child chunking.

The retrieval problem this solves: small chunks embed precisely -- a 200-token
passage is *about* one thing, so its vector is sharp and similarity search finds
it reliably. But 200 tokens is usually too little context for an LLM to answer
from. Large chunks have the opposite trade-off: great context, mushy vectors.

So we keep both, and link them:

    page documents
        |
        +-- PARENT block  (~1000 tokens)   <- what the LLM reads
              +-- child   (~200 tokens)    <- what FAISS searches
              +-- child
              +-- child

Only children are embedded. At query time a child hit is expanded to its parent
before the text reaches the LLM. Every child carries ``parent_id``, so that
expansion is a dictionary lookup rather than a second search.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.metadata import DocumentMetadata, chunk_metadata

logger = logging.getLogger(__name__)

# Split preference order: paragraph breaks first, then sentence ends, then any
# whitespace. This makes chunk boundaries land on a semantic seam when possible.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

# Page markers injected before splitting so each chunk can report its pages.
_PAGE_MARKER = "[[page:{}]]"
_PAGE_MARKER_RE = re.compile(r"\[\[page:(\d+)\]\]")


@dataclass
class ParentChunk:
    """A large context block handed to the LLM at generation time."""

    parent_id: str
    text: str
    doc_id: str
    pages: list[int] = field(default_factory=list)
    section: str | None = None
    child_ids: list[str] = field(default_factory=list)  # provenance: parent -> children
    index: int = 0                                      # position within the document
    # Full metadata record (size, page range, content signals, lineage).
    # Kept as a dict rather than more dataclass fields so new signals can be
    # added in app.core.metadata without changing this class or the on-disk format.
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise for the on-disk parent store (plain JSON)."""
        return {
            "parent_id": self.parent_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "pages": self.pages,
            "section": self.section,
            "child_ids": self.child_ids,
            "index": self.index,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParentChunk":
        """Rebuild a ParentChunk from its JSON form."""
        return cls(
            parent_id=data["parent_id"],
            text=data["text"],
            doc_id=data["doc_id"],
            pages=data.get("pages", []),
            section=data.get("section"),
            child_ids=data.get("child_ids", []),
            index=data.get("index", 0),
            meta=data.get("meta", {}),
        )


@dataclass
class ChildChunk:
    """A small, precisely-embeddable chunk pointing back at its parent."""

    child_id: str
    text: str
    parent_id: str
    doc_id: str
    pages: list[int] = field(default_factory=list)
    section: str | None = None
    index: int = 0  # position within the parent
    meta: dict = field(default_factory=dict)

    def to_document(self) -> Document:
        """Convert to the LangChain Document that FAISS will index.

        ``parent_id`` rides along in metadata -- that single field is what makes
        small-to-big expansion possible at query time.
        """
        return Document(
            page_content=self.text,
            metadata={
                # Identity and the parent link, which expansion depends on.
                "child_id": self.child_id,
                "parent_id": self.parent_id,
                "doc_id": self.doc_id,
                "pages": self.pages,
                "section": self.section,
                "index": self.index,
                # Everything else: page range, sizes, content signals, lineage.
                # Carried on the vector itself so a search result can be
                # filtered and explained without a second lookup.
                **self.meta,
            },
        )


def _stable_id(prefix: str, doc_id: str, text: str, index: object) -> str:
    """Derive a deterministic id from the content itself.

    Content-addressed ids make ingestion idempotent: re-running over the same
    PDF regenerates identical parent/child ids, so Neo4j MERGE updates instead
    of duplicating, and the FAISS index can be rebuilt without orphaning the
    graph's provenance links.
    """
    digest = hashlib.sha1(f"{doc_id}|{index}|{text[:512]}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _build_splitter(chunk_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    """Create a splitter that measures length in real tokens, not characters.

    ``from_tiktoken_encoder`` swaps the length function for a tiktoken count, so
    "1000 tokens" means 1000 tokens rather than a guess. Falls back to a
    4-characters-per-token estimate if tiktoken's encoding files cannot be
    downloaded (e.g. a fully offline machine on first run).
    """
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_tokens,
            chunk_overlap=overlap_tokens,
            separators=_SEPARATORS,
        )
    except Exception as exc:
        logger.warning("tiktoken unavailable (%s); approximating 1 token = 4 chars", exc)
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_tokens * 4,
            chunk_overlap=overlap_tokens * 4,
            separators=_SEPARATORS,
        )


def _merge_pages(pages: Iterable[Document]) -> list[Document]:
    """Concatenate consecutive page documents into one continuous document per doc_id.

    Splitting page-by-page would force a parent boundary at every page break,
    truncating sentences and tables that flow across pages. Instead we join the
    pages with explicit ``[[page:N]]`` markers, split the continuous text, then
    read the markers back out to recover which pages each chunk covers.
    """
    grouped: dict[str, list[Document]] = {}
    for page in pages:
        grouped.setdefault(page.metadata.get("doc_id", "unknown"), []).append(page)

    merged: list[Document] = []
    for doc_id, page_docs in grouped.items():
        page_docs.sort(key=lambda d: d.metadata.get("page", 0))
        buffer = [
            f"{_PAGE_MARKER.format(p.metadata.get('page', 0))}\n{p.page_content}"
            for p in page_docs
        ]
        merged.append(
            Document(
                page_content="\n\n".join(buffer),
                metadata={"doc_id": doc_id, "source": page_docs[0].metadata.get("source", "")},
            )
        )
    return merged


def _extract_pages(text: str) -> tuple[str, list[int]]:
    """Strip ``[[page:N]]`` markers from a chunk and return the page numbers found.

    Returns:
        ``(clean_text, sorted_unique_page_numbers)``. A chunk that spans a page
        break legitimately reports both pages.
    """
    pages = sorted({int(m) for m in _PAGE_MARKER_RE.findall(text)})
    clean = _PAGE_MARKER_RE.sub("", text).strip()
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean, pages


def chunk_documents(
    pages: list[Document],
    parent_tokens: int | None = None,
    parent_overlap: int | None = None,
    child_tokens: int | None = None,
    child_overlap: int | None = None,
    doc_meta: DocumentMetadata | None = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Split page documents into linked parent and child chunks.

    This is the heart of the hierarchical strategy. The document is first cut
    into ~1000-token parents; each parent is then independently cut into
    ~200-token children. Because children are produced *from* a parent's own
    text, the parent link is exact by construction -- no overlap heuristics.

    Args:
        pages: page-level documents from :func:`app.ingestion.loader.load_pdf`.
        parent_tokens: parent block size in tokens (default from settings).
        parent_overlap: token overlap between adjacent parents.
        child_tokens: child chunk size in tokens (default from settings).
        child_overlap: token overlap between adjacent children.
        doc_meta: source-document metadata stamped onto every chunk. Built
            here from the page documents if the caller does not supply it, so
            chunking remains usable standalone (tests, notebooks).

    Returns:
        ``(parents, children)``. Every child's ``parent_id`` exists in
        ``parents``, and every parent's ``child_ids`` lists its children, so the
        provenance map is navigable in both directions.
    """
    parent_tokens = parent_tokens or settings.parent_chunk_tokens
    parent_overlap = parent_overlap if parent_overlap is not None else settings.parent_chunk_overlap
    child_tokens = child_tokens or settings.child_chunk_tokens
    child_overlap = child_overlap if child_overlap is not None else settings.child_chunk_overlap

    parent_splitter = _build_splitter(parent_tokens, parent_overlap)
    child_splitter = _build_splitter(child_tokens, child_overlap)

    # Fall back to metadata derived from the pages themselves, so this function
    # does not require the full pipeline to have run.
    if doc_meta is None:
        first = pages[0].metadata if pages else {}
        doc_meta = DocumentMetadata.build(
            path=first.get("source", "unknown.pdf"),
            doc_id=first.get("doc_id", "unknown"),
            page_count=len({p.metadata.get("page") for p in pages}),
            parser=first.get("parser", "unknown"),
            embedding_model="",
            parent_tokens=parent_tokens,
            child_tokens=child_tokens,
        )

    # Map page -> section title so chunks can inherit a human-readable label
    # for the citation cards in the UI.
    section_by_page = {
        p.metadata.get("page"): p.metadata.get("section")
        for p in pages
        if p.metadata.get("section")
    }

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for merged in _merge_pages(pages):
        doc_id = merged.metadata["doc_id"]

        # --- Level 1: big context blocks the LLM will read ----------------
        for p_index, parent_raw in enumerate(parent_splitter.split_text(merged.page_content)):
            parent_text, parent_pages = _extract_pages(parent_raw)
            if len(parent_text) < 50:  # fragments left behind by marker stripping
                continue

            parent_id = _stable_id("parent", doc_id, parent_text, p_index)
            section = next(
                (section_by_page[pg] for pg in parent_pages if pg in section_by_page), None
            )

            parent = ParentChunk(
                parent_id=parent_id,
                text=parent_text,
                doc_id=doc_id,
                pages=parent_pages,
                section=section,
                index=p_index,
                meta=chunk_metadata(
                    text=parent_text, doc=doc_meta, pages=parent_pages,
                    section=section, index=p_index, kind="parent",
                ),
            )

            # --- Level 2: precise search units, cut from this parent ------
            for c_index, child_text in enumerate(child_splitter.split_text(parent_text)):
                child_text = child_text.strip()
                if len(child_text) < 30:  # punctuation-only slivers embed as noise
                    continue
                child_id = _stable_id("child", doc_id, child_text, f"{p_index}.{c_index}")
                children.append(
                    ChildChunk(
                        child_id=child_id,
                        text=child_text,
                        parent_id=parent_id,
                        doc_id=doc_id,
                        pages=parent_pages,
                        section=section,
                        index=c_index,
                        meta=chunk_metadata(
                            text=child_text, doc=doc_meta, pages=parent_pages,
                            section=section, index=c_index, kind="child",
                        ),
                    )
                )
                parent.child_ids.append(child_id)  # provenance: parent -> children

            # A parent with no children can never be retrieved, so don't keep it.
            if parent.child_ids:
                parent.meta["child_count"] = len(parent.child_ids)
                parents.append(parent)

    logger.info(
        "Chunked into %d parents (~%d tok) and %d children (~%d tok)",
        len(parents),
        parent_tokens,
        len(children),
        child_tokens,
    )
    return parents, children
