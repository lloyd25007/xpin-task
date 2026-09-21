"""
The vector side of retrieval: a FAISS index over *child* chunks plus a JSON
document store holding the *parent* blocks.

Splitting the two is deliberate. FAISS only ever sees the small, sharp child
chunks, which keeps the index tight and precise. The bulky parent text lives in
a plain JSON file, fetched by id after the search returns. That is the
"small-to-big" contract: search small, read big.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import settings
from app.ingestion.chunker import ChildChunk, ParentChunk
from app.chat.llm import get_embeddings

logger = logging.getLogger(__name__)


class ParentStore:
    """Disk-backed key-value store mapping ``parent_id`` -> :class:`ParentChunk`.

    A JSON file is the right tool here: parent blocks are read by exact id and
    never searched, the whole corpus for one annual report is a few megabytes,
    and keeping it human-readable makes retrieval problems easy to debug.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Open (or lazily create) the store at ``path``."""
        self.path = path or settings.parents_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._parents: dict[str, ParentChunk] = {}
        self._load()

    def _load(self) -> None:
        """Read the JSON file into memory, tolerating a missing or corrupt file.

        A corrupt store is recoverable by re-ingesting, so we warn and start
        empty rather than crashing the API at import time.
        """
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._parents = {pid: ParentChunk.from_dict(d) for pid, d in raw.items()}
            logger.info("Loaded %d parent chunks from %s", len(self._parents), self.path)
        except Exception as exc:
            logger.warning("Could not read parent store (%s); starting empty", exc)
            self._parents = {}

    def save(self) -> None:
        """Persist the in-memory map atomically.

        Writes to a temporary file and renames it, so a crash mid-write cannot
        leave a half-written store behind.
        """
        tmp = self.path.with_suffix(".tmp")
        payload = {pid: p.to_dict() for pid, p in self._parents.items()}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)
        logger.info("Saved %d parent chunks to %s", len(self._parents), self.path)

    def add(self, parents: list[ParentChunk]) -> None:
        """Insert or overwrite parent chunks, keyed by their stable id."""
        for parent in parents:
            self._parents[parent.parent_id] = parent

    def get(self, parent_id: str) -> ParentChunk | None:
        """Look up one parent block by id; ``None`` if it was never ingested."""
        return self._parents.get(parent_id)

    def get_many(self, parent_ids: list[str]) -> list[ParentChunk]:
        """Look up several parents, preserving caller order and skipping misses."""
        return [p for pid in parent_ids if (p := self._parents.get(pid)) is not None]

    def clear(self) -> None:
        """Drop every parent and delete the backing file (used by ``reset``)."""
        self._parents = {}
        self.path.unlink(missing_ok=True)

    def __len__(self) -> int:
        """Number of parent blocks currently stored."""
        return len(self._parents)

    @property
    def doc_ids(self) -> list[str]:
        """Distinct source documents represented in the store."""
        return sorted({p.doc_id for p in self._parents.values()})


class VectorStore:
    """FAISS index over child chunks, with small-to-big expansion built in.

    The index is loaded lazily: importing this module must stay cheap because
    FastAPI imports it at startup, while loading FAISS also loads the embedding
    model (seconds, hundreds of MB).
    """

    def __init__(self, parent_store: ParentStore | None = None) -> None:
        """Wire the vector index to the parent store used for expansion."""
        self.index_dir = settings.faiss_dir
        self.parent_store = parent_store or ParentStore()
        self._faiss: FAISS | None = None

    # ------------------------------------------------------------------ load --

    @property
    def faiss(self) -> FAISS | None:
        """Return the loaded FAISS index, loading it from disk on first access.

        ``None`` means nothing has been ingested yet -- callers treat that as an
        empty result rather than an error, so the API can start before ingestion.
        """
        if self._faiss is None:
            self._faiss = self._load_index()
        return self._faiss

    def _load_index(self) -> FAISS | None:
        """Deserialise the FAISS index from ``data/faiss_index``.

        ``allow_dangerous_deserialization`` is required by LangChain because the
        docstore is a pickle. It is safe here: the file is produced by this
        application on the same machine, never accepted from a user.
        """
        index_file = self.index_dir / "index.faiss"
        if not index_file.exists():
            logger.info("No FAISS index at %s yet -- ingest a document first", self.index_dir)
            return None
        try:
            store = FAISS.load_local(
                str(self.index_dir),
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded FAISS index with %d vectors", store.index.ntotal)
            return store
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return None

    # ----------------------------------------------------------------- write --

    def add_children(self, children: list[ChildChunk]) -> int:
        """Embed child chunks and add them to the index, creating it if needed.

        Args:
            children: the small chunks produced by the hierarchical splitter.

        Returns:
            Number of vectors added.
        """
        if not children:
            return 0

        documents = [c.to_document() for c in children]
        # Child ids double as FAISS document ids, which is what makes
        # re-ingestion idempotent -- but only once the duplicates are removed
        # first. FAISS does not upsert: ``add_documents`` raises outright on an
        # id it already holds, so re-running an ingest (for example to add graph
        # extraction to an existing vector index) would fail without this.
        ids = [c.child_id for c in children]

        if self.faiss is None:
            logger.info("Building new FAISS index from %d child chunks", len(documents))
            self._faiss = FAISS.from_documents(documents, get_embeddings(), ids=ids)
            return len(documents)

        existing = set(self._faiss.index_to_docstore_id.values())
        duplicates = [i for i in ids if i in existing]
        if duplicates:
            # Delete then re-add, so the new embedding wins if the chunk's text
            # or the embedding model has changed since the last run.
            logger.info("Replacing %d existing child chunks", len(duplicates))
            try:
                self._faiss.delete(duplicates)
            except Exception as exc:
                # If deletion is unsupported, skip the duplicates rather than
                # failing the whole ingest -- their content is unchanged anyway,
                # because the ids are content-addressed.
                logger.warning("Could not delete duplicates (%s); skipping them", exc)
                keep = [(d, i) for d, i in zip(documents, ids) if i not in existing]
                if not keep:
                    return 0
                documents, ids = [d for d, _ in keep], [i for _, i in keep]

        logger.info("Adding %d child chunks to existing FAISS index", len(documents))
        self._faiss.add_documents(documents, ids=ids)
        return len(documents)

    def add_parents(self, parents: list[ParentChunk]) -> int:
        """Store parent blocks in the document store (they are never embedded)."""
        self.parent_store.add(parents)
        return len(parents)

    def save(self) -> None:
        """Flush both the FAISS index and the parent store to disk."""
        if self._faiss is not None:
            self._faiss.save_local(str(self.index_dir))
            logger.info("Saved FAISS index (%d vectors)", self._faiss.index.ntotal)
        self.parent_store.save()

    def clear(self) -> None:
        """Delete the index and every parent block -- a full reset."""
        self._faiss = None
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir, ignore_errors=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.parent_store.clear()
        logger.info("Cleared vector store")

    # ---------------------------------------------------------------- search --

    def search_children(self, query: str, k: int | None = None,
                        filters: dict | None = None) -> list[tuple[Document, float]]:
        """Similarity search over child chunks, optionally metadata-filtered.

        Args:
            query: the user's question.
            k: number of children to return (default from settings).
            filters: metadata constraints applied *before* scoring, e.g.
                ``{"doc_id": "strategic_report_2025"}`` to search one document,
                ``{"has_table": True}`` to prefer chunks with recovered tables,
                or ``{"page_start": {"$gte": 10, "$lte": 25}}`` for a range.

        Returns:
            ``(document, score)`` pairs. Because embeddings are normalised, the
            score is a cosine similarity in roughly [0, 1] -- higher is better.
        """
        if self.faiss is None:
            return []
        k = k or settings.vector_top_k
        try:
            if filters:
                # Filtering happens inside FAISS's search, so `k` still yields
                # k *matching* results rather than k results then filtered down.
                return self.faiss.similarity_search_with_relevance_scores(
                    query, k=k, filter=self._build_filter(filters)
                )
            return self.faiss.similarity_search_with_relevance_scores(query, k=k)
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            return []

    @staticmethod
    def _build_filter(filters: dict):
        """Turn a filter spec into a predicate over document metadata.

        A callable rather than LangChain's dict form, because the dict form
        only supports equality and this needs ranges (page windows) and
        membership (several documents at once).

        Supported per key: a scalar for equality, a list for membership, or a
        dict with ``$gte`` / ``$lte`` for ranges.
        """
        def predicate(meta: dict) -> bool:
            for key, want in filters.items():
                have = meta.get(key)
                if isinstance(want, dict):
                    if have is None:
                        return False
                    if "$gte" in want and have < want["$gte"]:
                        return False
                    if "$lte" in want and have > want["$lte"]:
                        return False
                elif isinstance(want, (list, tuple, set)):
                    if have not in want:
                        return False
                elif have != want:
                    return False
            return True

        return predicate

    def search_parents(self, query: str, k: int | None = None, top_n: int | None = None,
                       filters: dict | None = None) -> list[dict]:
        """Search children, then expand the hits to their parent blocks.

        This is the small-to-big step. Several children often belong to the same
        parent; those hits are merged, the parent keeps its best child score,
        and the contributing child ids are recorded so the UI can show exactly
        which passage matched inside the larger block.

        Args:
            query: the user's question.
            k: how many child chunks to retrieve before expansion.
            top_n: how many distinct parents to keep afterwards.

        Returns:
            Parent records sorted by score, each containing the parent chunk,
            its score, and the ids/snippets of the children that matched.
        """
        k = k or settings.vector_top_k
        top_n = top_n or settings.parent_top_n

        hits = self.search_children(query, k=k, filters=filters)
        if not hits:
            return []

        # --- Group child hits under their parent -------------------------
        by_parent: dict[str, dict] = {}
        for doc, score in hits:
            parent_id = doc.metadata.get("parent_id")
            if not parent_id:
                continue
            entry = by_parent.setdefault(
                parent_id,
                {"parent_id": parent_id, "score": score, "matched_child_ids": [], "child_snippets": []},
            )
            # The parent inherits the *best* score among its children.
            entry["score"] = max(entry["score"], score)
            entry["matched_child_ids"].append(doc.metadata.get("child_id", ""))
            entry["child_snippets"].append(doc.page_content)

        # --- Attach the full parent text ---------------------------------
        results: list[dict] = []
        for parent_id, entry in by_parent.items():
            parent = self.parent_store.get(parent_id)
            if parent is None:
                # Index and parent store are out of sync (partial ingest).
                logger.debug("Parent %s missing from store; skipping", parent_id)
                continue
            entry["parent"] = parent
            results.append(entry)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_n]

    # ----------------------------------------------------------------- stats --

    @property
    def vector_count(self) -> int:
        """How many child vectors the index currently holds."""
        return self.faiss.index.ntotal if self.faiss is not None else 0

    @property
    def parent_count(self) -> int:
        """How many parent blocks are available for expansion."""
        return len(self.parent_store)
