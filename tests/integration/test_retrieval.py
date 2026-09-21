"""
Integration tests against live stores.

These verify the wiring that unit tests cannot reach: that FAISS really expands
child hits to parents, that Neo4j really returns typed edges, and that the
LangGraph topology really runs the branches the router selects.

Everything here auto-skips when the stores are empty or unreachable (see
``tests/conftest.py``), so a fresh clone still gets a clean ``pytest`` run.
No LLM is called except in the routing test, which tolerates failure.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #

class TestVectorRetrieval:
    """Small-to-big expansion, against the real index."""

    def test_search_returns_children(self, vector_store):
        """Basic sanity: the index answers queries."""
        hits = vector_store.search_children("Emirates NBD profit", k=5)
        assert hits, "no child chunks returned for an obviously in-domain query"
        assert all(len(doc.page_content) > 0 for doc, _ in hits)

    def test_child_hits_carry_parent_ids(self, vector_store):
        """Without parent_id in metadata, expansion is impossible."""
        hits = vector_store.search_children("banking", k=5)
        assert all(doc.metadata.get("parent_id") for doc, _ in hits)

    def test_expansion_returns_larger_context(self, vector_store):
        """The defining behaviour: parents must be bigger than the children.

        If this fails, the hierarchy costs storage and complexity while
        delivering no additional context to the model.
        """
        hits = vector_store.search_children("Emirates NBD", k=5)
        parents = vector_store.search_parents("Emirates NBD", k=5, top_n=3)
        assert parents, "expansion produced no parents"

        avg_child = sum(len(d.page_content) for d, _ in hits) / len(hits)
        avg_parent = sum(len(r["parent"].text) for r in parents) / len(parents)
        assert avg_parent > avg_child, (
            f"parents ({avg_parent:.0f} chars) are not larger than children ({avg_child:.0f})"
        )

    def test_expanded_parent_contains_the_matched_child(self, vector_store):
        """The expansion must return the block the match actually came from."""
        hits = vector_store.search_children("Emirates NBD", k=3)
        assert hits
        doc, _ = hits[0]
        parent = vector_store.parent_store.get(doc.metadata["parent_id"])
        assert parent is not None, "child points at a parent that is not stored"
        assert doc.page_content[:100] in parent.text

    def test_duplicate_parents_are_merged(self, vector_store):
        """Several children of one parent must yield that parent once."""
        results = vector_store.search_parents("banking services", k=12, top_n=10)
        ids = [r["parent_id"] for r in results]
        assert len(ids) == len(set(ids)), "the same parent was returned twice"

    def test_scores_are_ordered(self, vector_store):
        """Results must arrive best-first, or ranking downstream is meaningless."""
        results = vector_store.search_parents("net profit", k=10, top_n=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_reingestion_is_idempotent(self, vector_store):
        """Re-adding identical chunks must not grow the index.

        FAISS rejects duplicate ids rather than upserting, so the store has to
        delete-then-add. This is the regression test for that: without it,
        re-running an ingest to add graph extraction fails outright.
        """
        before = vector_store.vector_count
        sample = [
            doc for doc, _ in vector_store.search_children("Emirates NBD", k=3)
        ]
        assert sample, "index is empty"

        from app.ingestion.chunker import ChildChunk

        chunks = [
            ChildChunk(
                child_id=d.metadata["child_id"],
                text=d.page_content,
                parent_id=d.metadata["parent_id"],
                doc_id=d.metadata["doc_id"],
                pages=d.metadata.get("pages", []),
            )
            for d in sample
        ]
        vector_store.add_children(chunks)
        assert vector_store.vector_count == before, "re-ingestion duplicated vectors"


# --------------------------------------------------------------------------- #
# Graph store
# --------------------------------------------------------------------------- #

class TestGraphRetrieval:
    """Traversal and provenance, against the real Neo4j instance."""

    def test_entity_search_finds_the_subject_company(self, graph_store):
        """Full-text seed resolution must work for the document's main entity."""
        hits = graph_store.find_entities("Emirates NBD", limit=5)
        assert hits, "no entity matched 'Emirates NBD'"
        assert any("emirates" in h["id"].lower() for h in hits)

    def test_traversal_returns_typed_triples(self, graph_store):
        """Edges must carry real relationship types, not a generic label."""
        seeds = graph_store.find_entities("Emirates NBD", limit=3)
        if not seeds:
            pytest.skip("no seed entities in the graph")

        triples = graph_store.traverse([s["id"] for s in seeds], max_hops=2, limit=20)
        assert triples, "traversal returned nothing from a seeded entity"
        for row in triples:
            assert row["relation"], "a relationship has no type"
            assert row["relation"] == row["relation"].upper()
            assert row["relation"] not in ("MENTIONED_IN", "HAS_CHILD"), \
                "structural edges leaked into the semantic traversal"

    def test_hop_distance_is_reported_and_ordered(self, graph_store):
        """1-hop facts must precede 2-hop context."""
        seeds = graph_store.find_entities("Emirates NBD", limit=3)
        if not seeds:
            pytest.skip("no seed entities")
        triples = graph_store.traverse([s["id"] for s in seeds], max_hops=2, limit=20)
        if not triples:
            pytest.skip("no triples")
        hops = [t["hop"] for t in triples]
        assert all(h >= 1 for h in hops)
        assert hops == sorted(hops), "triples are not ordered by hop distance"

    def test_relationships_carry_chunk_provenance(self, graph_store):
        """Every edge must be traceable to the passage it came from."""
        seeds = graph_store.find_entities("Emirates NBD", limit=3)
        if not seeds:
            pytest.skip("no seed entities")
        triples = graph_store.traverse([s["id"] for s in seeds], max_hops=1, limit=20)
        if not triples:
            pytest.skip("no triples")
        assert any(t.get("parent_ids") for t in triples), \
            "no relationship records a parent chunk id"

    def test_entities_link_back_to_chunks(self, graph_store):
        """The MENTIONED_IN provenance edges must exist."""
        seeds = graph_store.find_entities("Emirates NBD", limit=2)
        if not seeds:
            pytest.skip("no seed entities")
        chunks = graph_store.chunks_for_entities([seeds[0]["id"]], limit=10)
        assert chunks, "entity has no MENTIONED_IN chunk links"
        assert all(c["kind"] in ("parent", "child") for c in chunks)

    def test_subgraph_edges_reference_present_nodes(self, graph_store):
        """A dangling edge would break the visualiser."""
        seeds = graph_store.find_entities("Emirates NBD", limit=3)
        if not seeds:
            pytest.skip("no seed entities")
        payload = graph_store.subgraph([s["id"] for s in seeds], max_hops=2, limit=50)
        node_ids = {n["id"] for n in payload["nodes"]}
        for edge in payload["edges"]:
            assert edge["source"] in node_ids and edge["target"] in node_ids


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #

class TestPipeline:
    """The LangGraph topology, run for real."""

    def test_pipeline_produces_a_grounded_answer(self, rag_graph):
        """The happy path: retrieve, fuse, generate."""
        state = asyncio.run(
            rag_graph.ainvoke(
                {"question": "What does Emirates NBD do?", "history": [], "use_graph": True}
            )
        )
        assert state.get("answer"), "pipeline produced no answer"
        assert state.get("strategy") in ("vector", "graph", "hybrid")

    def test_vector_only_path_skips_the_graph(self, rag_graph):
        """``use_graph=False`` must genuinely bypass the graph arm.

        This is the baseline used to compare hybrid retrieval against plain
        vector RAG, so it has to actually be plain vector RAG.
        """
        state = asyncio.run(
            rag_graph.ainvoke(
                {"question": "Summarise the outlook.", "history": [], "use_graph": False}
            )
        )
        assert state.get("strategy") == "vector"
        assert not state.get("used_triples"), "graph triples leaked into the vector-only path"

    def test_citations_expose_provenance(self, rag_graph):
        """Citations must carry the fields the UI renders."""
        state = asyncio.run(
            rag_graph.ainvoke(
                {"question": "What is Emirates NBD's total income?", "history": [], "use_graph": True}
            )
        )
        citations = state.get("citations", [])
        if not citations:
            pytest.skip("no citations retrieved")
        for citation in citations:
            assert citation.parent_id and citation.doc_id and citation.text
            assert citation.pages, "a citation has no page numbers"
