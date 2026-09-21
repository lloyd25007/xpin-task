"""
Unit tests for the pure logic — the parts that can break silently.

Deliberately no network, no Neo4j, no LLM: these test the invariants that hold
regardless of what any model returns. Retrieval quality is checked separately by
``scripts/smoke_test.py``, which needs live stores.

Run with::

    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunker import ChildChunk, ParentChunk, chunk_documents  # noqa: E402
from app.ingestion.loader import _clean_text, _guess_section                # noqa: E402
from app.retrieval.retriever import (                                          # noqa: E402
    dedupe_preserving_order, fuse_candidates, reciprocal_rank_fusion,
)
from app.retrieval.retriever import extract_entity_mentions, heuristic_route   # noqa: E402
from app.storage.graph_store import entity_key, sanitize_rel_type            # noqa: E402


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

class TestTextCleaning:
    """The cleaner is the first thing every downstream stage depends on."""

    def test_folds_unicode_spaces(self):
        """Typographic spaces must become plain spaces, not survive as-is.

        Left alone they break tokenisation and crash cp1252 consoles.
        """
        assert _clean_text("Emirates NBD Bank") == "Emirates NBD Bank"

    def test_normalises_smart_punctuation(self):
        """Smart quotes and en dashes fold to ASCII so one form reaches the index."""
        assert _clean_text("“profit” – 2025") == '"profit" - 2025'

    def test_strips_dot_leaders(self):
        """Contents-page dot leaders are noise and must not reach an embedding."""
        assert "...." not in _clean_text("Chairman's statement ........ 12")

    def test_rejoins_words_split_across_lines(self):
        """Column layout breaks sentences mid-clause; they must be rejoined."""
        assert _clean_text("the bank reported,\nstrong growth") == \
            "the bank reported, strong growth"

    def test_caps_blank_line_runs(self):
        """Long blank runs waste chunk budget."""
        assert "\n\n\n" not in _clean_text("a\n\n\n\n\nb")

    def test_guess_section_ignores_bare_numbers(self):
        """A page number is not a section title."""
        assert _guess_section("42\n\nsome body text here") != "42"


# --------------------------------------------------------------------------- #
# Hierarchical chunking — the core invariants
# --------------------------------------------------------------------------- #

class TestHierarchicalChunking:
    """If these break, small-to-big retrieval is silently broken."""

    @pytest.fixture
    def pages(self) -> list[Document]:
        """Three pages of prose, long enough to force several parent blocks."""
        body = (
            "Emirates NBD reported a record net profit for the year. "
            "The group expanded its international footprint across key markets. "
            "Retail Banking delivered strong deposit growth while Corporate and "
            "Institutional Banking grew its lending book substantially. "
        )
        return [
            Document(
                page_content=body * 12,
                metadata={"doc_id": "test_doc", "source": "test.pdf",
                          "page": page, "section": f"Section {page}"},
            )
            for page in (1, 2, 3)
        ]

    def test_produces_both_levels(self, pages):
        """Chunking must yield parents and strictly more children."""
        parents, children = chunk_documents(pages)
        assert parents, "no parent chunks produced"
        assert len(children) > len(parents), "children should outnumber parents"

    def test_every_child_has_a_real_parent(self, pages):
        """An orphan child can be retrieved but never expanded — a silent hole."""
        parents, children = chunk_documents(pages)
        parent_ids = {p.parent_id for p in parents}
        assert all(c.parent_id in parent_ids for c in children)

    def test_parent_child_links_are_bidirectional(self, pages):
        """Both directions of the provenance map must agree."""
        parents, children = chunk_documents(pages)
        declared = sum(len(p.child_ids) for p in parents)
        assert declared == len(children)

        by_id = {c.child_id: c for c in children}
        for parent in parents:
            for child_id in parent.child_ids:
                assert by_id[child_id].parent_id == parent.parent_id

    def test_child_text_is_contained_in_its_parent(self, pages):
        """The defining invariant: children are cut *from* parent text.

        If a child's text is not inside its parent, expansion returns a block
        that does not actually contain the matched passage.
        """
        parents, children = chunk_documents(pages)
        by_id = {p.parent_id: p for p in parents}
        for child in children:
            parent = by_id[child.parent_id]
            assert child.text[:80] in parent.text

    def test_children_are_smaller_than_parents(self, pages):
        """The whole point: search units small, context units big."""
        parents, children = chunk_documents(pages)
        avg_parent = sum(len(p.text) for p in parents) / len(parents)
        avg_child = sum(len(c.text) for c in children) / len(children)
        assert avg_child < avg_parent

    def test_ids_are_deterministic(self, pages):
        """Content-addressed ids are what make re-ingestion idempotent."""
        first, _ = chunk_documents(pages)
        second, _ = chunk_documents(pages)
        assert [p.parent_id for p in first] == [p.parent_id for p in second]

    def test_pages_are_tracked_across_boundaries(self, pages):
        """Citations are worthless without correct page attribution."""
        parents, _ = chunk_documents(pages)
        assert all(p.pages for p in parents), "a parent has no page numbers"
        assert {pg for p in parents for pg in p.pages} <= {1, 2, 3}

    def test_page_markers_never_leak_into_text(self, pages):
        """The internal ``[[page:N]]`` marker must not reach an embedding."""
        parents, children = chunk_documents(pages)
        assert all("[[page:" not in p.text for p in parents)
        assert all("[[page:" not in c.text for c in children)

    def test_roundtrip_serialisation(self, pages):
        """Parents must survive the JSON store unchanged."""
        parents, _ = chunk_documents(pages)
        restored = ParentChunk.from_dict(parents[0].to_dict())
        assert restored.parent_id == parents[0].parent_id
        assert restored.text == parents[0].text
        assert restored.child_ids == parents[0].child_ids

    def test_child_document_carries_parent_id(self, pages):
        """Without parent_id in metadata, FAISS hits cannot be expanded."""
        _, children = chunk_documents(pages)
        assert children[0].to_document().metadata["parent_id"] == children[0].parent_id


# --------------------------------------------------------------------------- #
# Graph identifier safety
# --------------------------------------------------------------------------- #

class TestGraphIdentifiers:
    """Relationship types are interpolated into Cypher, so sanitising matters."""

    @pytest.mark.parametrize("raw,expected", [
        ("agreed to acquire stake in", "AGREED_TO_ACQUIRE_STAKE_IN"),
        ("owns", "OWNS"),
        ("INVESTS_IN", "INVESTS_IN"),
        ("reported (net)", "REPORTED_NET"),
        ("  spaced  out  ", "SPACED_OUT"),
    ])
    def test_sanitises_to_legal_types(self, raw, expected):
        assert sanitize_rel_type(raw) == expected

    @pytest.mark.parametrize("hostile", [
        "]->(x) DETACH DELETE n //",
        "`; MATCH (n) DELETE n; //",
        "OWNS]-() CREATE (:Evil) //",
    ])
    def test_injection_attempts_yield_safe_identifiers(self, hostile):
        """Nothing outside [A-Z0-9_] may survive — that is the security property."""
        result = sanitize_rel_type(hostile)
        assert all(ch.isalnum() or ch == "_" for ch in result)
        assert result.isupper() or "_" in result

    @pytest.mark.parametrize("raw", ["", "   ", "123", "!!!"])
    def test_unusable_input_falls_back(self, raw):
        """A type must always be legal, even from garbage input."""
        assert sanitize_rel_type(raw) == "RELATED_TO"

    def test_reserved_type_is_renamed(self):
        """An extracted edge must not collide with our structural edge."""
        assert sanitize_rel_type("mentioned in") == "MENTIONED_IN_EXTRACTED"

    @pytest.mark.parametrize("variant", [
        "Emirates NBD", "emirates nbd", "  Emirates   NBD  ", "EMIRATES-NBD",
    ])
    def test_entity_key_collapses_variants(self, variant):
        """Case and spacing variants must resolve to one node, or the graph splits."""
        assert entity_key(variant) == "emirates_nbd"

    def test_entity_key_never_empty(self):
        """An empty key would create a shared junk node."""
        assert entity_key("!!!") == "unknown"


# --------------------------------------------------------------------------- #
# Rank fusion
# --------------------------------------------------------------------------- #

class TestRankFusion:
    """RRF must reward agreement between retrievers."""

    def test_agreement_beats_single_list_dominance(self):
        """A doc found by both retrievers should outrank one found by only one."""
        scores = reciprocal_rank_fusion([(["a", "b", "c"], 1.0), (["b", "a", "d"], 1.0)])
        assert scores["b"] > scores["c"]
        assert scores["a"] > scores["d"]

    def test_rank_order_is_respected(self):
        """Within one list, earlier ranks must score higher."""
        scores = reciprocal_rank_fusion([(["first", "second", "third"], 1.0)])
        assert scores["first"] > scores["second"] > scores["third"]

    def test_weights_shift_influence(self):
        """Weighting lets one retriever count for more."""
        weighted = reciprocal_rank_fusion([(["a"], 1.0), (["b"], 0.1)])
        assert weighted["a"] > weighted["b"]

    def test_damping_constant_flattens_differences(self):
        """A larger k narrows the gap between adjacent ranks."""
        tight = reciprocal_rank_fusion([(["a", "b"], 1.0)], k=1)
        flat = reciprocal_rank_fusion([(["a", "b"], 1.0)], k=1000)
        assert (tight["a"] - tight["b"]) > (flat["a"] - flat["b"])

    def test_missing_documents_are_simply_absent(self):
        """Absence contributes nothing; it is not a penalty."""
        assert "zzz" not in reciprocal_rank_fusion([(["a"], 1.0)])

    def test_empty_input_is_safe(self):
        assert reciprocal_rank_fusion([]) == {}

    def _record(self, pid: str, score: float) -> dict:
        """Minimal candidate record shaped like the retrievers produce."""
        parent = ParentChunk(parent_id=pid, text=f"text {pid}", doc_id="d")
        return {"parent_id": pid, "parent": parent, "score": score,
                "matched_child_ids": [], "hop": 1, "triples": []}

    def test_fusion_merges_and_tags_sources(self):
        """A parent found by both arms must be marked as such, not duplicated."""
        vector = [self._record("p1", 0.9), self._record("p2", 0.8)]
        graph = [self._record("p2", 0.7), self._record("p3", 0.6)]

        fused = fuse_candidates(vector, graph)
        assert len(fused) == 3, "parents must be merged, not duplicated"

        by_id = {r["parent_id"]: r for r in fused}
        assert set(by_id["p2"]["sources"]) == {"vector", "graph"}
        assert by_id["p1"]["sources"] == ["vector"]
        assert by_id["p3"]["sources"] == ["graph"]
        # p2 was found twice, so it must lead.
        assert fused[0]["parent_id"] == "p2"

    def test_fusion_handles_one_empty_arm(self):
        """Vector-only routing must not break fusion."""
        fused = fuse_candidates([self._record("p1", 0.9)], [])
        assert len(fused) == 1 and fused[0]["sources"] == ["vector"]

    def test_dedupe_preserves_order(self):
        """Ranking carries meaning, so dedupe must not reorder."""
        assert dedupe_preserving_order(["b", "a", "b", "c"]) == ["b", "a", "c"]


# --------------------------------------------------------------------------- #
# Intent routing
# --------------------------------------------------------------------------- #

class TestIntentRouter:
    """The heuristic is the fallback path, so it must stand on its own."""

    def test_structural_question_routes_to_graph_side(self):
        decision = heuristic_route("Who owns Emirates Islamic?")
        assert decision.strategy in ("graph", "hybrid")

    def test_narrative_question_routes_to_vector(self):
        decision = heuristic_route("Summarise the outlook and explain why.")
        assert decision.strategy == "vector"

    def test_entity_plus_figure_routes_to_hybrid(self):
        decision = heuristic_route("How much net profit did Emirates NBD report in 2025?")
        assert decision.strategy == "hybrid"

    def test_always_returns_a_valid_strategy(self):
        for question in ["", "?", "hello", "a" * 400]:
            assert heuristic_route(question).strategy in ("vector", "graph", "hybrid")

    def test_extracts_proper_nouns(self):
        mentions = extract_entity_mentions("Did Emirates NBD acquire RBL Bank?")
        joined = " ".join(mentions)
        assert "Emirates NBD" in joined and "RBL Bank" in joined

    def test_drops_sentence_initial_question_words(self):
        """'Which' is capitalised by grammar and is not an entity."""
        assert all(m not in ("Which", "What", "Who")
                   for m in extract_entity_mentions("Which banks were acquired?"))
