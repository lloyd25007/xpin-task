"""
RAG quality evaluation.

Runs the full pipeline over a labelled question set and scores retrieval,
groundedness and system behaviour. These are *quality* tests, not correctness
tests: they call the real LLM and can legitimately vary between runs, so they
are opt-in.

Run with::

    pytest tests/evaluation -m evaluation -v -s

The ``-s`` matters — the summary table is printed, not asserted, because the
per-metric numbers are more useful to read than to gate on. Only the coarse
thresholds are asserted, and they are set where a regression means something is
genuinely broken rather than merely different.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.evaluation.metrics import (
    EvalResult,
    aggregate,
    citation_validity,
    context_precision,
    expansion_ratio,
    graph_contribution,
    groundedness,
    hit_rate,
    mrr,
    numeric_grounding,
    recall_at_k,
    refusal_quality,
    routing_accuracy,
    triple_match,
)

pytestmark = [pytest.mark.evaluation, pytest.mark.slow]


def _evaluate_one(rag_graph, case: dict) -> EvalResult:
    """Run one question through the pipeline and score every metric.

    Args:
        rag_graph: the compiled LangGraph pipeline.
        case: one entry from ``tests/fixtures/eval_questions.json``.

    Returns:
        An :class:`EvalResult`. A pipeline exception is captured on the result
        rather than raised, so one failing question does not abort the run.
    """
    result = EvalResult(question_id=case["id"], question=case["question"])
    started = time.time()

    try:
        state = asyncio.run(
            rag_graph.ainvoke(
                {"question": case["question"], "history": [], "use_graph": True}
            )
        )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.latency_s = time.time() - started
        return result

    result.latency_s = time.time() - started
    citations = state.get("citations", []) or []
    triples = state.get("used_triples", []) or []
    answer = (state.get("answer") or "").strip()

    result.strategy = state.get("strategy", "")
    result.num_citations = len(citations)
    result.num_triples = len(triples)
    result.answer = answer

    texts = [c.text for c in citations]
    keywords = case.get("expected_keywords", [])

    result.scores = {
        # Retrieval quality
        "hit_rate": hit_rate(texts, keywords),
        "recall@3": recall_at_k(texts, keywords, 3),
        "mrr": mrr(texts, keywords),
        "context_precision": context_precision(texts, case["question"]),
        # Groundedness
        "groundedness": groundedness(answer, texts, triples),
        "citation_validity": citation_validity(answer, len(citations)),
        "numeric_grounding": numeric_grounding(answer, texts),
        # System behaviour
        "routing_accuracy": routing_accuracy(
            result.strategy, case.get("expected_strategy", ["vector", "graph", "hybrid"])
        ),
        "triple_match": triple_match(triples, case.get("expected_triples", [])),
        "graph_contribution": graph_contribution(citations, triples),
        "expansion_ratio": min(expansion_ratio(citations) / 5.0, 1.0),  # normalised
    }

    # Out-of-scope questions are scored on refusal instead of retrieval.
    if case.get("must_refuse"):
        result.scores["refusal_quality"] = refusal_quality(answer)

    return result


@pytest.fixture(scope="module")
def evaluation_run(rag_graph, eval_dataset) -> list[EvalResult]:
    """Evaluate every question once and share the results across tests.

    Module-scoped because each question costs an LLM call; re-running them per
    assertion would multiply cost and runtime for no benefit.
    """
    results = [_evaluate_one(rag_graph, case) for case in eval_dataset]

    # --- Printed report ---------------------------------------------------
    print("\n" + "=" * 92)
    print("RAG EVALUATION")
    print("=" * 92)
    header = f"{'question':<34} {'strategy':<8} {'cit':>3} {'tri':>3} {'hit':>5} {'grnd':>5} {'route':>5} {'s':>5}"
    print(header)
    print("-" * 92)
    for r in results:
        if r.error:
            print(f"{r.question[:33]:<34} ERROR: {r.error[:44]}")
            continue
        print(
            f"{r.question[:33]:<34} {r.strategy:<8} {r.num_citations:>3} {r.num_triples:>3} "
            f"{r.scores['hit_rate']:>5.2f} {r.scores['groundedness']:>5.2f} "
            f"{r.scores['routing_accuracy']:>5.0f} {r.latency_s:>5.1f}"
        )

    print("-" * 92)
    summary = aggregate(results)
    print("AGGREGATE")
    for key, value in summary.items():
        print(f"   {key:<22} {value:6.3f}")
    print("=" * 92)

    return results


# --------------------------------------------------------------------------- #
# Retrieval quality
# --------------------------------------------------------------------------- #

class TestRetrievalQuality:
    """Did retrieval fetch the right passages?"""

    def test_retrieval_finds_expected_facts(self, evaluation_run):
        """Mean hit rate must clear 0.5.

        Below that, the expected facts are missing from the context more often
        than not, and no prompt change will fix the answers.
        """
        scored = [r for r in evaluation_run if not r.error]
        assert scored, "every question errored"
        mean = sum(r.scores["hit_rate"] for r in scored) / len(scored)
        assert mean >= 0.5, f"mean hit_rate {mean:.2f} — retrieval is missing expected facts"

    def test_relevant_chunks_rank_highly(self, evaluation_run):
        """MRR must clear 0.4 — the answer should be near the top, not buried."""
        scored = [r for r in evaluation_run if not r.error]
        mean = sum(r.scores["mrr"] for r in scored) / len(scored)
        assert mean >= 0.4, f"mean MRR {mean:.2f} — relevant chunks are ranked too low"

    def test_every_question_retrieves_something(self, evaluation_run):
        """An empty context means the answer is ungrounded by construction."""
        empty = [r for r in evaluation_run
                 if not r.error and r.num_citations == 0 and r.num_triples == 0]
        assert not empty, f"no context retrieved for: {[r.question_id for r in empty]}"

    def test_context_is_not_mostly_noise(self, evaluation_run):
        """Precision must clear 0.3, or the prompt is being padded with junk."""
        scored = [r for r in evaluation_run if not r.error]
        mean = sum(r.scores["context_precision"] for r in scored) / len(scored)
        assert mean >= 0.3, f"mean context_precision {mean:.2f} — too much irrelevant context"


# --------------------------------------------------------------------------- #
# Groundedness
# --------------------------------------------------------------------------- #

class TestGroundedness:
    """Is the answer actually supported by what was retrieved?"""

    def test_answers_are_grounded_in_context(self, evaluation_run):
        """The anti-hallucination gate."""
        scored = [r for r in evaluation_run if not r.error and r.answer]
        mean = sum(r.scores["groundedness"] for r in scored) / len(scored)
        assert mean >= 0.35, f"mean groundedness {mean:.2f} — answers drift from the context"

    def test_figures_are_not_invented(self, evaluation_run):
        """Numbers must come from the document.

        The most dangerous failure in financial QA: a fluent answer quoting a
        profit figure that appears nowhere in the report.
        """
        scored = [r for r in evaluation_run if not r.error and r.answer]
        for r in scored:
            assert r.scores["numeric_grounding"] >= 0.5, (
                f"{r.question_id}: numbers in the answer are not in the retrieved text\n"
                f"  answer: {r.answer[:200]}"
            )

    def test_citations_point_at_real_sources(self, evaluation_run):
        """A `[7]` marker with five sources means dangling evidence in the UI."""
        scored = [r for r in evaluation_run if not r.error and r.answer]
        mean = sum(r.scores["citation_validity"] for r in scored) / len(scored)
        assert mean >= 0.5, f"mean citation_validity {mean:.2f} — citation markers are unreliable"

    def test_out_of_scope_questions_are_refused(self, evaluation_run, eval_dataset):
        """The system must decline rather than answer from prior knowledge."""
        refusal_ids = {c["id"] for c in eval_dataset if c.get("must_refuse")}
        for r in evaluation_run:
            if r.question_id in refusal_ids and not r.error:
                assert r.scores.get("refusal_quality", 0.0) == 1.0, (
                    f"{r.question_id}: should have declined but answered:\n  {r.answer[:220]}"
                )


# --------------------------------------------------------------------------- #
# System behaviour
# --------------------------------------------------------------------------- #

class TestSystemBehaviour:
    """Is the architecture doing what it claims?"""

    def test_router_chooses_sensible_strategies(self, evaluation_run):
        """Routing must be right for most questions.

        Not all: more than one strategy is often defensible, and the dataset
        allows a set of acceptable answers per question.
        """
        scored = [r for r in evaluation_run if not r.error]
        mean = sum(r.scores["routing_accuracy"] for r in scored) / len(scored)
        assert mean >= 0.6, f"routing accuracy {mean:.2f} — the intent router is misclassifying"

    def test_graph_actually_contributes(self, evaluation_run):
        """The knowledge graph must earn its place.

        If no question in the set draws a single triple, the graph is pure
        overhead and the architecture is vector RAG wearing a costume.
        """
        with_graph = [r for r in evaluation_run if not r.error and r.num_triples > 0]
        assert with_graph, "no question retrieved any graph triples"

    def test_expected_relationships_are_found(self, evaluation_run, eval_dataset):
        """Questions with declared expected triples must retrieve them."""
        expecting = {c["id"] for c in eval_dataset if c.get("expected_triples")}
        scored = [r for r in evaluation_run if r.question_id in expecting and not r.error]
        if not scored:
            pytest.skip("no questions declare expected triples")
        mean = sum(r.scores["triple_match"] for r in scored) / len(scored)
        assert mean >= 0.3, f"triple_match {mean:.2f} — expected relationships are missing"

    def test_small_to_big_expansion_is_real(self, evaluation_run):
        """Parents must be meaningfully larger than the children that matched.

        A ratio near 1.0 means the hierarchy costs storage and complexity while
        delivering no extra context — the strategy would be pointless.
        """
        scored = [r for r in evaluation_run if not r.error and r.num_citations > 0]
        mean = sum(r.scores["expansion_ratio"] for r in scored) / len(scored)
        assert mean > 0.15, f"expansion ratio {mean:.2f} — parents are barely bigger than children"

    def test_no_question_errors(self, evaluation_run):
        """The pipeline must not throw on any question in the set."""
        errored = [(r.question_id, r.error) for r in evaluation_run if r.error]
        assert not errored, f"pipeline errors: {errored}"
