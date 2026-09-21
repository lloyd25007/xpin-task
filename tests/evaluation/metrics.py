"""
RAG evaluation metrics.

Deliberately implemented from scratch rather than pulled from RAGAS or similar.
Two reasons: the heavyweight frameworks score almost everything with an LLM
judge, which is slow, costs quota, and is itself non-deterministic; and the
failure modes that actually matter in *this* system are structural — did
small-to-big expansion happen, did the graph contribute, is the answer traceable
to a retrieved chunk — which are measurable without a judge.

The metrics split into three groups:

**Retrieval quality** — did we fetch the right things?
    ``hit_rate``, ``recall_at_k``, ``mrr``, ``context_precision``

**Groundedness** — is the answer actually supported by what we fetched?
    ``groundedness``, ``citation_validity``, ``numeric_grounding``

**System behaviour** — did the architecture do what it claims?
    ``routing_accuracy``, ``graph_contribution``, ``expansion_ratio``

Every function returns a float in [0, 1] (or a small dict of them), so results
can be averaged across a dataset and tracked over time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for tolerant string comparison.

    Retrieval evaluation should not fail because the document wrote
    "Emirates  NBD" with two spaces.
    """
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _tokens(text: str) -> set[str]:
    """Content words of a string, for overlap-based scoring.

    Stop words are dropped because they appear in every passage and would
    inflate overlap between any two pieces of English text.
    """
    stop = {
        "the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "at", "by",
        "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
        "with", "as", "from", "has", "have", "had", "will", "would", "which",
    }
    words = re.findall(r"[a-z0-9']+", _normalise(text))
    return {w for w in words if len(w) > 2 and w not in stop}


def _contains_all(haystack: str, needles: list[str]) -> bool:
    """True when every needle appears in the haystack, case-insensitively."""
    hay = _normalise(haystack)
    return all(_normalise(n) in hay for n in needles if n)


# --------------------------------------------------------------------------- #
# Retrieval quality
# --------------------------------------------------------------------------- #

def hit_rate(retrieved_texts: list[str], expected_keywords: list[str]) -> float:
    """Did retrieval surface the expected facts *anywhere* in the context?

    The most basic retrieval question: if the keywords are absent from every
    retrieved chunk, no amount of prompting will produce a correct answer, and
    the problem is retrieval rather than generation.

    Args:
        retrieved_texts: the parent chunks handed to the LLM.
        expected_keywords: strings a correct context must contain.

    Returns:
        Fraction of keywords found, in [0, 1]. A question with no expected
        keywords scores 1.0 — there is nothing to miss.
    """
    if not expected_keywords:
        return 1.0
    combined = _normalise(" ".join(retrieved_texts))
    found = sum(1 for kw in expected_keywords if _normalise(kw) in combined)
    return found / len(expected_keywords)


def recall_at_k(retrieved_texts: list[str], expected_keywords: list[str], k: int) -> float:
    """Hit rate restricted to the top ``k`` retrieved chunks.

    Distinguishes "we found it" from "we found it *and ranked it highly*".
    A system with recall@10 = 1.0 but recall@3 = 0.3 is retrieving well and
    ranking badly — which points at the reranker, not the embeddings.
    """
    return hit_rate(retrieved_texts[:k], expected_keywords)


def mrr(retrieved_texts: list[str], expected_keywords: list[str]) -> float:
    """Mean Reciprocal Rank of the first chunk containing an expected keyword.

    Rewards putting the answer first: rank 1 scores 1.0, rank 2 scores 0.5,
    rank 5 scores 0.2. This is the metric that moves when reranking improves.

    Returns:
        ``1 / rank`` of the first relevant chunk, or 0.0 if none is relevant.
    """
    if not expected_keywords:
        return 1.0
    for rank, text in enumerate(retrieved_texts, start=1):
        haystack = _normalise(text)
        if any(_normalise(kw) in haystack for kw in expected_keywords):
            return 1.0 / rank
    return 0.0


def context_precision(retrieved_texts: list[str], question: str,
                      threshold: float = 0.06) -> float:
    """Fraction of retrieved chunks that are actually about the question.

    Precision matters independently of recall: padding the prompt with
    irrelevant parent blocks costs context budget and measurably degrades answer
    quality, even when the right chunk is also present.

    Relevance is approximated by content-word overlap. That is crude, but it is
    deterministic and free, and it reliably catches the failure it exists to
    catch — retrieval returning near-random chunks.

    Args:
        retrieved_texts: the retrieved parent chunks.
        question: the user's question.
        threshold: minimum overlap ratio for a chunk to count as relevant.

    Returns:
        Fraction of chunks clearing the threshold.
    """
    if not retrieved_texts:
        return 0.0
    q_tokens = _tokens(question)
    if not q_tokens:
        return 1.0

    relevant = 0
    for text in retrieved_texts:
        overlap = len(q_tokens & _tokens(text)) / len(q_tokens)
        if overlap >= threshold:
            relevant += 1
    return relevant / len(retrieved_texts)


# --------------------------------------------------------------------------- #
# Groundedness
# --------------------------------------------------------------------------- #

def groundedness(answer: str, retrieved_texts: list[str],
                 triples: list | None = None) -> float:
    """How much of the answer is traceable to the retrieved context?

    This is the anti-hallucination metric. Content words in the answer are
    checked against the union of the retrieved chunks and the graph triples;
    a word appearing in neither was invented by the model.

    It is a heuristic — a model can paraphrase using words that happen to be
    present — but it moves sharply in the right direction when a model starts
    fabricating, which is what it is for.

    Args:
        answer: the generated answer.
        retrieved_texts: parent chunks given to the model.
        triples: graph relationships given to the model.

    Returns:
        Fraction of the answer's content words present in the context.
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0

    context = " ".join(retrieved_texts)
    for triple in triples or []:
        source = getattr(triple, "source", None) or (triple.get("source", "") if isinstance(triple, dict) else "")
        target = getattr(triple, "target", None) or (triple.get("target", "") if isinstance(triple, dict) else "")
        relation = getattr(triple, "relation", None) or (triple.get("relation", "") if isinstance(triple, dict) else "")
        context += f" {source} {relation} {target}"

    context_tokens = _tokens(context)
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def citation_validity(answer: str, num_citations: int) -> float:
    """Do the ``[n]`` markers in the answer point at citations that exist?

    A model that cites `[7]` when five sources were supplied is signalling that
    it is not tracking its own evidence — and the citation cards in the UI
    would dangle.

    Returns:
        Fraction of markers that are in range. 1.0 when the answer cites
        nothing (nothing invalid), 0.0 when sources exist but none were cited
        *and* the answer is substantive.
    """
    markers = [int(m) for m in re.findall(r"\[(\d+)\]", answer or "")]
    if not markers:
        # No citations is only acceptable when there was nothing to cite.
        return 1.0 if num_citations == 0 else 0.0
    valid = sum(1 for m in markers if 1 <= m <= num_citations)
    return valid / len(markers)


def numeric_grounding(answer: str, retrieved_texts: list[str]) -> float:
    """Do the figures in the answer appear in the retrieved text?

    Financial QA fails most dangerously on numbers: a fluent answer quoting a
    fabricated profit figure is worse than no answer. Every number in the answer
    is checked against the context.

    Returns:
        Fraction of numbers found in the context, or 1.0 when the answer
        contains none.
    """
    # Strip citation markers first — [1] is a marker, not a claimed figure.
    cleaned = re.sub(r"\[\d+\]", " ", answer or "")
    numbers = re.findall(r"\d[\d,]*\.?\d*", cleaned)
    if not numbers:
        return 1.0

    context = " ".join(retrieved_texts)
    context_digits = re.sub(r"[,\s]", "", context)

    grounded = 0
    for number in numbers:
        bare = number.replace(",", "")
        if bare in context_digits:
            grounded += 1
    return grounded / len(numbers)


def refusal_quality(answer: str) -> float:
    """Does the answer correctly decline when the document cannot support it?

    Scored for out-of-scope questions. A system that confidently answers
    "what is Tesla's share price?" from a bank's annual report is worse than one
    that says it does not know.

    Returns:
        1.0 if the answer admits the limitation, 0.0 otherwise.
    """
    signals = [
        "could not find", "couldn't find", "not find", "does not contain",
        "doesn't contain", "no information", "not mentioned", "not covered",
        "unable to", "not available in", "outside the scope", "not present",
        "cannot answer", "can't answer", "not in the", "no mention",
    ]
    lowered = _normalise(answer)
    return 1.0 if any(signal in lowered for signal in signals) else 0.0


# --------------------------------------------------------------------------- #
# System behaviour
# --------------------------------------------------------------------------- #

def routing_accuracy(chosen: str, expected: list[str]) -> float:
    """Did the intent router pick an acceptable strategy?

    ``expected`` is a list rather than a single value because more than one
    choice is often defensible — a question about a named entity is reasonably
    served by either ``graph`` or ``hybrid``. The metric penalises only clearly
    wrong routing, such as ``vector`` for a pure ownership question.
    """
    return 1.0 if chosen in expected else 0.0


def triple_match(found_triples: list, expected_triples: list[dict]) -> float:
    """Fraction of expected graph relationships actually retrieved.

    Matching is loose on purpose: an empty field in the expectation is a
    wildcard, and comparison is substring-based, so
    ``AGREED_TO_ACQUIRE_STAKE_IN`` satisfies an expectation of ``ACQUIRE``.
    Extraction wording varies between runs; the *relationship* is what matters.
    """
    if not expected_triples:
        return 1.0
    if not found_triples:
        return 0.0

    def _fields(triple) -> tuple[str, str, str]:
        if isinstance(triple, dict):
            return (_normalise(triple.get("source", "")),
                    _normalise(triple.get("relation", "")),
                    _normalise(triple.get("target", "")))
        return (_normalise(getattr(triple, "source", "")),
                _normalise(getattr(triple, "relation", "")),
                _normalise(getattr(triple, "target", "")))

    found = [_fields(t) for t in found_triples]

    matched = 0
    for expected in expected_triples:
        want = (_normalise(expected.get("source", "")),
                _normalise(expected.get("relation", "")),
                _normalise(expected.get("target", "")))
        for actual in found:
            # An empty expectation field matches anything.
            if all(not w or w in a or a in w for w, a in zip(want, actual)):
                matched += 1
                break
    return matched / len(expected_triples)


def graph_contribution(citations: list, triples: list) -> float:
    """How much of the context came from the graph rather than vectors alone?

    Answers the question a reviewer will ask: is the knowledge graph actually
    doing anything, or is this vector RAG with extra infrastructure?

    Returns:
        Share of context items originating from the graph, in [0, 1].
    """
    total = len(citations) + len(triples)
    return len(triples) / total if total else 0.0


def expansion_ratio(citations: list, child_chunk_chars: int = 800) -> float:
    """Average size multiplier from child chunk to parent block.

    Verifies small-to-big is doing real work. A ratio near 1.0 means parents are
    barely larger than children, so the hierarchy is adding cost without adding
    context.

    Returns:
        Mean ``len(parent_text) / child_chunk_chars``.
    """
    if not citations:
        return 0.0
    sizes = [
        len(c.text if hasattr(c, "text") else c.get("text", ""))
        for c in citations
    ]
    return (sum(sizes) / len(sizes)) / child_chunk_chars


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

@dataclass
class EvalResult:
    """Scores for a single evaluated question."""

    question_id: str
    question: str
    strategy: str = ""
    num_citations: int = 0
    num_triples: int = 0
    latency_s: float = 0.0
    answer: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        """A question passes when retrieval found the facts and nothing was fabricated.

        Groundedness is thresholded well below 1.0 because paraphrase legitimately
        introduces words that are not in the source; the metric is there to catch
        wholesale invention, not rewording.
        """
        if self.error:
            return False
        return (
            self.scores.get("hit_rate", 0.0) >= 0.5
            and self.scores.get("groundedness", 0.0) >= 0.35
            and self.scores.get("numeric_grounding", 1.0) >= 0.5
        )


def aggregate(results: list[EvalResult]) -> dict[str, float]:
    """Average every metric across a set of results.

    Returns:
        Mean of each metric, plus ``pass_rate``, ``error_rate`` and mean latency.
        Questions that errored are excluded from metric means (their scores are
        absent, not zero) but still count against the pass and error rates.
    """
    if not results:
        return {}

    successful = [r for r in results if not r.error]
    keys = {k for r in successful for k in r.scores}

    summary = {
        key: sum(r.scores.get(key, 0.0) for r in successful) / len(successful)
        for key in sorted(keys)
    } if successful else {}

    summary["pass_rate"] = sum(1 for r in results if r.passed) / len(results)
    summary["error_rate"] = sum(1 for r in results if r.error) / len(results)
    summary["mean_latency_s"] = (
        sum(r.latency_s for r in successful) / len(successful) if successful else 0.0
    )
    return summary
