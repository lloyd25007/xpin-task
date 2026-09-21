"""Retrieval: routing, graph traversal and rank fusion.

The three stages that decide *what context the model sees*: an intent router
picks a strategy, the graph arm walks relationships, and fusion merges the two
rankings and reranks them.
"""

from __future__ import annotations

from app.core.config import settings
from app.chat.llm import build_llm
from app.core.schemas import Triple
from app.storage.graph_store import GraphStore
from app.storage.vector_store import VectorStore
from functools import lru_cache
from pydantic import BaseModel, Field
from typing import Any, Iterable, Sequence
from typing import Literal
import logging
import re


# ═══════════════════════════════════════════════════════════════════════
# Intent router -- vector, graph, or hybrid
# ═══════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

Strategy = Literal["vector", "graph", "hybrid"]


class RouteDecision(BaseModel):
    """Structured output returned by the LLM router."""

    strategy: Strategy = Field(
        description="'vector' for narrative/summary questions, 'graph' for pure "
                    "structural/relationship questions, 'hybrid' when the answer "
                    "needs both facts from text and relationships between entities."
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities mentioned in the question (companies, people, "
                    "products, segments). Used as graph traversal seeds.",
    )
    reasoning: str = Field(default="", description="One short sentence justifying the choice.")


# --------------------------------------------------------------------------- #
# Heuristic routing
# --------------------------------------------------------------------------- #

# Verbs and nouns that imply a relationship question -> the graph helps.
_GRAPH_SIGNALS = re.compile(
    r"\b(own|owns|owned|subsidiar|acquir|acquisition|stake|merge|parent compan|"
    r"relationship|related|connect|link|partner|joint venture|invest|invests|"
    r"invested|holding|structure|who (is|are|owns|leads|heads)|which compan|"
    r"portfolio|affiliat|collaborat|supplier|customer of|report(s|ed)? to)\b",
    re.I,
)

# Phrasing that implies free-text explanation -> chunks alone are fine.
_VECTOR_SIGNALS = re.compile(
    r"\b(summar|describe|explain|overview|outlook|statement|what does .* say|"
    r"how does|why|narrative|discuss|commentary|strategy for|approach to|"
    r"highlight|key takeaway)\b",
    re.I,
)

# Quantities usually live in the text, but are attached to an entity -> hybrid.
_FACT_SIGNALS = re.compile(
    r"\b(how much|how many|revenue|profit|income|aed|usd|percent|%|ratio|margin|"
    r"growth|total|amount|figure|number of|20\d\d)\b",
    re.I,
)

# Rough capitalised-phrase detector for seed entities when no LLM is used.
_PROPER_NOUN = re.compile(r"\b([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*)*)\b")

# Sentence-initial words that are capitalised by grammar, not because they name anything.
_STOP_CAPS = {
    "What", "Who", "Which", "How", "Why", "When", "Where", "Does", "Did", "Is",
    "Are", "Was", "Were", "The", "A", "An", "In", "On", "For", "And", "Or",
    "Tell", "List", "Show", "Give", "Explain", "Describe", "Summarise", "Summarize",
}


def extract_entity_mentions(question: str) -> list[str]:
    """Pull likely entity names out of a question without calling an LLM.

    Capitalised multi-word phrases are a decent proxy for named entities in
    financial prose ("Emirates Islamic", "RBL Bank"). Sentence-initial question
    words are filtered out so "Which banks..." does not seed on "Which".

    Args:
        question: the raw user question.

    Returns:
        Candidate entity mentions, longest first (more specific seeds first).
    """
    candidates: list[str] = []
    for match in _PROPER_NOUN.finditer(question):
        phrase = match.group(1).strip()
        words = phrase.split()
        # Drop leading grammar-capitalised words such as "Which" or "What".
        while words and words[0] in _STOP_CAPS:
            words = words[1:]
        phrase = " ".join(words)
        if len(phrase) > 2 and phrase not in candidates:
            candidates.append(phrase)
    candidates.sort(key=len, reverse=True)
    return candidates[:6]


def heuristic_route(question: str) -> RouteDecision:
    """Classify a question using keyword signals only.

    Cheap and deterministic. Used when ``ROUTING_MODE=heuristic``, and as the
    fallback whenever the LLM router errors or times out.

    The logic: a relationship signal means the graph is useful; a quantitative
    signal means the text is needed too. Both -> hybrid. Neither relationship
    signal nor entity mention -> vector, since there is nothing to traverse from.
    """
    entities = extract_entity_mentions(question)
    graph_like = bool(_GRAPH_SIGNALS.search(question))
    vector_like = bool(_VECTOR_SIGNALS.search(question))
    fact_like = bool(_FACT_SIGNALS.search(question))

    if graph_like and (fact_like or vector_like):
        strategy: Strategy = "hybrid"
        why = "relationship wording plus a factual/narrative component"
    elif graph_like:
        # A structural question with named entities can be answered by traversal,
        # but we still keep hybrid when the question is long enough to hide detail.
        strategy = "graph" if len(question.split()) <= 12 else "hybrid"
        why = "relationship wording dominates"
    elif vector_like and not entities:
        strategy = "vector"
        why = "narrative question with no named entity to traverse from"
    elif entities:
        strategy = "hybrid"
        why = "names an entity, so graph context may add relationships"
    else:
        strategy = "vector"
        why = "no relationship signal or entity mention"

    return RouteDecision(strategy=strategy, entities=entities, reasoning=why)


# --------------------------------------------------------------------------- #
# LLM routing
# --------------------------------------------------------------------------- #

_ROUTER_SYSTEM = """You route questions about a bank's annual report to the right retrieval strategy.

Choose exactly one strategy:
- "vector": the question asks for narrative, description, summary, commentary or explanation. Plain passage retrieval answers it.
- "graph": the question asks purely about relationships between named things (ownership, subsidiaries, acquisitions, who-does-what). A relationship graph answers it directly.
- "hybrid": the question involves named entities AND needs facts, figures or explanation from the text. This is the safe default when unsure.

Also list the named entities mentioned in the question (companies, subsidiaries, people, products, business segments). Return [] if there are none.
Be concise in your reasoning."""


def llm_route(question: str) -> RouteDecision:
    """Classify a question with the LLM, falling back to the heuristic on error.

    Uses structured output so the response is a validated :class:`RouteDecision`
    rather than free text that needs parsing. The router model is deliberately
    non-streaming and low temperature -- this is a classification, not prose.

    Args:
        question: the raw user question.

    Returns:
        A RouteDecision. On any failure (network, rate limit, schema mismatch)
        the heuristic result is returned instead, so routing always succeeds.
    """
    try:
        llm = build_llm(streaming=False, temperature=0.0, tags=["router"])
        structured = llm.with_structured_output(RouteDecision)
        decision = structured.invoke(
            [
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": question},
            ]
        )
        if isinstance(decision, RouteDecision):
            # The LLM sometimes returns an empty entity list even when the
            # question clearly names something; backfill from the heuristic so
            # the graph branch always has seeds to work with.
            if not decision.entities:
                decision.entities = extract_entity_mentions(question)
            return decision
        logger.warning("Router returned unexpected type %s; using heuristic", type(decision))
    except Exception as exc:
        logger.warning("LLM routing failed (%s); using heuristic", exc)
    return heuristic_route(question)


def route_query(question: str, mode: str | None = None) -> RouteDecision:
    """Public entry point: decide the retrieval strategy for a question.

    Args:
        question: the raw user question.
        mode: overrides ``settings.routing_mode`` for this call.
            ``"llm"`` classifies with the model, ``"heuristic"`` uses rules,
            ``"hybrid"`` skips classification and always runs both retrievers.

    Returns:
        The chosen :class:`RouteDecision`.
    """
    mode = (mode or settings.routing_mode).lower()

    if mode == "hybrid":
        # Force the full pipeline -- useful for demos and A/B comparison.
        return RouteDecision(
            strategy="hybrid",
            entities=extract_entity_mentions(question),
            reasoning="routing disabled; hybrid forced by configuration",
        )
    if mode == "heuristic":
        return heuristic_route(question)
    return llm_route(question)

# ═══════════════════════════════════════════════════════════════════════
# Graph arm -- seed resolution and N-hop traversal
# ═══════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)


def resolve_seeds(graph: GraphStore, question: str, mentions: list[str],
                  limit: int = 8) -> list[dict]:
    """Map question text and router-extracted mentions onto real graph nodes.

    The router says the question mentions "RBL Bank"; this function finds the
    node whose canonical id actually is RBL Bank. Mentions are tried first
    because they are precise, then the full question as a fallback so a question
    with no recognisable proper noun can still find seeds by description match.

    Args:
        graph: the Neo4j store.
        question: the raw user question.
        mentions: entity names produced by the router.
        limit: maximum seeds to return.

    Returns:
        Entity dicts (``id``, ``name``, ``type``, ``score``), deduplicated by id
        and ordered with mention-derived matches first.
    """
    seeds: dict[str, dict] = {}

    # --- Precise pass: one lookup per named mention -----------------------
    for mention in mentions:
        for hit in graph.find_entities(mention, limit=3):
            seeds.setdefault(hit["id"], hit)
        if len(seeds) >= limit:
            break

    # --- Fallback pass: match the whole question against name+description --
    if len(seeds) < limit:
        for hit in graph.find_entities(question, limit=limit - len(seeds)):
            seeds.setdefault(hit["id"], hit)

    resolved = list(seeds.values())[:limit]
    logger.info("Resolved %d graph seed entities from %d mentions", len(resolved), len(mentions))
    return resolved


def traverse_to_triples(graph: GraphStore, seed_ids: list[str],
                        max_hops: int | None = None,
                        limit: int | None = None) -> list[Triple]:
    """Run the N-hop traversal and convert rows into :class:`Triple` objects.

    Args:
        graph: the Neo4j store.
        seed_ids: canonical entity ids to start from.
        max_hops: traversal depth (clamped to 1-3 inside the store).
        limit: maximum triples.

    Returns:
        Triples ordered by hop distance, so 1-hop facts (directly about the
        question's entities) always precede 2-hop context.
    """
    rows = graph.traverse(seed_ids, max_hops=max_hops, limit=limit)
    triples = [
        Triple(
            source=row["source"],
            source_type=row.get("source_type", "Entity"),
            relation=row["relation"],
            target=row["target"],
            target_type=row.get("target_type", "Entity"),
            evidence=row.get("evidence", "") or "",
            parent_ids=row.get("parent_ids", []) or [],
            hop=int(row.get("hop", 1)),
            pages=row.get("pages", []) or [],
            confidence=float(row.get("confidence", 0.5)),
            mention_count=int(row.get("mention_count", 1)),
        )
        for row in rows
    ]
    logger.info("Traversal returned %d triples from %d seeds", len(triples), len(seed_ids))
    return triples


def triples_to_parent_candidates(triples: list[Triple], vector_store: VectorStore,
                                 top_n: int | None = None) -> list[dict]:
    """Convert graph triples into ranked parent-chunk candidates.

    This is the step that closes the provenance loop. Each relationship stores
    the ``parent_ids`` of the chunks it was extracted from, so a set of triples
    resolves directly to a set of passages -- retrieved *because of structure*
    rather than similarity.

    Ranking rules, in order of importance:

    1. **Hop distance.** A parent supporting a 1-hop fact is more relevant than
       one supporting a 2-hop fact, so the best (lowest) hop wins.
    2. **Supporting triple count.** Among parents at the same hop, the one that
       several relationships point at is more central to the question.

    Args:
        triples: output of :func:`traverse_to_triples`.
        vector_store: used to resolve parent ids to their text.
        top_n: how many candidates to keep.

    Returns:
        Records in the same shape the vector retriever produces -- ``parent_id``,
        ``parent``, ``score``, ``matched_child_ids`` -- plus ``hop`` and the
        ``triples`` that justified the hit.
    """
    top_n = top_n or settings.parent_top_n
    if not triples:
        return []

    # --- Group triples by the parent chunk that produced them -------------
    by_parent: dict[str, dict] = {}
    for triple in triples:
        for parent_id in triple.parent_ids:
            entry = by_parent.setdefault(
                parent_id, {"parent_id": parent_id, "hop": triple.hop, "triples": []}
            )
            entry["hop"] = min(entry["hop"], triple.hop)  # best hop wins
            entry["triples"].append(triple)

    # --- Attach parent text, dropping ids the vector store never saw ------
    candidates: list[dict] = []
    for parent_id, entry in by_parent.items():
        parent = vector_store.parent_store.get(parent_id)
        if parent is None:
            # Graph written but vector index rebuilt without it -- skip quietly.
            continue
        entry["parent"] = parent
        entry["matched_child_ids"] = []          # graph hits match a parent, not a child
        # Synthetic score so this list is orderable on its own; RRF only uses
        # the resulting rank, never this magnitude.
        entry["score"] = 1.0 / entry["hop"] + 0.05 * len(entry["triples"])
        candidates.append(entry)

    # Closest hop first, then most-supported.
    candidates.sort(key=lambda c: (c["hop"], -len(c["triples"])))
    logger.info("Graph produced %d parent candidates from %d triples", len(candidates), len(triples))
    return candidates[:top_n]


def graph_search(graph: GraphStore, vector_store: VectorStore, question: str,
                 mentions: list[str], max_hops: int | None = None,
                 top_n: int | None = None) -> tuple[list[dict], list[Triple], list[dict]]:
    """Run the full graph retrieval arm end to end.

    Args:
        graph: Neo4j store.
        vector_store: used to resolve parent ids to text.
        question: raw user question.
        mentions: entity names from the router.
        max_hops: traversal depth.
        top_n: how many parent candidates to return.

    Returns:
        ``(parent_candidates, triples, seeds)``. All three are returned because
        the API surfaces each separately: candidates feed fusion, triples are
        shown as graph citations, and seeds drive the subgraph visualiser.
    """
    seeds = resolve_seeds(graph, question, mentions)
    if not seeds:
        logger.info("No graph seeds matched; graph arm contributes nothing")
        return [], [], []

    seed_ids = [s["id"] for s in seeds]
    triples = traverse_to_triples(graph, seed_ids, max_hops=max_hops)
    candidates = triples_to_parent_candidates(triples, vector_store, top_n=top_n)
    return candidates, triples, seeds

# ═══════════════════════════════════════════════════════════════════════
# Rank fusion -- Reciprocal Rank Fusion + cross-encoder rerank
# ═══════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #

def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[Sequence[str], float]],
    k: int | None = None,
) -> dict[str, float]:
    """Fuse several ranked id lists into one score map.

    Args:
        ranked_lists: ``(ids_in_rank_order, weight)`` pairs. Rank is taken from
            list position, so the caller must pass the lists already sorted
            best-first. Weight scales one retriever's influence.
        k: RRF damping constant (default from settings). Larger ``k`` flattens
            the contribution of top ranks, making fusion more democratic.

    Returns:
        ``{id: fused_score}``. Higher is better. Ids absent from a list simply
        contribute nothing for that list -- no imputation, no penalty.
    """
    k = k or settings.rrf_k
    scores: dict[str, float] = {}
    for ids, weight in ranked_lists:
        for rank, doc_id in enumerate(ids, start=1):
            # The 1/(k+rank) shape is what makes RRF scale-free.
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


def fuse_candidates(
    vector_results: list[dict],
    graph_results: list[dict],
    k: int | None = None,
) -> list[dict]:
    """Fuse the vector and graph candidate lists into one ranked list.

    Both inputs are lists of parent-chunk records; the function merges records
    describing the same parent, records which retriever(s) found it, and orders
    the result by fused RRF score.

    Args:
        vector_results: records from ``VectorStore.search_parents`` (best first),
            each with ``parent_id``, ``parent``, ``score``, ``matched_child_ids``.
        graph_results: records derived from graph traversal (best first), same
            shape, plus ``hop`` and ``triples``.
        k: RRF damping constant.

    Returns:
        Merged records sorted by ``rrf_score`` descending. Each carries a
        ``sources`` list naming which retrievers contributed, which the UI shows
        on the citation card.
    """
    vector_ids = [r["parent_id"] for r in vector_results]
    graph_ids = [r["parent_id"] for r in graph_results]

    fused = reciprocal_rank_fusion(
        [
            (vector_ids, settings.rrf_vector_weight),
            (graph_ids, settings.rrf_graph_weight),
        ],
        k=k,
    )

    # Merge the two record sets, keeping the richest version of each field.
    merged: dict[str, dict] = {}
    for record in vector_results:
        entry = dict(record)
        entry["sources"] = ["vector"]
        merged[record["parent_id"]] = entry

    for record in graph_results:
        pid = record["parent_id"]
        if pid in merged:
            # Found by both retrievers -- the strongest possible signal.
            merged[pid]["sources"].append("graph")
            merged[pid]["hop"] = record.get("hop", merged[pid].get("hop"))
            # Union the triples that justified the graph hit.
            existing = merged[pid].get("triples", [])
            merged[pid]["triples"] = existing + [
                t for t in record.get("triples", []) if t not in existing
            ]
        else:
            entry = dict(record)
            entry["sources"] = ["graph"]
            merged[pid] = entry

    for pid, entry in merged.items():
        entry["rrf_score"] = fused.get(pid, 0.0)

    ranked = sorted(merged.values(), key=lambda r: r["rrf_score"], reverse=True)
    logger.info(
        "RRF fused %d vector + %d graph candidates into %d unique parents",
        len(vector_results), len(graph_results), len(ranked),
    )
    return ranked


# --------------------------------------------------------------------------- #
# Cross-encoder reranking
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _load_cross_encoder():
    """Load the cross-encoder reranker once per process.

    Returns ``None`` if the model cannot be loaded (not installed, no disk space,
    offline on first run). Callers must treat ``None`` as "skip reranking" rather
    than an error: RRF order alone is still a good ordering, so degrading
    gracefully keeps the app answering.
    """
    if not settings.use_reranker:
        logger.info("Reranker disabled by configuration")
        return None
    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder %s (first run downloads ~1.1GB)", settings.reranker_model)
        return CrossEncoder(settings.reranker_model, max_length=512, device="cpu")
    except Exception as exc:
        logger.warning("Cross-encoder unavailable (%s); falling back to RRF order", exc)
        return None


def rerank(question: str, candidates: list[dict], top_n: int | None = None,
           text_key: str = "parent") -> list[dict]:
    """Re-score fused candidates with a cross-encoder and return the best ones.

    Unlike the bi-encoder used for indexing -- which embeds question and passage
    separately and compares vectors -- a cross-encoder feeds both through the
    model together. It can therefore judge whether a passage actually answers
    *this* question, not merely whether it is topically nearby.

    Args:
        question: the user's question.
        candidates: fused records from :func:`fuse_candidates`.
        top_n: how many records to keep (default ``settings.parent_top_n``).
        text_key: attribute on each record holding the ParentChunk.

    Returns:
        The top ``top_n`` records, ordered by cross-encoder score where the model
        was available and by RRF score otherwise. A ``rerank_score`` field is
        added when reranking actually ran.
    """
    top_n = top_n or settings.parent_top_n
    if not candidates:
        return []

    model = _load_cross_encoder()
    if model is None:
        return candidates[:top_n]

    # Build (question, passage) pairs. Passages are truncated because the
    # cross-encoder's window is 512 tokens; the head of a parent block carries
    # the topic sentence, so the head is the right part to keep.
    pairs: list[tuple[str, str]] = []
    for record in candidates:
        parent = record.get(text_key)
        text = getattr(parent, "text", "") if parent is not None else record.get("text", "")
        pairs.append((question, text[:2000]))

    try:
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    except Exception as exc:
        logger.warning("Reranking failed (%s); keeping RRF order", exc)
        return candidates[:top_n]

    for record, score in zip(candidates, scores):
        record["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda r: r.get("rerank_score", 0.0), reverse=True)
    logger.info("Cross-encoder reranked %d candidates, keeping top %d", len(reranked), top_n)
    return reranked[:top_n]


def dedupe_preserving_order(items: Iterable[Any], key=lambda x: x) -> list[Any]:
    """Drop duplicates while keeping first-seen order.

    Used for triples and citations, where ranking carries meaning and Python's
    ``set`` would discard it.
    """
    seen: set = set()
    out: list[Any] = []
    for item in items:
        k = key(item)
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out
