"""
The LangGraph retrieval pipeline.

Topology::

                        +--> vector_search --+
    START --> route ----|                    |--> fuse --> generate --> END
                        +--> graph_search ---+

``route`` classifies the question and returns the *names of the nodes to run*.
LangGraph turns a list of names into a parallel fan-out, so the hybrid path
executes both retrieval arms concurrently rather than one after the other -- the
graph traversal overlaps with the embedding + FAISS search instead of adding to
it.

``fuse`` is deliberately one node for all three strategies. Single-arm routes
just take the top N; only the hybrid route pays for Reciprocal Rank Fusion and
cross-encoder reranking. That keeps the graph shape simple and the cost
proportional to what the question actually needs.

``generate`` streams. The LLM is tagged ``final_answer`` so the SSE layer can
forward exactly those tokens to the browser and ignore the router's tokens.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.chat.llm import get_answer_llm
from app.chat import prompts
from app.retrieval.retriever import dedupe_preserving_order, fuse_candidates, rerank
from app.retrieval.retriever import graph_search
from app.retrieval.retriever import route_query
from app.core.schemas import Citation, Triple
from app.storage.graph_store import GraphStore
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _take_last(_current: Any, new: Any) -> Any:
    """Reducer keeping the most recent write to a state key.

    Needed on keys that parallel branches may both touch: without a reducer,
    LangGraph raises ``InvalidUpdateError`` on concurrent writes.
    """
    return new


class RAGState(TypedDict, total=False):
    """State threaded through the graph. Every node returns a partial update."""

    # --- input ---
    question: str
    history: list[dict[str, str]]
    top_k: int | None
    max_hops: int | None
    use_graph: bool

    # --- set by route ---
    strategy: str                 # 'vector' | 'graph' | 'hybrid'
    entities: list[str]           # entity mentions found in the question
    routing_reason: str

    # --- set by the retrieval arms (distinct keys: safe to write in parallel) ---
    vector_candidates: list[dict]
    graph_candidates: list[dict]
    triples: list[Triple]
    seeds: list[dict]

    # --- set by fuse ---
    citations: Annotated[list[Citation], _take_last]
    used_triples: list[Triple]

    # --- set by generate ---
    answer: str


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def _node_route(state: RAGState) -> dict:
    """Decide which retrieval arms to run.

    Delegates the classification to :func:`app.retrieval.retriever.route_query` and
    records both the decision and its justification in state, so the UI can show
    *why* a question took a particular path.

    A caller can force plain vector RAG with ``use_graph=False`` -- useful for
    comparing hybrid against baseline retrieval on the same question.
    """
    question = state["question"]

    if not state.get("use_graph", True):
        return {
            "strategy": "vector",
            "entities": [],
            "routing_reason": "graph disabled by the request",
        }

    decision = route_query(question)
    logger.info(
        "Router chose '%s' (%s); entities=%s",
        decision.strategy, decision.reasoning, decision.entities,
    )
    return {
        "strategy": decision.strategy,
        "entities": decision.entities,
        "routing_reason": decision.reasoning,
    }


def _select_arms(state: RAGState) -> list[str]:
    """Conditional edge: map the strategy onto the nodes to execute.

    Returning a list is what makes LangGraph fan out. ``hybrid`` runs both arms
    concurrently; the single-strategy routes run one and skip the other's cost
    entirely.
    """
    strategy = state.get("strategy", "hybrid")
    if strategy == "vector":
        return ["vector_search"]
    if strategy == "graph":
        return ["graph_search"]
    return ["vector_search", "graph_search"]


def _make_vector_node(vector_store: VectorStore):
    """Build the vector retrieval node, closing over the store instance."""

    def _node_vector_search(state: RAGState) -> dict:
        """Dense retrieval over child chunks, expanded to parent blocks.

        Retrieves more children than we ultimately need (``top_k``), because
        several children usually collapse onto the same parent and fusion works
        better with a deeper candidate pool.
        """
        question = state["question"]
        top_k = state.get("top_k") or settings.vector_top_k
        # Over-fetch parents here; fuse/rerank narrows to parent_top_n later.
        results = vector_store.search_parents(question, k=top_k, top_n=settings.parent_top_n * 2)
        logger.info("Vector arm: %d parent candidates", len(results))
        return {"vector_candidates": results}

    return _node_vector_search


def _make_graph_node(graph_store: GraphStore, vector_store: VectorStore):
    """Build the graph retrieval node, closing over both stores."""

    def _node_graph_search(state: RAGState) -> dict:
        """Seed resolution plus N-hop traversal over the knowledge graph.

        Failures here are logged and swallowed: if Neo4j is down, hybrid should
        quietly degrade to vector-only rather than failing the whole request.
        """
        question = state["question"]
        mentions = state.get("entities", [])
        max_hops = state.get("max_hops") or settings.graph_max_hops
        try:
            candidates, triples, seeds = graph_search(
                graph_store, vector_store, question, mentions,
                max_hops=max_hops, top_n=settings.parent_top_n * 2,
            )
        except Exception as exc:
            logger.error("Graph arm failed (%s); continuing without it", exc)
            return {"graph_candidates": [], "triples": [], "seeds": []}

        logger.info("Graph arm: %d parent candidates, %d triples", len(candidates), len(triples))
        return {"graph_candidates": candidates, "triples": triples, "seeds": seeds}

    return _node_graph_search


def _to_citation(record: dict) -> Citation:
    """Convert an internal candidate record into the public Citation schema.

    Keeps the wire format free of retrieval internals (RRF scores, hop counts)
    while preserving the provenance the UI needs: which parent, which pages, and
    which child chunks actually matched.
    """
    parent = record["parent"]
    # Prefer the reranker's judgement when it ran, then RRF, then raw similarity.
    score = record.get("rerank_score", record.get("rrf_score", record.get("score", 0.0)))
    meta = parent.meta or {}
    return Citation(
        parent_id=parent.parent_id,
        doc_id=parent.doc_id,
        text=parent.text,
        pages=parent.pages,
        section=parent.section,
        score=float(score),
        matched_child_ids=[c for c in record.get("matched_child_ids", []) if c],
        source=meta.get("source", ""),
        char_count=meta.get("char_count", len(parent.text)),
        token_count=meta.get("token_count", 0),
        has_table=bool(meta.get("has_table", False)),
        has_figures=bool(meta.get("has_figures", False)),
        ingested_at=meta.get("ingested_at", ""),
        # Which arm(s) surfaced this passage -- shown on the citation card so a
        # reader can see whether it came from similarity, structure, or both.
        retrieved_by=record.get("sources", []),
    )


def _make_fuse_node(vector_store: VectorStore):
    """Build the fusion node, closing over the vector store for its fallback."""

    def _node_fuse(state: RAGState) -> dict:
        """Combine the retrieval arms into one ranked citation list.

        Three cases:

        * **Both arms ran** -- Reciprocal Rank Fusion merges the two rankings,
          then the cross-encoder re-scores the fused shortlist. This is the only
          path that pays for reranking, and it is the path where it matters: RRF
          mixes two very different notions of relevance, so a content-aware
          final pass is what keeps the ordering sensible.
        * **One arm ran** -- that arm's ordering is already coherent, so we
          simply truncate. The cross-encoder still runs when enabled, since it
          improves ordering cheaply on a short list.
        * **The chosen arm came back empty** -- fall back to dense retrieval
          rather than giving up. This matters in practice: the router sends a
          pure ownership question down the ``graph`` path, and if the graph has
          not been populated yet (or Neo4j is unreachable) that arm returns
          nothing -- while the vector index can very often still answer. Without
          this fallback the system says "I don't know" with the answer sitting
          in the index.

        Triples are carried through regardless of which arm produced the
        citations, because they are displayed as graph evidence alongside the
        answer.
        """
        vector_candidates = state.get("vector_candidates") or []
        graph_candidates = state.get("graph_candidates") or []
        triples = state.get("triples") or []
        question = state["question"]
        fell_back = False

        # --- Rescue an empty single-arm result -------------------------
        if not vector_candidates and not graph_candidates:
            logger.info("Both arms empty; falling back to dense retrieval")
            vector_candidates = vector_store.search_parents(
                question, k=settings.vector_top_k, top_n=settings.parent_top_n
            )
            fell_back = True
        elif not vector_candidates and state.get("strategy") == "graph":
            # Graph-only route that found triples but no usable parent chunks.
            logger.info("Graph arm produced no passages; adding dense retrieval")
            vector_candidates = vector_store.search_parents(
                question, k=settings.vector_top_k, top_n=settings.parent_top_n
            )
            fell_back = True

        if vector_candidates and graph_candidates:
            fused = fuse_candidates(vector_candidates, graph_candidates)
            ranked = rerank(question, fused, top_n=settings.parent_top_n)
        else:
            single = vector_candidates or graph_candidates
            for record in single:
                record.setdefault("sources", ["vector" if vector_candidates else "graph"])
            ranked = rerank(question, single, top_n=settings.parent_top_n)

        if fell_back:
            logger.info("Fallback retrieval supplied %d candidate(s)", len(ranked))

        citations = [_to_citation(record) for record in ranked]

        # Surface the triples that justify the citations we kept, first; then
        # any remaining triples as general context. Capped so the prompt stays lean.
        kept_parent_ids = {c.parent_id for c in citations}
        prioritised = [t for t in triples if any(p in kept_parent_ids for p in t.parent_ids)]
        remainder = [t for t in triples if t not in prioritised]
        used_triples = dedupe_preserving_order(
            prioritised + remainder,
            key=lambda t: (t.source, t.relation, t.target),
        )[: settings.graph_max_triples]

        logger.info("Fusion produced %d citations and %d triples", len(citations), len(used_triples))
        return {"citations": citations, "used_triples": used_triples}

    return _node_fuse


async def _node_generate(state: RAGState) -> dict:
    """Stream the grounded answer from the retrieved context.

    The LLM is invoked with ``astream`` so tokens surface as they are produced;
    the SSE route listens to LangGraph's event stream and forwards each chunk.
    The full text is also accumulated and returned, giving the client a
    canonical final answer that does not depend on having caught every token.

    When retrieval came back empty the model is not called at all -- there is
    nothing to ground an answer in, and inventing one is the failure mode this
    whole architecture exists to prevent.
    """
    citations = state.get("citations") or []
    triples = state.get("used_triples") or []

    if not citations and not triples:
        return {"answer": prompts.NO_CONTEXT_ANSWER}

    user_message = prompts.ANSWER_USER.format(
        history_block=prompts.format_history(state.get("history") or []),
        question=state["question"],
        context_block=prompts.format_context(citations),
        graph_block=prompts.format_triples(triples),
    )

    llm = get_answer_llm()
    parts: list[str] = []
    async for chunk in llm.astream(
        [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content": user_message},
        ]
    ):
        # ``chunk.content`` is a str for text deltas; some providers emit
        # structured content blocks, which we ignore for the plain-text answer.
        if isinstance(chunk.content, str):
            parts.append(chunk.content)

    return {"answer": "".join(parts)}


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #

def build_rag_graph(vector_store: VectorStore, graph_store: GraphStore):
    """Compile the LangGraph pipeline.

    Args:
        vector_store: FAISS + parent store.
        graph_store: Neo4j store.

    Returns:
        A compiled graph exposing ``ainvoke`` (full result) and
        ``astream_events`` (token-level streaming), both used by the API.
    """
    workflow = StateGraph(RAGState)

    workflow.add_node("route", _node_route)
    workflow.add_node("vector_search", _make_vector_node(vector_store))
    workflow.add_node("graph_search", _make_graph_node(graph_store, vector_store))
    workflow.add_node("fuse", _make_fuse_node(vector_store))
    workflow.add_node("generate", _node_generate)

    workflow.add_edge(START, "route")

    # The conditional edge returns node names; a list fans out in parallel.
    workflow.add_conditional_edges(
        "route",
        _select_arms,
        # Path map: every node the router may return must be declared here.
        {"vector_search": "vector_search", "graph_search": "graph_search"},
    )

    # Both arms converge on fusion. LangGraph waits for whichever arms ran.
    workflow.add_edge("vector_search", "fuse")
    workflow.add_edge("graph_search", "fuse")
    workflow.add_edge("fuse", "generate")
    workflow.add_edge("generate", END)

    compiled = workflow.compile()
    logger.info("RAG graph compiled: route -> [vector|graph] -> fuse -> generate")
    return compiled
