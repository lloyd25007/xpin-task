"""
Prompt templates for extraction and answer generation.

Kept in one module so prompt wording -- the part of a RAG system most likely to
need iteration -- can be tuned without touching pipeline code.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Graph extraction
# --------------------------------------------------------------------------- #

EXTRACTION_SYSTEM = """You extract a knowledge graph from excerpts of a bank's annual report.

Return entities (nodes) and typed relationships (edges) that are EXPLICITLY stated in the excerpt.

ENTITY RULES
- Use the exact name as written: "Emirates NBD", not "the bank" or "the Group".
- Resolve pronouns and generic references to the named entity they refer to. If an excerpt says "the Group reported...", and the document is Emirates NBD's report, the entity is "Emirates NBD".
- Assign one type from: Company, Subsidiary, BusinessSegment, Product, Service, Person, Role, Location, Metric, Technology, Initiative, Regulator, Award, Currency.
- Skip generic nouns that name nothing specific ("customers", "the market", "shareholders" in general).

RELATIONSHIP RULES
- The type must be an UPPER_SNAKE_CASE verb phrase, e.g. OWNS, ACQUIRED, AGREED_TO_ACQUIRE_STAKE_IN, OPERATES_IN, OFFERS, REPORTED, INVESTS_IN, LAUNCHED, PARTNERED_WITH, FACILITATED, APPOINTED, HEADQUARTERED_IN, SERVES, COMPLIES_WITH.
- Direction matters: (subject)-[VERB]->(object). "Emirates NBD owns Emirates Islamic" is Emirates NBD -[OWNS]-> Emirates Islamic.
- Financial results are relationships too: Emirates NBD -[REPORTED]-> "AED 24.0bn net profit". Keep the figure and its unit together in the target name.
- Include a short verbatim quote from the excerpt as evidence for each relationship.
- Only extract what the text states. Never infer, never use outside knowledge. If the excerpt contains no clear relationship, return empty lists.

QUALITY OVER QUANTITY
Prefer 3 precise, well-evidenced relationships to 15 vague ones."""

EXTRACTION_USER = """Extract entities and relationships from this excerpt of {doc_id}{page_hint}.

--- EXCERPT ---
{text}
--- END EXCERPT ---"""


# --------------------------------------------------------------------------- #
# Answer generation
# --------------------------------------------------------------------------- #

ANSWER_SYSTEM = """You are a financial analyst assistant answering questions about an annual report.

You are given two kinds of context:
1. DOCUMENT EXCERPTS - verbatim passages retrieved from the report.
2. KNOWLEDGE GRAPH FACTS - relationship triples extracted from the same report.

RULES
- Answer only from the context given. If the context does not contain the answer, say so plainly and state what is missing. Never fill gaps from prior knowledge.
- Cite the excerpts you used with bracketed markers like [1], [2], matching the numbering in the context. Put the marker right after the claim it supports.
- When a knowledge graph fact underpins a claim, mention the relationship in prose (e.g. "Emirates NBD owns Emirates Islamic") rather than printing the raw triple.
- Quote figures exactly as written, including currency and unit (AED 24.0bn, not 24 billion).
- Be direct. Lead with the answer, then the supporting detail. No preamble such as "Based on the provided context".
- Use short markdown: bold for key figures, bullet lists when enumerating. Keep it tight."""

ANSWER_USER = """{history_block}QUESTION
{question}

DOCUMENT EXCERPTS
{context_block}

KNOWLEDGE GRAPH FACTS
{graph_block}

Answer the question using only the context above, citing excerpts as [n]."""

NO_CONTEXT_ANSWER = (
    "I could not find anything in the ingested document that answers this question. "
    "The document may not cover this topic, or it may not have been ingested yet - "
    "check the index status in the sidebar."
)


def format_history(history: list[dict]) -> str:
    """Render prior turns as a compact block for follow-up questions.

    Only the last few turns are kept: enough to resolve "what about theirs?"
    style references, while leaving the context budget to retrieved passages.

    Args:
        history: ``[{"role": "user"|"assistant", "content": str}, ...]``.

    Returns:
        A formatted block ending in a blank line, or ``""`` when there is no
        history to include.
    """
    if not history:
        return ""
    recent = history[-4:]  # two exchanges is plenty for pronoun resolution
    lines = ["CONVERSATION SO FAR"]
    for turn in recent:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        # Truncate long prior answers -- they are context, not the subject.
        lines.append(f"{role}: {content[:400]}")
    lines.append("")
    return "\n".join(lines) + "\n"


def format_context(citations: list) -> str:
    """Render retrieved parent chunks as numbered excerpts.

    Numbering is what the ``[n]`` citation markers in the answer refer to, so
    the order here must match the order the UI displays citation cards in.

    Args:
        citations: :class:`app.core.schemas.Citation` objects, best first.

    Returns:
        Numbered excerpt block, or a placeholder when nothing was retrieved.
    """
    if not citations:
        return "(no excerpts retrieved)"
    blocks = []
    for i, citation in enumerate(citations, start=1):
        pages = ", ".join(str(p) for p in citation.pages) if citation.pages else "n/a"
        header = f"[{i}] {citation.doc_id} - page {pages}"
        if citation.section:
            header += f" - {citation.section}"
        blocks.append(f"{header}\n{citation.text}")
    return "\n\n".join(blocks)


def format_triples(triples: list) -> str:
    """Render graph relationships as readable arrow notation.

    ``(Emirates NBD)-[OWNS]->(Emirates Islamic)`` is compact, unambiguous about
    direction, and something LLMs parse reliably.

    Args:
        triples: :class:`app.core.schemas.Triple` objects.

    Returns:
        One triple per line, or a placeholder when the graph contributed nothing.
    """
    if not triples:
        return "(no graph relationships found for this query)"
    lines = []
    for triple in triples:
        line = f"({triple.source})-[{triple.relation}]->({triple.target})"
        if triple.evidence:
            line += f'   evidence: "{triple.evidence[:160]}"'
        lines.append(line)
    return "\n".join(lines)
