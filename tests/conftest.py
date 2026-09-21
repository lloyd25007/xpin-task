"""
Shared pytest configuration and fixtures.

Test layers, and what each is allowed to touch:

* ``tests/unit``        — pure logic. No network, no database, no LLM. Always runs.
* ``tests/integration`` — needs live stores (FAISS populated, Neo4j reachable).
  Skipped automatically when they are not available.
* ``tests/evaluation``  — RAG quality metrics. Needs live stores and the LLM, so
  it is opt-in via ``-m evaluation``.

The auto-skip matters: a contributor running ``pytest`` on a fresh clone should
get a clean pass from the unit layer rather than a wall of connection errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config: pytest.Config) -> None:
    """Register the custom markers so ``--strict-markers`` stays usable."""
    config.addinivalue_line("markers", "unit: pure logic, no external services")
    config.addinivalue_line("markers", "integration: requires FAISS index and/or Neo4j")
    config.addinivalue_line("markers", "evaluation: retrieval-quality metrics; needs stores + LLM")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


# --------------------------------------------------------------------------- #
# Service availability — computed once, reused by every fixture
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def vector_store():
    """The live FAISS store, or skip the test if nothing has been ingested.

    Session-scoped because loading FAISS also loads the embedding model, which
    costs seconds and hundreds of megabytes.
    """
    from app.storage.vector_store import VectorStore

    store = VectorStore()
    if store.vector_count == 0:
        pytest.skip("vector index is empty — run `python scripts/ingest_cli.py` first")
    return store


@pytest.fixture(scope="session")
def graph_store():
    """The live Neo4j store, or skip if it is unreachable or empty."""
    from app.storage.graph_store import GraphStore

    store = GraphStore()
    if not store.verify():
        pytest.skip("Neo4j is unreachable — check NEO4J_URI in .env")
    if store.stats()["nodes"] == 0:
        pytest.skip("graph is empty — ingest with graph extraction enabled")
    yield store
    store.close()


@pytest.fixture(scope="session")
def rag_graph(vector_store, graph_store):
    """The compiled LangGraph pipeline, built once per session."""
    from app.chat.rag_graph import build_rag_graph

    return build_rag_graph(vector_store, graph_store)


@pytest.fixture(scope="session")
def eval_dataset() -> list[dict]:
    """Load the evaluation question set.

    Kept as JSON rather than inline Python so the questions can be extended
    without touching test code.
    """
    import json

    path = Path(__file__).parent / "fixtures" / "eval_questions.json"
    if not path.exists():
        pytest.skip(f"evaluation dataset missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


@pytest.fixture
def sample_pages():
    """Synthetic page documents for chunking tests.

    Deliberately synthetic: chunking invariants must hold for any input, and
    tying them to the real PDF would make the unit layer depend on a 7MB file.
    """
    from langchain_core.documents import Document

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
