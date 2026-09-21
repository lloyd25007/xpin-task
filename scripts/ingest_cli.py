"""
Command-line ingestion.

Running ingestion outside the API is the practical way to load a long document:
there is no HTTP timeout to worry about, progress prints to the terminal, and
the FAISS index and Neo4j graph it produces are exactly what the API serves.

Usage::

    python scripts/ingest_cli.py                       # full report, graph on
    python scripts/ingest_cli.py --max-pages 20        # quick smoke test
    python scripts/ingest_cli.py --no-graph            # vectors only, fast
    python scripts/ingest_cli.py --reset               # wipe first
    python scripts/ingest_cli.py --path other.pdf --doc-id other
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make `import app.*` work when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings          # noqa: E402
from app.ingestion.pipeline import ingest_pdf  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    """Set up console logging and silence third-party noise.

    The HTTP and transformer libraries log a line per request and per batch,
    which buries the pipeline's own progress output.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers",
                  "transformers", "neo4j", "docling", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _progress_printer():
    """Return a progress callback that prints a single updating status line.

    Uses a carriage return rather than newlines so a long ingest does not
    produce hundreds of lines of scrollback.
    """
    start = time.time()

    def report(stage: str, fraction: float, counters: dict) -> None:
        elapsed = time.time() - start
        bar_width = 28
        filled = int(bar_width * fraction)
        bar = "#" * filled + "-" * (bar_width - filled)
        line = f"\r[{bar}] {fraction*100:5.1f}%  {elapsed:6.1f}s  {stage[:52]:<52}"
        sys.stdout.write(line)
        sys.stdout.flush()
        if fraction >= 1.0:
            sys.stdout.write("\n")

    return report


def main() -> int:
    """Parse arguments, run the pipeline, and print a summary.

    Returns:
        Process exit code: 0 on success, 1 on failure, 130 on Ctrl-C.
    """
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the FAISS index and Neo4j knowledge graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path", default=settings.default_pdf_path, help="PDF to ingest")
    parser.add_argument("--doc-id", default=None, help="Document id (defaults to the filename stem)")
    parser.add_argument("--max-pages", type=int, default=None, help="Ingest only the first N pages")
    parser.add_argument("--no-graph", action="store_true",
                        help="Skip LLM graph extraction (vectors only, much faster)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear the vector index and graph before ingesting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    pdf_path = settings.resolve(args.path)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    print("=" * 78)
    print("GraphRAG ingestion")
    print("=" * 78)
    print(f"  document   : {pdf_path.name}")
    print(f"  pages      : {args.max_pages or 'all'}")
    print(f"  parser     : Docling (layout + tables)")
    print(f"  chunking   : {settings.parent_chunk_tokens}-token parents / "
          f"{settings.child_chunk_tokens}-token children")
    print(f"  embeddings : {settings.embedding_model}")
    print(f"  graph      : {'OFF' if args.no_graph else settings.neo4j_uri}")
    print(f"  extract LLM: {settings.openrouter_model}")
    print("=" * 78)

    started = time.time()
    try:
        counters = ingest_pdf(
            pdf_path=pdf_path,
            doc_id=args.doc_id,
            extract_graph=not args.no_graph,
            reset=args.reset,
            max_pages=args.max_pages,
            progress=_progress_printer(),
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Any vectors written before the interrupt are saved.")
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        logging.getLogger(__name__).exception("Ingestion failed")
        return 1

    elapsed = time.time() - started
    print("=" * 78)
    print(f"Completed in {elapsed:.1f}s")
    for key, value in counters.items():
        print(f"  {key:<16}: {value:,}")
    print("=" * 78)
    print("\nNext:")
    print("  uvicorn app.api.main:app --port 8000")
    print("  open http://localhost:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
