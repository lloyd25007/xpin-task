"""
Load a hand-written Cypher file into Neo4j and make it retrievable.

Two steps, deliberately separate:

1. **Run the file unchanged.** Whatever schema it defines is the author's, and
   this script does not second-guess it. Statements are split on semicolons and
   executed in order.

2. **Project a shared ``:Entity`` label on top.** The retrieval layer cannot
   know a document's domain vocabulary, so it queries one label and one
   full-text index. Adding ``:Entity`` alongside ``:Organization`` (rather than
   replacing it) means both views coexist: domain queries keep reading well,
   and generic traversal starts working.

Nothing is deleted or renamed. Re-running is safe: every write is a MERGE or an
idempotent SET.

Usage::

    python scripts/load_cypher.py emirates_nbd_graph.cypher
    python scripts/load_cypher.py graph.cypher --wipe    # clear the graph first
    python scripts/load_cypher.py graph.cypher --dry-run # parse only
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings          # noqa: E402
from app.storage.graph_store import GraphStore, entity_key  # noqa: E402

logger = logging.getLogger(__name__)

# Labels that should NOT be folded into :Entity. Chunk nodes are structural
# provenance anchors, not things a question can be about, and pulling them into
# the entity space would flood every traversal with document plumbing.
STRUCTURAL_LABELS = {"Chunk", "Entity"}


def split_statements(text: str) -> list[str]:
    """Split a Cypher script into individual executable statements.

    The Neo4j driver runs one statement per call, so the file has to be broken
    up. Line comments are stripped first: a ``//`` inside a comment could
    otherwise hide a semicolon and silently merge two statements.

    String literals are respected, so a semicolon inside quotes does not split.
    """
    # Strip full-line and trailing // comments, but not inside quotes.
    cleaned_lines = []
    for line in text.splitlines():
        out, in_str, quote = [], False, ""
        i = 0
        while i < len(line):
            ch = line[i]
            if in_str:
                out.append(ch)
                if ch == quote and line[i - 1 : i] != "\\":
                    in_str = False
            elif ch in "'\"":
                in_str, quote = True, ch
                out.append(ch)
            elif ch == "/" and line[i + 1 : i + 2] == "/":
                break  # rest of the line is a comment
            else:
                out.append(ch)
            i += 1
        cleaned_lines.append("".join(out))

    cleaned = "\n".join(cleaned_lines)

    # Now split on semicolons that are not inside a string literal.
    statements, buf, in_str, quote = [], [], False, ""
    for ch in cleaned:
        if in_str:
            buf.append(ch)
            if ch == quote:
                in_str = False
        elif ch in "'\"":
            in_str, quote = True, ch
            buf.append(ch)
        elif ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def run_script(store: GraphStore, statements: list[str]) -> tuple[int, int]:
    """Execute each statement, continuing past individual failures.

    A constraint that already exists raises, and that is fine -- the script is
    meant to be re-runnable. Aborting the whole load because one idempotent
    statement was already satisfied would make re-running impossible.

    Returns:
        ``(succeeded, failed)``.
    """
    ok = failed = 0
    for i, stmt in enumerate(statements, start=1):
        preview = re.sub(r"\s+", " ", stmt)[:68]
        try:
            store._run(stmt)
            ok += 1
            print(f"  [{i:>2}/{len(statements)}] ok    {preview}")
        except Exception as exc:
            message = str(exc).split("\n")[0][:90]
            # "already exists" on a constraint is the expected re-run case.
            benign = "already exists" in message.lower() or "equivalent" in message.lower()
            if benign:
                ok += 1
                print(f"  [{i:>2}/{len(statements)}] skip  {preview}  (already present)")
            else:
                failed += 1
                print(f"  [{i:>2}/{len(statements)}] FAIL  {preview}\n           {message}")
    return ok, failed


def project_entities(store: GraphStore) -> dict:
    """Add the shared ``:Entity`` label and the properties retrieval needs.

    Every non-structural node gains:

    * ``:Entity`` -- so one full-text index and one traversal cover the graph.
    * ``id``      -- canonical, case-folded key. ``entity_key`` is the same
      function the LLM extractor uses, so a hand-written node and an extracted
      one describing the same company collapse onto one id.
    * ``type``    -- the node's own domain label, used for colour and filtering.
    * ``name`` / ``description`` -- normalised from whatever the source used.

    The original labels are kept, so the author's queries continue to work.

    Returns:
        Counts of what was projected.
    """
    # Pull candidates out first so ids can be computed in Python with the same
    # normalisation the rest of the app uses.
    rows = store._run(
        """
        MATCH (n)
        WHERE NOT n:Chunk AND NOT n:Entity AND n.name IS NOT NULL
        RETURN elementId(n) AS eid, n.name AS name, labels(n) AS labels,
               coalesce(n.description, n.desc, '') AS description
        """
    )
    if not rows:
        return {"projected": 0}

    payload = [
        {
            "eid": r["eid"],
            "id": entity_key(r["name"]),
            "name": r["name"],
            # First non-structural label is the node's kind.
            "type": next((l for l in r["labels"] if l not in STRUCTURAL_LABELS), "Entity"),
            "description": r["description"] or "",
        }
        for r in rows
    ]

    store._run(
        """
        UNWIND $rows AS row
        MATCH (n) WHERE elementId(n) = row.eid
        SET n:Entity,
            n.id = row.id,
            n.type = coalesce(n.type, row.type),
            n.description = CASE WHEN row.description = ''
                                 THEN coalesce(n.description, '') ELSE row.description END
        """,
        rows=payload,
    )

    # Nodes keyed only by name can collide on id after case-folding; report it
    # rather than silently merging, since merging is a judgement call.
    dupes = store._run(
        """
        MATCH (e:Entity) WITH e.id AS id, count(*) AS c
        WHERE c > 1 RETURN id, c ORDER BY c DESC LIMIT 10
        """
    )
    return {"projected": len(payload), "duplicate_ids": dupes}


def main() -> int:
    """Load the file, project entities, and report what the graph now holds."""
    parser = argparse.ArgumentParser(description="Load a Cypher file into Neo4j.")
    parser.add_argument("file", help="Path to the .cypher file")
    parser.add_argument("--wipe", action="store_true",
                        help="Delete existing Entity/Chunk nodes before loading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and list statements without executing")
    parser.add_argument("--no-project", action="store_true",
                        help="Load only; skip adding the :Entity label")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    path = settings.resolve(args.file)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    statements = split_statements(path.read_text(encoding="utf-8"))
    print("=" * 78)
    print(f"Loading {path.name}  ->  {settings.neo4j_uri}")
    print(f"{len(statements)} statement(s)")
    print("=" * 78)

    if args.dry_run:
        for i, s in enumerate(statements, 1):
            print(f"\n--- {i} ---\n{s}")
        return 0

    store = GraphStore()
    if not store.verify():
        print(f"ERROR: cannot reach Neo4j at {settings.neo4j_uri}", file=sys.stderr)
        return 1

    if args.wipe:
        print("\nClearing existing graph…")
        store.clear()

    ok, failed = run_script(store, statements)
    print(f"\n{ok} succeeded, {failed} failed")

    if not args.no_project:
        print("\nProjecting :Entity label for retrieval…")
        result = project_entities(store)
        print(f"  {result['projected']} node(s) labelled :Entity")
        for d in result.get("duplicate_ids") or []:
            print(f"  note: id '{d['id']}' is shared by {d['c']} nodes")

        # The full-text index must exist over the newly labelled nodes.
        store.init_schema()
        print("  full-text index ready")

    stats = store.stats()
    print("\n" + "=" * 78)
    print(f"  entities      : {stats['nodes']:,}")
    print(f"  relationships : {stats['relationships']:,}")
    print(f"  chunks        : {stats['chunks']:,}")
    print("=" * 78)
    store.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
