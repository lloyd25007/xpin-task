"""
Loading a hand-authored Cypher file into Neo4j.

Kept in the app rather than in ``scripts/`` because two callers need it: the
CLI (``scripts/load_cypher.py``) and the API's startup bootstrap, which loads
the file automatically when it finds an empty graph.

Two steps, deliberately separate:

1. **Run the file unchanged.** Whatever schema it defines is the author's, and
   this module does not second-guess it.

2. **Project a shared ``:Entity`` label on top.** The retrieval layer cannot
   know a document's domain vocabulary, so it queries one label and one
   full-text index. Adding ``:Entity`` alongside ``:Organization`` (rather than
   replacing it) means both views coexist.

Nothing is deleted or renamed. Re-running is safe: every write is a MERGE or an
idempotent SET.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.storage.graph_store import GraphStore, entity_key

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


def run_script(store: GraphStore, statements: list[str], quiet: bool = False) -> tuple[int, int]:
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
            if not quiet:
                print(f"  [{i:>2}/{len(statements)}] ok    {preview}")
        except Exception as exc:
            message = str(exc).split("\n")[0][:90]
            # "already exists" on a constraint is the expected re-run case.
            benign = "already exists" in message.lower() or "equivalent" in message.lower()
            if benign:
                ok += 1
                if not quiet:
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


def load_cypher_file(store: GraphStore, path: Path, project: bool = True) -> dict:
    """Load one Cypher file and optionally project the ``:Entity`` label.

    Args:
        store: an open graph store.
        path: the ``.cypher`` file.
        project: add ``:Entity`` and the properties retrieval needs.

    Returns:
        ``{"succeeded", "failed", "projected"}``.
    """
    statements = split_statements(path.read_text(encoding="utf-8"))
    ok, failed = run_script(store, statements, quiet=True)
    projected = project_entities(store).get("projected", 0) if project else 0
    if project:
        store.init_schema()   # the full-text index must cover the new nodes
    return {"succeeded": ok, "failed": failed, "projected": projected}
