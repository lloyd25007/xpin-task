"""
The graph side of retrieval: a Neo4j store for entities, typed relationships
and their provenance back to the vector chunks.

Data model
----------
::

    (:Entity {id, name, type, description})
        -[:REL_TYPE {evidence, parent_ids, child_ids, doc_id}]->
    (:Entity)

    (:Entity)-[:MENTIONED_IN]->(:Chunk {id, kind, parent_id, doc_id, pages})

Two things are worth calling out:

* **Relationship types are real Neo4j types**, not a ``type`` property on a
  generic edge. ``MATCH ()-[:ACQUIRED]->()`` is what makes the graph useful, so
  the type string is interpolated into the Cypher after being whitelisted
  against ``[A-Z0-9_]`` -- never taken raw from the LLM.
* **Chunk nodes close the provenance loop.** Each entity links to the parent and
  child chunks it was extracted from, so a graph hit can always be traced back
  to the exact passage that justifies it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.config import settings

logger = logging.getLogger(__name__)

# The driver logs every server-side notification (deprecation hints and the
# like) at WARNING. They are informational, extremely verbose, and drown the
# application's own logs, so the notification channel is quietened to ERROR.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# Relationship types are interpolated into Cypher, so they are strictly
# whitelisted. Anything the LLM produces outside this alphabet is rewritten.
_SAFE_REL_TYPE = re.compile(r"[^A-Z0-9_]")

# Fallback when the model emits an unusable relationship verb.
_DEFAULT_REL_TYPE = "RELATED_TO"

# Reserved: our own structural edge must never collide with an extracted one.
_RESERVED_REL_TYPES = {"MENTIONED_IN"}


def sanitize_rel_type(raw: str) -> str:
    """Convert a free-text relation verb into a legal Neo4j relationship type.

    ``"agreed to acquire stake in"`` becomes ``AGREED_TO_ACQUIRE_STAKE_IN``.
    This function is the only thing standing between LLM output and string
    interpolation into Cypher, so it is deliberately strict: uppercase, collapse
    to underscores, drop everything else, and fall back to a safe default.
    """
    if not raw:
        return _DEFAULT_REL_TYPE
    cleaned = _SAFE_REL_TYPE.sub("_", raw.strip().upper().replace(" ", "_"))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        return _DEFAULT_REL_TYPE
    if cleaned in _RESERVED_REL_TYPES:
        cleaned = f"{cleaned}_EXTRACTED"
    return cleaned[:60]  # keep types readable in the Neo4j browser


def entity_key(name: str) -> str:
    """Canonical, case-insensitive key used to deduplicate entities.

    "Emirates NBD", "emirates nbd" and "Emirates  NBD" must all resolve to one
    node, otherwise traversal fragments across near-duplicates. The display name
    is preserved separately on the node.
    """
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return key or "unknown"


class GraphStore:
    """Thin, purpose-built wrapper over the Neo4j Bolt driver.

    Holds one driver (which is itself a connection pool) for the process
    lifetime. Every write is idempotent via ``MERGE``, so re-ingesting a
    document updates the graph instead of duplicating it.
    """

    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None, database: str | None = None) -> None:
        """Create the driver. Connection is lazy -- Neo4j need not be up yet."""
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self._driver = None

    @property
    def driver(self):
        """Return the Bolt driver, creating it on first use."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
            )
        return self._driver

    def close(self) -> None:
        """Release the connection pool (called on FastAPI shutdown)."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def verify(self) -> bool:
        """Check that Neo4j is reachable and the credentials work.

        Returns ``True``/``False`` rather than raising, so /health can report a
        degraded state instead of returning a 500.
        """
        try:
            self.driver.verify_connectivity()
            return True
        except (ServiceUnavailable, Neo4jError, Exception) as exc:
            logger.warning("Neo4j not reachable at %s: %s", self.uri, exc)
            return False

    def _run(self, cypher: str, **params) -> list[dict[str, Any]]:
        """Execute one Cypher statement and materialise the rows as dicts.

        Materialising inside the session matters: Neo4j result objects are
        consumed lazily and become invalid once the session closes.
        """
        with self.driver.session(database=self.database) as session:
            return [record.data() for record in session.run(cypher, **params)]

    # ------------------------------------------------------------- schema --

    def init_schema(self) -> None:
        """Create constraints and indexes the app relies on.

        All statements are ``IF NOT EXISTS``, so this is safe to call on every
        startup. The uniqueness constraints also give us the backing indexes
        that make MERGE fast during ingestion.
        """
        statements = [
            # Uniqueness: one node per canonical entity key / chunk id.
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            # Lookup indexes for filtering by type / document.
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
            "CREATE INDEX chunk_parent IF NOT EXISTS FOR (c:Chunk) ON (c.parent_id)",
            # Full-text index: how a natural-language question finds seed entities.
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.description]",
        ]
        for stmt in statements:
            try:
                self._run(stmt)
            except Neo4jError as exc:
                # Equivalent-constraint-exists is benign; anything else is worth seeing.
                logger.debug("Schema statement skipped (%s): %s", exc.code, stmt)
        logger.info("Neo4j schema ready")

    def clear(self) -> None:
        """Delete every node and relationship this app owns.

        Batched with ``CALL { ... } IN TRANSACTIONS`` so a large graph does not
        blow the heap in a single transaction.
        """
        try:
            self._run(
                "MATCH (n) WHERE n:Entity OR n:Chunk "
                "CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS"
            )
        except Neo4jError:
            # Older servers / small graphs: plain delete is fine.
            self._run("MATCH (n) WHERE n:Entity OR n:Chunk DETACH DELETE n")
        logger.info("Cleared graph")

    # -------------------------------------------------------------- writes --

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """Create ``:Chunk`` nodes for parent and child chunks.

        These are the provenance anchors. They store only a short preview of the
        text -- the authoritative copy lives in the parent store / FAISS -- so
        the graph stays small and fast to traverse.

        Args:
            chunks: dicts with keys ``id``, ``kind`` ('parent'|'child'),
                ``doc_id``, ``parent_id``, ``pages``, ``preview``.

        Returns:
            Number of chunk nodes written.
        """
        if not chunks:
            return 0
        self._run(
            """
            UNWIND $rows AS row
            MERGE (c:Chunk {id: row.id})
            SET c.kind        = row.kind,
                c.doc_id      = row.doc_id,
                c.source      = row.source,
                c.parent_id   = row.parent_id,
                c.pages       = row.pages,
                c.page_start  = row.page_start,
                c.page_end    = row.page_end,
                c.section     = row.section,
                c.preview     = row.preview,
                c.char_count  = row.char_count,
                c.token_count = row.token_count,
                c.has_table   = row.has_table,
                c.has_figures = row.has_figures,
                c.ingested_at = row.ingested_at,
                c.parser      = row.parser
            """,
            rows=chunks,
        )
        return len(chunks)

    def link_parent_child_chunks(self, pairs: list[dict]) -> int:
        """Materialise the parent-child hierarchy inside the graph itself.

        Lets a Cypher query walk from a child chunk up to its parent without
        consulting the JSON parent store -- useful for graph-first exploration
        in the Neo4j browser.

        Args:
            pairs: dicts with ``parent_id`` and ``child_id``.
        """
        if not pairs:
            return 0
        self._run(
            """
            UNWIND $rows AS row
            MATCH (p:Chunk {id: row.parent_id})
            MATCH (c:Chunk {id: row.child_id})
            MERGE (p)-[:HAS_CHILD]->(c)
            """,
            rows=pairs,
        )
        return len(pairs)

    def upsert_entities(self, entities: list[dict]) -> int:
        """Create or update ``:Entity`` nodes.

        Uses ``coalesce`` so a later, richer mention can fill in a description
        that an earlier mention left blank, without ever overwriting good data
        with an empty string.

        Args:
            entities: dicts with ``id``, ``name``, ``type``, ``description``, ``doc_id``.
        """
        if not entities:
            return 0
        self._run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {id: row.id})
            ON CREATE SET e.name          = row.name,
                          e.type          = row.type,
                          e.description   = row.description,
                          e.created_at    = row.seen_at,
                          e.mention_count = 0
            SET e.updated_at = row.seen_at,
                e.type       = coalesce(row.type, e.type),
                // Never overwrite a real description with an empty one: a later
                // mention in a table may carry no prose, and losing the earlier
                // definition would make the node less useful than before.
                e.description = CASE
                                  WHEN row.description IS NULL OR row.description = ''
                                  THEN e.description ELSE row.description
                                END,
                e.doc_ids     = CASE
                                  WHEN row.doc_id IN coalesce(e.doc_ids, [])
                                  THEN e.doc_ids ELSE coalesce(e.doc_ids, []) + row.doc_id
                                END,
                // Surface forms actually seen in the text. The canonical id
                // collapses casing, so this is where the variants survive.
                e.aliases     = CASE
                                  WHEN row.name IN coalesce(e.aliases, [])
                                  THEN e.aliases ELSE coalesce(e.aliases, []) + row.name
                                END,
                // Pages the entity appears on, for page-scoped filtering.
                e.pages       = CASE
                                  WHEN row.page IS NULL OR row.page IN coalesce(e.pages, [])
                                  THEN coalesce(e.pages, []) ELSE coalesce(e.pages, []) + row.page
                                END,
                // How many chunks mentioned it: a cheap importance signal that
                // ranks the graph's hubs without running a centrality algorithm.
                e.mention_count = coalesce(e.mention_count, 0) + 1
            """,
            rows=entities,
        )
        return len(entities)

    def upsert_relationships(self, relationships: list[dict]) -> int:
        """Create typed edges between entities, carrying chunk provenance.

        Neo4j will not accept a parameterised relationship type, so rows are
        grouped by type and the (already sanitised) type is interpolated into
        one statement per distinct type -- still a single round trip per type
        rather than per row.

        Args:
            relationships: dicts with ``source_id``, ``target_id``, ``type``,
                ``evidence``, ``parent_ids``, ``child_ids``, ``doc_id``.

        Returns:
            Number of edges written.
        """
        if not relationships:
            return 0

        # Group by relationship type so each type needs only one UNWIND query.
        by_type: dict[str, list[dict]] = {}
        for rel in relationships:
            rel_type = sanitize_rel_type(rel.get("type", ""))
            by_type.setdefault(rel_type, []).append(rel)

        written = 0
        for rel_type, rows in by_type.items():
            # rel_type is safe by construction -- sanitize_rel_type() guarantees [A-Z0-9_].
            cypher = f"""
            UNWIND $rows AS row
            MATCH (s:Entity {{id: row.source_id}})
            MATCH (t:Entity {{id: row.target_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            ON CREATE SET r.created_at = row.seen_at, r.mention_count = 0
            SET r.updated_at = row.seen_at,
                r.type       = '{rel_type}',
                r.doc_id     = row.doc_id,
                r.model      = row.model,
                r.evidence   = CASE WHEN row.evidence = '' THEN r.evidence ELSE row.evidence END,
                // Provenance, unioned across every chunk that produced this
                // same edge -- the list is what lets a triple be traced back.
                r.parent_ids = CASE
                                 WHEN r.parent_ids IS NULL THEN row.parent_ids
                                 ELSE [x IN r.parent_ids WHERE NOT x IN row.parent_ids] + row.parent_ids
                               END,
                r.child_ids  = CASE
                                 WHEN r.child_ids IS NULL THEN row.child_ids
                                 ELSE [x IN r.child_ids WHERE NOT x IN row.child_ids] + row.child_ids
                               END,
                r.pages      = CASE
                                 WHEN r.pages IS NULL THEN row.pages
                                 ELSE [x IN r.pages WHERE NOT x IN row.pages] + row.pages
                               END,
                // Independent extractions of the same triple. One mention may
                // be a misread caption; three from different chunks is a fact
                // the document states repeatedly.
                r.mention_count = coalesce(r.mention_count, 0) + 1,
                // Confidence derived from that count, saturating at 1.0 by the
                // third mention. Deliberately not a model-reported score:
                // self-reported confidence is poorly calibrated, whereas
                // repeated independent extraction is real evidence.
                r.confidence = CASE
                                 WHEN coalesce(r.mention_count, 0) + 1 >= 3 THEN 1.0
                                 ELSE 0.5 + 0.25 * (coalesce(r.mention_count, 0))
                               END
            """
            try:
                self._run(cypher, rows=rows)
                written += len(rows)
            except Neo4jError as exc:
                logger.error("Failed writing %d '%s' relationships: %s", len(rows), rel_type, exc)
        return written

    def link_entities_to_chunks(self, links: list[dict]) -> int:
        """Connect entities to the chunks they were extracted from.

        This is the provenance mapping the brief calls for: from any graph node
        you can reach both the parent block and the child chunks that mention it.

        Args:
            links: dicts with ``entity_id`` and ``chunk_id``.
        """
        if not links:
            return 0
        self._run(
            """
            UNWIND $rows AS row
            MATCH (e:Entity {id: row.entity_id})
            MATCH (c:Chunk  {id: row.chunk_id})
            MERGE (e)-[:MENTIONED_IN]->(c)
            """,
            rows=links,
        )
        return len(links)

    # -------------------------------------------------------------- search --

    def find_entities(self, query: str, limit: int = 8) -> list[dict]:
        """Find seed entities for a natural-language question.

        Strategy, cheapest first:

        1. Full-text index over name + description (handles fuzzy phrasing).
        2. Substring fallback, in case the full-text index is missing or the
           query is a single short token the analyser discards.

        Args:
            query: raw user question or an extracted entity mention.
            limit: maximum seeds to return.

        Returns:
            Entity dicts with ``id``, ``name``, ``type``, ``score``.
        """
        if not query.strip():
            return []

        # Lucene syntax: quote-strip, then OR the terms with fuzzy matching so
        # "Emirates NBD net profit" still matches the "Emirates NBD" node.
        terms = [t for t in re.findall(r"[A-Za-z0-9&']+", query) if len(t) > 2]
        if not terms:
            return []
        lucene = " OR ".join(f"{t}~0.8" for t in terms[:12])

        try:
            rows = self._run(
                """
                CALL db.index.fulltext.queryNodes('entity_fulltext', $q, {limit: $limit})
                YIELD node, score
                RETURN node.id AS id, node.name AS name,
                       coalesce(node.type, 'Entity') AS type, score
                ORDER BY score DESC
                """,
                q=lucene,
                limit=limit,
            )
            if rows:
                return rows
        except Neo4jError as exc:
            logger.debug("Full-text entity search unavailable (%s); falling back", exc)

        # --- Fallback: plain substring match on the name ------------------
        return self._run(
            """
            UNWIND $terms AS term
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower(term)
            RETURN DISTINCT e.id AS id, e.name AS name,
                   coalesce(e.type,'Entity') AS type, 1.0 AS score
            LIMIT $limit
            """,
            terms=terms[:12],
            limit=limit,
        )

    def traverse(self, entity_ids: list[str], max_hops: int | None = None,
                 limit: int | None = None) -> list[dict]:
        """Walk outward from seed entities and return the relationships found.

        This is the N-hop traversal that gives the RAG pipeline its structural
        context: facts that are *connected* to the question's entities but might
        never appear in a chunk that is textually similar to the question.

        ``max_hops`` is inlined into the pattern because Cypher does not accept a
        parameter inside a variable-length bound; it is clamped to 1..3 first, so
        no untrusted value ever reaches the query.

        Args:
            entity_ids: canonical ids from :meth:`find_entities`.
            max_hops: traversal depth (clamped to 1-3; deeper explodes fast).
            limit: maximum triples returned.

        Returns:
            Triple dicts with source/target names and types, relation type,
            evidence, provenance chunk ids, and the hop distance.
        """
        if not entity_ids:
            return []
        max_hops = max(1, min(int(max_hops or settings.graph_max_hops), 3))
        limit = limit or settings.graph_max_triples

        cypher = f"""
        MATCH (seed:Entity) WHERE seed.id IN $ids
        MATCH path = (seed)-[rels*1..{max_hops}]-(other:Entity)
        WHERE ALL(r IN rels WHERE type(r) <> 'MENTIONED_IN')
        WITH rels, length(path) AS hop
        UNWIND range(0, size(rels)-1) AS i
        WITH rels[i] AS r, hop
        WITH DISTINCT r, min(hop) AS hop
        WITH r, hop, startNode(r) AS s, endNode(r) AS t
        RETURN s.name AS source,
               coalesce(s.type,'Entity')  AS source_type,
               type(r)                    AS relation,
               t.name AS target,
               coalesce(t.type,'Entity')  AS target_type,
               coalesce(r.evidence,'')      AS evidence,
               coalesce(r.parent_ids,[])    AS parent_ids,
               coalesce(r.pages,[])         AS pages,
               coalesce(r.confidence,0.5)   AS confidence,
               coalesce(r.mention_count,1)  AS mention_count,
               hop
        ORDER BY hop ASC, confidence DESC, relation ASC
        LIMIT $limit
        """
        try:
            return self._run(cypher, ids=entity_ids, limit=limit)
        except Neo4jError as exc:
            logger.error("Graph traversal failed: %s", exc)
            return []

    def subgraph(self, entity_ids: list[str], max_hops: int | None = None,
                 limit: int = 100) -> dict[str, list[dict]]:
        """Return a node/edge payload for the graph visualiser.

        Differs from :meth:`traverse` in shape, not in intent: the visualiser
        needs a deduplicated node list plus edges referencing node ids, whereas
        the LLM prompt wants flat triples.

        Args:
            entity_ids: seed entity ids.
            max_hops: traversal depth (clamped 1-3).
            limit: maximum edges to include.

        Returns:
            ``{"nodes": [...], "edges": [...]}``.
        """
        if not entity_ids:
            return {"nodes": [], "edges": []}
        max_hops = max(1, min(int(max_hops or settings.graph_max_hops), 3))

        cypher = f"""
        MATCH (seed:Entity) WHERE seed.id IN $ids
        MATCH path = (seed)-[rels*1..{max_hops}]-(other:Entity)
        WHERE ALL(r IN rels WHERE type(r) <> 'MENTIONED_IN')
        UNWIND rels AS r
        WITH DISTINCT r, startNode(r) AS s, endNode(r) AS t
        RETURN s.id AS source_id, s.name AS source_name,
               coalesce(s.type,'Entity') AS source_type,
               coalesce(s.description,'') AS source_desc,
               t.id AS target_id, t.name AS target_name,
               coalesce(t.type,'Entity') AS target_type,
               coalesce(t.description,'') AS target_desc,
               type(r) AS rel_type, coalesce(r.evidence,'') AS evidence
        LIMIT $limit
        """
        try:
            rows = self._run(cypher, ids=entity_ids, limit=limit)
        except Neo4jError as exc:
            logger.error("Subgraph query failed: %s", exc)
            return {"nodes": [], "edges": []}

        # Deduplicate nodes; edges may legitimately repeat between the same pair
        # with different types, so key them on the full triple.
        nodes: dict[str, dict] = {}
        edges: dict[tuple, dict] = {}
        for row in rows:
            for side in ("source", "target"):
                nodes.setdefault(
                    row[f"{side}_id"],
                    {
                        "id": row[f"{side}_id"],
                        "label": row[f"{side}_name"],
                        "type": row[f"{side}_type"],
                        "description": row[f"{side}_desc"],
                    },
                )
            key = (row["source_id"], row["rel_type"], row["target_id"])
            edges.setdefault(
                key,
                {
                    "source": row["source_id"],
                    "target": row["target_id"],
                    "type": row["rel_type"],
                    "evidence": row["evidence"],
                },
            )
        return {"nodes": list(nodes.values()), "edges": list(edges.values())}

    def chunks_for_entities(self, entity_ids: list[str], limit: int = 20) -> list[dict]:
        """Resolve entities back to the chunks that mention them.

        The reverse provenance direction: given graph hits, recover the passages
        that produced them so the answer can cite text rather than bare triples.
        """
        if not entity_ids:
            return []
        return self._run(
            """
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.id IN $ids
            RETURN DISTINCT c.id AS chunk_id, c.kind AS kind,
                   c.parent_id AS parent_id, c.doc_id AS doc_id, c.pages AS pages
            LIMIT $limit
            """,
            ids=entity_ids,
            limit=limit,
        )

    def stats(self) -> dict:
        """Counters for the health endpoint and the UI sidebar.

        Deliberately four small queries rather than one with CALL subqueries:
        subquery syntax changed across Neo4j 5.x minor versions, and these
        counts are cheap enough that portability is worth more than a single
        round trip.
        """
        out = {"nodes": 0, "relationships": 0, "chunks": 0, "documents": []}
        queries = {
            "nodes": "MATCH (e:Entity) RETURN count(e) AS v",
            "chunks": "MATCH (c:Chunk) RETURN count(c) AS v",
            "relationships": "MATCH ()-[r]->() "
                             "WHERE type(r) <> 'MENTIONED_IN' AND type(r) <> 'HAS_CHILD' "
                             "RETURN count(r) AS v",
            "documents": "MATCH (e:Entity) UNWIND coalesce(e.doc_ids, []) AS d "
                         "RETURN collect(DISTINCT d) AS v",
        }
        for key, cypher in queries.items():
            try:
                rows = self._run(cypher)
                out[key] = rows[0]["v"] if rows else out[key]
            except Neo4jError as exc:
                logger.warning("Graph stat '%s' unavailable: %s", key, exc)
        return out
