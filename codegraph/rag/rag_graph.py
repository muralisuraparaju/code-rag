"""
codegraph/rag/graph_rag.py

Hybrid GraphRAG interface for the code knowledge graph.

Why GraphRAG over plain RAG?
-----------------------------
Plain vector RAG would embed and chunk all function summaries and retrieve
the top-k nearest by cosine similarity.  That works for "find similar
functions" queries but fails at structural questions like:
  "What does UserService depend on?"
  "Show me the call chain from login() to the database"
  "Which classes inherit from BaseModel?"

GraphRAG solves this by:
  1. Embedding query → find seed nodes via vector similarity
  2. From seed nodes, traverse the graph (N hops) to collect neighbourhood
  3. Assemble collected nodes+edges into a rich context string for the LLM

Architecture
------------
- VectorIndex       — ChromaDB collection of function/class summaries
- GraphTraverser    — Cypher-based neighbourhood expansion from seed nodes
- ContextAssembler  — converts graph data → readable LLM prompt context
- CodeGraphRAG      — orchestrates the above; single `.retrieve(query)` call

The RAG layer is intentionally read-only; it never writes to Neo4j.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Optional

from codegraph.config import Neo4jConfig, RAGConfig
from codegraph.graph.neo4j_writer import Neo4jWriter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector index (ChromaDB — embedded, no extra server)
# ---------------------------------------------------------------------------

class VectorIndex:
    """
    Thin wrapper around ChromaDB.
    Indexes function and class summaries so we can do semantic seed lookup.
    """

    def __init__(self, cfg: RAGConfig):
        try:
            import chromadb                                    # type: ignore
            from chromadb.utils import embedding_functions    # type: ignore
        except ImportError:
            raise ImportError("pip install chromadb")

        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)

        if cfg.embed_provider == "openai":
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=cfg.openai_api_key,
                model_name=cfg.embed_model,
            )
        else:
            # Default: sentence-transformers (local, no key needed)
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=cfg.embed_model or "all-MiniLM-L6-v2"
            )

        self._col = client.get_or_create_collection(
            name=cfg.collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, qname: str, text: str, metadata: dict):
        """Add or update one node's text embedding."""
        if not text.strip():
            return
        self._col.upsert(
            ids=[qname],
            documents=[text],
            metadatas=[metadata],
        )

    def search(self, query: str, k: int = 8) -> list[dict]:
        """Return up to k nearest results with id, document, metadata, distance."""
        results = self._col.query(
            query_texts=[query],
            n_results=min(k, self._col.count() or 1),
        )
        out = []
        for i, doc_id in enumerate(results["ids"][0]):
            out.append({
                "id":       doc_id,
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    1 - results["distances"][0][i],  # cosine → similarity
            })
        return out

    def index_from_neo4j(self, writer: Neo4jWriter):
        """
        Pull all Function and Class nodes from Neo4j and index their summaries.
        Call this once after a full graph build (or incrementally).
        """
        functions = writer.query("""
            MATCH (f:Function) WHERE f.stub = false AND f.summary <> ''
            RETURN f.qualified_name AS qname, f.name AS name,
                   f.summary AS summary, f.docstring AS docstring,
                   f.module_path AS module_path, f.repo AS repo,
                   f.return_type AS return_type, f.params_json AS params_json
        """)
        for row in functions:
            text = f"{row['name']}: {row['summary'] or row['docstring'] or ''}"
            self.upsert(row["qname"], text, {
                "kind": "function", "name": row["name"],
                "module": row["module_path"] or "", "repo": row["repo"] or "",
            })

        classes = writer.query("""
            MATCH (c:Class) WHERE c.stub = false
            RETURN c.qualified_name AS qname, c.name AS name,
                   c.docstring AS docstring, c.module_path AS module_path,
                   c.repo AS repo
        """)
        for row in classes:
            text = f"Class {row['name']}: {row['docstring'] or ''}"
            self.upsert(row["qname"], text, {
                "kind": "class", "name": row["name"],
                "module": row["module_path"] or "", "repo": row["repo"] or "",
            })

        log.info("Vector index: %d documents total", self._col.count())


# ---------------------------------------------------------------------------
# Graph traverser (Cypher neighbourhood expansion)
# ---------------------------------------------------------------------------

class GraphTraverser:
    """
    Given a set of seed node qualified_names, expand the neighbourhood
    via graph traversal and return a structured context dict.
    """

    def __init__(self, writer: Neo4jWriter):
        self._writer = writer

    def expand(self, seed_qnames: list[str], hops: int = 2) -> dict[str, Any]:
        """
        Return a dict with keys: nodes, call_edges, inherit_edges,
        contains_edges, import_edges.
        """
        if not seed_qnames:
            return {"nodes": [], "call_edges": [], "inherit_edges": [],
                    "contains_edges": [], "import_edges": []}

        qnames_param = list(seed_qnames)

        # Expand neighbourhood up to `hops` hops (union of all relationship types)
        nodes = self._writer.query(f"""
            MATCH (seed) WHERE seed.qualified_name IN $qnames OR seed.path IN $qnames
            CALL apoc.path.subgraphNodes(seed, {{
                maxLevel: {hops},
                relationshipFilter: 'CALLS>|INHERITS>|HAS_METHOD>|CONTAINS>|IMPORTS>'
            }}) YIELD node
            RETURN DISTINCT
                labels(node)[0]         AS kind,
                coalesce(node.qualified_name, node.path) AS qname,
                node.name               AS name,
                node.module_path        AS module_path,
                node.summary            AS summary,
                node.docstring          AS docstring,
                node.return_type        AS return_type,
                node.params_json        AS params_json,
                node.is_method          AS is_method,
                node.stub               AS stub
        """, qnames=qnames_param)

        # Fall back if APOC not available
        if not nodes:
            nodes = self._expand_no_apoc(qnames_param, hops)

        call_edges     = self._fetch_edges("CALLS",      qnames_param)
        inherit_edges  = self._fetch_edges("INHERITS",   qnames_param)
        contains_edges = self._fetch_edges("CONTAINS",   qnames_param)
        import_edges   = self._fetch_edges("IMPORTS",    qnames_param)

        return {
            "nodes":          nodes,
            "call_edges":     call_edges,
            "inherit_edges":  inherit_edges,
            "contains_edges": contains_edges,
            "import_edges":   import_edges,
        }

    def _expand_no_apoc(self, qnames: list[str], hops: int) -> list[dict]:
        """Iterative BFS expansion when APOC plugin is unavailable."""
        visited = set(qnames)
        frontier = set(qnames)
        all_nodes: list[dict] = []

        for _ in range(hops):
            if not frontier:
                break
            rows = self._writer.query("""
                MATCH (a)-[r]-(b)
                WHERE (a.qualified_name IN $q OR a.path IN $q)
                  AND b.stub <> true
                RETURN DISTINCT
                    labels(b)[0] AS kind,
                    coalesce(b.qualified_name, b.path) AS qname,
                    b.name AS name, b.module_path AS module_path,
                    b.summary AS summary, b.docstring AS docstring,
                    b.return_type AS return_type, b.params_json AS params_json,
                    b.is_method AS is_method, b.stub AS stub
            """, q=list(frontier))
            new_frontier = set()
            for row in rows:
                q = row.get("qname") or ""
                if q and q not in visited:
                    visited.add(q)
                    new_frontier.add(q)
                    all_nodes.append(row)
            frontier = new_frontier

        # Also fetch the seeds themselves
        seeds = self._writer.query("""
            MATCH (n) WHERE n.qualified_name IN $q OR n.path IN $q
            RETURN labels(n)[0] AS kind,
                   coalesce(n.qualified_name, n.path) AS qname,
                   n.name AS name, n.module_path AS module_path,
                   n.summary AS summary, n.docstring AS docstring,
                   n.return_type AS return_type, n.params_json AS params_json,
                   n.is_method AS is_method, n.stub AS stub
        """, q=qnames)
        all_nodes = seeds + all_nodes
        return all_nodes

    def _fetch_edges(self, rel_type: str, qnames: list[str]) -> list[dict]:
        cypher = f"""
            MATCH (a)-[r:{rel_type}]->(b)
            WHERE (a.qualified_name IN $q OR a.path IN $q)
               OR (b.qualified_name IN $q OR b.path IN $q)
            RETURN coalesce(a.qualified_name, a.path) AS from_q,
                   coalesce(b.qualified_name, b.path) AS to_q,
                   properties(r) AS props
            LIMIT 200
        """
        return self._writer.query(cypher, q=qnames)


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------

class ContextAssembler:
    """
    Converts raw graph data (nodes + edges dicts) into a structured
    text block that an LLM can reason over.
    """

    MAX_NODES    = 40
    MAX_EDGES    = 60
    SUMMARY_LEN  = 300

    def assemble(self, graph_data: dict, query: str) -> str:
        nodes    = graph_data.get("nodes", [])[:self.MAX_NODES]
        c_edges  = graph_data.get("call_edges",     [])[:self.MAX_EDGES]
        i_edges  = graph_data.get("inherit_edges",  [])
        im_edges = graph_data.get("import_edges",   [])

        sections = [f"# Code graph context for query: {query}\n"]

        # --- Nodes ---
        fn_nodes  = [n for n in nodes if n.get("kind") == "Function" and not n.get("stub")]
        cls_nodes = [n for n in nodes if n.get("kind") == "Class"    and not n.get("stub")]
        mod_nodes = [n for n in nodes if n.get("kind") == "Module"   and not n.get("stub")]

        if fn_nodes:
            sections.append("## Functions\n")
            for n in fn_nodes:
                summary = (n.get("summary") or n.get("docstring") or "")[:self.SUMMARY_LEN]
                params  = self._fmt_params(n.get("params_json") or "[]")
                ret     = n.get("return_type") or ""
                marker  = "async " if n.get("is_async") else ""
                sections.append(
                    f"- **{n.get('name')}**({params}) → {ret}  [{marker}{'method' if n.get('is_method') else 'function'}]\n"
                    f"  Module: `{n.get('module_path', '')}`\n"
                    f"  {summary}\n"
                )

        if cls_nodes:
            sections.append("## Classes\n")
            for n in cls_nodes:
                doc = (n.get("docstring") or "")[:200]
                sections.append(
                    f"- **{n.get('name')}**  [{n.get('module_path', '')}]\n"
                    + (f"  {doc}\n" if doc else "")
                )

        if mod_nodes:
            sections.append("## Modules\n")
            for n in mod_nodes:
                sections.append(f"- `{n.get('qname', '')}`\n")

        # --- Edges ---
        if c_edges:
            sections.append("## Call relationships\n")
            for e in c_edges[:self.MAX_EDGES]:
                props = e.get("props") or {}
                ret   = props.get("return_type", "")
                sections.append(f"- `{e['from_q']}` → CALLS → `{e['to_q']}`" +
                                 (f"  (returns: {ret})" if ret else "") + "\n")

        if i_edges:
            sections.append("## Inheritance\n")
            for e in i_edges[:20]:
                sections.append(f"- `{e['from_q']}` → INHERITS → `{e['to_q']}`\n")

        if im_edges:
            sections.append("## Module imports\n")
            for e in im_edges[:20]:
                sections.append(f"- `{e['from_q']}` → IMPORTS → `{e['to_q']}`\n")

        return "\n".join(sections)

    def _fmt_params(self, params_json: str) -> str:
        try:
            params = json.loads(params_json)
            parts = []
            for p in params:
                s = p["name"]
                if p.get("type"):
                    s += f": {p['type']}"
                if p.get("default"):
                    s += f" = {p['default']}"
                parts.append(s)
            return ", ".join(parts)
        except Exception:
            return "..."


# ---------------------------------------------------------------------------
# Main RAG orchestrator
# ---------------------------------------------------------------------------

class CodeGraphRAG:
    """
    Single entry point for the agent.

    retrieve(query) → context_string  (ready to inject into LLM prompt)

    Steps
    -----
    1. Vector search → seed qnames (semantic similarity)
    2. Graph traverse → neighbourhood expansion
    3. Context assemble → structured text
    """

    def __init__(
        self,
        writer:    Neo4jWriter,
        rag_cfg:   RAGConfig,
        seed_k:    int = 6,
        hop_depth: int = 2,
    ):
        self._writer    = writer
        self._vector    = VectorIndex(rag_cfg)
        self._traverser = GraphTraverser(writer)
        self._assembler = ContextAssembler()
        self._seed_k    = seed_k
        self._hop_depth = hop_depth

    def index(self):
        """Rebuild the vector index from current Neo4j contents. Call after graph build."""
        self._vector.index_from_neo4j(self._writer)

    def retrieve(self, query: str) -> str:
        """Return a context string for the given user query."""
        # 1. Semantic seed retrieval
        hits = self._vector.search(query, k=self._seed_k)
        seed_qnames = [h["id"] for h in hits]
        log.info("RAG seeds for %r: %s", query[:60], seed_qnames)

        if not seed_qnames:
            return "No relevant code found in the knowledge graph for this query."

        # 2. Graph expansion
        graph_data = self._traverser.expand(seed_qnames, hops=self._hop_depth)

        # 3. Assemble context
        context = self._assembler.assemble(graph_data, query)
        return context

    def retrieve_by_cypher(self, cypher: str) -> str:
        """Escape hatch: run a custom Cypher and format results as context."""
        rows = self._writer.query(cypher)
        if not rows:
            return "No results."
        lines = ["## Custom Cypher results\n"]
        for row in rows[:50]:
            lines.append(str(dict(row)))
        return "\n".join(lines)