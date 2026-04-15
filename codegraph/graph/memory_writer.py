"""
codegraph/graph/memory_writer.py

In-memory graph store with the exact same interface as Neo4jWriter.
Nothing is persisted to disk.  Use it for:
  - Rapid local testing (no Neo4j needed)
  - Unit tests
  - Quick one-off exploration

The graph is stored as two plain dicts:
  _nodes : {qualified_name/path → node_dict}
  _edges : list of {from, to, kind, props}

A minimal Cypher-like query method is provided so the RAG and agent layers
work unchanged — it handles the subset of Cypher patterns actually issued
by GraphTraverser and ToolExecutor.  Complex APOC calls are silently
skipped and return empty lists (the code already has a no-APOC fallback).

print_summary() gives a human-readable snapshot for debugging.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from codegraph.models import ClassNode, EdgeKind, FunctionNode, Language, ModuleNode, Param

log = logging.getLogger(__name__)

_NOW = lambda: datetime.now(timezone.utc).isoformat()


def _params_to_json(params: list[Param]) -> str:
    return json.dumps([
        {"name": p.name, "type": p.type_hint, "default": p.default}
        for p in params
    ])


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

class InMemoryWriter:
    """
    Drop-in replacement for Neo4jWriter.

    Every write_* method stores data in plain Python dicts.
    The `.query()` method implements a tiny subset of Cypher so that
    GraphTraverser, ToolExecutor, and the stats command work without Neo4j.
    """

    def __init__(self):
        # primary store: qname/path → property dict
        self._nodes: dict[str, dict] = {}
        # adjacency: from_key → list of {to, kind, props}
        self._adj:   dict[str, list] = defaultdict(list)
        # flat edge list for reverse lookups
        self._edges: list[dict]      = []
        self._open   = False

    # ------------------------------------------------------------------
    # Lifecycle (mirrors Neo4jWriter)
    # ------------------------------------------------------------------

    def setup_schema(self):
        log.info("[InMemory] setup_schema — no-op")

    def clear_repo(self, repo: str):
        keys = [k for k, v in self._nodes.items() if v.get("repo") == repo]
        for k in keys:
            self._nodes.pop(k, None)
            self._adj.pop(k, None)
        self._edges = [
            e for e in self._edges
            if self._nodes.get(e["from"], {}).get("repo") != repo
            and self._nodes.get(e["to"],   {}).get("repo") != repo
        ]
        log.info("[InMemory] cleared %d nodes for repo=%s", len(keys), repo)

    def begin(self):
        self._open = True

    def commit(self):
        self._open = False

    def close(self):
        self.commit()

    # ------------------------------------------------------------------
    # Node writes
    # ------------------------------------------------------------------

    def _upsert(self, key: str, props: dict):
        existing = self._nodes.get(key, {})
        existing.update(props)
        existing["_key"] = key
        self._nodes[key] = existing

    def write_module(self, mod: ModuleNode):
        self._upsert(mod.path, {
            "kind": "Module", "path": mod.path, "repo": mod.repo,
            "language": mod.language.value, "docstring": mod.docstring or "",
            "stub": False, "qualified_name": mod.path,
        })
        for cls in mod.classes:
            self.write_class(cls)
            self.write_edge(EdgeKind.CONTAINS, mod.path, cls.qualified_name, {})
            for fn in cls.methods:
                self.write_function(fn)
                self.write_edge(EdgeKind.HAS_METHOD, cls.qualified_name, fn.qualified_name, {})
        for fn in mod.functions:
            self.write_function(fn)
            self.write_edge(EdgeKind.CONTAINS, mod.path, fn.qualified_name, {})

    def write_class(self, cls: ClassNode):
        self._upsert(cls.qualified_name, {
            "kind": "Class", "qualified_name": cls.qualified_name,
            "name": cls.name, "module_path": cls.module_path, "repo": cls.repo,
            "start_line": cls.start_line, "end_line": cls.end_line,
            "docstring": cls.docstring or "",
            "decorators": json.dumps(cls.decorators), "stub": False,
        })

    def write_function(self, fn: FunctionNode):
        self._upsert(fn.qualified_name, {
            "kind": "Function", "qualified_name": fn.qualified_name,
            "name": fn.name, "module_path": fn.module_path, "repo": fn.repo,
            "start_line": fn.start_line, "end_line": fn.end_line,
            "is_method": fn.is_method, "is_async": fn.is_async,
            "params_json": _params_to_json(fn.params),
            "return_type": fn.return_type or "",
            "decorators": json.dumps(fn.decorators),
            "docstring": fn.docstring or "",
            "summary": fn.summary or "", "stub": False,
        })

    # ------------------------------------------------------------------
    # Stub writes
    # ------------------------------------------------------------------

    def write_module_stub(self, path: str):
        if path not in self._nodes:
            self._upsert(path, {
                "kind": "Module", "path": path, "repo": "__external__",
                "language": "unknown", "stub": True, "qualified_name": path,
            })

    def write_class_stub(self, qname: str):
        if qname not in self._nodes:
            self._upsert(qname, {
                "kind": "Class", "qualified_name": qname,
                "name": qname.split("::")[-1], "module_path": "__stub__",
                "repo": "__stub__", "start_line": 0, "end_line": 0, "stub": True,
            })

    def write_function_stub(self, qname: str):
        if qname not in self._nodes:
            self._upsert(qname, {
                "kind": "Function", "qualified_name": qname,
                "name": qname.split("::")[-1], "module_path": "__external__",
                "repo": "__external__", "is_method": False, "is_async": False,
                "start_line": 0, "end_line": 0, "summary": "", "stub": True,
            })

    # ------------------------------------------------------------------
    # Edge write
    # ------------------------------------------------------------------

    def write_edge(self, kind: EdgeKind, from_q: str, to_q: str, props: dict[str, Any]):
        edge = {"from": from_q, "to": to_q, "kind": kind.value, "props": props}
        # Deduplicate
        sig = (from_q, to_q, kind.value)
        for e in self._adj[from_q]:
            if (e["from"], e["to"], e["kind"]) == sig:
                e["props"].update(props)
                return
        self._adj[from_q].append(edge)
        self._edges.append(edge)

    # ------------------------------------------------------------------
    # Query (minimal Cypher interpreter)
    # ------------------------------------------------------------------

    def query(self, cypher: str, **params) -> list[dict]:
        """
        Handles the Cypher patterns used by GraphTraverser + ToolExecutor.
        Unsupported patterns (APOC calls) return [].
        """
        cypher_stripped = cypher.strip()

        # Skip APOC calls — caller has a no-APOC fallback
        if "apoc." in cypher_stripped:
            return []

        # Route to handler
        upper = cypher_stripped.upper()
        try:
            if "MATCH (f:FUNCTION)" in upper:
                return self._q_function(cypher_stripped, params)
            if "MATCH (C:CLASS)" in upper:
                return self._q_class(cypher_stripped, params)
            if "MATCH (M:MODULE)" in upper:
                return self._q_module(cypher_stripped, params)
            if re.search(r"MATCH\s+\(N\)", upper) or "LABELS(N)" in upper:
                return self._q_stats(cypher_stripped, params)
            if "MATCH ()-[R]->()" in upper or "MATCH ()-[R]->" in upper:
                return self._q_rel_stats(cypher_stripped, params)
            if re.search(r"MATCH\s+\(A\)-\[R:", upper):
                return self._q_edges(cypher_stripped, params)
            if re.search(r"MATCH\s+\(A\)", upper) or re.search(r"MATCH\s+\(SEED\)", upper):
                return self._q_neighbourhood(cypher_stripped, params)
        except Exception as e:
            log.debug("InMemory query error: %s\nCypher: %s", e, cypher_stripped[:200])
        return []

    # ---- query helpers ----

    def _all_by_kind(self, kind: str) -> list[dict]:
        return [n for n in self._nodes.values() if n.get("kind") == kind]

    def _match_name(self, node: dict, name_fragment: str) -> bool:
        return (
            name_fragment in (node.get("name") or "")
            or name_fragment in (node.get("qualified_name") or "")
        )

    def _q_function(self, cypher: str, params: dict) -> list[dict]:
        name = params.get("name", "")
        rows = []
        for fn in self._all_by_kind("Function"):
            if fn.get("stub"):
                continue
            if name and not self._match_name(fn, name):
                continue
            callee_names = [
                self._nodes[e["to"]].get("name", "")
                for e in self._edges
                if e["from"] == fn["qualified_name"] and e["kind"] == "CALLS"
                and e["to"] in self._nodes
            ]
            caller_names = [
                self._nodes[e["from"]].get("name", "")
                for e in self._edges
                if e["to"] == fn["qualified_name"] and e["kind"] == "CALLS"
                and e["from"] in self._nodes
            ]
            rows.append({
                "qname":       fn.get("qualified_name"),
                "name":        fn.get("name"),
                "module":      fn.get("module_path"),
                "return_type": fn.get("return_type"),
                "params":      fn.get("params_json"),
                "summary":     fn.get("summary"),
                "docstring":   fn.get("docstring"),
                "is_method":   fn.get("is_method"),
                "is_async":    fn.get("is_async"),
                "calls":       callee_names,
                "called_by":   caller_names,
            })
        return rows[:20]

    def _q_class(self, cypher: str, params: dict) -> list[dict]:
        name = params.get("name", "")
        rows = []
        for cls in self._all_by_kind("Class"):
            if cls.get("stub"):
                continue
            if name and not self._match_name(cls, name):
                continue
            methods  = [self._nodes[e["to"]].get("name","") for e in self._edges
                        if e["from"] == cls["qualified_name"] and e["kind"] == "HAS_METHOD"]
            parents  = [self._nodes[e["to"]].get("name","")  for e in self._edges
                        if e["from"] == cls["qualified_name"] and e["kind"] == "INHERITS"]
            children = [self._nodes[e["from"]].get("name","") for e in self._edges
                        if e["to"] == cls["qualified_name"]   and e["kind"] == "INHERITS"]
            rows.append({
                "qname":    cls.get("qualified_name"),
                "name":     cls.get("name"),
                "module":   cls.get("module_path"),
                "docstring":cls.get("docstring"),
                "methods":  methods,
                "parents":  parents,
                "children": children,
            })
        return rows[:10]

    def _q_module(self, cypher: str, params: dict) -> list[dict]:
        repo = params.get("repo")
        rows = []
        for mod in self._all_by_kind("Module"):
            if mod.get("stub"):
                continue
            if repo and mod.get("repo") != repo:
                continue
            if "summarized" in cypher.lower():
                # stats query
                count = sum(
                    1 for n in self._all_by_kind("Function")
                    if not n.get("stub") and n.get("summary")
                )
                return [{"summarized": count}]
            rows.append({
                "path":     mod.get("path"),
                "repo":     mod.get("repo"),
                "language": mod.get("language"),
                "kind":     "modules",
                "n":        1,
            })
        return rows[:50]

    def _q_stats(self, cypher: str, params: dict) -> list[dict]:
        counts: dict[str, int] = defaultdict(int)
        for n in self._nodes.values():
            counts[n.get("kind", "Unknown")] += 1
        return [{"kind": k, "n": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]

    def _q_rel_stats(self, cypher: str, params: dict) -> list[dict]:
        counts: dict[str, int] = defaultdict(int)
        for e in self._edges:
            counts[e["kind"]] += 1
        return [{"kind": k, "total": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]

    def _q_edges(self, cypher: str, params: dict) -> list[dict]:
        """Generic edge query for GraphTraverser._fetch_edges."""
        qnames = params.get("q", params.get("qnames", []))
        if isinstance(qnames, str):
            qnames = [qnames]
        qnames_set = set(qnames)

        # Extract relationship type from Cypher e.g. MATCH (a)-[r:CALLS]->(b)
        m = re.search(r"\[r:(\w+)\]", cypher, re.IGNORECASE)
        rel_filter = m.group(1).upper() if m else None

        rows = []
        for e in self._edges:
            if rel_filter and e["kind"] != rel_filter:
                continue
            if e["from"] in qnames_set or e["to"] in qnames_set:
                rows.append({
                    "from_q": e["from"],
                    "to_q":   e["to"],
                    "props":  e["props"],
                })
        return rows[:200]

    def _q_neighbourhood(self, cypher: str, params: dict) -> list[dict]:
        """BFS expansion for GraphTraverser._expand_no_apoc."""
        qnames = params.get("q", params.get("qnames", []))
        if isinstance(qnames, str):
            qnames = [qnames]

        visited = set(qnames)
        result  = []

        for q in qnames:
            node = self._nodes.get(q)
            if node:
                result.append(self._node_to_row(node))

        # one hop of neighbours
        for e in self._edges:
            if e["from"] in visited and e["to"] not in visited:
                nb = self._nodes.get(e["to"])
                if nb and not nb.get("stub"):
                    visited.add(e["to"])
                    result.append(self._node_to_row(nb))

        return result

    def _node_to_row(self, n: dict) -> dict:
        return {
            "kind":        n.get("kind"),
            "qname":       n.get("qualified_name") or n.get("path"),
            "name":        n.get("name"),
            "module_path": n.get("module_path"),
            "summary":     n.get("summary"),
            "docstring":   n.get("docstring"),
            "return_type": n.get("return_type"),
            "params_json": n.get("params_json"),
            "is_method":   n.get("is_method"),
            "stub":        n.get("stub"),
        }

    # ------------------------------------------------------------------
    # Inspection / debugging helpers
    # ------------------------------------------------------------------

    def print_summary(self, max_nodes: int = 30):
        """Pretty-print the current graph state for debugging."""
        from collections import Counter
        kind_counts = Counter(n.get("kind") for n in self._nodes.values())
        edge_counts = Counter(e["kind"] for e in self._edges)

        print("\n" + "=" * 60)
        print("  IN-MEMORY GRAPH SUMMARY")
        print("=" * 60)
        print(f"  Nodes : {len(self._nodes)}")
        for k, v in kind_counts.most_common():
            print(f"    {k:15s} {v}")
        print(f"  Edges : {len(self._edges)}")
        for k, v in edge_counts.most_common():
            print(f"    {k:20s} {v}")
        print()

        non_stub = [n for n in self._nodes.values() if not n.get("stub")]
        print(f"  Non-stub nodes ({min(len(non_stub), max_nodes)} of {len(non_stub)}):")
        for n in sorted(non_stub, key=lambda x: x.get("kind", ""))[:max_nodes]:
            kind   = n.get("kind", "?")
            name   = n.get("name") or n.get("path", "?")
            module = n.get("module_path", "")
            summ   = (n.get("summary") or "")[:60]
            print(f"    [{kind:8s}] {name:30s} {module}")
            if summ:
                print(f"             → {summ}")
        print("=" * 60 + "\n")

    def to_dot(self) -> str:
        """Export as Graphviz DOT for visual inspection."""
        lines = ["digraph codegraph {", "  rankdir=LR;", '  node [shape=box fontsize=10];']
        id_map: dict[str, str] = {}

        def safe_id(k: str) -> str:
            if k not in id_map:
                id_map[k] = f"n{len(id_map)}"
            return id_map[k]

        for k, n in self._nodes.items():
            if n.get("stub"):
                continue
            label = (n.get("name") or k).replace('"', "'")[:40]
            kind  = n.get("kind", "?")
            color = {"Module": "lightblue", "Class": "lightyellow",
                     "Function": "lightgreen"}.get(kind, "white")
            lines.append(f'  {safe_id(k)} [label="{label}\\n{kind}" fillcolor="{color}" style=filled];')

        for e in self._edges:
            if e["from"] in id_map and e["to"] in id_map:
                lines.append(f'  {safe_id(e["from"])} -> {safe_id(e["to"])} [label="{e["kind"]}"];')

        lines.append("}")
        return "\n".join(lines)