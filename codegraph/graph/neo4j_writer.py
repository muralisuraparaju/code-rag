"""
codegraph/graph/neo4j_writer.py

All Cypher lives here.  GraphBuilder calls write_* methods;
this class handles batching and transactions.

Schema summary
--------------
Nodes
  (:Module   {path, repo, language, docstring, stub})
  (:Class    {qualified_name, name, module_path, repo,
              start_line, end_line, docstring, decorators, stub})
  (:Function {qualified_name, name, module_path, repo,
              start_line, end_line, is_method, is_async,
              params_json, return_type, decorators,
              docstring, summary, stub})

Relationships (all carry `since` timestamp + any extra props)
  (Module)-[:CONTAINS]->(Class|Function)
  (Class)-[:HAS_METHOD]->(Function)
  (Class)-[:INHERITS]->(Class)          props: {}
  (Function)-[:CALLS]->(Function)       props: {args_repr, return_type}
  (Module)-[:IMPORTS]->(Module)         props: {names}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase, Session

from codegraph.config import Neo4jConfig
from codegraph.models import ClassNode, EdgeKind, FunctionNode, Language, ModuleNode, Param

log = logging.getLogger(__name__)

_NOW = lambda: datetime.now(timezone.utc).isoformat()


def _params_to_json(params: list[Param]) -> str:
    return json.dumps([
        {"name": p.name, "type": p.type_hint, "default": p.default}
        for p in params
    ])


# ---------------------------------------------------------------------------
# Cypher statements
# ---------------------------------------------------------------------------

_CONSTRAINTS = [
    "CREATE CONSTRAINT cg_module_unique   IF NOT EXISTS FOR (n:Module)   REQUIRE (n.path, n.repo)                    IS UNIQUE",
    "CREATE CONSTRAINT cg_class_unique    IF NOT EXISTS FOR (n:Class)    REQUIRE (n.qualified_name)                  IS UNIQUE",
    "CREATE CONSTRAINT cg_function_unique IF NOT EXISTS FOR (n:Function) REQUIRE (n.qualified_name)                  IS UNIQUE",
]
_INDEXES = [
    "CREATE INDEX cg_module_repo   IF NOT EXISTS FOR (n:Module)   ON (n.repo)",
    "CREATE INDEX cg_class_repo    IF NOT EXISTS FOR (n:Class)    ON (n.repo)",
    "CREATE INDEX cg_func_repo     IF NOT EXISTS FOR (n:Function) ON (n.repo)",
    "CREATE INDEX cg_func_name     IF NOT EXISTS FOR (n:Function) ON (n.name)",
    "CREATE INDEX cg_func_summary  IF NOT EXISTS FOR (n:Function) ON (n.summary)",
]

_UPSERT_MODULE = """
MERGE (m:Module {path: $path, repo: $repo})
SET   m.language  = $language,
      m.docstring = $docstring,
      m.stub      = false,
      m.updated   = $now
"""

_UPSERT_CLASS = """
MERGE (c:Class {qualified_name: $qualified_name})
ON CREATE SET
      c.name        = $name,
      c.module_path = $module_path,
      c.repo        = $repo,
      c.start_line  = $start_line,
      c.end_line    = $end_line,
      c.docstring   = $docstring,
      c.decorators  = $decorators,
      c.stub        = false,
      c.created     = $now
ON MATCH SET
      c.name        = $name,
      c.start_line  = $start_line,
      c.end_line    = $end_line,
      c.docstring   = $docstring,
      c.decorators  = $decorators,
      c.stub        = false,
      c.updated     = $now
"""

_UPSERT_FUNCTION = """
MERGE (f:Function {qualified_name: $qualified_name})
ON CREATE SET
      f.name        = $name,
      f.module_path = $module_path,
      f.repo        = $repo,
      f.start_line  = $start_line,
      f.end_line    = $end_line,
      f.is_method   = $is_method,
      f.is_async    = $is_async,
      f.params_json = $params_json,
      f.return_type = $return_type,
      f.decorators  = $decorators,
      f.docstring   = $docstring,
      f.summary     = $summary,
      f.stub        = false,
      f.created     = $now
ON MATCH SET
      f.name        = $name,
      f.start_line  = $start_line,
      f.end_line    = $end_line,
      f.is_method   = $is_method,
      f.is_async    = $is_async,
      f.params_json = $params_json,
      f.return_type = $return_type,
      f.decorators  = $decorators,
      f.docstring   = $docstring,
      f.summary     = $summary,
      f.stub        = false,
      f.updated     = $now
"""

_LINK_MODULE_CONTAINS = """
MATCH (m:Module {path: $module_path, repo: $repo})
MATCH (n {qualified_name: $child_qname})
MERGE (m)-[:CONTAINS]->(n)
"""

_LINK_CLASS_METHOD = """
MATCH (c:Class    {qualified_name: $class_qname})
MATCH (f:Function {qualified_name: $fn_qname})
MERGE (c)-[:HAS_METHOD]->(f)
"""

_UPSERT_EDGE = """
MATCH (a {qualified_name: $from_qname})
MATCH (b {qualified_name: $to_qname})
MERGE (a)-[r:{rel_type}]->(b)
SET   r += $props,
      r.updated = $now
"""

_STUB_MODULE = """
MERGE (m:Module {path: $path, repo: '__external__'})
ON CREATE SET m.stub = true, m.language = 'unknown', m.created = $now
"""

_STUB_CLASS = """
MERGE (c:Class {qualified_name: $qname})
ON CREATE SET c.stub = true, c.name = $name, c.module_path = '__stub__',
              c.repo = '__stub__', c.start_line = 0, c.end_line = 0, c.created = $now
"""

_STUB_FUNCTION = """
MERGE (f:Function {qualified_name: $qname})
ON CREATE SET f.stub = true, f.name = $name, f.module_path = '__external__',
              f.repo = '__external__', f.is_method = false, f.is_async = false,
              f.start_line = 0, f.end_line = 0, f.created = $now
"""

_CLEAR_REPO = "MATCH (n {repo: $repo}) DETACH DELETE n"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class Neo4jWriter:
    """
    Buffered writer.  Accumulates Cypher work in a single transaction per
    flush batch.  Call begin() before a batch and commit() at the end.
    """

    def __init__(self, cfg: Neo4jConfig):
        self._driver = GraphDatabase.driver(
            cfg.uri, auth=(cfg.user, cfg.password), database=cfg.database
        )
        self._session: Session | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_schema(self):
        with self._driver.session() as s:
            for stmt in _CONSTRAINTS + _INDEXES:
                try:
                    s.run(stmt)
                except Exception as e:
                    log.debug("Schema: %s — %s", stmt[:60], e)
        log.info("Neo4j schema ready.")

    def clear_repo(self, repo: str):
        with self._driver.session() as s:
            s.run(_CLEAR_REPO, repo=repo)
        log.info("Cleared repo nodes: %s", repo)

    def begin(self):
        self._session = self._driver.session()

    def commit(self):
        if self._session:
            self._session.close()
            self._session = None

    def close(self):
        self.commit()
        self._driver.close()

    # ------------------------------------------------------------------
    # Node writes
    # ------------------------------------------------------------------

    def write_module(self, mod: ModuleNode):
        s = self._session or self._driver.session()
        s.run(_UPSERT_MODULE, path=mod.path, repo=mod.repo,
              language=mod.language.value, docstring=mod.docstring or "",
              now=_NOW())
        for cls in mod.classes:
            self.write_class(cls)
            s.run(_LINK_MODULE_CONTAINS, module_path=mod.path, repo=mod.repo,
                  child_qname=cls.qualified_name)
            for fn in cls.methods:
                self.write_function(fn)
                s.run(_LINK_CLASS_METHOD, class_qname=cls.qualified_name,
                      fn_qname=fn.qualified_name)
        for fn in mod.functions:
            self.write_function(fn)
            s.run(_LINK_MODULE_CONTAINS, module_path=mod.path, repo=mod.repo,
                  child_qname=fn.qualified_name)

    def write_class(self, cls: ClassNode):
        s = self._session or self._driver.session()
        s.run(_UPSERT_CLASS,
              qualified_name=cls.qualified_name,
              name=cls.name,
              module_path=cls.module_path,
              repo=cls.repo,
              start_line=cls.start_line,
              end_line=cls.end_line,
              docstring=cls.docstring or "",
              decorators=json.dumps(cls.decorators),
              now=_NOW())

    def write_function(self, fn: FunctionNode):
        s = self._session or self._driver.session()
        s.run(_UPSERT_FUNCTION,
              qualified_name=fn.qualified_name,
              name=fn.name,
              module_path=fn.module_path,
              repo=fn.repo,
              start_line=fn.start_line,
              end_line=fn.end_line,
              is_method=fn.is_method,
              is_async=fn.is_async,
              params_json=_params_to_json(fn.params),
              return_type=fn.return_type or "",
              decorators=json.dumps(fn.decorators),
              docstring=fn.docstring or "",
              summary=fn.summary or "",
              now=_NOW())

    # ------------------------------------------------------------------
    # Stub writes
    # ------------------------------------------------------------------

    def write_module_stub(self, path: str):
        s = self._session or self._driver.session()
        s.run(_STUB_MODULE, path=path, now=_NOW())

    def write_class_stub(self, qname: str):
        s = self._session or self._driver.session()
        name = qname.split("::")[-1]
        s.run(_STUB_CLASS, qname=qname, name=name, now=_NOW())

    def write_function_stub(self, qname: str):
        s = self._session or self._driver.session()
        name = qname.split("::")[-1]
        s.run(_STUB_FUNCTION, qname=qname, name=name, now=_NOW())

    # ------------------------------------------------------------------
    # Edge write
    # ------------------------------------------------------------------

    def write_edge(self, kind: EdgeKind, from_qname: str, to_qname: str, props: dict[str, Any]):
        s = self._session or self._driver.session()
        # Build Cypher with literal rel type (cannot parameterise rel type in Neo4j)
        cypher = f"""
            MATCH (a) WHERE a.qualified_name = $from_q OR a.path = $from_q
            MATCH (b) WHERE b.qualified_name = $to_q   OR b.path = $to_q
            MERGE (a)-[r:{kind.value}]->(b)
            SET r += $props, r.updated = $now
        """
        try:
            s.run(cypher, from_q=from_qname, to_q=to_qname,
                  props=props, now=_NOW())
        except Exception as e:
            log.debug("Edge write failed %s→%s: %s", from_qname, to_qname, e)

    # ------------------------------------------------------------------
    # Query helpers (used by RAG layer)
    # ------------------------------------------------------------------

    def query(self, cypher: str, **params) -> list[dict]:
        with self._driver.session() as s:
            return [dict(r) for r in s.run(cypher, **params)]