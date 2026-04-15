"""
codegraph/pipeline.py

Top-level orchestrator.  Wires together all components.

Connector selection (in order of precedence)
--------------------------------------------
1. Explicit connector passed to Pipeline(connector=...)
2. cfg.local_path is set  → LocalConnector
3. cfg.gitlab.token set   → GitLabConnector
4. Error

Graph backend selection
-----------------------
1. Explicit writer passed to Pipeline(writer=...)
2. cfg.graph_backend == "memory" → InMemoryWriter
3. cfg.graph_backend == "neo4j"  → Neo4jWriter (default)
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from codegraph.config import AppConfig
from codegraph.connectors.local_connector import LocalConnector, BaseConnector
from codegraph.parsers.ast_parser import get_default_registry
from codegraph.summarizer.function_summarizer import SummarizerFactory
from codegraph.graph.graph_builder import GraphBuilder
from codegraph.graph.neo4j_writer import Neo4jWriter
from codegraph.graph.memory_writer import InMemoryWriter
from codegraph.rag.graph_rag import CodeGraphRAG
from codegraph.agent.code_agent import CodeAgent, BaseAgent

log = logging.getLogger(__name__)

AnyWriter = Union[Neo4jWriter, InMemoryWriter]


def _make_connector(cfg: AppConfig) -> BaseConnector:
    if cfg.local_path:
        log.info("Using LocalConnector: %s", cfg.local_path)
        return LocalConnector(
            root_dir=cfg.local_path,
            repo_name=cfg.local_repo_name or None,
        )
    if cfg.gitlab.token and cfg.gitlab.project:
        log.info("Using GitLabConnector: %s", cfg.gitlab.project)
        from codegraph.connectors.gitlab_connector import GitLabConnector
        return GitLabConnector(cfg.gitlab).connect()
    raise ValueError(
        "No source configured. Set cfg.local_path or cfg.gitlab.token+project."
    )


def _make_writer(cfg: AppConfig) -> AnyWriter:
    if cfg.graph_backend == "memory":
        log.info("Using InMemoryWriter (no Neo4j required)")
        return InMemoryWriter()
    log.info("Using Neo4jWriter: %s", cfg.neo4j.uri)
    return Neo4jWriter(cfg.neo4j)


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        connector: Optional[BaseConnector] = None,
        writer: Optional[AnyWriter] = None,
    ):
        self._cfg       = cfg
        self._connector = connector or _make_connector(cfg)
        self._writer    = writer    or _make_writer(cfg)
        self._rag:   Optional[CodeGraphRAG] = None
        self._agent: Optional[BaseAgent]    = None

    @property
    def writer(self) -> AnyWriter:
        return self._writer

    @property
    def connector(self) -> BaseConnector:
        return self._connector

    def build(self, clear: bool = True):
        cfg       = self._cfg
        repo_name = self._connector.repo_name

        self._writer.setup_schema()
        if clear:
            self._writer.clear_repo(repo_name)

        parser_reg = get_default_registry()
        summarizer = SummarizerFactory.build(cfg.summarizer)
        builder    = GraphBuilder(
            writer=self._writer,
            summarizer=summarizer,
            summarizer_workers=cfg.workers,
        )

        total = parsed = skipped = 0

        for file_entry in self._connector.iter_files(
            extensions=cfg.supported_extensions,
            max_size_kb=cfg.max_file_size_kb,
        ):
            total += 1
            try:
                source = file_entry.content
            except Exception as e:
                log.warning("Failed to read %s: %s", file_entry.path, e)
                skipped += 1
                continue

            module = parser_reg.parse(source, file_entry.path, repo_name)
            if module is None:
                skipped += 1
                continue

            builder.add_module(module)
            parsed += 1

            if parsed % 20 == 0:
                log.info("Progress: %d / %d files processed", parsed, total)

        log.info(
            "All files read. Flushing graph (%d parsed, %d skipped / %d total)...",
            parsed, skipped, total,
        )
        builder.flush()
        log.info("Graph build complete.")

    def index(self):
        rag = self._get_rag()
        rag.index()
        log.info("Vector index built.")

    def get_agent(self) -> BaseAgent:
        if self._agent is None:
            self._agent = CodeAgent.from_config(
                cfg=self._cfg,
                writer=self._writer,
                rag=self._get_rag(),
            )
        return self._agent

    def _get_rag(self) -> CodeGraphRAG:
        if self._rag is None:
            self._rag = CodeGraphRAG(
                writer=self._writer,
                rag_cfg=self._cfg.rag,
            )
        return self._rag

    def close(self):
        self._writer.close()

    def print_graph_summary(self):
        if isinstance(self._writer, InMemoryWriter):
            self._writer.print_summary()
        else:
            rows = self._writer.query(
                "MATCH (n) RETURN labels(n)[0] AS kind, count(*) AS n ORDER BY n DESC"
            )
            print("\n=== Graph node counts ===")
            for r in rows:
                print(f"  {r['kind']:15s} {r['n']}")