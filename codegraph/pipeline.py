"""
codegraph/pipeline.py

Top-level orchestrator.  Ties together all 6 components:

  1. GitLabConnector   — stream files from GitLab
  2. ParserRegistry    — AST → ModuleNode
  3. FunctionSummarizer— LLM summaries on each function
  4. GraphBuilder      — resolve cross-file deps + build edge lists
  5. Neo4jWriter       — write to graph DB
  6. CodeGraphRAG      — index for retrieval
  7. CodeAgent         — answer user questions

Usage
-----
from codegraph.pipeline import Pipeline
from codegraph.config import AppConfig

cfg      = AppConfig.from_env()
pipeline = Pipeline(cfg)

# Build / refresh the graph
pipeline.build()

# Rebuild vector index (call after build)
pipeline.index()

# Interactive chat
agent = pipeline.get_agent()
while True:
    q = input("You: ")
    print("Agent:", agent.chat(q))
"""

from __future__ import annotations

import logging
from typing import Optional

from codegraph.config import AppConfig
from codegraph.connectors.gitlab_connector import GitLabConnector
from codegraph.parsers.ast_parser import get_default_registry
from codegraph.summarizer.function_summarizer import SummarizerFactory
from codegraph.graph.graph_builder import GraphBuilder
from codegraph.graph.neo4j_writer import Neo4jWriter
from codegraph.rag.graph_rag import CodeGraphRAG
from codegraph.agent.code_agent import CodeAgent, BaseAgent

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: AppConfig):
        self._cfg     = cfg
        self._writer  = Neo4jWriter(cfg.neo4j)
        self._rag: Optional[CodeGraphRAG] = None
        self._agent: Optional[BaseAgent]  = None

    # ------------------------------------------------------------------
    # Step 1-5: Build the graph from GitLab
    # ------------------------------------------------------------------

    def build(self, clear: bool = True):
        """Full pipeline: clone → parse → summarize → write to Neo4j."""
        cfg = self._cfg

        # Schema + optional wipe
        self._writer.setup_schema()

        connector  = GitLabConnector(cfg.gitlab).connect()
        repo_name  = connector.repo_name
        if clear:
            self._writer.clear_repo(repo_name)

        parser_reg  = get_default_registry()
        summarizer  = SummarizerFactory.build(cfg.summarizer)
        builder     = GraphBuilder(
            writer=self._writer,
            summarizer=summarizer,
            summarizer_workers=cfg.workers,
        )

        total = parsed = skipped = 0

        for file_entry in connector.iter_files(
            extensions=cfg.supported_extensions,
            max_size_kb=cfg.max_file_size_kb,
        ):
            total += 1
            try:
                source = file_entry.content
            except Exception as e:
                log.warning("Failed to fetch %s: %s", file_entry.path, e)
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

        log.info("All files fetched. Flushing graph (%d parsed, %d skipped)...", parsed, skipped)
        builder.flush()
        log.info("Graph build complete.")

    # ------------------------------------------------------------------
    # Step 6: Build vector index
    # ------------------------------------------------------------------

    def index(self):
        """Populate ChromaDB vector index from current Neo4j contents."""
        rag = self._get_rag()
        rag.index()
        log.info("Vector index built.")

    # ------------------------------------------------------------------
    # Step 7: Get agent
    # ------------------------------------------------------------------

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