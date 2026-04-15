#!/usr/bin/env python3
"""
scripts/run.py — CLI for CodeGraph

Commands
--------
  build   — Ingest GitLab repo into Neo4j
  index   — (Re)build ChromaDB vector index from Neo4j
  chat    — Interactive Q&A with the agent
  query   — One-shot query (non-interactive)
  stats   — Print graph statistics

Examples
--------
  # Full build from env vars
  export GITLAB_TOKEN=glpat-xxx
  export GITLAB_PROJECT=mygroup/myrepo
  export NEO4J_PASSWORD=password
  export ANTHROPIC_API_KEY=sk-ant-xxx
  python scripts/run.py build

  # Build with explicit flags
  python scripts/run.py build \
    --gitlab-url https://gitlab.com \
    --token glpat-xxx \
    --project mygroup/myrepo \
    --neo4j-password password \
    --summarizer-provider anthropic \
    --agent-provider anthropic

  # Interactive chat
  python scripts/run.py chat

  # One-shot query
  python scripts/run.py query "How does authentication work?"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from codegraph.config import (
    AppConfig, GitLabConfig, Neo4jConfig, SummarizerConfig, RAGConfig, AgentConfig
)
from codegraph.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("codegraph.cli")


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def add_gitlab_args(p):
    p.add_argument("--gitlab-url",  default=None)
    p.add_argument("--token",       default=None, help="GitLab PAT (or set GITLAB_TOKEN)")
    p.add_argument("--project",     default=None, help="group/repo (or set GITLAB_PROJECT)")
    p.add_argument("--branch",      default=None)


def add_neo4j_args(p):
    p.add_argument("--neo4j-uri",      default=None)
    p.add_argument("--neo4j-user",     default=None)
    p.add_argument("--neo4j-password", default=None)


def add_summarizer_args(p):
    p.add_argument("--summarizer-provider", default=None, choices=["anthropic","openai","ollama","noop"])
    p.add_argument("--summarizer-model",    default=None)


def add_agent_args(p):
    p.add_argument("--agent-provider", default=None, choices=["anthropic","openai"])
    p.add_argument("--agent-model",    default=None)


def apply_overrides(cfg: AppConfig, args) -> AppConfig:
    """Merge CLI flags on top of env-var config."""
    if getattr(args, "gitlab_url",  None): cfg.gitlab.url     = args.gitlab_url
    if getattr(args, "token",       None): cfg.gitlab.token   = args.token
    if getattr(args, "project",     None): cfg.gitlab.project = args.project
    if getattr(args, "branch",      None): cfg.gitlab.branch  = args.branch
    if getattr(args, "neo4j_uri",   None): cfg.neo4j.uri      = args.neo4j_uri
    if getattr(args, "neo4j_user",  None): cfg.neo4j.user     = args.neo4j_user
    if getattr(args, "neo4j_password", None): cfg.neo4j.password = args.neo4j_password
    if getattr(args, "summarizer_provider", None): cfg.summarizer.provider = args.summarizer_provider
    if getattr(args, "summarizer_model",    None): cfg.summarizer.model    = args.summarizer_model
    if getattr(args, "agent_provider", None): cfg.agent.provider = args.agent_provider
    if getattr(args, "agent_model",    None): cfg.agent.model    = args.agent_model
    return cfg


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_build(args):
    cfg = apply_overrides(AppConfig.from_env(), args)
    pipeline = Pipeline(cfg)
    try:
        pipeline.build(clear=not args.no_clear)
        if not args.skip_index:
            pipeline.index()
    finally:
        pipeline.close()


def cmd_index(args):
    cfg = apply_overrides(AppConfig.from_env(), args)
    pipeline = Pipeline(cfg)
    try:
        pipeline.index()
    finally:
        pipeline.close()


def cmd_chat(args):
    cfg = apply_overrides(AppConfig.from_env(), args)
    pipeline = Pipeline(cfg)
    agent = pipeline.get_agent()
    print("\nCodeGraph Agent ready. Type 'exit' or Ctrl-C to quit.\n")
    try:
        while True:
            try:
                user = input("You: ").strip()
            except EOFError:
                break
            if user.lower() in ("exit", "quit", "q"):
                break
            if not user:
                continue
            print("\nAgent: ", end="", flush=True)
            response = agent.chat(user)
            print(response)
            print()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close()
        print("\nGoodbye.")


def cmd_query(args):
    cfg = apply_overrides(AppConfig.from_env(), args)
    pipeline = Pipeline(cfg)
    try:
        agent = pipeline.get_agent()
        response = agent.chat(args.question)
        print(response)
    finally:
        pipeline.close()


def cmd_stats(args):
    cfg = apply_overrides(AppConfig.from_env(), args)
    from codegraph.graph.neo4j_writer import Neo4jWriter
    writer = Neo4jWriter(cfg.neo4j)
    rows = writer.query("""
        MATCH (n) RETURN labels(n)[0] AS kind, count(*) AS total
        ORDER BY total DESC
    """)
    print("\n=== Node counts ===")
    for r in rows:
        print(f"  {r['kind']:15s} {r['total']}")

    rows = writer.query("""
        MATCH ()-[r]->() RETURN type(r) AS kind, count(*) AS total ORDER BY total DESC
    """)
    print("\n=== Relationship counts ===")
    for r in rows:
        print(f"  {r['kind']:20s} {r['total']}")

    rows = writer.query("""
        MATCH (f:Function) WHERE f.stub=false AND f.summary <> ''
        RETURN count(f) AS summarized
    """)
    print(f"\n  Functions with summaries: {rows[0]['summarized'] if rows else 0}")
    writer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CodeGraph — GitLab → AST → Neo4j → LLM agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # build
    p_build = sub.add_parser("build", help="Ingest repo into Neo4j")
    add_gitlab_args(p_build); add_neo4j_args(p_build); add_summarizer_args(p_build)
    p_build.add_argument("--no-clear",    action="store_true")
    p_build.add_argument("--skip-index",  action="store_true", help="Skip vector indexing after build")

    # index
    p_index = sub.add_parser("index", help="Rebuild vector index")
    add_neo4j_args(p_index)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive agent chat")
    add_neo4j_args(p_chat); add_agent_args(p_chat)

    # query
    p_query = sub.add_parser("query", help="One-shot question")
    add_neo4j_args(p_query); add_agent_args(p_query)
    p_query.add_argument("question", help="Question to ask the agent")

    # stats
    p_stats = sub.add_parser("stats", help="Print graph statistics")
    add_neo4j_args(p_stats)

    args = parser.parse_args()
    dispatch = {
        "build": cmd_build,
        "index": cmd_index,
        "chat":  cmd_chat,
        "query": cmd_query,
        "stats": cmd_stats,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()