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
  export GITLAB_TOKEN=glp#!/usr/bin/env python3
"""
scripts/run.py — CodeGraph CLI

COMMANDS
--------
  test-connection       Verify GitLab credentials and list repo files
  test-parse            Parse a local directory, print AST results (no DB)
  build                 Full pipeline: read → parse → summarize → write graph
  index                 (Re)build ChromaDB vector index from graph
  chat                  Interactive agent Q&A
  query                 One-shot question to the agent
  stats                 Print node/edge counts from the graph
  export-dot            Export in-memory graph to Graphviz DOT file

SOURCE FLAGS  (pick one)
  --local-path PATH     Use a local cloned directory instead of GitLab
  --repo-name NAME      Logical name for local repo (default: dir basename)

  --gitlab-url URL      GitLab server (default: https://gitlab.com)
  --token TOKEN         GitLab personal access token
  --project GROUP/REPO  e.g. mygroup/myrepo
  --branch BRANCH       Branch to read (default: main)

GRAPH BACKEND FLAGS
  --backend neo4j       Write to Neo4j (default)
  --backend memory      In-memory graph, no DB required

NEO4J FLAGS  (only needed with --backend neo4j)
  --neo4j-uri URI
  --neo4j-user USER
  --neo4j-password PASS

SUMMARIZER FLAGS
  --summarizer noop         No LLM summaries (fast, for testing)
  --summarizer anthropic    Claude  (needs ANTHROPIC_API_KEY)
  --summarizer openai       GPT-4   (needs OPENAI_API_KEY)
  --summarizer ollama       Local Ollama
  --summarizer-model MODEL  Override default model

AGENT FLAGS
  --agent-provider anthropic|openai
  --agent-model MODEL

QUICK-START EXAMPLES
--------------------
# 1. Check GitLab credentials
python scripts/run.py test-connection --token glpat-xxx --project mygroup/myrepo

# 2. Parse a local directory, print results, no DB needed
python scripts/run.py test-parse --local-path /path/to/repo --backend memory

# 3. Full local build into in-memory graph
python scripts/run.py build --local-path /path/to/repo --backend memory --summarizer noop

# 4. Full local build into Neo4j with real summaries
python scripts/run.py build --local-path /path/to/repo --backend neo4j --neo4j-password pw

# 5. Interactive chat on top of in-memory graph
python scripts/run.py chat --local-path /path/to/repo --backend memory --summarizer noop

# 6. Export DOT graph after in-memory build
python scripts/run.py export-dot --local-path /path/to/repo --output graph.dot
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from codegraph.config import (
    AppConfig, GitLabConfig, Neo4jConfig,
    SummarizerConfig, RAGConfig, AgentConfig,
)
from codegraph.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("codegraph.cli")


# ─────────────────────────────────────────────
#  Shared argument groups
# ─────────────────────────────────────────────

def _add_source(p: argparse.ArgumentParser):
    src = p.add_mutually_exclusive_group()
    src.add_argument("--local-path", metavar="PATH",
                     help="Local repo directory (skip GitLab)")
    p.add_argument("--repo-name", metavar="NAME", default="",
                   help="Logical repo name for --local-path (default: dir basename)")
    p.add_argument("--gitlab-url",  default=None, metavar="URL")
    p.add_argument("--token",       default=None, metavar="TOKEN",
                   help="GitLab PAT (or GITLAB_TOKEN env var)")
    p.add_argument("--project",     default=None, metavar="GROUP/REPO")
    p.add_argument("--branch",      default=None, metavar="BRANCH")


def _add_backend(p: argparse.ArgumentParser):
    p.add_argument("--backend", choices=["neo4j", "memory"], default=None,
                   help="Graph backend (default: neo4j, or memory for testing)")


def _add_neo4j(p: argparse.ArgumentParser):
    p.add_argument("--neo4j-uri",      default=None)
    p.add_argument("--neo4j-user",     default=None)
    p.add_argument("--neo4j-password", default=None)


def _add_summarizer(p: argparse.ArgumentParser):
    p.add_argument("--summarizer",
                   choices=["noop", "anthropic", "openai", "ollama"],
                   default=None,
                   help="LLM summarizer (default: noop for fast testing)")
    p.add_argument("--summarizer-model", default=None)


def _add_agent(p: argparse.ArgumentParser):
    p.add_argument("--agent-provider",
                   choices=["anthropic", "openai"], default=None)
    p.add_argument("--agent-model", default=None)


def _add_extensions(p: argparse.ArgumentParser):
    p.add_argument("--ext", default=None, metavar=".py,.js",
                   help="Comma-separated extensions to parse")


# ─────────────────────────────────────────────
#  Config assembly
# ─────────────────────────────────────────────

def _build_config(args) -> AppConfig:
    cfg = AppConfig.from_env()

    # Source
    if getattr(args, "local_path", None):
        cfg.local_path       = args.local_path
        cfg.local_repo_name  = getattr(args, "repo_name", "") or ""
    if getattr(args, "gitlab_url",  None): cfg.gitlab.url     = args.gitlab_url
    if getattr(args, "token",       None): cfg.gitlab.token   = args.token
    if getattr(args, "project",     None): cfg.gitlab.project = args.project
    if getattr(args, "branch",      None): cfg.gitlab.branch  = args.branch

    # Backend
    if getattr(args, "backend", None):
        cfg.graph_backend = args.backend

    # Neo4j
    if getattr(args, "neo4j_uri",      None): cfg.neo4j.uri      = args.neo4j_uri
    if getattr(args, "neo4j_user",     None): cfg.neo4j.user     = args.neo4j_user
    if getattr(args, "neo4j_password", None): cfg.neo4j.password = args.neo4j_password

    # Summarizer
    if getattr(args, "summarizer",       None): cfg.summarizer.provider = args.summarizer
    if getattr(args, "summarizer_model", None): cfg.summarizer.model    = args.summarizer_model

    # Agent
    if getattr(args, "agent_provider", None): cfg.agent.provider = args.agent_provider
    if getattr(args, "agent_model",    None): cfg.agent.model    = args.agent_model

    # Extensions
    if getattr(args, "ext", None):
        cfg.supported_extensions = tuple(args.ext.split(","))

    return cfg


# ─────────────────────────────────────────────
#  Command: test-connection
# ─────────────────────────────────────────────

def cmd_test_connection(args):
    """Verify GitLab credentials and list the first 20 source files."""
    cfg = _build_config(args)
    print("\n── GitLab connection test ──────────────────────────────")
    print(f"  URL     : {cfg.gitlab.url}")
    print(f"  Project : {cfg.gitlab.project}")
    print(f"  Branch  : {cfg.gitlab.branch}")
    print(f"  Token   : {'[set]' if cfg.gitlab.token else '[NOT SET]'}")
    print()

    if not cfg.gitlab.token:
        print("✗  No token provided. Use --token or set GITLAB_TOKEN.")
        sys.exit(1)
    if not cfg.gitlab.project:
        print("✗  No project provided. Use --project or set GITLAB_PROJECT.")
        sys.exit(1)

    try:
        import gitlab
        gl = gitlab.Gitlab(cfg.gitlab.url, private_token=cfg.gitlab.token)
        gl.auth()
        user = gl.auth()
        print(f"✓  Authenticated as: {gl.users.get(gl.auth()).username if hasattr(gl, 'auth') else 'OK'}")
    except Exception as e:
        print(f"✗  Authentication failed: {e}")
        sys.exit(1)

    try:
        from codegraph.connectors.gitlab_connector import GitLabConnector
        connector = GitLabConnector(cfg.gitlab).connect()
        print(f"✓  Connected to project: {connector.project.path_with_namespace}")
        print(f"   Default branch: {connector.project.default_branch}")
        print(f"   Repo name (graph tag): {connector.repo_name}")
        print()
        print("  First 20 matching source files:")
        count = 0
        for fe in connector.iter_files(
            extensions=cfg.supported_extensions,
            max_size_kb=cfg.max_file_size_kb,
        ):
            print(f"    {fe.path}  ({fe.size:,} bytes)")
            count += 1
            if count >= 20:
                print("    … (truncated)")
                break
        if count == 0:
            print("    (no matching files found — check --ext flag)")
        print(f"\n✓  Connection test passed.")
    except Exception as e:
        print(f"✗  Project access failed: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────
#  Command: test-parse
# ─────────────────────────────────────────────

def cmd_test_parse(args):
    """
    Parse files from a local directory (or GitLab), build an in-memory graph,
    and print a human-readable summary.  No DB required.
    """
    cfg = _build_config(args)
    # Force in-memory and noop summarizer for quick feedback
    cfg.graph_backend      = "memory"
    cfg.summarizer.provider = "noop"

    max_files = getattr(args, "max_files", 0) or 0

    print("\n── Parse test ──────────────────────────────────────────")
    if cfg.local_path:
        print(f"  Source : local  {cfg.local_path}")
    else:
        print(f"  Source : GitLab {cfg.gitlab.project}")
    print(f"  Extensions: {cfg.supported_extensions}")
    print(f"  Max files : {max_files or 'unlimited'}")
    print()

    from codegraph.connectors.local_connector import LocalConnector
    from codegraph.parsers.ast_parser import get_default_registry
    from codegraph.graph.memory_writer import InMemoryWriter

    if cfg.local_path:
        connector = LocalConnector(cfg.local_path, cfg.local_repo_name or None)
    else:
        from codegraph.connectors.gitlab_connector import GitLabConnector
        connector = GitLabConnector(cfg.gitlab).connect()

    registry = get_default_registry()
    writer   = InMemoryWriter()
    writer.setup_schema()

    from codegraph.graph.graph_builder import GraphBuilder
    from codegraph.summarizer.function_summarizer import SummarizerFactory
    builder  = GraphBuilder(writer=writer, summarizer=SummarizerFactory.build(cfg.summarizer))

    total = parsed = skipped = 0
    errors = []

    for fe in connector.iter_files(
        extensions=cfg.supported_extensions,
        max_size_kb=cfg.max_file_size_kb,
    ):
        total += 1
        if max_files and total > max_files:
            log.info("Reached --max-files %d, stopping.", max_files)
            break
        try:
            source = fe.content
        except Exception as e:
            errors.append((fe.path, str(e)))
            skipped += 1
            continue

        mod = registry.parse(source, fe.path, connector.repo_name)
        if mod is None:
            skipped += 1
            continue

        # Print per-file summary
        n_cls = len(mod.classes)
        n_fn  = len(mod.functions) + sum(len(c.methods) for c in mod.classes)
        print(f"  ✓ {fe.path:<55} {n_cls} class(es)  {n_fn} fn(s)")

        builder.add_module(mod)
        parsed += 1

    print(f"\n  Total: {total} files  |  parsed: {parsed}  |  skipped: {skipped}")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for path, msg in errors[:10]:
            print(f"    {path}: {msg}")

    print("\n  Flushing graph (resolving cross-file references)...")
    builder.flush()

    writer.print_summary()

    if getattr(args, "dot", None):
        dot_content = writer.to_dot()
        Path(args.dot).write_text(dot_content)
        print(f"  DOT file written: {args.dot}")


# ─────────────────────────────────────────────
#  Command: build
# ─────────────────────────────────────────────

def cmd_build(args):
    cfg = _build_config(args)
    pipeline = Pipeline(cfg)
    try:
        pipeline.build(clear=not args.no_clear)
        pipeline.print_graph_summary()
        if not args.skip_index:
            try:
                pipeline.index()
            except ImportError as e:
                log.warning("Skipping vector index (missing dep): %s", e)
    finally:
        pipeline.close()


# ─────────────────────────────────────────────
#  Command: index
# ─────────────────────────────────────────────

def cmd_index(args):
    cfg = _build_config(args)
    pipeline = Pipeline(cfg)
    try:
        pipeline.index()
    finally:
        pipeline.close()


# ─────────────────────────────────────────────
#  Command: chat
# ─────────────────────────────────────────────

def cmd_chat(args):
    cfg = _build_config(args)

    # If a local path is given and --build is set, build first
    pipeline = Pipeline(cfg)
    if getattr(args, "build_first", False):
        pipeline.build()

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
            print("\nAgent:", flush=True)
            response = agent.chat(user)
            print(response)
            print()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close()
        print("\nGoodbye.")


# ─────────────────────────────────────────────
#  Command: query
# ─────────────────────────────────────────────

def cmd_query(args):
    cfg = _build_config(args)
    pipeline = Pipeline(cfg)
    try:
        if getattr(args, "build_first", False):
            pipeline.build()
        agent    = pipeline.get_agent()
        response = agent.chat(args.question)
        print(response)
    finally:
        pipeline.close()


# ─────────────────────────────────────────────
#  Command: stats
# ─────────────────────────────────────────────

def cmd_stats(args):
    cfg = _build_config(args)
    pipeline = Pipeline(cfg)
    try:
        pipeline.print_graph_summary()

        # Extra: show sample function summaries
        writer = pipeline.writer
        fns = writer.query("""
            MATCH (f:Function) WHERE f.stub = false AND f.summary <> ''
            RETURN f.name AS name, f.module_path AS module, f.summary AS summary
            LIMIT 10
        """)
        if fns:
            print("\n=== Sample function summaries ===")
            for f in fns:
                print(f"  {f.get('name')} [{f.get('module','')}]")
                print(f"    {(f.get('summary') or '')[:120]}")
    finally:
        pipeline.close()


# ─────────────────────────────────────────────
#  Command: export-dot
# ─────────────────────────────────────────────

def cmd_export_dot(args):
    """Build in-memory graph then export to Graphviz DOT."""
    cfg = _build_config(args)
    cfg.graph_backend       = "memory"
    cfg.summarizer.provider = cfg.summarizer.provider or "noop"

    from codegraph.graph.memory_writer import InMemoryWriter
    pipeline = Pipeline(cfg)
    pipeline.build(clear=True)

    writer = pipeline.writer
    if not isinstance(writer, InMemoryWriter):
        print("export-dot requires --backend memory")
        sys.exit(1)

    dot = writer.to_dot()
    out = getattr(args, "output", "codegraph.dot") or "codegraph.dot"
    Path(out).write_text(dot)
    print(f"DOT written to: {out}")
    print("Render with:  dot -Tsvg codegraph.dot -o codegraph.svg")
    pipeline.close()


# ─────────────────────────────────────────────
#  Argument parser assembly
# ─────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codegraph",
        description="GitLab → AST → Graph → LLM agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── test-connection ─────────────────────────────────────
    p = sub.add_parser("test-connection",
                       help="Verify GitLab credentials and list files")
    _add_source(p)
    _add_extensions(p)

    # ── test-parse ──────────────────────────────────────────
    p = sub.add_parser("test-parse",
                       help="Parse local/GitLab files into in-memory graph, print summary")
    _add_source(p)
    _add_extensions(p)
    p.add_argument("--max-files", type=int, default=0,
                   help="Stop after N files (0 = all)")
    p.add_argument("--dot", metavar="FILE", default=None,
                   help="Also write DOT graph to FILE")

    # ── build ───────────────────────────────────────────────
    p = sub.add_parser("build", help="Full pipeline build")
    _add_source(p); _add_backend(p); _add_neo4j(p); _add_summarizer(p); _add_extensions(p)
    p.add_argument("--no-clear",    action="store_true")
    p.add_argument("--skip-index",  action="store_true")

    # ── index ───────────────────────────────────────────────
    p = sub.add_parser("index", help="Rebuild vector index")
    _add_backend(p); _add_neo4j(p)

    # ── chat ────────────────────────────────────────────────
    p = sub.add_parser("chat", help="Interactive agent Q&A")
    _add_source(p); _add_backend(p); _add_neo4j(p); _add_agent(p); _add_summarizer(p)
    p.add_argument("--build", dest="build_first", action="store_true",
                   help="Build graph before starting chat")

    # ── query ───────────────────────────────────────────────
    p = sub.add_parser("query", help="One-shot question")
    _add_source(p); _add_backend(p); _add_neo4j(p); _add_agent(p); _add_summarizer(p)
    p.add_argument("--build", dest="build_first", action="store_true")
    p.add_argument("question", help="Natural language question")

    # ── stats ───────────────────────────────────────────────
    p = sub.add_parser("stats", help="Print graph node/edge counts")
    _add_backend(p); _add_neo4j(p); _add_source(p)

    # ── export-dot ──────────────────────────────────────────
    p = sub.add_parser("export-dot",
                       help="Build in-memory graph and export Graphviz DOT")
    _add_source(p); _add_summarizer(p); _add_extensions(p)
    p.add_argument("--output", default="codegraph.dot", metavar="FILE")

    return parser


# ─────────────────────────────────────────────
#  Dispatch
# ─────────────────────────────────────────────

_DISPATCH = {
    "test-connection": cmd_test_connection,
    "test-parse":      cmd_test_parse,
    "build":           cmd_build,
    "index":           cmd_index,
    "chat":            cmd_chat,
    "query":           cmd_query,
    "stats":           cmd_stats,
    "export-dot":      cmd_export_dot,
}

if __name__ == "__main__":
    p   = _make_parser()
    args = p.parse_args()
    _DISPATCH[args.cmd](args)at-xxx
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