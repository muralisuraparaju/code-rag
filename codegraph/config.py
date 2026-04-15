"""
codegraph/config.py

Single source-of-truth for all runtime configuration.
Can be loaded from env-vars or constructed directly.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass
class GitLabConfig:
    url: str   = "https://gitlab.com"
    token: str = ""
    project:str= ""          # "group/repo"
    branch: str= "main"


@dataclass
class Neo4jConfig:
    uri:      str = "bolt://localhost:7687"
    user:     str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


@dataclass
class SummarizerConfig:
    provider: str  = "anthropic"   # "anthropic" | "openai" | "ollama" | "noop"
    model:    str  = "claude-haiku-4-5-20251001"
    api_key:  str  = ""
    base_url: str  = ""            # for ollama / custom endpoints
    max_tokens: int= 256
    # Cache summaries to avoid re-generating on re-runs
    cache_file: str= ".codegraph_summary_cache.json"


@dataclass
class RAGConfig:
    # Embedding model for vector search
    embed_model: str = "text-embedding-3-small"  # openai
    embed_provider: str = "openai"
    openai_api_key: str = ""
    # ChromaDB for vector store (embedded, no server needed)
    chroma_persist_dir: str = ".codegraph_chroma"
    collection_name: str = "codegraph"


@dataclass
class AgentConfig:
    provider: str = "anthropic"
    model:    str = "claude-sonnet-4-6"
    api_key:  str = ""
    max_tokens: int = 4096


@dataclass
class AppConfig:
    gitlab:     GitLabConfig     = field(default_factory=GitLabConfig)
    neo4j:      Neo4jConfig      = field(default_factory=Neo4jConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    rag:        RAGConfig        = field(default_factory=RAGConfig)
    agent:      AgentConfig      = field(default_factory=AgentConfig)

    supported_extensions: tuple  = (".py", ".js", ".ts", ".java", ".go")
    max_file_size_kb: int        = 500
    workers: int                 = 4    # parallel summarization threads

    @classmethod
    def from_env(cls) -> "AppConfig":
        cfg = cls()
        # GitLab
        cfg.gitlab.url     = os.getenv("GITLAB_URL",   cfg.gitlab.url)
        cfg.gitlab.token   = os.getenv("GITLAB_TOKEN", cfg.gitlab.token)
        cfg.gitlab.project = os.getenv("GITLAB_PROJECT", cfg.gitlab.project)
        cfg.gitlab.branch  = os.getenv("GITLAB_BRANCH",  cfg.gitlab.branch)
        # Neo4j
        cfg.neo4j.uri      = os.getenv("NEO4J_URI",      cfg.neo4j.uri)
        cfg.neo4j.user     = os.getenv("NEO4J_USER",     cfg.neo4j.user)
        cfg.neo4j.password = os.getenv("NEO4J_PASSWORD", cfg.neo4j.password)
        # Summarizer
        cfg.summarizer.provider  = os.getenv("SUMMARIZER_PROVIDER", cfg.summarizer.provider)
        cfg.summarizer.model     = os.getenv("SUMMARIZER_MODEL",    cfg.summarizer.model)
        cfg.summarizer.api_key   = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        # RAG / embeddings
        cfg.rag.embed_provider   = os.getenv("EMBED_PROVIDER",  cfg.rag.embed_provider)
        cfg.rag.embed_model      = os.getenv("EMBED_MODEL",     cfg.rag.embed_model)
        cfg.rag.openai_api_key   = os.getenv("OPENAI_API_KEY",  cfg.rag.openai_api_key)
        # Agent
        cfg.agent.provider       = os.getenv("AGENT_PROVIDER",  cfg.agent.provider)
        cfg.agent.model          = os.getenv("AGENT_MODEL",     cfg.agent.model)
        cfg.agent.api_key        = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        return cfg