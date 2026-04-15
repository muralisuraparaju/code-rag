"""
codegraph/summarizer/function_summarizer.py

Generates natural-language summaries for FunctionNode objects.

Architecture
------------
- BaseSummarizer  — interface
- AnthropicSummarizer  — claude-* models
- OpenAISummarizer     — gpt-* / compatible endpoints
- OllamaSummarizer     — local models via Ollama REST API
- NoOpSummarizer       — returns docstring or empty string (testing / offline)
- CachingSummarizer    — decorator that wraps any backend with JSON file cache
- SummarizerFactory    — builds the right backend from config

The summarizer is intentionally decoupled from everything else: it takes a
FunctionNode and returns a str.  Swap backends without touching the graph code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import textwrap
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from codegraph.config import SummarizerConfig
from codegraph.models import FunctionNode

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(fn: FunctionNode) -> str:
    """Craft a tight prompt that extracts the function's purpose."""
    params_str = ", ".join(
        f"{p.name}: {p.type_hint or 'Any'}" + (f" = {p.default}" if p.default else "")
        for p in fn.params
    )
    ret_str = fn.return_type or "unknown"
    lines = [
        "You are a senior software engineer. Summarize the following function in 2-3 sentences.",
        "Focus on: what it does, key inputs/outputs, and any important side-effects.",
        "Be concise and precise. Do not repeat the function signature.",
        "",
        f"Function: {fn.name}",
        f"Module:   {fn.module_path}",
        f"Params:   ({params_str})",
        f"Returns:  {ret_str}",
    ]
    if fn.docstring:
        lines += ["", f"Docstring: {fn.docstring[:400]}"]
    if fn.source_snippet:
        code = textwrap.indent(fn.source_snippet[:800], "  ")
        lines += ["", "Source:", "```", code, "```"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, fn: FunctionNode) -> str:
        """Return a short natural-language summary of fn."""

    def summarize_batch(
        self,
        functions: list[FunctionNode],
        workers: int = 4,
    ) -> list[FunctionNode]:
        """
        Summarize a list of functions in parallel.
        Mutates each FunctionNode in place (sets .summary) and returns the list.
        """
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_fn = {pool.submit(self.summarize, fn): fn for fn in functions}
            for future in as_completed(future_to_fn):
                fn = future_to_fn[future]
                try:
                    fn.summary = future.result()
                except Exception as e:
                    log.warning("Summary failed for %s: %s", fn.qualified_name, e)
                    fn.summary = fn.docstring or ""
        return functions


# ---------------------------------------------------------------------------
# NoOp (offline / testing)
# ---------------------------------------------------------------------------

class NoOpSummarizer(BaseSummarizer):
    """Returns the docstring if available, otherwise empty string."""

    def summarize(self, fn: FunctionNode) -> str:
        if fn.docstring:
            return fn.docstring[:300]
        sig = f"Function {fn.name}({', '.join(p.name for p in fn.params)})"
        if fn.return_type:
            sig += f" -> {fn.return_type}"
        return sig


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------

class AnthropicSummarizer(BaseSummarizer):
    def __init__(self, cfg: SummarizerConfig):
        try:
            import anthropic  # type: ignore
        except ImportError:
            raise ImportError("pip install anthropic")
        self._client = anthropic.Anthropic(api_key=cfg.api_key or os.getenv("ANTHROPIC_API_KEY", ""))
        self._model  = cfg.model
        self._max_tokens = cfg.max_tokens

    def summarize(self, fn: FunctionNode) -> str:
        import anthropic  # type: ignore
        prompt = _build_prompt(fn)
        try:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except anthropic.APIError as e:
            log.warning("Anthropic API error for %s: %s", fn.name, e)
            return fn.docstring or ""


# ---------------------------------------------------------------------------
# OpenAI (GPT / compatible)
# ---------------------------------------------------------------------------

class OpenAISummarizer(BaseSummarizer):
    def __init__(self, cfg: SummarizerConfig):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise ImportError("pip install openai")
        kwargs: dict = {"api_key": cfg.api_key or os.getenv("OPENAI_API_KEY", "")}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        from openai import OpenAI
        self._client = OpenAI(**kwargs)
        self._model  = cfg.model
        self._max_tokens = cfg.max_tokens

    def summarize(self, fn: FunctionNode) -> str:
        prompt = _build_prompt(fn)
        try:
            rsp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": "You summarize code functions concisely."},
                    {"role": "user",   "content": prompt},
                ],
            )
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            log.warning("OpenAI API error for %s: %s", fn.name, e)
            return fn.docstring or ""


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

class OllamaSummarizer(BaseSummarizer):
    def __init__(self, cfg: SummarizerConfig):
        try:
            import httpx  # type: ignore
        except ImportError:
            raise ImportError("pip install httpx")
        import httpx
        base = cfg.base_url or "http://localhost:11434"
        self._client = httpx.Client(base_url=base, timeout=60)
        self._model  = cfg.model or "codellama"
        self._max_tokens = cfg.max_tokens

    def summarize(self, fn: FunctionNode) -> str:
        prompt = _build_prompt(fn)
        try:
            rsp = self._client.post("/api/generate", json={
                "model":  self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": self._max_tokens},
            })
            rsp.raise_for_status()
            return rsp.json().get("response", "").strip()
        except Exception as e:
            log.warning("Ollama error for %s: %s", fn.name, e)
            return fn.docstring or ""


# ---------------------------------------------------------------------------
# Caching decorator
# ---------------------------------------------------------------------------

class CachingSummarizer(BaseSummarizer):
    """
    Wraps any BaseSummarizer with a JSON file cache keyed by a hash of the
    function's source snippet + name.  Persists across runs so you don't
    re-summarize unchanged functions.
    """

    def __init__(self, inner: BaseSummarizer, cache_file: str = ".codegraph_summary_cache.json"):
        self._inner      = inner
        self._cache_path = Path(cache_file)
        self._cache: dict[str, str] = {}
        self._dirty = False
        self._load()

    def _load(self):
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
                log.info("Loaded %d cached summaries from %s", len(self._cache), self._cache_path)
            except Exception:
                self._cache = {}

    def _save(self):
        if self._dirty:
            self._cache_path.write_text(json.dumps(self._cache, indent=2))
            self._dirty = False

    def _key(self, fn: FunctionNode) -> str:
        raw = f"{fn.qualified_name}|{fn.source_snippet}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def summarize(self, fn: FunctionNode) -> str:
        key = self._key(fn)
        if key in self._cache:
            return self._cache[key]
        result = self._inner.summarize(fn)
        self._cache[key] = result
        self._dirty = True
        # Save after every 20 new entries to avoid data loss on crash
        if len(self._cache) % 20 == 0:
            self._save()
        return result

    def flush(self):
        self._save()

    def __del__(self):
        try:
            self._save()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class SummarizerFactory:
    """
    Build a fully-configured (and cached) summarizer from SummarizerConfig.

    Example
    -------
    cfg = SummarizerConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    summarizer = SummarizerFactory.build(cfg)
    fn.summary = summarizer.summarize(fn)
    """

    @staticmethod
    def build(cfg: SummarizerConfig) -> BaseSummarizer:
        provider = cfg.provider.lower()
        if provider == "anthropic":
            inner: BaseSummarizer = AnthropicSummarizer(cfg)
        elif provider == "openai":
            inner = OpenAISummarizer(cfg)
        elif provider == "ollama":
            inner = OllamaSummarizer(cfg)
        elif provider == "noop":
            inner = NoOpSummarizer()
        else:
            log.warning("Unknown summarizer provider %r — using NoOp", provider)
            inner = NoOpSummarizer()

        if cfg.cache_file and provider != "noop":
            return CachingSummarizer(inner, cfg.cache_file)
        return inner