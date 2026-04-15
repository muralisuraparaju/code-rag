"""
codegraph/connectors/local_connector.py

Reads source files from a local directory (e.g. an already-cloned repo).
Drop-in replacement for GitLabConnector — both expose the same interface:
  .repo_name          str
  .iter_files(...)    Iterator[FileEntry]

Also contains the BaseConnector protocol so callers can type-annotate
against the interface rather than a concrete class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared FileEntry (re-exported from here so callers import one place)
# ---------------------------------------------------------------------------

class LocalFileEntry:
    """FileEntry backed by an on-disk file — content is read lazily."""

    __slots__ = ("path", "size", "_abs_path", "_content")

    def __init__(self, rel_path: str, abs_path: Path):
        self.path      = rel_path           # relative to repo root (same as GitLab)
        self.size      = abs_path.stat().st_size
        self._abs_path = abs_path
        self._content: Optional[str] = None

    @property
    def content(self) -> str:
        if self._content is None:
            self._content = self._abs_path.read_text(encoding="utf-8", errors="replace")
        return self._content

    def __repr__(self) -> str:
        return f"<LocalFileEntry {self.path} ({self.size} bytes)>"


# ---------------------------------------------------------------------------
# Protocol (structural subtyping — no inheritance required)
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseConnector(Protocol):
    """
    Minimal interface that Pipeline and tests depend on.
    Both GitLabConnector and LocalConnector satisfy this protocol.
    """

    @property
    def repo_name(self) -> str: ...

    def iter_files(
        self,
        extensions: tuple[str, ...],
        max_size_kb: int,
    ) -> Iterator: ...


# ---------------------------------------------------------------------------
# LocalConnector
# ---------------------------------------------------------------------------

_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", "vendor", "dist", "build", ".venv", "venv", "env",
    ".tox", "target",  # Java/Rust
}


class LocalConnector:
    """
    Walks a local directory and yields LocalFileEntry objects.

    Parameters
    ----------
    root_dir : str | Path
        Absolute (or relative) path to the cloned repository root.
    repo_name : str, optional
        Logical name to tag graph nodes with.
        Defaults to the directory's basename.
    """

    def __init__(self, root_dir: str | Path, repo_name: Optional[str] = None):
        self._root = Path(root_dir).resolve()
        if not self._root.is_dir():
            raise FileNotFoundError(f"Directory not found: {self._root}")
        self._repo_name = repo_name or self._root.name

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    @property
    def repo_name(self) -> str:
        return self._repo_name

    def iter_files(
        self,
        extensions: tuple[str, ...] = (".py",),
        max_size_kb: int = 500,
    ) -> Iterator[LocalFileEntry]:
        """Yield one LocalFileEntry per matching source file."""
        found = skipped = 0
        max_bytes = max_size_kb * 1024

        for abs_path in sorted(self._root.rglob("*")):
            if not abs_path.is_file():
                continue

            # Skip hidden / generated directories anywhere in the path
            parts = abs_path.relative_to(self._root).parts
            if any(p in _SKIP_DIRS or p.startswith(".") for p in parts):
                skipped += 1
                continue

            if abs_path.suffix.lower() not in extensions:
                continue

            if abs_path.stat().st_size > max_bytes:
                log.debug("Skipping large file: %s", abs_path)
                skipped += 1
                continue

            rel_path = str(abs_path.relative_to(self._root))
            found += 1
            yield LocalFileEntry(rel_path=rel_path, abs_path=abs_path)

        log.info(
            "LocalConnector walk: %d files found, %d skipped  (root: %s)",
            found, skipped, self._root,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_file(self, rel_path: str) -> Optional[str]:
        """Read a single file by relative path."""
        abs_path = self._root / rel_path
        if not abs_path.exists():
            return None
        return abs_path.read_text(encoding="utf-8", errors="replace")

    def __repr__(self) -> str:
        return f"<LocalConnector root={self._root!r} repo={self._repo_name!r}>"