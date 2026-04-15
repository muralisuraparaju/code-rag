"""
codegraph/connectors/gitlab_connector.py

Streams file contents from a GitLab project via the REST API.
No git clone required — files are fetched on-demand, which lets us
process one file at a time without holding the full repo on disk.

For large repos a shallow clone is also supported as a fallback.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import gitlab
from gitlab.v4.objects import Project
from git import Repo

from codegraph.config import GitLabConfig

log = logging.getLogger(__name__)


@dataclass_workaround = None  # noqa – plain class, no dataclass needed


class FileEntry:
    """Metadata + lazy content for one repository file."""

    __slots__ = ("path", "size", "_project", "_ref", "_content")

    def __init__(self, path: str, size: int, project: Project, ref: str):
        self.path     = path
        self.size     = size
        self._project = project
        self._ref     = ref
        self._content: Optional[str] = None

    @property
    def content(self) -> str:
        """Fetch and decode file content on first access (lazy)."""
        if self._content is None:
            raw = self._project.files.get(file_path=self.path, ref=self._ref)
            self._content = base64.b64decode(raw.content).decode("utf-8", errors="replace")
        return self._content

    def __repr__(self) -> str:
        return f"<FileEntry {self.path} ({self.size} bytes)>"


class GitLabConnector:
    """
    Connects to a GitLab project and iterates over source files.

    Usage
    -----
    connector = GitLabConnector(cfg)
    for file_entry in connector.iter_files(extensions=(".py",)):
        source = file_entry.content   # fetched on demand
        ...
    """

    def __init__(self, cfg: GitLabConfig):
        self._cfg = cfg
        self._gl  = gitlab.Gitlab(cfg.url, private_token=cfg.token)
        self._project: Optional[Project] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> "GitLabConnector":
        """Authenticate and resolve the project. Returns self for chaining."""
        self._gl.auth()
        self._project = self._gl.projects.get(self._cfg.project)
        log.info(
            "Connected to GitLab project: %s (id=%s)",
            self._project.path_with_namespace,
            self._project.id,
        )
        return self

    @property
    def project(self) -> Project:
        if self._project is None:
            raise RuntimeError("Call connect() first.")
        return self._project

    @property
    def repo_name(self) -> str:
        return self.project.path_with_namespace.replace("/", "__")

    # ------------------------------------------------------------------
    # File iteration
    # ------------------------------------------------------------------

    def iter_files(
        self,
        extensions: tuple[str, ...] = (".py",),
        max_size_kb: int = 500,
        path_prefix: str = "",
    ) -> Iterator[FileEntry]:
        """
        Yield FileEntry objects for every matching source file.
        Content is NOT fetched yet — caller pulls it via entry.content.
        """
        if self._project is None:
            self.connect()

        ref     = self._cfg.branch
        seen    = 0
        skipped = 0

        # GitLab repository tree API (recursive=True walks all subdirs)
        items = self._project.repository_tree(
            path=path_prefix,
            ref=ref,
            recursive=True,
            all=True,           # paginate automatically
        )

        for item in items:
            if item["type"] != "blob":
                continue
            fpath = item["path"]
            if not any(fpath.endswith(ext) for ext in extensions):
                continue
            # Size guard — use the blob size from the tree listing
            size = item.get("size") or 0
            if size > max_size_kb * 1024:
                log.debug("Skipping large file %s (%d KB)", fpath, size // 1024)
                skipped += 1
                continue
            # Skip hidden / generated paths
            parts = Path(fpath).parts
            if any(p.startswith(".") or p in ("node_modules", "__pycache__", "vendor", "dist") for p in parts):
                skipped += 1
                continue

            seen += 1
            yield FileEntry(path=fpath, size=size, project=self._project, ref=ref)

        log.info("GitLab tree walk: %d files found, %d skipped", seen, skipped)

    # ------------------------------------------------------------------
    # Single file
    # ------------------------------------------------------------------

    def get_file(self, path: str) -> Optional[str]:
        """Fetch and return the decoded content of a single file."""
        try:
            raw = self.project.files.get(file_path=path, ref=self._cfg.branch)
            return base64.b64decode(raw.content).decode("utf-8", errors="replace")
        except Exception as e:
            log.warning("Could not fetch %s: %s", path, e)
            return None

    # ------------------------------------------------------------------
    # Shallow clone fallback (for repos too large for the API)
    # ------------------------------------------------------------------

    def shallow_clone(self, target_dir: str) -> Path:
        """
        Shallow-clone the repo into target_dir and return the root Path.
        Use this when you need full filesystem access (e.g. for running
        build tools or language servers).
        """
        clone_url = self.project.http_url_to_repo
        if self._cfg.token:
            proto, rest = clone_url.split("://", 1)
            clone_url = f"{proto}://oauth2:{self._cfg.token}@{rest}"

        dest = Path(target_dir) / self.repo_name
        if dest.exists():
            log.info("Pulling %s", dest)
            Repo(dest).remotes.origin.pull(self._cfg.branch)
        else:
            log.info("Shallow-cloning %s → %s", self._cfg.project, dest)
            Repo.clone_from(clone_url, dest, branch=self._cfg.branch, depth=1)
        return dest

    def __repr__(self) -> str:
        return f"<GitLabConnector project={self._cfg.project!r}>"