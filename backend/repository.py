"""
repository.py
=============
Gets a public GitHub repository onto disk (and removes it again afterwards).

Public functions
----------------
parse_github_url(repo_url)   -> GitHubRepo   Validate a URL with urllib.parse
clone_repository(repo_url)   -> str          Shallow-clone into a temp folder, return its path
cleanup_repository(repo_path)                Delete the temp folder

Every problem is reported as a RepositoryError whose message is safe to show
directly to the user (no stack traces).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# GitPython normally raises an ugly error at *import time* if the `git`
# executable is missing. "quiet" delays the problem until we actually try to
# clone, where we can turn it into a friendly message.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from git import Repo  # noqa: E402
from git.exc import GitCommandError, GitCommandNotFound  # noqa: E402

logger = logging.getLogger(__name__)

# Temp folders created by this module always start with this prefix.
# cleanup_repository() refuses to delete anything that does not.
TEMP_PREFIX = "repolens_"

# Repositories bigger than this (working tree, MB) are rejected.
MAX_REPO_SIZE_MB = 300

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

# URL parts that mean "this is a page inside a repo, not the repo itself".
_SUB_PAGES = {
    "blob", "tree", "raw", "blame", "commit", "commits", "issues", "pull",
    "pulls", "releases", "actions", "wiki", "discussions", "branches", "tags",
}


class RepositoryError(Exception):
    """A problem with the repository. The message is meant for end users."""


@dataclass(frozen=True)
class GitHubRepo:
    owner: str
    name: str

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"

    @property
    def web_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"


# --------------------------------------------------------------------------
# URL validation
# --------------------------------------------------------------------------
def parse_github_url(repo_url: str) -> GitHubRepo:
    """Validate a GitHub *repository* URL and return its owner and name.

    Accepted : https://github.com/user/repo   (also with .git or a trailing /)
    Rejected : https://github.com/user/repo/blob/main/file.py  (file URL)
               https://github.com/user/repo/tree/main/src      (folder URL)
               anything that is not on github.com
    """
    if not repo_url or not repo_url.strip():
        raise RepositoryError("Please enter a GitHub repository URL.")

    raw = repo_url.strip()
    if "://" not in raw:  # allow "github.com/user/repo"
        raw = "https://" + raw

    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
    except ValueError:
        raise RepositoryError("That does not look like a valid URL.") from None

    if parsed.scheme not in ("http", "https"):
        raise RepositoryError("The URL must start with https://")
    if host not in ("github.com", "www.github.com"):
        raise RepositoryError(
            "Only public github.com repositories are supported "
            "(example: https://github.com/psf/requests)."
        )

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise RepositoryError(
            "Please give the full repository URL, like https://github.com/owner/repository"
        )
    if len(parts) > 2:
        if parts[2].lower() in _SUB_PAGES:
            raise RepositoryError(
                "This looks like a link to a file, folder or page inside a repository. "
                "Please paste the repository URL itself: "
                f"https://github.com/{parts[0]}/{parts[1]}"
            )
        raise RepositoryError(
            "Please use the plain repository URL: https://github.com/owner/repository"
        )

    owner, name = parts[0], parts[1]
    if name.lower().endswith(".git"):
        name = name[:-4]

    if (
        not _NAME_PATTERN.match(owner)
        or not _NAME_PATTERN.match(name)
        or name in (".", "..")
        or owner in (".", "..")
    ):
        raise RepositoryError("The owner or repository name contains invalid characters.")

    return GitHubRepo(owner=owner, name=name)


# --------------------------------------------------------------------------
# Cloning
# --------------------------------------------------------------------------
def clone_repository(repo_url: str) -> str:
    """Shallow-clone a public GitHub repository into a temporary directory.

    Returns the path of the cloned repository. The caller must pass that path
    to cleanup_repository() when finished (use try/finally).
    """
    repo = parse_github_url(repo_url)

    # <tmp>/repolens_xxxx/<repo-name>  -> the repo keeps its real name,
    # while the random parent folder keeps parallel users from colliding.
    temp_root = tempfile.mkdtemp(prefix=TEMP_PREFIX)
    target = os.path.join(temp_root, repo.name)

    # Fail fast instead of hanging or opening a login prompt:
    #  - never ask for a username/password (private repos just fail)
    #  - give up if the download stalls (< 1 KB/s for 30 s)
    git_env = {
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "GIT_HTTP_LOW_SPEED_LIMIT": "1000",
        "GIT_HTTP_LOW_SPEED_TIME": "30",
    }

    try:
        # depth=1 -> only the latest commit: much smaller and faster.
        Repo.clone_from(
            repo.clone_url, target, depth=1, single_branch=True, env=git_env
        )
        size_mb = _folder_size_mb(Path(target))
        if size_mb > MAX_REPO_SIZE_MB:
            raise RepositoryError(
                f"This repository is very large (about {size_mb:.0f} MB). "
                f"Please try a repository smaller than {MAX_REPO_SIZE_MB} MB."
            )
    except Exception as exc:  # always clean up, then raise a friendly error
        cleanup_repository(target)
        if isinstance(exc, RepositoryError):
            raise
        logger.warning("Clone of %s failed: %s", repo.web_url, exc)
        raise RepositoryError(_friendly_git_error(exc)) from exc

    return target


def _friendly_git_error(exc: Exception) -> str:
    if isinstance(exc, GitCommandNotFound):
        return (
            "Git is not installed on this machine. "
            "Install Git from https://git-scm.com and try again."
        )
    text = str(exc).lower()
    if any(s in text for s in (
        "repository not found", "not found", "could not read username",
        "authentication failed", "terminal prompts disabled", "error: 403", "error: 404",
    )):
        return (
            "Repository not found. It may not exist, or it may be private. "
            "Only public GitHub repositories can be analyzed."
        )
    if any(s in text for s in (
        "could not resolve host", "unable to access", "failed to connect",
        "timed out", "timeout", "rpc failed", "early eof", "low speed",
    )):
        return "Could not download the repository from GitHub. Check your internet connection and try again."
    if isinstance(exc, GitCommandError):
        return "Git could not clone this repository. Please check the URL and try again."
    return "Something went wrong while downloading the repository."


def _folder_size_mb(path: Path) -> float:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        if ".git" in dirnames:
            dirnames.remove(".git")  # history is not part of the "size" we care about
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                pass
    return total / (1024 * 1024)


# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------
def _force_remove(func, path, _exc_info) -> None:
    """rmtree error handler: Git marks some files read-only (a problem on Windows)."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def cleanup_repository(repo_path: str | os.PathLike | None) -> None:
    """Delete a repository previously returned by clone_repository().

    Safe to call with None, with a path that no longer exists, or twice.
    It will only ever delete folders created by this module.
    """
    if not repo_path:
        return
    path = Path(repo_path)
    root = path.parent if path.parent.name.startswith(TEMP_PREFIX) else path
    if not root.name.startswith(TEMP_PREFIX):
        logger.warning("Refusing to delete a folder this module did not create: %s", root)
        return
    if root.exists():
        kwargs = {"onexc": _force_remove} if sys.version_info >= (3, 12) else {"onerror": _force_remove}
        shutil.rmtree(root, **kwargs)
