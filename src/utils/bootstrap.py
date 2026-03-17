
from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """
    Find repository root by walking upward until we find `pyproject.toml` or `.git/`.

    Parameters
    ----------
    start:
        Directory to start searching from. If None, uses current working directory.

    Returns
    -------
    Path
        Path to the repository root.
    """
    start = (start or Path.cwd()).resolve()

    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p

    # Fallback: if not found, return cwd (safer than assuming a fixed parents[] depth).
    return start


def set_repo_root_env(var_name: str = "UNET_VARIANTS_ROOT", start: Path | None = None) -> Path:
    """
    Ensure a stable repo-root environment variable exists.

    This is useful because Hydra may change run directories, and we want
    path resolution (data, runs) to be stable and independent of cwd.

    Parameters
    ----------
    var_name:
        Environment variable name to set.
    start:
        Optional search start path for locating the repo root.

    Returns
    -------
    Path
        Resolved repository root.
    """
    if os.environ.get(var_name):
        return Path(os.environ[var_name]).resolve()

    root = find_repo_root(start=start)
    os.environ[var_name] = str(root)
    return root
