from __future__ import annotations
import os
from pathlib import Path

def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root by walking upwards until we find pyproject.toml or .git folder."""
    start = start or Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # fallback: assume 2 levels up from src/unet_variants/utils/bootstrap.py -> repo root
    return Path(__file__).resolve().parents[3]

def set_repo_root_env(var_name: str = "UNET_VARIANTS_ROOT") -> Path:
    """Set env var used by Hydra configs for stable path resolution."""
    if os.environ.get(var_name):
        return Path(os.environ[var_name])

    root = find_repo_root()
    os.environ[var_name] = str(root)
    return root
