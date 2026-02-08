
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Union


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create directory if it doesn't exist and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_non_empty_dir(path: PathLike) -> bool:
    """Return True if path exists and contains at least one entry."""
    p = Path(path)
    return p.exists() and p.is_dir() and any(p.iterdir())


def save_text(text: str, path: PathLike, encoding: str = "utf-8") -> Path:
    """Save text to a file, ensuring parent directory exists."""
    p = Path(path)
    ensure_dir(p.parent if p.parent.as_posix() else ".")
    p.write_text(text + "\n", encoding=encoding)
    return p


def save_json(obj: Any, path: PathLike, indent: int = 2, encoding: str = "utf-8") -> Path:
    """Save a JSON-serializable object to a file, ensuring parent directory exists."""
    p = Path(path)
    ensure_dir(p.parent if p.parent.as_posix() else ".")
    p.write_text(json.dumps(obj, indent=indent), encoding=encoding)
    return p


def extract_zip(zip_path: PathLike, dst_dir: PathLike, *, overwrite: bool = True) -> Path:
    """
    Extract a .zip archive into dst_dir.

    Parameters
    ----------
    zip_path:
        Path to the .zip archive.
    dst_dir:
        Destination directory.
    overwrite:
        If True, remove dst_dir first if it exists.

    Returns
    -------
    Path
        Destination directory path.
    """
    zip_path = Path(zip_path).expanduser().resolve()
    dst_dir = Path(dst_dir).expanduser().resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")

    if dst_dir.exists() and overwrite:
        shutil.rmtree(dst_dir)

    ensure_dir(dst_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)

    return dst_dir
