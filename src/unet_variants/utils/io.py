from __future__ import annotations

import os
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Union, Dict, List
from omegaconf import DictConfig, OmegaConf

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create directory if it doesn't exist and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_path(path:str, file:str) -> str:
    """Create path if it doesn't exist and return it as Path."""
    file_path = os.path.join(path, file)
    ensure_dir(os.path.dirname(file_path))
    return file_path

def is_non_empty_dir(path: PathLike) -> bool:
    """Return True if path exists and contains at least one entry."""
    p = Path(path)
    return p.exists() and p.is_dir() and any(p.iterdir())

def is_file(path: PathLike) -> bool:
    """Return True if path exists and contains at least one entry."""
    p = Path(path)
    return p.is_file()

def file_not_exist(path: PathLike) -> bool:
    return not os.path.exists(path)

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

def save_config_yaml(cfg: DictConfig, path: PathLike) -> None:
    """Save a YAML configuration file to a file, ensuring parent directory exists."""
    p = Path(path)
    ensure_dir(p.parent if p.parent.as_posix() else ".")
    OmegaConf.save(config=cfg, f=p)

def extract_zip(zip_path: PathLike, dst_dir: PathLike, *, overwrite: bool = True) -> Path:
    """Extract a .zip archive into dst_dir."""
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

def append_to_aggregated_csv(
        records: Dict[str, Any],
        columns: List[str],
        path: str,
        file_name: str
    ) -> None:
    """
    Append one or more benchmark records to a consolidated CSV table.
    If the CSV doesn't exist, it's created with a header.
    Returns the CSV path for convenience.
    """
    csv_path = make_path(path, file_name)
    write_csv_row(csv_path, columns, records)

def write_csv_row(csv_path:str, columns:List[str], records: Dict[str, Any]) -> None:
    """Write a CSV row to a CSV file."""
    is_new = file_not_exist(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow(records)
