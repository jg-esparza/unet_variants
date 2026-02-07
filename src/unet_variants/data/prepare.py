
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def ensure_extracted_dataset(cfg) -> Path:
    """
    If cfg.data.root already exists and looks non-empty -> use it.
    Otherwise, if cfg.data.archive_path exists -> extract it into cfg.data.root.

    Expected extracted structure:
      root/train/images, root/train/masks
      root/val/images,   root/val/masks
    """
    root = Path(cfg.data.root)
    if root.exists() and any(root.iterdir()):
        return root

    archive_path = str(getattr(cfg.data, "archive_path", "")).strip()
    if not archive_path:
        raise ValueError(
            "Dataset not found at data.root and no data.archive_path provided. "
            "Please download the zip and set data.archive_path."
        )

    ap = Path(archive_path).expanduser().resolve()
    if not ap.exists():
        raise FileNotFoundError(f"archive_path not found: {ap}")

    root.parent.mkdir(parents=True, exist_ok=True)

    # extract to a temp folder then move contents to root
    tmp = root.parent / (root.name + "_tmp_extract")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ap, "r") as zf:
        zf.extractall(tmp)

    # If zip contains a single folder, unwrap it
    children = list(tmp.iterdir())
    if len(children) == 1 and children[0].is_dir():
        shutil.move(str(children[0]), str(root))
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        root.mkdir(parents=True, exist_ok=True)
        for item in children:
            shutil.move(str(item), str(root / item.name))
        shutil.rmtree(tmp, ignore_errors=True)

    return root
