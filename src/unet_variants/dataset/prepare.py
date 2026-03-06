
from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import DictConfig

from unet_variants.utils.io import ensure_dir, is_non_empty_dir, extract_zip


def ensure_extracted_dataset(cfg:DictConfig) -> Path:
    """
    Ensure dataset root exists and contains extracted files.

    Policy:
    - If cfg.dataset.root exists and is non-empty -> use it.
    - Else if cfg.dataset.archive_path exists -> extract into cfg.dataset.root.
    - Else raise an error.

    Expected structure under root:
      root/train/images, root/train/masks
      root/val/images,   root/val/masks
    """
    root = Path(cfg.dataset.root).expanduser().resolve()

    if is_non_empty_dir(root):
        return root

    archive_path = str(getattr(cfg.dataset, "archive_path", "")).strip()
    if not archive_path:
        raise ValueError(
            f"Dataset not found at dataset.root and no data.archive_path provided. "
            f"Please download the zip and set data.archive_path."
        )

    ap = Path(archive_path).expanduser().resolve()
    if not ap.exists():
        raise FileNotFoundError(f"archive_path not found: {ap}")

    ensure_dir(root.parent)

    # Extract to temp folder first
    tmp = root.parent / (root.name + "_tmp_extract")
    extract_zip(ap, tmp, overwrite=True)

    # Unwrap if zip contains a single top-level folder
    children = list(tmp.iterdir())
    if len(children) == 1 and children[0].is_dir():
        if root.exists():
            shutil.rmtree(root)
        shutil.move(str(children[0]), str(root))
    else:
        ensure_dir(root)
        for item in children:
            shutil.move(str(item), str(root / item.name))

    shutil.rmtree(tmp, ignore_errors=True)
    return root
