from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, root: str | Path, phase: str = "train", transform=None,
                 return_raw: bool = False, return_paths: bool = False):
        """
        Expected structure:
          root/train/images, root/train/masks
          root/val/images,   root/val/masks

        return_raw:
          - False (default): returns transformed (img, mask)
          - True: returns (img, mask, raw_img, raw_mask) or a dict (see below)

        return_paths:
          adds img_path + mask_path for debugging
        """
        root = Path(root)
        self.phase = phase
        self.image_dir = root / phase / "images"
        self.mask_dir = root / phase / "masks"
        self.transform = transform
        self.return_raw = return_raw
        self.return_paths = return_paths

        self.image_filenames = sorted([p.name for p in self.image_dir.iterdir() if p.is_file()])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        file_name = self.image_filenames[idx]
        img_path = self.image_dir / file_name
        mask_path = self.mask_dir / file_name

        raw_img = np.array(Image.open(img_path).convert("RGB"))  # (H,W,3) uint8
        raw_mask = np.expand_dims(np.array(Image.open(mask_path).convert("L")), axis=2)  # (H,W,1) uint8

        img, mask = raw_img, raw_mask
        if self.transform is not None:
            img, mask = self.transform((img, mask))

        # Return style: dict makes notebooks super convenient
        out = {"image": img, "mask": mask}
        if self.return_raw:
            out["raw_image"] = raw_img
            out["raw_mask"] = raw_mask
        if self.return_paths:
            out["image_path"] = str(img_path)
            out["mask_path"] = str(mask_path)
        # access batch["image"] and batch["mask"]
        return out
