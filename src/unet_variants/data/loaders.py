
from __future__ import annotations

from torch.utils.data import DataLoader
import torch

from unet_variants.data.prepare import ensure_extracted_dataset
from unet_variants.data.dataset import SegmentationDataset, SegmentationDatasetToTuple
from unet_variants.data.transforms import build_transforms


def build_dataloaders(cfg):
    root = ensure_extracted_dataset(cfg)

    train_tf = build_transforms(cfg, phase="train")
    val_tf   = build_transforms(cfg, phase="val")

    # train_ds = SegmentationDatasetToTuple(root=root, phase="train", transform=train_tf)
    # val_ds   = SegmentationDatasetToTuple(root=root, phase="val",   transform=val_tf)

    train_ds = SegmentationDataset(
        root=root,
        phase="train",
        transform=train_tf,
        return_raw=False,
        return_paths=False,
    )

    val_ds = SegmentationDataset(
        root=root,
        phase="val",
        transform=val_tf,
        return_raw=False,
        return_paths=False,
    )

    pin = torch.cuda.is_available() and str(cfg.project.device).startswith("cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=pin,
        persistent_workers=(int(cfg.train.num_workers) > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=pin,
        persistent_workers=(int(cfg.train.num_workers) > 0),
    )


    return train_loader, val_loader
