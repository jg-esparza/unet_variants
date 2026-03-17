
from __future__ import annotations

import random
import numpy as np

from omegaconf import DictConfig

import torch
import torchvision.transforms.functional as F


# ---------- Composers ----------
class PairCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


# ---------- Tensor conversion ----------
class ToTensorNoScale:
    """Convert numpy uint8 image/mask to torch float tensors."""
    def __call__(self, data):
        image, mask = data
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return image, mask

# ---------- Resize ----------
class ResizePair:
    def __init__(self, size: int):
        self.size = [int(size), int(size)]

    def __call__(self, data):
        image, mask = data
        image = F.resize(image, self.size, antialias=True)
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST, antialias=False)
        return image, mask

# ---------- Augmentation ----------
class RandomHFlip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return F.hflip(image), F.hflip(mask)
        return image, mask


class RandomVFlip:
    def __init__(self, p=0.0): self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return F.vflip(image), F.vflip(mask)
        return image, mask


class RandomRotation:
    def __init__(self, p=0.2, degree=(0, 360)):
        self.p = p
        self.degree = degree

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
            mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
        return image, mask

# ---------- Normalization (VM-Mamba style: numpy in, numpy out) ----------
class CustomNormaization:
    """Reproduce your original behavior on numpy uint8 image/mask.
    Returns image in [0,1] via per-image min-max after standardization.
    Mask returned as {0,1}.
    """
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def __call__(self, data):
        image, mask = data
        img = image.astype(np.float32)  # 0..255
        img = (img - self.mean) / (self.std + 1e-8)

        # per-image min-max -> [0,1] (matches your pipeline)
        mn, mx = np.min(img), np.max(img)
        img = (img - mn) / (mx - mn + 1e-8)

        m = mask.astype(np.float32) / 255.0
        m = (m >= 0.5).astype(np.float32)  # {0,1}
        return img, m


# ---------- Normalization (standard: torch tensor in/out) ----------
class NormalizeTorch:
    """Normalize torch image tensor (C,H,W) using per-channel mean/std.
    Assumes image already in [0,1].
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1).float()
        self.std = torch.tensor(std).view(-1, 1, 1).float()

    def __call__(self, data):
        image, mask = data
        device = image.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        image = (image - mean) / (std + 1e-8)

        # ensure mask is {0,1}
        if mask.max() > 1.0:
            mask = (mask / 255.0 >= 0.5).float()
        else:
            mask = (mask >= 0.5).float()
        return image, mask


def _get_norm_stats(cfg, phase: str):
    """Get scalar mean/std for normalization."""
    stats = cfg.dataset.norm.stats.train if phase == "train" else cfg.dataset.norm.stats.val
    return float(stats.mean), float(stats.std)

def build_transforms(cfg:DictConfig, phase: str):
    """
    Returns a paired transform pipeline:
      - vm_mamba: CustomNormaization (numpy) -> ToTensorNoScale -> resize -> aug
      - standard_train: ToTensorScale255 -> resize -> aug -> NormalizeTorch (train stats for all splits)
      - imagenet: same but ImageNet stats
      - none: ToTensorScale255 -> resize -> aug (mask binarized)
    """
    aug = cfg.dataset.augment
    do_aug = (phase == "train") and bool(aug.enabled)

    # Common geometric transforms (operate on torch tensors)
    geom = [ResizePair(cfg.dataset.image_size)]
    if do_aug:
        geom += [
            RandomHFlip(aug.hflip),
            RandomVFlip(aug.vflip),
            RandomRotation(aug.rotate_p, degree=tuple(aug.rotate_deg)),
        ]

    mean, std = _get_norm_stats(cfg, phase)
    return PairCompose([
        CustomNormaization(mean=mean, std=std),
        ToTensorNoScale(),
        *geom,
    ])
