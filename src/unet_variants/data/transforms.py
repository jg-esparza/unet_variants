
from __future__ import annotations

import random
import numpy as np
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
class ToTensorScale255:
    """Convert numpy uint8 image/mask to torch float tensors.
    Image scaled to [0,1]. Mask kept in [0,255] unless binarized earlier.
    """
    def __call__(self, data):
        image, mask = data
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return image, mask


class ToTensorNoScale:
    """Use when image is already in [0,1] (e.g., after VM-Mamba normalization)."""
    def __call__(self, data):
        image, mask = data
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return image, mask


# ---------- Resize & Aug (mask always NEAREST) ----------
class ResizePair:
    def __init__(self, size: int):
        self.size = [int(size), int(size)]

    def __call__(self, data):
        image, mask = data
        image = F.resize(image, self.size, antialias=True)
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST, antialias=False)
        return image, mask


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
            angle = random.uniform(self.degree[0], self.degree[1])  # per sample ✅
            image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
            mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
        return image, mask


# ---------- Normalization (VM-Mamba style: numpy in, numpy out) ----------
class VmMambaNormalization:
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


def _get_vm_stats(cfg, phase: str):
    """Get scalar mean/std for vm_mamba mode."""
    # optional best practice switch: use train stats everywhere
    use_train = bool(getattr(cfg.data.norm, "use_train_stats_for_all", False))
    split = "train" if (phase == "train" or use_train) else "val"
    stats = cfg.data.norm.stats.train if split == "train" else cfg.data.norm.stats.val
    return float(stats.mean), float(stats.std)


def build_transforms(cfg, phase: str):
    """
    Returns a paired transform pipeline according to cfg.data.norm.mode:
      - vm_mamba: VmMambaNormalization (numpy) -> ToTensorNoScale -> resize -> aug
      - standard_train: ToTensorScale255 -> resize -> aug -> NormalizeTorch (train stats for all splits)
      - imagenet: same but ImageNet stats
      - none: ToTensorScale255 -> resize -> aug (mask binarized)
    """
    mode = str(cfg.data.norm.mode).lower()

    aug = cfg.data.augment
    do_aug = (phase == "train") and bool(aug.enabled)

    # Common geometric transforms (operate on torch tensors)
    geom = [ResizePair(cfg.data.image_size)]
    if do_aug:
        geom += [
            RandomHFlip(aug.hflip),
            RandomVFlip(aug.vflip),
            RandomRotation(aug.rotate_p, degree=tuple(aug.rotate_deg)),
        ]

    if mode == "vmunet":
        mean, std = _get_vm_stats(cfg, phase)
        return PairCompose([
            VmMambaNormalization(mean=mean, std=std),  # numpy -> numpy in [0,1]
            ToTensorNoScale(),                         # numpy [0,1] -> tensor [0,1]
            *geom,
        ])

    if mode == "imagenet":
        # imagenet_mean = [0.485, 0.456, 0.406]
        # imagenet_std  = [0.229, 0.224, 0.225]
        imagenet_mean = list(cfg.data.norm.imagenet_stats.mean)
        imagenet_std = list(cfg.data.norm.imagenet_stats.std)
        return PairCompose([
            ToTensorScale255(),                        # numpy uint8 -> tensor [0,1]
            *geom,
            NormalizeTorch(imagenet_mean, imagenet_std),
        ])

    if mode == "standard_train":
        # best practice: use TRAIN stats for all splits
        mean = list(cfg.data.norm.train_stats_channel.mean)
        std = list(cfg.data.norm.train_stats_channel.std)
        return PairCompose([
            ToTensorScale255(),
            *geom,
            NormalizeTorch(mean, std),
        ])

    if mode == "none":
        return PairCompose([
            ToTensorScale255(),
            *geom,
            # ensure mask is binary
            NormalizeTorch([0, 0, 0], [1, 1, 1]),  # cheap trick to binarize mask without changing image
        ])

    raise ValueError(f"Unknown normalization mode: {mode}")
