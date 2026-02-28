from __future__ import annotations

import torch

from omegaconf import DictConfig

class BinarySegmentationMetrics:
    """
    Streaming evaluator for **binary segmentation**.

    Accumulates confusion matrix counts across batches and exposes common metrics.

    Parameters
    ----------
    cfg : DictConfig

        Expected keys:
                - task (str): task type; must be "bin" for this class
                - threshold (float): binarization threshold in [0, 1]
        from_logits : bool, optional
            If True, apply sigmoid to predictions before thresholding. Default: True.
        ignore_index : Optional[int], optional
            Label to ignore in `target` (e.g., 255). Pixels at this index are excluded
            from counts. If None, no ignoring is performed. Default: None.
        device : Optional[torch.device], optional
            Device to keep the internal counters on. Default: CPU.

    Notes
    -----
    - Assumes predictions have shape (N, 1, H, W) and targets are (N, 1, H, W)
        or (N, H, W). Targets should be {0,1} or boolean. If not boolean, they are
        binarized with the same `threshold` applied to floats.
    - Call `reset()` before a new evaluation run, then `update(...)` per batch,
        and finally read the computed metrics.
    """
    def __init__(self, cfg: DictConfig) -> None:

        self.task = cfg.task
        self.threshold = cfg.threshold
        self.tp, self.fp, self.fn, self.tn = 0.0, 0.0, 0.0, 0.0

    # -------- public API --------

    def reset(self) -> None:
        """Reset running confusion counts."""
        self.tp, self.fp, self.fn, self.tn = 0.0, 0.0, 0.0, 0.0

    def update(self, pred, target):
        """
        Update confusion counts from a batch.

        Args:
            pred: Model outputs; logits or probabilities.
                Shape: (N, 1, H, W).
            target: Ground truth binary masks in {0,1} (int/bool/float),
                Shape: (N, 1, H, W).
        """
        if self.task == "bin":
            tp_batch, fp_batch, fn_batch, tn_batch = self._get_binary_stats(pred, target)
            self.tp += tp_batch
            self.fp += fp_batch
            self.fn += fn_batch
            self.tn += tn_batch
        else:
            raise NotImplementedError

    # -------- metrics --------

    def compute_f1_score(self):
        return (2 * self.tp) / ((2 * self.tp) + self.fn + self.fp) if ((2 * self.tp) + self.fn + self.fp) != 0 else 0

    def compute_iou_score(self):
        return self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) != 0 else 0

    def compute_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn) if (self.tp + self.fp + self.fn + self.tn) != 0 else 0

    def compute_sensitivity(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) != 0 else 0

    def compute_specificity(self):
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) != 0 else 0

    # -------- helpers --------

    def binarize(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.where(tensor >= self.threshold, 1, 0)

    def _get_binary_stats(self, pred, target):
        pred = self.binarize(pred)
        target = self.binarize(target)

        b, c, *dims = target.shape
        pred = pred.view(b, c, -1)
        target = target.view(b, c, -1)

        tp = (pred * target).sum(2)
        fp = pred.sum(2) - tp
        fn = target.sum(2) - tp
        tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)
        return tp.sum(), fp.sum(), fn.sum(), tn.sum()