from __future__ import annotations

class EarlyStopping:
    def __init__(self, cfg):
        """
        Simple early stopping utility for monitoring validation loss.

        The stopper tracks the best (lowest) validation loss observed so far and
        requests training to stop when there has been no improvement for a number
        of consecutive epochs defined by `patience`.

        Improvement is defined as:
            current_val_loss < best_val_loss - min_delta
        """
        self.patience = cfg.patience
        self.min_delta =  float(cfg.min_delta)
        self.verbose = cfg.verbose

        self.best_loss = float("inf")
        self.wait = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Update the stopper with the latest validation loss.
        """
        # Improvement if current loss is at least `min_delta` lower than best.
        improved = val_loss < (self.best_loss - self.min_delta)

        if improved:
            self.best_loss = float(val_loss)
            self.wait = 0
            return self.should_stop

        # No improvement
        self.wait += 1

        if self.wait >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"[EarlyStopping] Patience exhausted. Stopping.")
            return self.should_stop

        return self.should_stop

    def reset(self) -> None:
        """Reset the stopper's internal state."""
        self.best_loss = float("inf")
        self.wait = 0
        self.should_stop = False
