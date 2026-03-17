from typing import Optional, Callable

import torch
import matplotlib.pyplot as plt

def single_prediction(subtitle: str, images: torch.Tensor, masks: torch.Tensor, preds: torch.Tensor, sample_size: Optional[int]) -> plt.Figure:
    """
    Render a single [image | mask | predicted] triplet as a 1x3 figure.
    """
    images = images.numpy().transpose((0, 2, 3, 1))
    masks = masks.numpy()
    preds = preds.numpy()

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(subtitle, fontsize=10)
    ax[0].imshow(images[0], cmap='gray')
    ax[0].set_title("Image", fontsize=8)
    ax[0].axis('off')
    ax[1].imshow(masks[0].squeeze(), cmap='gray')
    ax[1].set_title("Mask", fontsize=8)
    ax[1].axis('off')
    ax[2].imshow(preds[0].squeeze(), cmap='gray')
    ax[2].set_title("Predicted mask", fontsize=8)
    ax[2].axis('off')
    return fig

def multiple_predictions(subtitle: str, images: torch.Tensor, masks: torch.Tensor, preds: torch.Tensor, sample_size: int) -> plt.Figure:
    """
     Render a grid of `sample_size` rows and 3 columns: [image | mask | predicted].
    """
    images = images.numpy().transpose((0, 2, 3, 1))
    masks = masks.numpy()
    preds = preds.numpy()

    fig, ax = plt.subplots(sample_size, 3, figsize=(11, 7))
    fig.suptitle(subtitle, fontsize=10)
    for i in range(sample_size):
        ax[i, 0].imshow(images[i], cmap='gray')
        ax[i, 0].set_title("Image", fontsize=8)
        ax[i, 0].axis('off')
        ax[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        ax[i, 1].set_title("Mask", fontsize=8)
        ax[i, 1].axis('off')
        ax[i, 2].imshow(preds[i].squeeze(), cmap='gray')
        ax[i, 2].set_title("Predicted mask", fontsize=8)
        ax[i, 2].axis('off')
    return fig

def choose_visualizer(sample_size: int) -> Callable[..., plt.Figure]:
    """
    Return the appropriate visualization function based on sample_size.
    """
    if sample_size == 1:
        return single_prediction
    else:
        return multiple_predictions
