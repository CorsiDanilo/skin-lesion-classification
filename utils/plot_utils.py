from typing import Optional
import matplotlib.pyplot as plt
import os
import torchvision
import torch
import numpy as np


def plot_image_grid(inp: torch.Tensor, title=None):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inp)
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_image(inp: torch.Tensor or np.ndarray, title=None):
    """Imshow for Tensor."""
    if isinstance(inp, torch.Tensor):
        inp = inp.permute(1, 2, 0).cpu().numpy()
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_segmentations_batch(epoch: int,
                             tr_i: int,
                             images: torch.Tensor,
                             upscaled_mask: torch.Tensor,
                             pred_bin_mask: torch.Tensor,
                             gt_mask: torch.Tensor,
                             boxes: Optional[torch.Tensor],
                             name: str):
    # Convert tensors to numpy arrays
    upscaled_mask = upscaled_mask.detach().cpu().numpy()
    pred_bin_mask = pred_bin_mask.detach().cpu().numpy()
    gt_mask = gt_mask.detach().cpu().numpy()
    images = images.detach().cpu().numpy()
    if boxes is not None:
        boxes = boxes.detach().cpu().numpy()
    # Get the batch size
    batch_size = upscaled_mask.shape[0]

    # Create a grid of subplots
    fig, axs = plt.subplots(
        batch_size, 4, figsize=(10, 5 * batch_size))

    # Iterate over each sample in the batch
    for i in range(batch_size):
        axs[i][0].imshow(upscaled_mask[i][0], cmap='gray')
        axs[i][0].set_title(f'Upscaled Masks')
        axs[i][1].imshow(pred_bin_mask[i][0], cmap='gray')
        axs[i][1].set_title(
            f'Pred Bin Masks')
        axs[i][2].imshow(gt_mask[i][0], cmap='gray')
        axs[i][2].set_title(
            f'GT Bin Masks')
        axs[i][3].imshow(images[i][0])
        if boxes is not None:
            box = boxes[i]
            rect = plt.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red')
            axs[i][3].add_patch(rect)
            plt.axis('off')
            axs[i][3].set_title(
                f'Image w/ box')

    plot_dir = os.path.join('plots', 'sam_comparison_masks_plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(
        plot_dir, f'{name}_epoch_{epoch}_step_{tr_i}_comparison.png'))
    plt.close(fig)
