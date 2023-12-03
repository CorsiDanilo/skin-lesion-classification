import matplotlib.pyplot as plt
import os


def plot_segmentations_single_sample(epoch, tr_i, pred_mask, gt_mask, name):
    # Convert tensors to numpy arrays
    pred_mask = pred_mask.detach().numpy()
    gt_mask = gt_mask.detach().numpy()

    # Plot the images side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(pred_mask[0][0], cmap='gray')
    axs[0].set_title('Upscaled Masks')
    axs[1].imshow(gt_mask[0][0], cmap='gray')
    axs[1].set_title('tr_segmentation Masks')

    plot_dir = os.path.join('plots', 'sam_comparison_masks_plots')
    # if os.path.exists(plot_dir):
    #     shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    # Save the plot
    plt.savefig(os.path.join(
        plot_dir, f'{name}_epoch_{epoch}_step_{tr_i}_comparison.png'))
    plt.close(fig)


def plot_segmentations_batch(epoch, tr_i, pred_mask, gt_mask, name):
    # Convert tensors to numpy arrays
    pred_mask = pred_mask.detach().numpy()
    gt_mask = gt_mask.detach().numpy()

    # Get the batch size
    batch_size = pred_mask.shape[0]

    # Create a grid of subplots
    fig, axs = plt.subplots(
        batch_size, 2, figsize=(10, 5 * batch_size))

    # Iterate over each sample in the batch
    for i in range(batch_size):
        axs[i][0].imshow(pred_mask[i][0], cmap='gray')
        axs[i][0].set_title(f'Upscaled Masks - Sample {i+1}')
        axs[i][1].imshow(gt_mask[i][0], cmap='gray')
        axs[i][1].set_title(
            f'tr_segmentation Masks - Sample {i+1}')

    plot_dir = os.path.join('plots', 'sam_comparison_masks_plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(
        plot_dir, f'{name}_epoch_{epoch}_step_{tr_i}_comparison.png'))
    plt.close(fig)
