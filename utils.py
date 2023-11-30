from typing import Tuple
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
from PIL import Image


def plot_image_grid(inp: torch.Tensor, title=None):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inp)
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_image(inp: torch.Tensor or np.ndarray, title=None):
    """Imshow for Tensor."""
    if isinstance(inp, torch.Tensor):
        inp = inp.permute(1, 2, 0).numpy()
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# def center_crop(image: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
#     if size[0] > image.shape[0] or size[1] > image.shape[1]:
#         image = cv2.resize(image.numpy(), (size[0] * 3, size[1] * 3))
#         image = torch.from_numpy(image)
#     assert image.ndim == 3, f"Input must be a 3D array. Input shape is {image.shape}"
#     assert image.shape[2] == 3, "Input must have 3 channels (RGB image)"
#     assert size[0] <= image.shape[0] and size[1] <= image.shape[
#         1], f"Crop size must be smaller than input size. Current input size is {image.shape} and crop size is {size}"

#     h, w = image.shape[0], image.shape[1]
#     top = (h - size[0]) // 2
#     left = (w - size[1]) // 2
#     bottom = top + size[0]
#     right = left + size[1]

#     cropped_image = image[top:bottom, left:right, :]

#     return cropped_image.numpy()


def crop_roi(images: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if images.dim() == 3:
        images = images.unsqueeze(0)
    assert images.dim(
    ) == 4, f"Input must be a 4D tensor. Input shape is {images.shape}"
    # assert images.shape[1:] == (
    # 3, size[0], size[1]), "Input must be a 4D tensor of shape (N, 3, H, W)"
    batch_images = np.array([(image_tensor.permute(
        1, 2, 0) * 255).numpy().astype(np.uint8) for image_tensor in images])

    cropped_images = []
    for image_array in batch_images:
        # Convert image to grayscale
        image_gray = cv2.cvtColor(
            image_array, cv2.COLOR_BGR2GRAY)

        # plot_image_grid(torch.from_numpy(image_gray))

        ret, thresh = cv2.threshold(image_gray, 0, 1, cv2.THRESH_BINARY)

        # plot_image_grid(torch.from_numpy(thresh.astype(np.uint8)))

        # Find contours in the edge image
        contours, _ = cv2.findContours(
            thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If there are no contours, save the image as it is
        if len(contours) == 0:
            cropped_images.append(torch.from_numpy(
                image_array).permute(2, 0, 1))
            print("No contours found")
            continue

        # Find the contour with the maximum area (assuming it corresponds to the item)
        max_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(max_contour)

        # Crop the image using the bounding box
        cropped_image = image_array[y:y + h, x:x + w]
        # cropped_image = center_crop(
        #     torch.from_numpy(cropped_image), (150, 150))
        cropped_image = cv2.resize(cropped_image, size)
        cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1)
        cropped_images.append(cropped_image / 255)

    cropped_images = torch.stack(cropped_images, dim=0)
    return cropped_images


def zoom_out(image: torch.Tensor or np.ndarray, size=(700, 700)) -> torch.Tensor:
    # Create a new black image
    if image.shape[-1] != 3:
        # image = image.permute(2, 0, 1)
        image = image.permute(1, 2, 0).numpy()
    new_image = np.zeros((size[0], size[1], 3))

    # Calculate the position to paste the original image
    y_offset = (size[0] - image.shape[0]) // 2
    x_offset = (size[1] - image.shape[1]) // 2

    # Paste the original image into the new image
    new_image[y_offset:y_offset+image.shape[0],
              x_offset:x_offset+image.shape[1]] = image
    new_image = torch.from_numpy(new_image)
    new_image = new_image.permute(2, 0, 1)
    return new_image
