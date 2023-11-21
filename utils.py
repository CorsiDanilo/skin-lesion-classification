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


def crop_roi(images: torch.Tensor) -> torch.Tensor:
    assert images.dim(
    ) == 4, f"Input must be a 4D tensor. Input shape is {images.shape}"
    assert images.shape[1:] == (
        3, 224, 224), "Input must be a 4D tensor of shape (N, 3, 224, 224)"
    batch_images = np.array([image_tensor.permute(1, 2, 0).numpy()
                            for image_tensor in images])
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                torchvision.transforms.ToTensor()])

    cropped_images = []
    for image_array in batch_images:
        # Convert image to grayscale
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

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
            continue

        # Find the contour with the maximum area (assuming it corresponds to the item)
        max_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(max_contour)

        # Crop the image using the bounding box
        cropped_image = image_array[y:y + h, x:x + w]
        cropped_image = Image.fromarray((cropped_image * 255).astype(np.uint8))
        cropped_image = transform(cropped_image)
        cropped_images.append(cropped_image)

    cropped_images = torch.stack(cropped_images, dim=0)
    return cropped_images
