import cv2
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np
import random
import os
import wandb

from detection_dataloaders import create_dataloaders
from tqdm import tqdm

from sklearn.metrics import recall_score, accuracy_score

from utilities import approximate_bounding_box_to_square, crop_image_from_box, get_bounding_boxes_from_segmentation, shift_boxes, zoom_out
import numpy as np
import matplotlib.pyplot as plt

# Configurations
DATASET_LIMIT = None
NORMALIZE = False
BALANCE_UNDERSAMPLING = 1
BATCH_SIZE = 10

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
# print('Using device: %s' % device)


RESUME = False
FROM_EPOCH = 0


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


RANDOM_SEED = 42
resnet_mean = torch.tensor([0.485, 0.456, 0.406])
resnet_std = torch.tensor([0.229, 0.224, 0.225])


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def create_loaders():
    # Load the dataset
    resnet_mean = torch.tensor([0.485, 0.456, 0.406])
    resnet_std = torch.tensor([0.229, 0.224, 0.225])

    train_loader, val_loader, test_loader = create_dataloaders(
        mean=resnet_mean,
        std=resnet_std,
        normalize=NORMALIZE,
        limit=DATASET_LIMIT,
        batch_size=BATCH_SIZE,
        dynamic_load=True)
    return train_loader, val_loader, test_loader


def save_model(model, model_name, epoch):
    torch.save(model.state_dict(), f"{model_name}_{epoch}.pt")


def test_boxes():
    train_loader, _, _ = create_loaders()
    for tr_i, (tr_images, tr_labels, tr_segmentations) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        tr_images = tr_images.to(torch.float32)
        tr_images = tr_images.to(device)
        # tr_images = torch.stack([zoom_out(img)
        # for img in tr_images], dim=0)
        # print(f"Tr_images shape is {tr_images.shape}")
        tr_outputs = get_boxes(tr_images)
        tr_outputs = get_larger_bounding_box(tr_outputs)
        tr_outputs = [approximate_bounding_box_to_square(
            box) for box in tr_outputs]
        plot_images_with_boxes(tr_images, tr_outputs)
        plot_cropped_images(tr_images, tr_outputs)


def bounding_box_pipeline(images):
    outputs = get_boxes(images)
    outputs = get_larger_bounding_box(outputs)
    outputs = [approximate_bounding_box_to_square(
        box) for box in outputs]
    boxes = shift_boxes(outputs, w_shift=700-600, h_shift=700-450)
    # print(f"Images shape is {images.shape}")
    images = [zoom_out(img) for img in images]
    # print(f"Image shape is {images[0].shape}")

    cropped_images: List[torch.Tensor] = [torch.from_numpy(crop_image_from_box(
        image, box)) for image, box in zip(images, boxes)]

    cropped_images = torch.stack(cropped_images, dim=0)
    cropped_images = cropped_images.permute(0, 3, 1, 2)
    # print(f"Cropped image shape is {cropped_images.shape}")
    return cropped_images


def parse_target(segmentation: torch.Tensor) -> Dict[torch.Tensor, torch.Tensor]:
    target = {}
    # Replace with your function to get the bounding boxes
    target['boxes'] = get_bounding_boxes_from_segmentation(segmentation)
    # Replace with your function to get the labels
    target['labels'] = [0 for _ in range(len(target['boxes']))]
    target["boxes"] = target["boxes"].to(device)
    target["labels"] = torch.tensor(
        target["labels"]).to(torch.int64).to(device)
    return target


def get_boxes(batched_images):
    bounding_boxes = []
    for image in batched_images:
        image = (image.permute(2, 1, 0) * 255).numpy().astype(np.uint8)
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply a threshold
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # print(f"Found {len(contours)} contours")

        # For each contour, calculate the bounding box and add it to the list
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(torch.tensor([x, y, x + w, y + h]))
        bounding_boxes.append(boxes)

    return bounding_boxes


def plot_images_with_boxes(images, boxes):
    new_images = []
    boxes = shift_boxes(boxes, w_shift=700-600, h_shift=700-450)
    # print(f"Boxes are {boxes}")
    for image, box in zip(images, boxes):
        image = zoom_out(image)
        image = (image.permute(2, 1, 0) * 255).numpy().astype(np.uint8).copy()
        # print(f"Box is {box}")
        box_0 = box[0]
        box_1 = box[1]
        box_2 = box[2]
        box_3 = box[3]
        image = cv2.rectangle(
            image, (box_0, box_1), (box_2, box_3), (0, 255, 0), 2)
        # print(f"Image shape after bounding box is {image.shape}")
        new_images.append(image)

    # print(f"Images are {len(images)}")
    for i, image in enumerate(new_images):
        plt.imshow(image)
        plt.savefig(f"boxes_outputs/opencv_boxes{i}.png")


def plot_cropped_images(images, boxes):
    new_images = []
    boxes = shift_boxes(boxes, w_shift=700-600, h_shift=700-450)
    for image, box in zip(images, boxes):
        image = zoom_out(image)
        # print(f"Image shape in plot_cropped_images is {image.shape}")
        cropped_image = crop_image_from_box(image, box)
        new_images.append(cropped_image)
    for i, image in enumerate(new_images):
        plt.imshow(image)
        plt.savefig(f"boxes_outputs/cropped_image{i}.png")


def get_larger_bounding_box(bounding_boxes):
    largest_boxes = []
    for boxes in bounding_boxes:
        # Calculate the area of each bounding box
        # print(f"First bounding box is: {boxes[0]}")
        areas = [abs((box[0] - box[2]) * (box[3] - box[1]))
                 for box in boxes]

        # Find the index of the bounding box with the largest area
        largest_index = np.argmax(areas)

        # print(f"areas are {areas}")
        # print(f"Largest indexes are {largest_index}")

        # Return the largest bounding box
        largest_box = boxes[largest_index]
        largest_boxes.append(largest_box)
    return largest_boxes


def main():
    set_seed(RANDOM_SEED)
    test_boxes()


if __name__ == "__main__":
    main()
