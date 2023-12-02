import cv2
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np
import random
import os

from dataloaders.detection_dataloaders import create_dataloaders
from tqdm import tqdm

from utils.utils import approximate_bounding_box_to_square, crop_image_from_box, get_bounding_boxes_from_segmentation, shift_boxes, zoom_out
import numpy as np
import matplotlib.pyplot as plt

# Configurations
DATASET_LIMIT = None
NORMALIZE = False
BALANCE_UNDERSAMPLING = 1
BATCH_SIZE = 128

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
        squared_boxes = []
        for boxes in tr_outputs:
            squared_boxes.append([approximate_bounding_box_to_square(
                box) for box in boxes])
        tr_outputs = select_best_box(squared_boxes)
        gt_boxes = []
        for segmentation in tr_segmentations:
            gt_boxes.append(get_bounding_boxes_from_segmentation(
                segmentation)[0])
        # gt_boxes = shift_boxes(gt_boxes, w_shift=600 - 700, h_shift=450 - 700)
        # plot_images_with_gt_and_pred(tr_images, gt_boxes, tr_outputs)

        def calculate_iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2

            # Calculate the (x, y)-coordinates of the intersection rectangle
            xA = max(x1, x2)
            yA = max(y1, y2)
            xB = min(x1 + w1, x2 + w2)
            yB = min(y1 + h1, y2 + h2)

            # Compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            # Compute the area of both the prediction and ground-truth rectangles
            box1Area = w1 * h1
            box2Area = w2 * h2

            # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
            iou = interArea / float(box1Area + box2Area - interArea)

            return iou

        ious = []
        for box, gt_box in tqdm(zip(tr_outputs, gt_boxes), desc="Calculating IoU between gt and preds"):
            ious.append(calculate_iou(box, gt_box))
        print(f"Mean IoU is {sum(ious) / len(ious)}")
        if sum(ious) / len(ious) < 0.77:
            plot_images_with_gt_and_pred(tr_images, gt_boxes, tr_outputs)


def bounding_box_pipeline(images):
    outputs = get_boxes(images)
    squared_boxes = []
    for boxes in outputs:
        squared_boxes.append([approximate_bounding_box_to_square(
            box) for box in boxes])
    outputs = select_best_box(squared_boxes)
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
    for i, image in enumerate(batched_images):
        image = (image.permute(2, 1, 0) * 255).numpy().astype(np.uint8)
        # Convert the image to grayscale

        def increase_contrast(image):
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)

            # Applying CLAHE to L-channel
            # feel free to try different values for the limit and grid size:
            clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(80, 80))
            cl = clahe.apply(l_channel)

            # merge the CLAHE enhanced L-channel with the a and b channel
            limg = cv2.merge((cl, a, b))

            # Converting image from LAB Color model to BGR color spcae
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced_img

        def gamma_correction(image):
            # Convert the image to float32 to prevent overflow or underflow
            image = image.astype(np.float32) / 255.0

            # Apply gamma correction with gamma > 1 to make darker colors darker and lighter colors less light
            gamma = 0.5
            image = cv2.pow(image, gamma)

            # Convert the image back to uint8
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

            return image
        # image = gamma_correction(image)
        image = increase_contrast(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # plt.imshow(gray)
        # plt.savefig(f"boxes_outputs/contrast_img_{i}.png")
        for _ in range(15):
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # plt.imshow(blurred)
        # plt.savefig(f"boxes_outputs/blurred_img_{i}.png")

        # Apply a threshold
        _, thresh = cv2.threshold(
            gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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


def plot_images_with_gt_and_pred(images, gt_boxes, pred_boxes):
    new_images = []
    gt_boxes = shift_boxes(gt_boxes, w_shift=700-600, h_shift=700-450)
    pred_boxes = shift_boxes(pred_boxes, w_shift=700-600, h_shift=700-450)

    for image, gt_box, pred_box in zip(images, gt_boxes, pred_boxes):
        image = zoom_out(image)
        image = (image.permute(2, 1, 0) * 255).numpy().astype(np.uint8).copy()

        # Plot ground truth (gt) box
        gt_box_0 = gt_box[0]
        gt_box_1 = gt_box[1]
        gt_box_2 = gt_box[2]
        gt_box_3 = gt_box[3]
        image_with_gt_box = cv2.rectangle(
            image.copy(), (gt_box_0, gt_box_1), (gt_box_2, gt_box_3), (0, 255, 0), 2)
        cv2.putText(image_with_gt_box, 'Ground Truth', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        # Plot predicted (pred) box
        pred_box_0 = pred_box[0]
        pred_box_1 = pred_box[1]
        pred_box_2 = pred_box[2]
        pred_box_3 = pred_box[3]
        image_with_pred_box = cv2.rectangle(
            image.copy(), (pred_box_0, pred_box_1), (pred_box_2, pred_box_3), (255, 0, 0), 2)
        cv2.putText(image_with_pred_box, 'Prediction', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # Create a combined image with gt box on the left and pred box on the right
        combined_image = np.concatenate(
            (image_with_gt_box, image_with_pred_box), axis=1)
        new_images.append(combined_image)

    for i, image in enumerate(new_images):
        plt.imshow(image)
        plt.savefig(f"boxes_outputs/gt_pred_boxes{i}.png")


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


# def get_larger_bounding_box(bounding_boxes):
#     largest_boxes = []
#     for boxes in bounding_boxes:
#         # Calculate the area of each bounding box
#         # print(f"First bounding box is: {boxes[0]}")
#         areas = [abs((box[0] - box[2]) * (box[3] - box[1]))
#                  for box in boxes]

#         # Find the index of the bounding box with the largest area
#         largest_index = np.argmax(areas)

#         # print(f"areas are {areas}")
#         # print(f"Largest indexes are {largest_index}")

#         # Return the largest bounding box
#         largest_box = boxes[largest_index]
#         largest_boxes.append(largest_box)
#     return largest_boxes

def crop_black_borders(image):
    cs_image = (image.permute(2, 1, 0) * 255).numpy().astype(np.uint8)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(cs_image, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to create a binary mask
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find the contours of the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (which corresponds to the black borders)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to remove the black borders
    cropped_image = image[:, y:y+h, x:x+w]

    return cropped_image


def select_best_box(bounding_boxes, image_shape=(600, 450)):
    closest_boxes = []
    for boxes in bounding_boxes:
        # Calculate the area of each bounding box
        MIN_AREA = 300
        areas = [abs((box[2] - box[0]) * (box[3] - box[1])) for box in boxes]

        # Get the indices of the boxes in descending order of their areas
        sorted_indices = np.argsort(areas)[::-1]

        # Select the top 3 boxes
        top_boxes = [boxes[i]
                     for i in sorted_indices[:3] if areas[i] > MIN_AREA]

        # Calculate the center of the image
        image_center = [image_shape[1] / 2, image_shape[0] / 2]

        # Calculate the center of each bounding box
        box_centers = [[(box[2] + box[0]) / 2, (box[3] + box[1]) / 2]
                       for box in top_boxes]

        # Calculate the distance between the center of each bounding box and the center of the image
        distances = [np.sqrt((box_center[0] - image_center[0]) ** 2 +
                             (box_center[1] - image_center[1]) ** 2) for box_center in box_centers]

        # Find the index of the bounding box with the smallest distance
        closest_index = np.argmin(distances)

        # Return the largest bounding box that is closest to the center of the image
        closest_box = top_boxes[closest_index]
        closest_boxes.append(closest_box)
    return closest_boxes


def get_bbox_closer_to_mean_area(bounding_boxes):
    MEAN_AREA = 119005
    closest_boxes = []
    for boxes in bounding_boxes:
        # Calculate the area of each bounding box
        areas = [abs((box[0] - box[2]) * (box[3] - box[1]))
                 for box in boxes]

        # Find the index of the bounding box with the largest area
        closest_index = np.argmin([abs(area - MEAN_AREA) for area in areas])

        # print(f"areas are {areas}")
        # print(f"Largest indexes are {largest_index}")

        # Return the largest bounding box
        closest_box = boxes[closest_index]
        closest_boxes.append(closest_box)
    return closest_boxes


def main():
    set_seed(RANDOM_SEED)
    test_boxes()


if __name__ == "__main__":
    main()
