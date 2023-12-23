import torch
from tqdm import tqdm
from dataloaders.DynamicSegmentationDataLoader import DynamicSegmentationDataLoader, DynamicSegmentationStrategy
from utils.opencv_segmentation import get_boxes, plot_images_with_gt_and_pred, select_best_box
from utils.utils import approximate_bounding_box_to_square, get_bounding_boxes_from_segmentation
import random
import numpy as np
import os
device = torch.device("cpu")

# Configurations
DATASET_LIMIT = None
NORMALIZE = False
BALANCE_DOWNSAMPLING = 1
BATCH_SIZE = 128
RANDOM_SEED = 42


# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
# print('Using device: %s' % device)


RESUME = False
FROM_EPOCH = 0


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


def eval_boxes():
    dataloader = DynamicSegmentationDataLoader(
        dynamic_load=True, train=False, segmentation_strategy=DynamicSegmentationStrategy.OPENCV)
    train_loader = dataloader.get_train_dataloder()
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


if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    eval_boxes()
