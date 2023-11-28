import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm


# def find_bounding_box(segmentation_mask):
#     # Find the indices of the white pixels
#     rows, cols = np.where(segmentation_mask > 0)
#     # Get the coordinates of the bounding box
#     top = np.min(rows)
#     left = np.min(cols)
#     bottom = np.max(rows)
#     right = np.max(cols)
#     return top, left, bottom, right
def find_bounding_box(segmentation_mask):
    # Find the indices of the white pixels
    rows, cols = np.where(segmentation_mask > 0)
    # Get the coordinates of the bounding box
    top = np.min(rows)
    left = np.min(cols)
    bottom = np.max(rows)
    right = np.max(cols)
    # Calculate the center of the bounding box
    center_y = (top + bottom) // 2
    center_x = (left + right) // 2
    # Calculate the half-length of the side of the square
    half_length = max(bottom - top, right - left) // 2
    # Calculate the top, bottom, left, and right of the square
    # Ensure the coordinates are not out of bounds
    square_top = max(0, center_y - half_length)
    square_left = max(0, center_x - half_length)
    square_bottom = min(segmentation_mask.shape[0], center_y + half_length)
    square_right = min(segmentation_mask.shape[1], center_x + half_length)
    return square_top, square_left, square_bottom, square_right


def draw_bounding_box(segmentation_mask):
    top, left, bottom, right = find_bounding_box(segmentation_mask)
    # Create a copy of the segmentation mask
    segmentation_mask_with_bounding_box = segmentation_mask.copy()
    # Draw the bounding box on the segmentation mask
    segmentation_mask_with_bounding_box[top:bottom+1, left:right+1] = 255
    return segmentation_mask_with_bounding_box


def draw_empty_bounding_box(segmentation_mask):
    top, left, bottom, right = find_bounding_box(segmentation_mask)
    # Create a copy of the segmentation mask
    segmentation_mask_with_bounding_box = segmentation_mask.copy()
    # Draw the bounding box on the segmentation mask
    segmentation_mask_with_bounding_box[top:bottom+1, left] = 255
    segmentation_mask_with_bounding_box[top:bottom+1, right] = 255
    segmentation_mask_with_bounding_box[top, left:right+1] = 255
    segmentation_mask_with_bounding_box[bottom, left:right+1] = 255
    return segmentation_mask_with_bounding_box


def main():
    segmentation_dir = "./data/HAM10000_segmentations_lesion_tschandl"
    new_segmentation_dir = "./data/HAM10000_segmentations_lesion_tschandl_with_bounding_box"
    if os.path.exists(new_segmentation_dir):
        shutil.rmtree(new_segmentation_dir)
    os.makedirs(new_segmentation_dir, exist_ok=True)
    for file in tqdm(os.listdir(segmentation_dir), desc="Drawing bounding boxes"):
        if file.endswith(".png"):
            segmentation_mask = cv2.imread(os.path.join(
                segmentation_dir, file), cv2.IMREAD_GRAYSCALE)
            segmentation_mask_with_bounding_box = draw_bounding_box(
                segmentation_mask)
            cv2.imwrite(os.path.join(new_segmentation_dir, file),
                        segmentation_mask_with_bounding_box)


if __name__ == "__main__":
    main()
