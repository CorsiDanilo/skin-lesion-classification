import pandas as pd
from config import SYNTHETIC_METADATA_TRAIN_DIR, AUGMENTED_SEGMENTATION_DIR, AUGMENTED_IMAGES_DIR
from tqdm import tqdm
import os


def main():
    synthetic_metadata = pd.read_csv(SYNTHETIC_METADATA_TRAIN_DIR)
    not_found_images = []
    not_found_segmentations = []
    for index, row in tqdm(synthetic_metadata.iterrows()):
        image_id = row['image_id']
        augmented_image_path = f'{AUGMENTED_IMAGES_DIR}/{image_id}.png'
        augmented_segmentation_path = f'{AUGMENTED_SEGMENTATION_DIR}/{image_id}_segmentation.png'
        if not os.path.exists(augmented_image_path):
            not_found_images.append(image_id)
            print(f"Image {augmented_image_path} not found.")
        if not os.path.exists(augmented_segmentation_path):
            not_found_segmentations.append(image_id)
            print(f"Segmentation {augmented_segmentation_path} not found.")

    print(f"Total images not found: {len(not_found_images)}")
    print(f"Total segmentations not found: {len(not_found_segmentations)}")

    # Remove the not found segmentations from metadata
    synthetic_metadata = synthetic_metadata[~synthetic_metadata['image_id'].isin(
        not_found_segmentations)]
    synthetic_metadata.to_csv(SYNTHETIC_METADATA_TRAIN_DIR, index=False)
    # print(f"List of images not found: {not_found_images}")
    # print(f"List of segmentations not found: {not_found_segmentations}")


if __name__ == "__main__":
    main()
