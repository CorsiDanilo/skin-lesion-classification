import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from config import DATA_DIR, DATASET_TRAIN_DIR, METADATA_TRAIN_DIR
from models.GradCAM import GradCAM


def main():
    threshold = 110
    cam_instance = GradCAM()
    for image in tqdm(os.listdir(DATASET_TRAIN_DIR)):
        image_path = os.path.join(DATASET_TRAIN_DIR, image)
        _, cropped_image, _ = cam_instance.generate_cam(image_path, threshold)
        cropped_image = cropped_image.permute(1, 2, 0)
        os.makedirs(os.path.join(
            DATA_DIR, f"gradcam_output_{threshold}"), exist_ok=True)
        output_dir = os.path.join(DATA_DIR, "gradcam_output", image)
        cropped_image = (cropped_image - cropped_image.min()) / \
            (cropped_image.max() - cropped_image.min())

        Image.fromarray((cropped_image.cpu().numpy() * 255).astype(np.uint8)).save(
            output_dir, format="PNG")


if __name__ == "__main__":
    main()
