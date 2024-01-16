import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from config import DATA_DIR, DATASET_TRAIN_DIR, METADATA_TRAIN_DIR
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from dataloaders.MSLANetDataLoader import MSLANetDataLoader
from models.GradCAM import GradCAM
from shared.constants import IMAGENET_STATISTICS
from utils.utils import select_device


def generate_gradcam_from_train_dir():
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


def generate_gradcam_from_dataloader():
    dataloader = ImagesAndSegmentationDataLoader(
        limit=None,
        load_segmentations=False,
        dynamic_load=True,
        normalize=False,
        normalization_statistics=IMAGENET_STATISTICS,
        batch_size=8,
        load_synthetic=True,
        return_image_name=True
    )
    train_loader = dataloader.get_train_dataloder()
    val_loader = dataloader.get_val_dataloader()
    test_loader = dataloader.get_test_dataloader()

    low_threshold = 70
    high_threshold = 110

    device = select_device()
    cam_instance = GradCAM().to(device)

    for batch in tqdm(train_loader, desc=f"Generating GradCAMs for train"):
        images, labels, images_id = batch
        for image, label, image_id in zip(images, labels, images_id):
            image = image.to(device)
            image_id = image_id + ".png"
            save_gradcam(cam_instance, image, image_id, low_threshold,
                         save_dir=os.path.join(DATA_DIR, f"train_gradcam_{low_threshold}"))
            save_gradcam(cam_instance, image, image_id, high_threshold,
                         save_dir=os.path.join(DATA_DIR, f"train_gradcam_{high_threshold}"))
            save_image(image, image_id, save_dir=os.path.join(
                DATA_DIR, "train_images"))

    for batch in tqdm(val_loader, desc=f"Generating GradCAMs for val"):
        images, labels, images_id = batch
        for image, label, image_id in zip(images, labels, images_id):
            image = image.to(device)
            image_id = image_id + ".png"
            save_gradcam(cam_instance, image, image_id, low_threshold,
                         save_dir=os.path.join(DATA_DIR, f"val_gradcam_{low_threshold}"))
            save_gradcam(cam_instance, image, image_id, high_threshold,
                         save_dir=os.path.join(DATA_DIR, f"val_gradcam_{high_threshold}"))
            save_image(image, image_id, save_dir=os.path.join(
                DATA_DIR, "val_images"))

    for batch in tqdm(test_loader, desc=f"Generating GradCAMs for test"):
        images, labels, images_id = batch
        for image, label, image_id in zip(images, labels, images_id):
            image = image.to(device)
            image_id = image_id + ".png"
            save_gradcam(cam_instance, image, image_id, low_threshold,
                         save_dir=os.path.join(DATA_DIR, f"test_gradcam_{low_threshold}"))
            save_gradcam(cam_instance, image, image_id, high_threshold,
                         save_dir=os.path.join(DATA_DIR, f"test_gradcam_{high_threshold}"))
            save_image(image, image_id, save_dir=os.path.join(
                DATA_DIR, "test_images"))


def save_image(image: torch.Tensor, image_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    output_dir = os.path.join(save_dir, "", image_name)
    image = image.permute(1, 2, 0)
    Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8)).save(
        output_dir, format="PNG")


def save_gradcam(gradcam_instance: GradCAM, image: torch.Tensor, image_name: str, threshold: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    _, cropped_image, _ = gradcam_instance.generate_cam(
        image=image,
        threshold=threshold)
    cropped_image = cropped_image.permute(1, 2, 0)

    image_path = os.path.join(save_dir, image_name)
    Image.fromarray((cropped_image.cpu().numpy() * 255).astype(np.uint8)).save(
        image_path, format="PNG")


if __name__ == "__main__":
    generate_gradcam_from_dataloader()
