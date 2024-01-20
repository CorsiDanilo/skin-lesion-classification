import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from config import DATA_DIR, DATASET_TRAIN_DIR, METADATA_TRAIN_DIR
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
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
        batch_size=2,
        load_synthetic=False,
        return_image_name=True,
        upscale_train=True,
        shuffle_train=False  # NOTE: this is crucial to restore the process.
    )
    train_loader = dataloader.get_train_dataloder()
    val_loader = dataloader.get_val_dataloader()
    test_loader = dataloader.get_test_dataloader()

    low_threshold = 70
    high_threshold = 110

    device = select_device()
    cam_instance = GradCAM().to(device)
    data_dir = os.path.join(DATA_DIR, "offline_computed_dataset_no_synthetic")
    os.makedirs(data_dir, exist_ok=True)

    augmentation_tracking = {}

    if not os.path.exists(os.path.join(data_dir, "offline_images", "train")):
        total_images_generated = 0
    else:
        total_images_generated = sum(1 for _ in os.listdir(os.path.join(
            data_dir, "offline_images", "train")))

    print(f"Total train images generated: {total_images_generated}")

    image_generated_now = 0
    pbar = tqdm(total=len(train_loader),
                desc="Generating GradCAMs for train")
    for batch in train_loader:
        images, labels, image_ids, is_augmented_list = batch
        pbar.update(1)
        for image, label, image_id, is_augmented in zip(images, labels, image_ids, is_augmented_list):
            image_generated_now += 1
            image = image.to(device)
            if image_id not in augmentation_tracking:
                augmentation_tracking[image_id] = 0
            if is_augmented:
                augmentation_tracking[image_id] += 1
            image_id = f"{image_id}_{augmentation_tracking[image_id]}.png"

            if image_generated_now < total_images_generated:
                pbar.set_postfix_str(
                    f"Skipping...")
                continue

            if image_generated_now == total_images_generated:
                pbar.set_postfix_str(
                    f"Generating...")
                print(
                    f"Restoring with augmentation {augmentation_tracking}")

            if os.path.exists(os.path.join(data_dir, f"gradcam_{low_threshold}", "train", image_id)) and \
                    os.path.exists(os.path.join(data_dir, f"gradcam_{high_threshold}", "train", image_id)) and \
                os.path.exists(os.path.join(data_dir, "offline_images", "train", image_id)):
                continue
            save_gradcam(cam_instance, image, image_id, low_threshold,
                         save_dir=os.path.join(data_dir, f"gradcam_{low_threshold}", "train"))
            save_gradcam(cam_instance, image, image_id, high_threshold,
                         save_dir=os.path.join(data_dir, f"gradcam_{high_threshold}", "train"))
            save_image(image, image_id, save_dir=os.path.join(
                data_dir, "offline_images", "train"))
    pbar.close()

    for batch in tqdm(val_loader, desc=f"Generating GradCAMs for val"):
        images, labels, image_ids, is_augmented_list = batch
        for image, label, image_id, is_augmented in zip(images, labels, image_ids, is_augmented_list):
            image = image.to(device)
            image_id = image_id + ".png"
            if os.path.exists(os.path.join(data_dir, f"gradcam_{low_threshold}", "val", image_id)) and \
                    os.path.exists(os.path.join(data_dir, f"gradcam_{high_threshold}", "val", image_id)) and \
                os.path.exists(os.path.join(data_dir, "offline_images", "val", image_id)):
                continue
            save_gradcam(cam_instance, image, image_id, low_threshold,
                         save_dir=os.path.join(data_dir, f"gradcam_{low_threshold}", "val"))
            save_gradcam(cam_instance, image, image_id, high_threshold,
                         save_dir=os.path.join(data_dir, f"gradcam_{high_threshold}", "val"))
            save_image(image, image_id, save_dir=os.path.join(
                data_dir, "offline_images", "val"))

    for batch in tqdm(test_loader, desc=f"Generating GradCAMs for test"):
        images, labels, image_ids, is_augmented_list = batch
        for image, label, image_id, is_augmented in zip(images, labels, image_ids, is_augmented_list):
            image = image.to(device)
            image_id = image_id + ".png"
            if os.path.exists(os.path.join(data_dir, f"gradcam_{low_threshold}", "test", image_id)) and \
                    os.path.exists(os.path.join(data_dir, f"gradcam_{high_threshold}", "test", image_id)) and \
                os.path.exists(os.path.join(data_dir, "offline_images", "test", image_id)):
                continue
            save_gradcam(cam_instance, image, image_id, low_threshold,
                         save_dir=os.path.join(data_dir, f"gradcam_{low_threshold}", "test"))
            save_gradcam(cam_instance, image, image_id, high_threshold,
                         save_dir=os.path.join(data_dir, f"gradcam_{high_threshold}", "test"))
            save_image(image, image_id, save_dir=os.path.join(
                data_dir, "offline_images", "test"))


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
