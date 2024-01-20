import os
from config import BATCH_SIZE, IMAGE_SIZE, KEEP_BACKGROUND, NORMALIZE
from dataloaders.DataLoader import DataLoader
from typing import Optional
import torch

from typing import Optional

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
from dataloaders.ImagesAndSegmentationDataLoader import StatefulTransform
from models.SAM import SAM
from shared.enums import DynamicSegmentationStrategy
from train_loops.SAM_pretrained import preprocess_images
from utils.opencv_segmentation import bounding_box_pipeline
from torchvision.transforms import functional as TF
from utils.utils import approximate_bounding_box_to_square, crop_image_from_box, get_bounding_boxes_from_segmentation, resize_images, resize_segmentations


class DynamicSegmentationDataLoader(DataLoader):
    """
    This class is used to load the images and create the dataloaders.
    The dataloder will output a tuple of (images, labels).
    The images are already segmented using the segmentation strategy specified in the constructor.
    """

    def __init__(self,
                 limit: Optional[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 dynamic_load: bool = False,
                 upscale_train: bool = True,
                 segmentation_strategy: DynamicSegmentationStrategy = DynamicSegmentationStrategy.OPENCV,
                 normalize: bool = NORMALIZE,
                 keep_background: Optional[bool] = KEEP_BACKGROUND,
                 normalization_statistics: tuple = None,
                 batch_size: int = BATCH_SIZE,
                 load_synthetic: bool = False):
        super().__init__(limit=limit,
                         transform=transform,
                         dynamic_load=dynamic_load,
                         upscale_train=upscale_train,
                         normalize=normalize,
                         normalization_statistics=normalization_statistics,
                         batch_size=batch_size,
                         always_rotate=False,
                         load_synthetic=load_synthetic)
        self.segmentation_strategy = segmentation_strategy
        self.load_synthetic = load_synthetic
        self.segmentation_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.stateful_transform = StatefulTransform(
            always_rotate=self.always_rotate)
        if self.segmentation_strategy == DynamicSegmentationStrategy.OPENCV.value:
            print(f"NOOOOOO, DON'T USE OPEN_CV AS A STRATEGY, IT'S DEPRECATED!! ò_ó")
        self.keep_background = keep_background
        if segmentation_strategy == DynamicSegmentationStrategy.SAM.value:
            sam_checkpoint_path = "checkpoints/sam_checkpoint.pt"
            SAM_IMG_SIZE = 128
            self.sam_model = SAM(
                custom_size=True,
                img_size=SAM_IMG_SIZE,
                checkpoint_path=sam_checkpoint_path).to(self.device)
            self.sam_model.model.eval()

            self.preprocess_params = {
                'adjust_contrast': 1.5,
                'adjust_brightness': 1.2,
                'adjust_saturation': 2,
                'adjust_gamma': 1.5,
                'gaussian_blur': 5}

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int, transform: transforms.Compose = None):
        img = metadata.iloc[idx]
        label = img['label']
        segmentation_available = img['train'] and not img["synthetic"]

        if not segmentation_available:
            image = Image.open(img['image_path'])
            image = TF.to_tensor(image)
            if self.segmentation_strategy == DynamicSegmentationStrategy.OPENCV.value:
                # NOTE: This is deprecated, use SAM instead
                segmented_image = bounding_box_pipeline(
                    image.unsqueeze(0)).squeeze(0)
            elif self.segmentation_strategy == DynamicSegmentationStrategy.SAM.value:
                segmented_image = self.sam_segmentation_pipeline(
                    image).squeeze(0)
            else:
                raise NotImplementedError(
                    f"Dynamic segmentation strategy {self.segmentation_strategy} not implemented")

            assert segmented_image.shape[-2:
                                         ] == IMAGE_SIZE, f"Image shape is {segmented_image.shape}, expected last two dimensions to be {IMAGE_SIZE}"

            return segmented_image, label

        ti, ts = Image.open(img['image_path']), Image.open(
            img['segmentation_path']).convert('1')
        # ti, ts = TF.to_tensor(ti), TF.to_tensor(ts)
        ti, ts = self.stateful_transform(ti, ts)
        if img["augmented"]:
            if not self.keep_background:
                ti *= ts
            image = ti
        else:
            image = ti
            if not self.keep_background:
                image *= ts

        image = self.crop_to_background(
            image.unsqueeze(0), ts.unsqueeze(0)).squeeze(0)

        assert image.shape[-2:
                           ] == IMAGE_SIZE, f"Image shape is {image.shape[-2:]}, expected {IMAGE_SIZE}"

        return image, label

    def load_images_and_labels(self, metadata: pd.DataFrame):
        not_found_files = []
        images = []
        labels = []
        for index, (row_index, img) in tqdm(enumerate(metadata.iterrows()), desc=f'Loading images'):
            image, label = self.load_images_and_labels_at_idx(
                idx=index, metadata=metadata)
            images.append(image)
            labels.append(label)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        print(f"---Data Loader--- Images uploaded: " + str(len(images)))

        print(
            f"Loading complete, some files ({len(not_found_files)}) were not found: {not_found_files}")
        return images, labels

    def crop_to_background(self, images: torch.Tensor,
                           segmentations: torch.Tensor,
                           resize: bool = True):
        bboxes = [get_bounding_boxes_from_segmentation(
            mask)[0] for mask in segmentations]

        cropped_images = []
        for image, bbox in zip(images, bboxes):
            bbox = approximate_bounding_box_to_square(bbox)
            cropped_image = crop_image_from_box(
                image, bbox, size=IMAGE_SIZE if resize else None)
            cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1)
            cropped_images.append(cropped_image)

        cropped_images = torch.stack(cropped_images)

        return cropped_images

    def sam_segmentation_pipeline(self, images: torch.Tensor):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        THRESHOLD = 0.5
        resized_images = resize_images(images, new_size=(
            self.sam_model.get_img_size(), self.sam_model.get_img_size())).to(self.device)

        resized_images = preprocess_images(
            resized_images, params=self.preprocess_params)

        upscaled_masks = self.sam_model(resized_images)
        binary_masks = torch.sigmoid(upscaled_masks)
        binary_masks = (binary_masks > THRESHOLD).float()
        binary_masks = resize_segmentations(
            binary_masks, new_size=(600, 450)).to(self.device)

        images = images.to(self.device)
        # images = resize_images(images, new_size=(450, 450)).to(self.device)
        if not self.keep_background:
            images = binary_masks * images

        cropped_images = self.crop_to_background(images, binary_masks)
        cropped_images = cropped_images.to(self.device)
        return cropped_images
