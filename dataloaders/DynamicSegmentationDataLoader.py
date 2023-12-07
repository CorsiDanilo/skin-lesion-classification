from enum import Enum
from config import BATCH_SIZE, IMAGE_SIZE, KEEP_BACKGROUND, NORMALIZE
from dataloaders.DataLoader import DataLoader
from typing import Optional
import torch

from typing import Optional

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
from models.SAM import SAM
from shared.enums import DynamicSegmentationStrategy
from utils.opencv_segmentation import bounding_box_pipeline
from torchvision.transforms import functional as TF
from utils.utils import approximate_bounding_box_to_square, crop_image_from_box, get_bounding_boxes_from_segmentation, resize_images, zoom_out


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
                 batch_size: int = BATCH_SIZE,
                 keep_background: Optional[bool] = KEEP_BACKGROUND):
        super().__init__(limit=limit,
                         transform=transform,
                         dynamic_load=dynamic_load,
                         upscale_train=upscale_train,
                         normalize=normalize,
                         batch_size=batch_size)
        self.segmentation_strategy = segmentation_strategy
        self.segmentation_transform = transforms.Compose([
            transforms.ToTensor()
        ])
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

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int, transform: transforms.Compose = None):
        img = metadata.iloc[idx]
        # Augment the data if balance_data is true and load segmentations
        label = img['label']
        segmentation_available = "segmentation_path" in img

        if not segmentation_available:
            image = Image.open(img['image_path'])
            image = TF.to_tensor(image)
            if self.segmentation_strategy == DynamicSegmentationStrategy.OPENCV.value:
                segmented_image = bounding_box_pipeline(
                    image.unsqueeze(0)).squeeze(0)
            elif self.segmentation_strategy == DynamicSegmentationStrategy.SAM.value:
                segmented_image = self.sam_segmentation_pipeline(
                    image).squeeze(0)
            else:
                raise NotImplementedError(
                    f"Dynamic segmentation strategy {self.segmentation_strategy} not implemented")
            return segmented_image, label

        ti, ts, ts_bbox = Image.open(img['image_path']), Image.open(
            img['segmentation_path']).convert('1'), Image.open(img['segmentation_bbox_path']).convert('1')
        ti, ts, ts_bbox = TF.to_tensor(ti), TF.to_tensor(ts), TF.to_tensor(
            ts_bbox)
        if img["augmented"]:
            if not self.keep_background:
                ti = ti * ts
            pil_image = TF.to_pil_image(ti)
            image = self.transform(pil_image)
            image = image * ts_bbox
        else:
            image = ti
            if not self.keep_background:
                image = ti * ts
            image = image * ts_bbox

        bbox = get_bounding_boxes_from_segmentation(ts_bbox)[0]
        image = crop_image_from_box(image, bbox)
        image = torch.from_numpy(image).permute(2, 0, 1)
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

    def sam_segmentation_pipeline(self, images: torch.Tensor):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        THRESHOLD = 0.5
        images = resize_images(images, new_size=(
            self.sam_model.get_img_size(), self.sam_model.get_img_size()))

        upscaled_masks = self.sam_model(images)
        binary_masks = torch.sigmoid(upscaled_masks)
        binary_masks = (binary_masks > THRESHOLD).float()

        if not self.keep_background:
            images = binary_masks * images

        bboxes = [get_bounding_boxes_from_segmentation(
            mask)[0] for mask in binary_masks]

        cropped_images = []
        for image, bbox in zip(images, bboxes):
            bbox = approximate_bounding_box_to_square(bbox)
            cropped_image = crop_image_from_box(image, bbox)
            cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1)
            cropped_images.append(cropped_image)

        cropped_images = torch.stack(cropped_images)
        cropped_image = resize_images(
            cropped_images, new_size=IMAGE_SIZE)
        return cropped_images
