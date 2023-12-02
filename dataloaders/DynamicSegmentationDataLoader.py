from enum import Enum
from dataloaders.DataLoader import DataLoader
from typing import Optional
import torch

from typing import Optional

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
from utils.opencv_segmentation import bounding_box_pipeline
from torchvision.transforms import functional as TF
from utils.utils import crop_image_from_box, get_bounding_boxes_from_segmentation, zoom_out


class DynamicSegmentationStrategy(Enum):
    OPENCV = "opencv"
    SAM = "sam"


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
                 train: bool = True,
                 segmentation_strategy: DynamicSegmentationStrategy = DynamicSegmentationStrategy.OPENCV):
        super().__init__(limit, transform, dynamic_load)
        self.train = train
        self.segmentation_strategy = segmentation_strategy
        self.segmentation_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int, transform: transforms.Compose = None):
        img = metadata.iloc[idx]
        # Augment the data if balance_data is true and load segmentations
        label = img['label']
        if not self.train:
            image = Image.open(img['image_path'])
            image = TF.to_tensor(image)
            if self.segmentation_strategy == DynamicSegmentationStrategy.OPENCV:
                segmented_image = bounding_box_pipeline(
                    image.unsqueeze(0)).squeeze(0)
            else:
                raise NotImplementedError(
                    f"Dynamic segmentation strategy {self.segmentation_strategy} not implemented")
            return segmented_image, label
        # Augment the data if balance_data is true and load segmentations
        ti, ts = Image.open(img['image_path']), Image.open(
            img['segmentation_path']).convert('1')
        ti, ts = TF.to_tensor(ti), TF.to_tensor(ts)
        ti = zoom_out(ti)
        if img["augmented"]:
            # TODO: verify that the box is squared and doesn't go out of borders for the augmented images
            pil_image = TF.to_pil_image(ti)
            image = self.transform(pil_image)
            # image = image * ts
        else:
            image = ti
        bbox = get_bounding_boxes_from_segmentation(ts)[0]
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
