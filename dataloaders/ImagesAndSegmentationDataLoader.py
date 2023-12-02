import random
from dataloaders.DataLoader import DataLoader
from typing import Optional, Tuple
import torch

from typing import Optional

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
import torchvision.transforms.functional as TF


class ImagesAndSegmentationDataLoader(DataLoader):
    """
    This class is used to load the images and create the dataloaders.
    The dataloder will output a tuple of (images, labels, segmentations), if segmentations are available (for training and validation, not for testing).
    The images are not segmented, and they are resized only if the resize_dim parameter is set.
    """

    def __init__(self,
                 limit: Optional[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 dynamic_load: bool = False,
                 resize_dim: Optional[Tuple[int, int]] = (None, None)):
        super().__init__(limit, transform, dynamic_load)
        self.stateful_transform = StatefulTransform(
            height=resize_dim[0], width=resize_dim[1])

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int, transform: transforms.Compose = None):
        img = metadata.iloc[idx]
        load_segmentations = "segmentation_path" in img
        label = img['label']
        if load_segmentations:
            segmentation = Image.open(img['segmentation_path']).convert('1')
            image = Image.open(img['image_path'])
            image, segmentation = self.stateful_transform(image, segmentation)
        # Only load images
        else:
            image = self.transform(Image.open(img['image_path']))
        if load_segmentations:
            return image, label, segmentation
        return image, label

    def load_images_and_labels(self, metadata: pd.DataFrame):
        images = []
        segmentations = []
        labels = []
        load_segmentations = "segmentation_path" in img

        for index, (row_index, img) in tqdm(enumerate(metadata.iterrows()), desc=f'Loading images'):
            if load_segmentations:
                image, label, segmentation = self.load_images_and_labels_at_idx(
                    idx=index, metadata=metadata)
                segmentations.append(segmentation)
            else:
                image, label = self.load_images_and_labels_at_idx(
                    idx=index, metadata=metadata)
            images.append(image)
            labels.append(label)
        images = torch.stack(images)
        if load_segmentations:
            segmentations = torch.stack(segmentations)
        labels = torch.tensor(labels, dtype=torch.long)

        print(f"---Data Loader--- Images uploaded: " + str(len(images)))

        if load_segmentations:
            return images, labels, segmentations
        return images, labels


class StatefulTransform:
    def __init__(self, height: Optional[int] = None, width: Optional[int] = None):
        self.height = height
        self.width = width

    def __call__(self, img, seg):
        # Resize
        # img = transforms.Resize((self.height, self.width),
        # interpolation=Image.BILINEAR)(img)

        # Horizonal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            seg = TF.hflip(seg)

        # Vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            seg = TF.vflip(seg)

        # Random rotation
        if random.random() > 0.5:
            angle = random.randint(1, 360)
            img = TF.rotate(img, angle)
            seg = TF.rotate(seg, angle)

        img = transforms.ToTensor()(img)
        seg = transforms.ToTensor()(seg)

        return img, seg
