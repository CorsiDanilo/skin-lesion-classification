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
from config import BATCH_SIZE, IMAGE_SIZE, NORMALIZE


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
                 # If None, no resize is performed
                 resize_dim: Optional[Tuple[int, int]] = IMAGE_SIZE,
                 upscale_train: bool = True,
                 normalize: bool = NORMALIZE,
                 normalization_statistics: tuple = None,
                 batch_size: int = BATCH_SIZE):
        super().__init__(limit=limit,
                         transform=transform,
                         dynamic_load=dynamic_load,
                         upscale_train=upscale_train,
                         normalize=normalize,
                         normalization_statistics=normalization_statistics,
                         batch_size=batch_size)
        self.resize_dim = resize_dim
        if self.resize_dim is not None:
            self.stateful_transform = StatefulTransform(
                height=resize_dim[0], width=resize_dim[1])  # TODO: check if this is needed
            self.transform = transforms.Compose([
                # TODO: check if this is needed
                transforms.Resize((resize_dim[0], resize_dim[1]),),
                transforms.ToTensor()
            ])
        else:
            self.stateful_transform = StatefulTransform()

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int, transform: transforms.Compose = None):
        img = metadata.iloc[idx]
        load_segmentations = "segmentation_path" in img
        label = img['label']
        image = Image.open(img['image_path'])
        if load_segmentations:
            segmentation = Image.open(img['segmentation_path']).convert('1')
            if img["augmented"]:
                image, segmentation = self.stateful_transform(
                    image, segmentation)
            else:
                image = self.transform(image)
                segmentation = self.transform(segmentation)
        # Only load images
        else:
            image = self.transform(image)
        if load_segmentations:
            return image, label, segmentation
        return image, label

    def load_images_and_labels(self, metadata: pd.DataFrame):
        images = []
        segmentations = []
        labels = []

        for index, (row_index, img) in tqdm(enumerate(metadata.iterrows()), desc=f'Loading images'):
            load_segmentations = "segmentation_path" in img
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

        if self.height is not None and self.width is not None:
            # Resize
            img = transforms.Resize((self.height, self.width),
                                    interpolation=Image.BILINEAR)(img)
            seg = transforms.Resize((self.height, self.width),
                                    interpolation=Image.BILINEAR)(seg)

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
