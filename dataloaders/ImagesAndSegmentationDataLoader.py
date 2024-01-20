import numpy as np
from augmentation.StatefulTransform import StatefulTransform
from augmentation.Augmentations import Augmentations
from dataloaders.DataLoader import DataLoader
from typing import Optional, Tuple
import torch

from typing import Optional

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
from config import BATCH_SIZE, IMAGE_SIZE, NORMALIZE, RANDOM_SEED
import random
from datasets.HAM10K import HAM10K

from utils.utils import calculate_normalization_statistics

random.seed(RANDOM_SEED)


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
                 batch_size: int = BATCH_SIZE,
                 load_segmentations: bool = True,
                 load_synthetic: bool = True,
                 return_image_name: bool = False,
                 shuffle_train: bool = True):
        super().__init__(limit=limit,
                         transform=transform,
                         dynamic_load=dynamic_load,
                         upscale_train=upscale_train,
                         normalize=normalize,
                         normalization_statistics=normalization_statistics,
                         batch_size=batch_size,
                         always_rotate=False,
                         load_synthetic=load_synthetic)
        self.resize_dim = resize_dim
        self.load_segmentations = load_segmentations
        self.return_image_name = return_image_name
        self.shuffle_train = shuffle_train

        assert return_image_name or load_segmentations, "Returning both image name and segmentation is still not supported"
        if self.resize_dim is not None:
            self.stateful_transform = StatefulTransform(
                height=resize_dim[0],
                width=resize_dim[1],
                always_rotate=self.always_rotate)
            self.transform = transforms.Compose([
                transforms.Resize(resize_dim,
                                  interpolation=Image.BILINEAR),
                transforms.ToTensor()
            ])
        else:
            self.stateful_transform = StatefulTransform(
                always_rotate=self.always_rotate)

        self.mslanet_transform = Augmentations(
            resize_dim=self.resize_dim).transform

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int):
        img = metadata.iloc[idx]
        load_segmentations = "train" in img and self.load_segmentations
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
            if img["augmented"]:
                image = (np.array(image)).astype(np.uint8)
                image = self.mslanet_transform(image=image)["image"] / 255
            else:
                image = self.transform(image)
        if load_segmentations:
            return image, label, segmentation

        if self.return_image_name:
            return image, label, img["image_id"], img["augmented"]
        return image, label

    def load_images_and_labels(self, metadata: pd.DataFrame):
        images = []
        segmentations = []
        labels = []

        for index, (row_index, img) in tqdm(enumerate(metadata.iterrows()), desc=f'Loading images'):
            load_segmentations = "train" in img and self.load_segmentations
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

    def get_train_dataloder(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        if self.normalize:
            print(
                "--Normalization-- Normalization flag set to True: Images will be normalized with z-score normalization")
            if self.normalization_statistics is None:
                self.normalization_statistics = calculate_normalization_statistics(
                    self.train_df)
                print(
                    "--Normalization-- Statistics not provided. They will be computed on the training set.")
            print(
                f"--Normalization-- Statistics for normalization (per channel) -> Mean: {self.normalization_statistics[0].view(-1)}, Variance: {self.normalization_statistics[1].view(-1)}, Epsilon (adjustment value): 0.01")

        train_dataset = HAM10K(
            self.train_df,
            load_data_fn=self.load_data,
            normalize=self.normalize,
            mean=self.normalization_statistics[0] if self.normalize else None,
            std=self.normalization_statistics[1] if self.normalize else None,
            balance_data=self.upscale_train,
            resize_dims=IMAGE_SIZE,
            dynamic_load=self.dynamic_load)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            pin_memory=False,
        )
        print(f"Train dataloader has shuffle train on? {self.shuffle_train}")
        return train_dataloader
