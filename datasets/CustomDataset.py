from abc import ABC
from typing import Callable, Optional, Tuple
import pandas as pd
from torch.utils.data import Dataset
from config import BALANCE_UNDERSAMPLING
import torch

from utils.utils import select_device


class CustomDataset(Dataset, ABC):
    """
    Abstract Class that defines the custom dataset for the project.
    """

    def __init__(
            self,
            metadata: pd.DataFrame,
            load_data_fn: Callable,
        # Control the data augmentation process aim to solve class imbalance
        balance_data: bool = True,
        # Percentage of data to keep from the majority class
        balance_undersampling: float = BALANCE_UNDERSAMPLING,
        normalize: bool = False,  # Control the application of z-score normalization
        # Mean (per channel) for the z-score normalization
        mean: Optional[torch.Tensor] = None,
        # Standard deviation (per channel) for the z-score normalization
        std: Optional[torch.Tensor] = None,
        # Adjustment value to avoid division per zero during normalization
        std_epsilon: float = 0.01,
        # Sizes (height, width) for resize the images
        resize_dims: Tuple[int] = (224, 224),
            dynamic_load: bool = False):
        self.metadata = metadata
        self.dynamic_load = dynamic_load
        self.load_data_fn = load_data_fn
        self.device = select_device()

        self.metadata['augmented'] = False
        self.metadata = self.metadata
        self.balance_data = balance_data
        self.normalize = normalize
        if self.normalize:
            self.mean = mean.to(self.device)
            self.std = std.to(self.device)
        self.resize_dims = resize_dims
        if std_epsilon <= 0:
            raise ValueError("std_epsilon must be a positive number.")
        else:
            self.std_epsilon = std_epsilon
        if balance_undersampling <= 0 and balance_undersampling > 1:
            raise ValueError(
                "balance_undersampling must be a value in the range (0, 1].")
        else:
            self.balance_undersampling = balance_undersampling
        if self.normalize and (self.mean is None or self.std is None):
            raise ValueError(
                "Normalization flag set to True. Please specify the mean a standard deviation for z-score normalization.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.dynamic_load:
            result = self.load_data_fn(metadata=self.metadata, idx=idx)
            if len(result) == 3:
                image, label, segmentation = result
            elif len(result) == 2:
                image, label = result
            else:
                raise ValueError(
                    "load_data_fn must return a tuple of length 2 or 3.")

            image = image.to(self.device)

            if self.normalize:
                image = (image - self.mean) / self.std
            if len(result) == 3:
                return image, label, segmentation
            return image, label
        else:
            image = self.images[idx].to(self.device)
            label = self.labels[idx]
            if self.normalize:
                image = (image - self.mean) / self.std
            try:
                segmentation = self.segmentations[idx].to(self.device)
                return image, label, segmentation
            except:
                return image, label

    def load_images_and_labels(self):
        result = self.load_data_fn(metadata=self.metadata)
        if len(result) == 3:
            self.images, self.labels, self.segmentations = result
        elif len(result) == 2:
            self.images, self.labels = result
        else:
            raise ValueError(
                "load_data_fn must return a tuple of length 2 or 3.")
