from collections import Counter
import math
import random
from typing import Callable
import pandas as pd
import torch
from config import BALANCE_UNDERSAMPLING
from datasets.CustomDataset import CustomDataset
import random
import math


class HAM10K(CustomDataset):
    """
    Class that defines the HAM10K dataset for the project.
    """

    def __init__(self,
                 metadata: pd.DataFrame,
                 load_data_fn: Callable,
                 # Control the data augmentation process aim to solve class imbalance
                 balance_data: bool = True,
                 # Percentage of data to keep from the majority class
                 balance_undersampling: float = BALANCE_UNDERSAMPLING,
                 normalize: bool = False,  # Control the application of z-score normalization
                 # Mean (per channel) for the z-score normalization
                 mean: torch.Tensor = None,
                 # Standard deviation (per channel) for the z-score normalization
                 std: torch.Tensor = None,
                 # Adjustment value to avoid division per zero during normalization
                 std_epsilon: float = 0.01,
                 # Sizes (height, width) for resize the images
                 resize_dims=(224, 224),
                 dynamic_load: bool = False):
        super().__init__(metadata, load_data_fn, balance_data, balance_undersampling,
                         normalize, mean, std, std_epsilon, resize_dims, dynamic_load)

        if self.balance_data:
            self.balance_dataset()

        if not dynamic_load:
            self.load_images_and_labels()

    def balance_dataset(self):
        print(
            "--Data Balance-- balance_data set to True. Training data will be balanced.")
        # Count images associated to each label
        labels_counts = Counter(self.metadata['label'])
        max_label, max_count = max(
            labels_counts.items(), key=lambda x: x[1])  # Majority class
        second_max_label, second_max_count = labels_counts.most_common(
            2)[-1]  # Second majority class
        print(
            f"--Data Balance-- The most common class is {max_label} with {max_count} images.")
        print(
            f"--Data Balance-- The second common class is {second_max_label} with {second_max_count} images with a difference of {max_count-second_max_count} images from the most common class.")

        # Undersampling most common class
        max_label_images_to_remove = max(math.floor(
            max_count*self.balance_undersampling), second_max_count)
        print(
            f"--Data Balance (Undersampling)-- Keeping {max_label_images_to_remove} from {max_label} class..")
        label_indices = self.metadata[self.metadata['label']
                                      == max_label].index
        removal_indices = random.sample(
            label_indices.tolist(), k=max_count-max_label_images_to_remove)
        self.metadata = self.metadata.drop(index=removal_indices)
        self.metadata.reset_index(drop=True, inplace=True)
        labels_counts = Counter(self.metadata['label'])
        max_label, max_count = max(
            labels_counts.items(), key=lambda x: x[1])
        print(
            f"--Data Balance (Undersampling)-- {max_label} now has {max_count} images")

        # Oversampling of the other classes
        for label in self.metadata['label'].unique():
            label_indices = self.metadata[self.metadata['label']
                                          == label].index
            current_images = len(label_indices)

            if current_images < max_count:
                num_images_to_add = max_count - current_images
                print(
                    f"-- Data Balance (Oversampling) -- Adding {num_images_to_add} from {label} class..")
                aug_indices = random.choices(
                    label_indices.tolist(), k=num_images_to_add)
                self.metadata = pd.concat(
                    [self.metadata, self.metadata.loc[aug_indices]])
                # Apply data augmentation only to the augmented subset
                self.metadata.loc[aug_indices, 'augmented'] = True
                label_indices = self.metadata[self.metadata['label']
                                              == label].index
