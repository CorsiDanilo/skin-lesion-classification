from collections import Counter
import math
import random
from typing import Callable
import pandas as pd
import torch
from config import BALANCE_DOWNSAMPLING
from datasets.CustomDataset import CustomDataset
import random
import math


class StyleGANPairsDataset(CustomDataset):

    def __init__(self,
                 metadata: pd.DataFrame,
                 load_data_fn: Callable,
                 resize_dims=(224, 224),
                 balance_data: bool = False,
                 balance_downsampling: float = BALANCE_DOWNSAMPLING,
                 dynamic_load: bool = False):
        super().__init__(
            metadata=metadata,
            load_data_fn=load_data_fn,
            balance_data=balance_data,
            balance_downsampling=balance_downsampling,
            # NOTE: normalization is not needed for StyleGAN images
            normalize=False,
            mean=None,
            std=None,
            std_epsilon=0.01,
            resize_dims=resize_dims,
            dynamic_load=dynamic_load)

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

        # Downsampling most common class
        max_label_images_to_remove = max(math.floor(
            max_count*self.balance_downsampling), second_max_count)
        print(
            f"--Data Balance (Downsampling)-- Keeping {max_label_images_to_remove} from {max_label} class..")
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
            f"--Data Balance (Downsampling)-- {max_label} now has {max_count} images")

        # Oversampling of the other classes
        for label in self.metadata['label'].unique():
            label_indices = self.metadata[self.metadata['label']
                                          == label].index
            current_images = len(label_indices)

            if current_images < max_count:
                num_images_to_add = max_count - current_images
                print(
                    f"--Data Balance (Oversampling)-- Adding {num_images_to_add} from {label} class..")
                aug_indices = random.choices(
                    label_indices.tolist(), k=num_images_to_add)
                self.metadata = pd.concat(
                    [self.metadata, self.metadata.loc[aug_indices]])
                # Apply data augmentation only to the augmented subset
                self.metadata.loc[aug_indices, 'augmented'] = True
                label_indices = self.metadata[self.metadata['label']
                                              == label].index

    def __getitem__(self, idx):
        if self.dynamic_load:
            result = self.load_data_fn(metadata=self.metadata, idx=idx)
            image, label, image_path, augmented = result
            image = image.to(self.device)
            return image, label, image_path, augmented
        else:
            raise NotImplementedError()
