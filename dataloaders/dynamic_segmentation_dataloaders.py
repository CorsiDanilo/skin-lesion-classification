from typing import Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
import os
from PIL import Image
from tqdm import tqdm
import random
import math

from config import DATASET_TEST_DIR, DATASET_TRAIN_DIR, METADATA_TEST_DIR, METADATA_NO_DUPLICATES_DIR, SEGMENTATION_DIR, BATCH_SIZE, SEGMENTATION_WITH_BOUNDING_BOX_DIR, SEGMENTATION_BOUNDING_BOX, BALANCE_UNDERSAMPLING
from utils.opencv_boxes_test import bounding_box_pipeline
from utils.utils import crop_image_from_box, crop_roi, get_bounding_boxes_from_segmentation, zoom_out

"""
This dataloader uses a dynamic segmentation technique on the validation set in order to produce the segmented image given the image.
On the training set it uses the ground truth segmentation already provided.

Both the training and validation set images that are output of the dataloader are already normalized and segmented, so no further operation are needed.
"""


class ImageDataset(Dataset):
    def __init__(self,
                 metadata: pd.DataFrame,
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
                 transform: Optional[transforms.Compose] = None,
                 dynamic_load: bool = False,
                 train: bool = True):
        self.metadata = metadata
        self.transform = transform
        self.train = train

        self.metadata['augmented'] = False
        self.metadata = self.metadata
        self.balance_data = balance_data
        self.normalize = normalize
        self.mean = mean
        self.std = std
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

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
                # transforms.RandomAffine(0, scale=(0.8, 1.2)),
                transforms.ToTensor()
            ])

        if self.balance_data:
            self.balance_dataset()

        self.dynamic_load = dynamic_load
        if not dynamic_load:
            self.images, self.labels = self.load_images_and_labels()

        # TODO: maybe load other information,
        # encode it in one-hot vectors and concatenate them to the images in order to feed it to the NN

    def balance_dataset(self):
        print("--Data Balance-- balance_data set to True. Training data will be balanced.")
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
        max_label, max_count = max(labels_counts.items(), key=lambda x: x[1])
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

    def load_images_and_labels_at_idx(self, idx):
        img = self.metadata.iloc[idx]
        # Augment the data if balance_data is true and load segmentations
        label = img['label']
        if not self.train:
            image = Image.open(img['image_path'])
            image = TF.to_tensor(image)
            segmented_image = bounding_box_pipeline(
                image.unsqueeze(0)).squeeze(0)
            return segmented_image, label
        # Augment the data if balance_data is true and load segmentations
        ti, ts = Image.open(img['image_path']), Image.open(
            img['segmentation_path']).convert('1')
        ti, ts = TF.to_tensor(ti), TF.to_tensor(ts)
        ti = zoom_out(ti)
        if self.balance_data and img["augmented"]:
            # TODO: verify that the box is squared and doesn't go out of borders for the augmented images
            pil_image = TF.to_pil_image(ti)
            image = self.transform(pil_image)
            # image = image * ts
        else:
            image = ti
        bbox = get_bounding_boxes_from_segmentation(ts)[0]
        image = crop_image_from_box(image, bbox)
        image = torch.from_numpy(image).permute(2, 0, 1)
        # image = ti * ts
        # image = crop_roi(image, self.resize_dims)
        # image = image.squeeze(0)
        return image, label

    def load_images_and_labels(self):
        not_found_files = []
        images = []
        labels = []
        for index, (row_index, img) in tqdm(enumerate(self.metadata.iterrows()), desc=f'Loading images'):
            image, label = self.load_images_and_labels_at_idx(index)
            images.append(image)
            labels.append(label)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        print(f"---Data Loader--- Images uploaded: " + str(len(images)))

        print(
            f"Loading complete, some files ({len(not_found_files)}) were not found: {not_found_files}")
        return images, labels

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.dynamic_load:
            image, label = self.load_images_and_labels_at_idx(idx)
            if self.normalize:
                image = (image - self.mean.view(3, 1, 1)) / \
                    (self.std + self.std_epsilon).view(3, 1, 1)
            return image, label
        else:
            image = self.images[idx]
            label = self.labels[idx]
            if self.normalize:
                image = (image - self.mean.view(3, 1, 1)) / \
                    (self.std + self.std_epsilon).view(3, 1, 1)
            return image, label


def load_metadata(train: bool = True,
                  limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame] or pd.DataFrame:
    # TODO: BE AWARE: the test set (not train and val) labels may be different from the train and val labels with this implementation
    metadata = pd.read_csv(
        METADATA_NO_DUPLICATES_DIR if train else METADATA_TEST_DIR)
    unique_labels = metadata['dx'].unique()
    label_dict = {label: idx for idx, label in enumerate(unique_labels)}
    labels_encoded = metadata['dx'].map(label_dict)
    assert len(
        label_dict) == 7, "There should be 7 unique labels, increase the limit"
    metadata['label'] = labels_encoded
    # df_count = metadata.groupby('label').count()
    # print(df_count)
    print(f"LOADED METADATA HAS LENGTH {len(metadata)}")
    if limit is not None and limit > len(metadata):
        print(
            f"Ignoring limit for {METADATA_NO_DUPLICATES_DIR if train else METADATA_TEST_DIR} because it is bigger than the dataset size")
        limit = None
    if limit is not None:
        metadata = metadata.sample(n=limit, random_state=42)
    metadata['image_path'] = metadata['image_id'].apply(
        lambda x: os.path.join(DATASET_TRAIN_DIR if train else DATASET_TEST_DIR, x + '.jpg'))

    if train:
        segmentation_path = SEGMENTATION_WITH_BOUNDING_BOX_DIR if SEGMENTATION_BOUNDING_BOX else SEGMENTATION_DIR
        metadata['segmentation_path'] = metadata['image_id'].apply(
            lambda x: os.path.join(segmentation_path, x + '_segmentation.png'))

        print(f"Metadata before split has length {len(metadata)}")
        # Assuming `df` is your DataFrame
        df_train, df_val = train_test_split(
            metadata,
            test_size=0.2,
            random_state=42,
            stratify=metadata['label'])

        print(f"DF_TRAIN LENGTH: {len(df_train)}")
        print(f"DF_VAL LENGTH: {len(df_val)}")
        return df_train, df_val

    return metadata


# def remove_non_existing_images(metadata: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, set]:
#     initial_length = len(metadata)
#     metadata['image_exists'] = metadata['image_path'].apply(
#         lambda x: os.path.exists(x))
#     if 'segmentation_path' in metadata.columns:
#         metadata["segmentation_exists"] = metadata['segmentation_path'].apply(
#             lambda x: os.path.exists(x))
#         metadata = metadata[metadata['segmentation_exists']]
#         metadata.reset_index(drop=True, inplace=True)
#     metadata = metadata[metadata['image_exists']]
#     if "is_duplicated" in metadata.columns:
#         metadata.drop(columns=['is_duplicated'], inplace=True)
#     metadata.reset_index(drop=True, inplace=True)
#     metadata.to_csv(
#         f"new_metadata_{name}.csv")
#     print(
#         f"Removed {initial_length - len(metadata)} images from {name} out of {initial_length} total images")

#     return metadata


def create_dataloaders(normalize: bool = True,
                       mean: Optional[torch.Tensor] = None,
                       std: Optional[torch.Tensor] = None,
                       limit: Optional[int] = None,
                       size: Tuple[int, int] = (224, 224),
                       batch_size: int = BATCH_SIZE,
                       dynamic_load: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:

    df_train, df_val = load_metadata(limit=limit)
    df_test = load_metadata(train=False, limit=limit)

    # df_train.reset_index(drop=True, inplace=True)
    # df_val.reset_index(drop=True, inplace=True)

    # Calculate and store normalization statistics for the training dataset
    # if normalize and (mean is None or std is None):
    #     mean, std = calculate_normalization_statistics(df_train)

    train_dataset = ImageDataset(
        df_train,
        train=True,
        normalize=normalize,
        mean=mean,
        std=std,
        balance_data=True,
        resize_dims=size,
        dynamic_load=dynamic_load)
    val_dataset = ImageDataset(
        df_val,
        normalize=normalize,
        mean=mean,
        std=std,
        train=False,
        balance_data=False,
        resize_dims=size,
        dynamic_load=dynamic_load)
    # test_dataset = ImageDataset(
    #     df_test,
    #     load_segmentations=False,
    #     normalize=normalize,
    #     mean=mean,
    #     std=std,
    #     balance_data=False,
    #     resize_dims=size,
    #     dynamic_load=dynamic_load)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, pin_memory=True)
    # test_loader = DataLoader(
    #     test_dataset, batch_size=batch_size, pin_memory=True)
    return train_loader, val_loader, None


if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_dataloaders(
        normalize=True, limit=None)

    batch: torch.Tensor
    labels: torch.Tensor
    segmentations: torch.Tensor
    for (batch, labels, segmentations) in train_loader:
        print(f"Batch shape is {batch.shape}")
        print(f"Labels shape is {labels.shape}")
        print(f"Segmentation shape is {segmentations.shape}")
        break
