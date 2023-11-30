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

from config import DATASET_TEST_DIR, DATASET_TRAIN_DIR, METADATA_TEST_DIR, METADATA_NO_DUPLICATES_DIR, SEGMENTATION_DIR, BATCH_SIZE, SEGMENTATION_WITH_BOUNDING_BOX_DIR


class ImageDataset(Dataset):
    def __init__(self,
                 metadata: pd.DataFrame,
                 load_segmentations: bool = True,
                 # Control the data augmentation process aim to solve class imbalance
                 balance_data: bool = True,
                 # Percentage of data to keep from the majority class
                 balance_undersampling: float = 1,
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
                 dynamic_load: bool = False):
        self.metadata = metadata
        # self.remove_duplicates()
        self.transform = transform

        self.metadata['augmented'] = False
        self.metadata = self.metadata
        self.load_segmentations = load_segmentations
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
                # transforms.Resize(
                # (self.resize_dims[0], self.resize_dims[1]), interpolation=Image.BILINEAR),
                # transforms.RandomEqualize(p=1),
                # transforms.RandomAdjustSharpness(sharpness_factor=10, p=1),
                transforms.ToTensor()
            ])

            self.segmentation_transform = transforms.Compose([
                # transforms.Resize((self.resize_dims[0], self.resize_dims[1])),
                transforms.ToTensor()
            ])

        self.dynamic_load = dynamic_load
        if not dynamic_load:
            if self.load_segmentations:
                self.images, self.labels, self.segmentations = self.load_images_and_labels()
            else:
                self.images, self.labels = self.load_images_and_labels()

        # TODO: maybe load other information,
        # encode it in one-hot vectors and concatenate them to the images in order to feed it to the NN

    def remove_duplicates(self):
        # Find duplicates in 'lesion_id'
        duplicates = self.metadata[self.metadata.duplicated(
            'lesion_id', keep=False)]

        print(f"Original metadata length: {len(self.metadata)}")
        # Print duplicates
        print(duplicates)

        # Add 'is_duplicated' column
        self.metadata['is_duplicated'] = self.metadata.duplicated(
            'lesion_id', keep=False)

        # Sort by 'lesion_id' and 'is_duplicated'
        self.metadata.sort_values(
            ['lesion_id', 'is_duplicated'], inplace=True)

        # Drop duplicates, keeping the first occurrence
        matadata_train_no_duplicates = self.metadata.drop_duplicates(
            'lesion_id', keep='first')

        matadata_train_no_duplicates.drop(
            'is_duplicated', axis=1, inplace=True)

        print(
            f"Metadata length without duplicates: {len(matadata_train_no_duplicates)}")

        self.metadata = matadata_train_no_duplicates

    def load_images_and_labels_at_idx(self, idx):
        img = self.metadata.iloc[idx]
        if not os.path.exists(img['image_path']):
            self.load_images_and_labels_at_idx(idx+1)
        label = img['label']
        # Augment the data if balance_data is true and load segmentations
        # Load segmentations without augmenting the data
        if self.load_segmentations:
            if not os.path.exists(img['segmentation_path']):
                self.load_images_and_labels_at_idx(idx+1)
            segmentation = self.segmentation_transform(
                Image.open(img['segmentation_path']).convert('1'))
            image = self.transform(Image.open(img['image_path']))
        # Only load images
        else:
            image = self.transform(Image.open(img['image_path']))
        if self.load_segmentations:
            return image, label, segmentation
        return image, label

    def load_images_and_labels(self):
        not_found_files = []
        images = []
        segmentations = []
        labels = []
        for _, img in tqdm(self.metadata.iterrows(), desc=f'Loading images'):
            if not os.path.exists(img['image_path']):
                not_found_files.append(img['image_path'])
                continue
            labels.append(img['label'])
            # Augment the data if balance_data is true and load segmentations
            if self.load_segmentations:
                if not os.path.exists(img['segmentation_path']):
                    not_found_files.append(img['segmentation_path'])
                    continue
                segmentations.append(self.segmentation_transform(
                    Image.open(img['segmentation_path']).convert('1')))
                images.append(self.transform(Image.open(img['image_path'])))
            # Only load images
            else:
                images.append(self.transform(Image.open(img['image_path'])))
        if self.load_segmentations:
            segmentations = torch.stack(segmentations)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        print(f"---Data Loader--- Images uploaded: " + str(len(images)))

        print(
            f"Loading complete, some files ({len(not_found_files)}) were not found: {not_found_files}")
        if self.load_segmentations:
            return images, labels, segmentations
        return images, labels

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.dynamic_load:
            if self.load_segmentations:
                image, label, segmentation = self.load_images_and_labels_at_idx(
                    idx)
            else:
                image, label = self.load_images_and_labels_at_idx(idx)
            if self.normalize:
                image = (image - self.mean.view(3, 1, 1)) / \
                    (self.std + self.std_epsilon).view(3, 1, 1)
            if self.load_segmentations:
                return image, label, segmentation
            return image, label
        else:
            image = self.images[idx]
            label = self.labels[idx]
            if self.normalize:
                image = (image - self.mean.view(3, 1, 1)) / \
                    (self.std + self.std_epsilon).view(3, 1, 1)
            if self.load_segmentations:
                segmentation = self.segmentations[idx]
                return image, label, segmentation
            return image, label


def calculate_normalization_statistics(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    images_for_normalization = []

    for _, img in tqdm(df[:100].iterrows(), desc=f'Calculating normalization statistics'):
        if not os.path.exists(img['image_path']):
            continue
        image = transforms.ToTensor()(Image.open(img['image_path']))
        images_for_normalization.append(image)

    images_for_normalization = torch.stack(images_for_normalization)
    mean = torch.tensor([torch.mean(images_for_normalization[:, channel, :, :])
                        for channel in range(3)]).reshape(3, 1, 1)
    std = torch.tensor([torch.std(images_for_normalization[:, channel, :, :])
                       for channel in range(3)]).reshape(3, 1, 1)

    print("---Normalization--- Normalization flag set to True: Images will be normalized with z-score normalization")
    print(
        f"---Normalization--- Statistics for normalization (per channel) -> Mean: {mean.view(-1)}, Variance: {std.view(-1)}, Epsilon (adjustment value): 0.01")

    return mean, std


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
        metadata['segmentation_path'] = metadata['image_id'].apply(
            lambda x: os.path.join(SEGMENTATION_WITH_BOUNDING_BOX_DIR, x + '_segmentation.png'))

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


def create_dataloaders(normalize: bool = True,
                       mean: Optional[torch.Tensor] = None,
                       std: Optional[torch.Tensor] = None,
                       limit: Optional[int] = None,
                       batch_size: int = BATCH_SIZE,
                       dynamic_load: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:

    df_train, df_val = load_metadata(limit=limit)
    df_test = load_metadata(train=False, limit=limit)

    # df_train.reset_index(drop=True, inplace=True)
    # df_val.reset_index(drop=True, inplace=True)

    # Calculate and store normalization statistics for the training dataset
    if normalize and (mean is None or std is None):
        mean, std = calculate_normalization_statistics(df_train)

    train_dataset = ImageDataset(
        df_train,
        load_segmentations=True,
        normalize=normalize,
        mean=mean,
        std=std,
        dynamic_load=dynamic_load)
    val_dataset = ImageDataset(
        df_val,
        load_segmentations=True,
        normalize=normalize,
        mean=mean,
        std=std,
        dynamic_load=dynamic_load)
    test_dataset = ImageDataset(
        df_test,
        load_segmentations=False,
        normalize=normalize,
        mean=mean,
        std=std,
        dynamic_load=dynamic_load)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, pin_memory=True)
    return train_loader, val_loader, test_loader


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
