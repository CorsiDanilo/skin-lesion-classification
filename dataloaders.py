from typing import Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from tqdm import tqdm

from config import DATASET_TEST_DIR, DATASET_TRAIN_DIR, METADATA_TEST_DIR, METADATA_TRAIN_DIR, SEGMENTATION_DIR, BATCH_SIZE


class ImageDataset(Dataset):
    def __init__(self,
                 metadata: pd.DataFrame,
                 train: bool = True,
                 transform: Optional[transforms.Compose] = None):
        self.metadata = metadata
        self.transform = transform

        unique_labels = self.metadata['dx'].unique()
        label_dict = {label: idx for idx, label in enumerate(unique_labels)}
        labels_encoded = self.metadata['dx'].map(label_dict)
        self.metadata['label'] = labels_encoded
        self.train = train
        if self.train:
            self.images, self.labels, self.segmentations = self.load_images_and_labels()
        else:
            self.images, self.labels = self.load_images_and_labels()

        # TODO: maybe load other information,
        # encode it in one-hot vectors and concatenate them to the images in order to feed it to the NN

    def load_images_and_labels(self):
        scale_factor = 0.1
        ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 450, 600
        height, width = int(
            ORIGINAL_HEIGHT * scale_factor), int(ORIGINAL_WIDTH * scale_factor)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor()
            ])

        not_found_files = []
        images = []
        segmentations = []
        for img in tqdm(self.metadata['image_path'], desc=f'Loading {"train" if self.train else "test"} images'):
            if not os.path.exists(img):
                not_found_files.append(img)
                continue
            images.append(self.transform(Image.open(img)))
        if self.train:
            for img in tqdm(self.metadata['segmentation_path'], desc='Loading train segmentations'):
                if not os.path.exists(img):
                    not_found_files.append(img)
                    continue
                segmentations.append(self.transform(Image.open(img)))
            segmentations = torch.stack(segmentations)
        images = torch.stack(images)

        labels = torch.tensor(self.metadata['label'].tolist(
        ), dtype=torch.long)

        print(
            f"Loading complete, some files ({len(not_found_files)}) were not found: {not_found_files}")
        if self.train:
            return images, labels, segmentations
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.train:
            segmentation = self.segmentations[idx]
            return image, label, segmentation
        return image, label


def load_metadata(train: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame] or pd.DataFrame:
    metadata = pd.read_csv(METADATA_TRAIN_DIR if train else METADATA_TEST_DIR)
    metadata['image_path'] = metadata['image_id'].apply(
        lambda x: os.path.join(DATASET_TRAIN_DIR if train else DATASET_TEST_DIR, x + '.jpg'))

    if train:
        metadata['segmentation_path'] = metadata['image_id'].apply(
            lambda x: os.path.join(SEGMENTATION_DIR, x + '_segmentation.png'))

        # Assuming `df` is your DataFrame
        df_train, df_val = train_test_split(
            metadata, test_size=0.2, random_state=42)

        return df_train, df_val

    return metadata


def create_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    df_train, df_val = load_metadata()
    df_test = load_metadata(train=False)

    train_dataset = ImageDataset(df_train)
    val_dataset = ImageDataset(df_val, train=False)
    test_dataset = ImageDataset(df_test, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_dataloaders()

    batch: torch.Tensor
    labels: torch.Tensor
    segmentations: torch.Tensor
    for (batch, labels, segmentations) in train_loader:
        print(f"Batch shape is {batch.shape}")
        print(f"Labels shape is {labels.shape}")
        print(f"Segmentation shape is {segmentations.shape}")
        break
