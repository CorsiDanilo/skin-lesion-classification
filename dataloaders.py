from typing import Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from tqdm import tqdm

from config import DATASET_DIR, SEGMENTATION_DIR, METADATA_DIR, BATCH_SIZE


class ImageDataset(Dataset):
    def __init__(self,
                 metadata: pd.DataFrame,
                 transform: Optional[transforms.Compose] = None):
        self.metadata = metadata
        self.transform = transform

        unique_labels = self.metadata['dx'].unique()
        label_dict = {label: idx for idx, label in enumerate(unique_labels)}
        labels_encoded = self.metadata['dx'].map(label_dict)
        self.metadata['label'] = labels_encoded

        self.images, self.labels = self.load_images_and_labels()

        # TODO: also load segmentations

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

        images = []
        for img in tqdm(self.metadata['image_path'], desc='Loading images'):
            images.append(self.transform(Image.open(img)))
        images = torch.stack(images)

        labels = torch.tensor(self.metadata['label'].tolist(
        ), dtype=torch.long)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label


def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(METADATA_DIR)
    metadata['image_path'] = metadata['image_id'].apply(
        lambda x: os.path.join(DATASET_DIR, x + '.jpg'))
    metadata['segmentation_path'] = metadata['image_id'].apply(
        lambda x: os.path.join(SEGMENTATION_DIR, x + '_segmentation.png'))

    # Assuming `df` is your DataFrame
    df_train, df_test = train_test_split(
        metadata, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(
        df_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    return df_train, df_val, df_test


def create_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    df_train, df_val, df_test = load_metadata()

    train_dataset = ImageDataset(df_train)
    val_dataset = ImageDataset(df_val)
    test_dataset = ImageDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_dataloaders()

    for batch in train_loader:
        print(batch)
        break
