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

from config import DATASET_TEST_DIR, DATASET_TRAIN_DIR, METADATA_TEST_DIR, METADATA_TRAIN_DIR, SEGMENTATION_DIR, BATCH_SIZE

seed = 42
random.seed(seed)
torch.manual_seed(seed)


class ImageDataset(Dataset):
    def __init__(self,
                 metadata: pd.DataFrame,
                 train: bool = True,
                 balance_data: bool = True,
                 normalize: bool = False,
                 transform: Optional[transforms.Compose] = None,
                 balance_transform: Optional[transforms.Compose] = None,
                 limit: int = None):
        if limit is not None:
            self.metadata = metadata[:limit]
        else:
            self.metadata = metadata
        self.transform = transform

        unique_labels = self.metadata['dx'].unique()
        label_dict = {label: idx for idx, label in enumerate(unique_labels)}
        labels_encoded = self.metadata['dx'].map(label_dict)
        self.metadata['label'] = labels_encoded
        self.metadata['augmented'] = False
        self.train = train
        self.balance_data = balance_data
        self.balance_transform = balance_transform
        self.normalize = normalize

        scale_factor = 0.1
        ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 450, 600
        height, width = int(
            ORIGINAL_HEIGHT * scale_factor), int(ORIGINAL_WIDTH * scale_factor)
        if self.transform is None:
            self.transform = transforms.Compose([
                # transforms.Resize((height, width)),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        '''
        if self.balance_transform is None:
            self.balance_transform = transforms.Compose([
                transforms.Resize((height, width)), #TO DO: TRY TO PUT TOGETHER SELF.TRASFORM AND BALANCE TRANSFORM
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                #transforms.RandomResizedCrop(size=(self.height, self.width), scale=(0.9, 1.1)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor()
        ])
        '''

        if self.balance_data:
            self.balance_dataset()

        if self.train:
            self.images, self.labels, self.segmentations = self.load_images_and_labels()
        else:
            self.images, self.labels = self.load_images_and_labels()

        # TODO: maybe load other information,
        # encode it in one-hot vectors and concatenate them to the images in order to feed it to the NN

    def balance_dataset(self):
        labels_counts = Counter(self.metadata['label'])
        max_label, max_count = max(labels_counts.items(), key=lambda x: x[1])
        _, second_max_count = labels_counts.most_common(2)[-1]

        # Undersampling most common class
        max_label_images_to_remove = max(
            math.floor(max_count*0.5), second_max_count)
        label_indices = self.metadata[self.metadata['label']
                                      == max_label].index
        removal_indices = random.sample(
            label_indices.tolist(), k=max_label_images_to_remove)
        self.metadata = self.metadata.drop(index=removal_indices)
        self.metadata.reset_index(drop=True, inplace=True)

        labels_counts = Counter(self.metadata['label'])
        # Compute max label again - Maybe create a function to avoid doing this operation twice
        max_label, max_count = max(labels_counts.items(), key=lambda x: x[1])

        # Oversampling of the other classes
        for label in self.metadata['label'].unique():
            label_indices = self.metadata[self.metadata['label']
                                          == label].index
            current_images = len(label_indices)

            if current_images < max_count:
                num_images_to_add = max_count - current_images
                aug_indices = random.choices(
                    label_indices.tolist(), k=num_images_to_add)
                self.metadata = pd.concat(
                    [self.metadata, self.metadata.loc[aug_indices]])
                # Apply data augmentation only to the augmented subset
                self.metadata.loc[aug_indices, 'augmented'] = True
                label_indices = self.metadata[self.metadata['label']
                                              == label].index
        self.metadata.reset_index(drop=True, inplace=True)

    def load_images_and_labels(self):
        not_found_files = []
        images = []
        segmentations = []
        labels = []
        for _, img in tqdm(self.metadata.iterrows(), desc=f'Loading {"train" if self.train else "test"} images'):
            if not os.path.exists(img['image_path']):
                not_found_files.append(img['image_path'])
                continue
            labels.append(img['label'])
            if self.balance_data:
                # CHANGE WITH NEW SIZES DINAMICALLY!
                stateful_transform = StatefulTransform(45, 60)
                if not os.path.exists(img['segmentation_path']):
                    not_found_files.append(img['segmentation_path'])
                    continue
                if img['augmented']:
                    # images.append(self.balance_transform(Image.open(img['image_path'])))
                    # segmentations.append(self.balance_transform(Image.open(img['segmentation_path'])))
                    ti, ts = stateful_transform(Image.open(
                        img['image_path']), Image.open(img['segmentation_path']))
                    # segmentations.append(stateful_transform(Image.open(img['segmentation_path'])))
                    images.append(ti)
                    segmentations.append(ts)
            elif self.train:
                images.append(self.transform(
                    Image.open(img['image_path'])))
                segmentations.append(self.transform(
                    Image.open(img['segmentation_path'])))
            else:
                images.append(self.transform(Image.open(img['image_path'])))
        if self.train:
            segmentations = torch.stack(segmentations)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        self.mean = torch.tensor(
            [torch.mean(images[:, channel, :, :]) for channel in range(3)]).reshape(3, 1, 1)
        self.std = torch.tensor([torch.std(images[:, channel, :, :])
                                for channel in range(3)]).reshape(3, 1, 1)

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
        # TODO: you should first crop the roi and then normalize the image
        if self.normalize:
            image -= self.mean
            image /= self.std
        if self.train:
            segmentation = self.segmentations[idx]
            return image, label, segmentation
        return image, label


class StatefulTransform:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, seg):
        img = transforms.Resize((224, 224))(img)
        seg = transforms.Resize((224, 224))(seg)

        if random.random() > 0.5:
            img = TF.hflip(img)
            seg = TF.hflip(seg)

        if random.random() > 0.5:
            img = TF.vflip(img)
            seg = TF.vflip(seg)

        if random.random() > 0.5:
            angle = random.randint(1, 360)
            img = TF.rotate(img, angle)
            seg = TF.rotate(seg, angle)

        # img = transforms.RandomRotation(180)(img)
        # img = transforms.RandomHorizontalFlip()(img)
        # img = transforms.RandomVerticalFlip()(img)
        # img = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))(img)
        img = transforms.ToTensor()(img)
        seg = transforms.ToTensor()(seg)

        return img, seg


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


def create_dataloaders(limit: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    df_train, df_val = load_metadata()
    df_test = load_metadata(train=False)

    train_dataset = ImageDataset(
        df_train,
        train=True,
        limit=limit,
        balance_data=True)

    # TODO: remove the train here, it's just to test if the segmentation helps the model
    # I'm puttingbalance data false in order to not augment the validation even if train is true.
    val_dataset = ImageDataset(
        df_val,
        train=True,
        limit=limit,
        balance_data=False)

    test_dataset = ImageDataset(
        df_test,
        train=False,
        limit=limit,
        balance_data=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE)
    
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


'''
class ImageDataset(Dataset):
    def __init__(self,
                 metadata: pd.DataFrame,
                 train: bool = True,
                 transform: Optional[transforms.Compose] = None):
        self.metadata = metadata[:100]
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
'''
