
from abc import ABC, abstractmethod
from typing import Optional

import os
import pandas as pd
import torch
from config import DATASET_LIMIT, DATASET_TEST_DIR, DATASET_TRAIN_DIR, METADATA_TEST_DIR, METADATA_NO_DUPLICATES_DIR, NORMALIZE, SEGMENTATION_DIR, BATCH_SIZE, SEGMENTATION_WITH_BOUNDING_BOX_DIR, SEGMENTATION_BOUNDING_BOX, BALANCE_UNDERSAMPLING
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from torchvision import transforms

from datasets.HAM10K import HAM10K


class DataLoader(ABC):
    def __init__(self,
                 limit: Optional[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 dynamic_load: bool = False,
                 upscale_train: bool = True,
                 normalize: bool = NORMALIZE,
                 batch_size: int = BATCH_SIZE):
        super().__init__()
        self.limit = limit
        self.transform = transform
        self.dynamic_load = dynamic_load
        self.upscale_train = upscale_train
        self.normalize = normalize
        self.batch_size = batch_size
        if self.transform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomRotation(degrees=90),
                # transforms.RandomAffine(0, scale=(0.8, 1.2)),
                transforms.ToTensor()
            ])

    @abstractmethod
    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int, transform: transforms.Compose = None):
        pass

    @abstractmethod
    def load_images_and_labels(self, metadata: pd.DataFrame):
        pass

    def load_metadata(self,
                      train: bool = True,
                      limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame] or pd.DataFrame:
        metadata = pd.read_csv(
            METADATA_NO_DUPLICATES_DIR if train else METADATA_TEST_DIR)
        label_dict = {'nv': 0, 'bkl': 1, 'mel': 2,
                      'akiec': 3, 'bcc': 4, 'df': 5, 'vasc': 6} #2, 3, 4 malignant, otherwise begign 
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
                lambda x: os.path.join(SEGMENTATION_DIR, x + '_segmentation.png'))
            metadata['segmentation_bbox_path'] = metadata['image_id'].apply(
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

    def load_data(self, metadata: pd.DataFrame, idx: Optional[int] = None):
        if idx is not None:
            return self.load_images_and_labels_at_idx(metadata, idx)
        return self.load_images_and_labels(metadata)

    def get_train_val_dataloders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        resnet_mean, resnet_std = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([
            0.229, 0.224, 0.225]).view(3, 1, 1)
        self.df_train, self.df_val = self.load_metadata(limit=self.limit)
        train_dataset = HAM10K(
            self.df_train,
            load_data_fn=self.load_data,
            normalize=self.normalize,
            mean=resnet_mean,
            std=resnet_std,
            balance_data=self.upscale_train,
            resize_dims=(224, 224),  # TODO: make dynamic
            dynamic_load=self.dynamic_load)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        val_dataset = HAM10K(
            self.df_val,
            load_data_fn=self.load_data,
            normalize=self.normalize,
            mean=resnet_mean,
            std=resnet_std,
            balance_data=False,
            resize_dims=(224, 224),
            dynamic_load=self.dynamic_load)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        return train_dataloader, val_dataloader

    def get_test_dataloader(self):
        resnet_mean, resnet_std = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([
            0.229, 0.224, 0.225]).view(3, 1, 1)
        self.df_test = self.load_metadata(limit=self.limit, train=False)
        test_dataset = HAM10K(
            self.df_test,
            load_data_fn=self.load_data,
            normalize=self.normalize,
            mean=resnet_mean,
            std=resnet_std,
            balance_data=False,
            resize_dims=(224, 224),
            dynamic_load=self.dynamic_load)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        return test_dataloader
