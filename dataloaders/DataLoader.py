
from abc import ABC, abstractmethod
from typing import Optional

import os
import pandas as pd
import torch
from config import IMAGE_SIZE, DATASET_TRAIN_DIR, METADATA_TRAIN_DIR, NORMALIZE, SEGMENTATION_DIR, BATCH_SIZE
from constants import DEFAULT_STATISTICS
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from utils.utils import calculate_normalization_statistics
from torchvision import transforms

from datasets.HAM10K import HAM10K
from utils.utils import select_device


class DataLoader(ABC):
    def __init__(self,
                 limit: Optional[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 dynamic_load: bool = False,
                 upscale_train: bool = True,
                 normalize: bool = NORMALIZE,
                 normalization_statistics: tuple = None,
                 batch_size: int = BATCH_SIZE,
                 always_rotate: bool = False):
        super().__init__()
        self.limit = limit
        self.transform = transform
        self.dynamic_load = dynamic_load
        self.upscale_train = upscale_train
        self.normalize = normalize
        self.normalization_statistics = normalization_statistics
        self.batch_size = batch_size
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        self.device = select_device()
        self.train_df, self.val_df, self.test_df = self._init_metadata(
            limit=limit)
        self.always_rotate = always_rotate

    @abstractmethod
    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int, transform: transforms.Compose = None):
        pass

    @abstractmethod
    def load_images_and_labels(self, metadata: pd.DataFrame):
        pass

    def _init_metadata(self,
                       limit: Optional[int] = None):
        metadata = pd.read_csv(METADATA_TRAIN_DIR)
        label_dict = {'nv': 0, 'bkl': 1, 'mel': 2,
                      'akiec': 3, 'bcc': 4, 'df': 5, 'vasc': 6}  # 2, 3, 4 malignant, otherwise begign
        labels_encoded = metadata['dx'].map(label_dict)
        metadata['label'] = labels_encoded

        print(f"LOADED METADATA HAS LENGTH {len(metadata)}")
        if limit is not None and limit > len(metadata):
            print(
                f"Ignoring limit for because it is bigger than the dataset size")
            limit = None
        if limit is not None:
            print(f"---LIMITING DATASET TO {limit} ENTRIES---")
            metadata = metadata.sample(n=limit, random_state=42)
        metadata['image_path'] = metadata['image_id'].apply(
            lambda x: os.path.join(DATASET_TRAIN_DIR, x + '.jpg'))

        metadata['segmentation_path'] = metadata['image_id'].apply(
            lambda x: os.path.join(SEGMENTATION_DIR, x + '_segmentation.png'))
        # metadata['segmentation_bbox_path'] = metadata['image_id'].apply(
        # lambda x: os.path.join(SEGMENTATION_WITH_BOUNDING_BOX_DIR, x + '_segmentation.png'))

        df_train, df_test = train_test_split(
            metadata,
            test_size=0.15,  # 15% test, 85% train
            random_state=42,
            stratify=metadata['dx'])

        df_train, df_val = train_test_split(
            df_train,
            test_size=0.1,  # Of the 85% train, 10% val, 90% train
            random_state=42,
            stratify=df_train['dx'])

        assert len(df_train['label'].unique(
        )) == 7, f"Number of unique labels in metadata is not 7, it's {len(df_train['label'].unique())}, increase the limit"
        assert len(df_val['label'].unique(
        )) == 7, f"Number of unique labels in metadata is not 7, it's {len(df_val['label'].unique())}, increase the limit"
        # TODO: Uncomment
        # assert len(df_test['label'].unique(
        # )) == 7, f"Number of unique labels in metadata is not 7, it's {len(df_test['label'].unique())}, increase the limit"

        df_train["train"] = True
        # df_val["train"] = False
        # df_test["train"] = False

        # Remove segmentation path from test and val just to be sure not to use them
        # TODO: Uncomment
        # df_val.drop(columns=['segmentation_path'], inplace=True)
        # df_test.drop(columns=['segmentation_path'], inplace=True)

        print(f"---TRAIN---: {len(df_train)} entries")
        print(f"---VAL---: {len(df_val)} entries")
        print(f"---TEST---: {len(df_test)} entries")
        return df_train, df_val, df_test

    def load_data(self, metadata: pd.DataFrame, idx: Optional[int] = None):
        if idx is not None:
            return self.load_images_and_labels_at_idx(metadata, idx)
        return self.load_images_and_labels(metadata)

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
            shuffle=True,
            pin_memory=False,
        )
        return train_dataloader

    def get_val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.normalize:
            if self.normalization_statistics is None:
                print(
                    "--Normalization-- Normalization statistics not defined during test. Using default ones.")
                self.normalization_statistics = DEFAULT_STATISTICS
            print(
                f"--Normalization-- Statistics for normalization (per channel) -> Mean: {self.normalization_statistics[0].view(-1)}, Variance: {self.normalization_statistics[1].view(-1)}, Epsilon (adjustment value): 0.01")
        val_dataset = HAM10K(
            self.val_df,
            load_data_fn=self.load_data,
            normalize=self.normalize,
            mean=self.normalization_statistics[0] if self.normalize else None,
            std=self.normalization_statistics[1] if self.normalize else None,
            balance_data=False,
            resize_dims=IMAGE_SIZE,
            dynamic_load=self.dynamic_load)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return val_dataloader

    def get_test_dataloader(self):
        if self.normalize:
            if self.normalization_statistics is None:
                print(
                    "--Normalization (Test)-- Normalization statistics not defined during test. Using default ones.")
                self.normalization_statistics = DEFAULT_STATISTICS
            print(
                f"--Normalization (Test)-- Statistics for normalization (per channel) -> Mean: {self.normalization_statistics[0].view(-1)}, Variance: {self.normalization_statistics[1].view(-1)}, Epsilon (adjustment value): 0.01")
        test_dataset = HAM10K(
            self.test_df,
            load_data_fn=self.load_data,
            normalize=self.normalize,
            mean=self.normalization_statistics[0] if self.normalize else None,
            std=self.normalization_statistics[1] if self.normalize else None,
            balance_data=False,
            resize_dims=IMAGE_SIZE,
            dynamic_load=self.dynamic_load)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
        )
        return test_dataloader
