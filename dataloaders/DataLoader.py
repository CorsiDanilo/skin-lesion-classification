
from abc import ABC, abstractmethod
from typing import Optional

import os
import pandas as pd
import torch
from config import IMAGE_SIZE, DATASET_TEST_DIR, DATASET_TRAIN_DIR, METADATA_TEST_DIR, METADATA_NO_DUPLICATES_DIR, NORMALIZE, SEGMENTATION_DIR, BATCH_SIZE, SEGMENTATION_WITH_BOUNDING_BOX_DIR, BALANCE_UNDERSAMPLING
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
                 take_val_from_test: bool = True):
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
        self.take_val_from_test = take_val_from_test

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
                      'akiec': 3, 'bcc': 4, 'df': 5, 'vasc': 6}  # 2, 3, 4 malignant, otherwise begign
        labels_encoded = metadata['dx'].map(label_dict)
        metadata['label'] = labels_encoded
        # label_percentages = metadata['label'].value_counts(
        # normalize=True) * 100
        # print(f"Label percentages are: {label_percentages}")
        assert len(
            label_dict) == 7, "There should be 7 unique labels, increase the limit"
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

            if not self.take_val_from_test:
                df_train, df_val = train_test_split(
                    metadata,
                    test_size=0.2,
                    random_state=42,
                    stratify=metadata['label'])

                df_train["train"] = True
                return df_train, df_val

            metadata["train"] = train
            return metadata

        if not self.take_val_from_test:
            return metadata

        df_test, df_val = train_test_split(
            metadata,
            test_size=0.5,
            random_state=42,
            stratify=metadata['label'])

        return df_val, df_test

    def load_data(self, metadata: pd.DataFrame, idx: Optional[int] = None):
        if idx is not None:
            return self.load_images_and_labels_at_idx(metadata, idx)
        return self.load_images_and_labels(metadata)

    def get_train_dataloder(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        if self.take_val_from_test:
            self.df_train = self.load_metadata(limit=self.limit, train=True)

            if self.normalize:
                print(
                    "--Normalization-- Normalization flag set to True: Images will be normalized with z-score normalization")
                if self.normalization_statistics is None:
                    self.normalization_statistics = calculate_normalization_statistics(
                        self.df_train)
                    print(
                        "--Normalization-- Statistics not provided. They will be computed on the training set.")
                print(
                    f"--Normalization-- Statistics for normalization (per channel) -> Mean: {self.normalization_statistics[0].view(-1)}, Variance: {self.normalization_statistics[1].view(-1)}, Epsilon (adjustment value): 0.01")

            train_dataset = HAM10K(
                self.df_train,
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
        else:
            self.df_train, self.df_val = self.load_metadata(
                limit=self.limit, train=True)
            if self.normalize:
                print(
                    "--Normalization-- Normalization flag set to True: Images will be normalized with z-score normalization")
                if self.normalization_statistics is None:
                    self.normalization_statistics = calculate_normalization_statistics(
                        self.df_train)
                    print(
                        "--Normalization-- Statistics not provided. They will be computed on the training set.")
                print(
                    f"--Normalization-- Statistics for normalization (per channel) -> Mean: {self.normalization_statistics[0].view(-1)}, Variance: {self.normalization_statistics[1].view(-1)}, Epsilon (adjustment value): 0.01")

            train_dataset = HAM10K(
                self.df_train,
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
            val_dataset = HAM10K(
                self.df_val,
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
            return train_dataloader, val_dataloader

    def get_val_test_dataloader(self):
        if self.take_val_from_test:
            self.df_val, self.df_test = self.load_metadata(
                limit=self.limit, train=False)
            if self.normalize:
                if self.normalization_statistics is None:
                    print(
                        "--Normalization-- Normalization statistics not defined during test. Using default ones.")
                    self.normalization_statistics = DEFAULT_STATISTICS
                print(
                    f"--Normalization-- Statistics for normalization (per channel) -> Mean: {self.normalization_statistics[0].view(-1)}, Variance: {self.normalization_statistics[1].view(-1)}, Epsilon (adjustment value): 0.01")
            val_dataset = HAM10K(
                self.df_val,
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
            test_dataset = HAM10K(
                self.df_test,
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
            return val_dataloader, test_dataloader
        else:
            self.df_test = self.load_metadata(
                limit=self.limit, train=False)
            if self.normalize:
                if self.normalization_statistics is None:
                    print(
                        "--Normalization-- Normalization statistics not defined during test. Using default ones.")
                    self.normalization_statistics = DEFAULT_STATISTICS
                print(
                    f"--Normalization-- Statistics for normalization (per channel) -> Mean: {self.normalization_statistics[0].view(-1)}, Variance: {self.normalization_statistics[1].view(-1)}, Epsilon (adjustment value): 0.01")
            test_dataset = HAM10K(
                self.df_test,
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
