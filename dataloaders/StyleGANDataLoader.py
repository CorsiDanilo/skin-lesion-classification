import os

from sklearn.model_selection import train_test_split
from torchvision.transforms.transforms import Compose
from dataloaders.DataLoader import DataLoader
from typing import Dict, List, Optional, Tuple
import torch
from torchvision import transforms
from typing import Optional
from augmentation.Augmentations import Augmentations
from PIL import Image
from tqdm import tqdm
import pandas as pd
from config import BATCH_SIZE, DATA_DIR, DATASET_TRAIN_DIR, IMAGE_SIZE, METADATA_TRAIN_DIR,  RANDOM_SEED
import random
from datasets.StyleGANPairsDataset import StyleGANPairsDataset

random.seed(RANDOM_SEED)


class StyleGANDataLoader(DataLoader):

    def __init__(self,
                 limit: Optional[int] = None,
                 dynamic_load: bool = False,
                 resize_dim: Optional[Tuple[int, int]] = IMAGE_SIZE,
                 batch_size: int = BATCH_SIZE,):
        super().__init__(limit=limit,
                         transform=None,
                         dynamic_load=dynamic_load,
                         upscale_train=False,
                         normalize=False,
                         normalization_statistics=None,
                         batch_size=batch_size,
                         always_rotate=False)
        self.resize_dim = resize_dim
        self.transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.ToTensor()
        ])
        self.augmented_transform = MSLANetAugmentation(
            resize_dim=resize_dim).transform
        self.split_metadatas = self.split_metadata_by_label(self.train_df)

    def split_metadata_by_label(self, metadata: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        unique_labels = metadata['label'].unique()
        split_metadatas = {
            label: metadata[metadata['label'] == label] for label in unique_labels}
        return split_metadatas

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int):
        img = metadata.iloc[idx]
        label = img['label']
        image_path = img['image_path']
        augmented = img['augmented']
        image = Image.open(image_path)
        if augmented:
            image = self.augmented_transform(image)
        else:
            image = self.transform(image)
        return image, label, image_path, augmented

    def load_images_and_labels(self, metadata: pd.DataFrame):
        raise NotImplementedError(
            "Non-Dynamic load is still not implemented for StyleGAN")
        images = []
        labels = []

        for index, (_, _) in tqdm(enumerate(metadata.iterrows()), desc=f'Loading images'):
            image, label = self.load_images_and_labels_at_idx(
                idx=index, metadata=metadata)
            images.append(image)
            labels.append(label)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        print(f"---Data Loader--- Images uploaded: " + str(len(images)))

        return images, labels

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
        ori_data_dir = DATASET_TRAIN_DIR
        low_data_dir = os.path.join(DATA_DIR, "gradcam_output_70")
        high_data_dir = os.path.join(DATA_DIR, "gradcam_output_110")
        metadata['image_path'] = metadata['image_id'].apply(
            lambda x: os.path.join(ori_data_dir, x + '.jpg'))
        metadata['image_path_low'] = metadata['image_id'].apply(
            lambda x: os.path.join(low_data_dir, x + '.jpg'))
        metadata['image_path_high'] = metadata['image_id'].apply(
            lambda x: os.path.join(high_data_dir, x + '.jpg'))

        df_train, df_test = train_test_split(
            metadata,
            test_size=0.1,  # 15% test, 85% train
            random_state=RANDOM_SEED,
            stratify=metadata['dx'])

        df_train, df_val = train_test_split(
            df_train,
            test_size=0.2,  # Of the 85% train, 10% val, 90% train
            random_state=RANDOM_SEED,
            stratify=df_train['dx'])

        assert len(df_train['label'].unique(
        )) == 7, f"Number of unique labels in metadata is not 7, it's {len(df_train['label'].unique())}, increase the limit"
        assert len(df_val['label'].unique(
        )) == 7, f"Number of unique labels in metadata is not 7, it's {len(df_val['label'].unique())}, increase the limit"
        assert len(df_test['label'].unique(
        )) == 7, f"Number of unique labels in metadata is not 7, it's {len(df_test['label'].unique())}, increase the limit"

        df_train["train"] = True
        df_val["train"] = False
        df_test["train"] = False

        print(f"---TRAIN---: {len(df_train)} entries")
        print(f"---VAL---: {len(df_val)} entries")
        print(f"---TEST---: {len(df_test)} entries")
        return df_train, df_val, df_test

    def get_train_dataloder(self) -> List[torch.utils.data.DataLoader]:
        train_datasets = {}
        train_dataloaders = {}
        for label, metadata in self.split_metadatas.items():
            train_datasets[label] = StyleGANPairsDataset(
                metadata,
                load_data_fn=self.load_data,
                resize_dims=IMAGE_SIZE,
                dynamic_load=self.dynamic_load)

            train_dataloaders[label] = torch.utils.data.DataLoader(
                train_datasets[label],
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=False,
            )
        return train_dataloaders

    def get_val_dataloader(self):
        raise NotImplementedError(
            "Validation dataloader should not be used for StyleGAN")

    def get_test_dataloader(self):
        raise NotImplementedError(
            "Test dataloader should not be used for StyleGAN")
