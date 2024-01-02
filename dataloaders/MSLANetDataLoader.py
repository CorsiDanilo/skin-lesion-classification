import os
from typing import Optional, Tuple
import torch
import numpy as np

from typing import Optional

from PIL import Image, ImageDraw
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
import torchvision.transforms.functional as TF
from config import BATCH_SIZE, DATA_DIR, IMAGE_SIZE, NORMALIZE, RANDOM_SEED
import random

from dataloaders.DataLoader import DataLoader

random.seed(RANDOM_SEED)


class MSLANetDataLoader(DataLoader):
    def __init__(self,
                 limit: Optional[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 dynamic_load: bool = False,
                 resize_dim: Optional[Tuple[int, int]] = IMAGE_SIZE,
                 upscale_train: bool = False,
                 normalize: bool = NORMALIZE,
                 normalization_statistics: tuple = None,
                 batch_size: int = BATCH_SIZE):
        super().__init__(limit=limit,
                         transform=transform,
                         dynamic_load=dynamic_load,
                         upscale_train=upscale_train,
                         normalize=normalize,
                         normalization_statistics=normalization_statistics,
                         batch_size=batch_size,
                         always_rotate=False,
                         data_dir=os.path.join(DATA_DIR, "gradcam_output"))
        self.resize_dim = resize_dim

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int):
        img = metadata.iloc[idx]
        label = img['label']
        image = Image.open(img['image_path'])
        image = TF.to_tensor(image)
        # image = self.transform(image)
        return image, label

    def load_images_and_labels(self, metadata: pd.DataFrame):
        images = []
        segmentations = []
        labels = []

        for index, (row_index, img) in tqdm(enumerate(metadata.iterrows()), desc=f'Loading images'):
            image, label = self.load_images_and_labels_at_idx(
                idx=index, metadata=metadata)
            images.append(image)
            labels.append(label)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        print(f"---Data Loader--- Images uploaded: " + str(len(images)))

        return images, labels
