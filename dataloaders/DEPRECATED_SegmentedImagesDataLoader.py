import random
from config import BATCH_SIZE, KEEP_BACKGROUND, NORMALIZE
from dataloaders.DataLoader import DataLoader
from typing import Optional
import torch
import pandas as pd

from typing import Optional

import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from utils.utils import crop_image_from_box, get_bounding_boxes_from_segmentation


class DEPRECTED_SegmentedImagesDataLoader(DataLoader):
    """
    This class is used to load the images and create the dataloaders.
    The dataloder will output a tuple of (images, labels).
    The images are already segmented.
    """

    def __init__(self,
                 limit: Optional[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 dynamic_load: bool = False,
                 upscale_train: bool = True,
                 normalize: bool = NORMALIZE,
                 batch_size: int = BATCH_SIZE,
                 keep_background: bool = KEEP_BACKGROUND):
        super().__init__(limit=limit,
                         transform=transform,
                         dynamic_load=dynamic_load,
                         upscale_train=upscale_train,
                         normalize=normalize,
                         batch_size=batch_size)
        self.keep_background = keep_background
        print(f"Dynamic Load for Segmentation Dataloader: {self.dynamic_load}")

    def load_images_and_labels_at_idx(self, metadata: pd.DataFrame, idx: int, transform: transforms.Compose = None):
        img = metadata.iloc[idx]
        label = img['label']
        ti, ts, ts_bbox = Image.open(img['image_path']), Image.open(
            img['segmentation_path']).convert('1'), Image.open(img['segmentation_bbox_path']).convert('1')
        ti, ts, ts_bbox = TF.to_tensor(ti), TF.to_tensor(ts), TF.to_tensor(
            ts_bbox)
        # ti = zoom_out(ti)
        if img["augmented"]:
            if not self.keep_background:
                ti = ti * ts
            pil_image = TF.to_pil_image(ti)
            image = self.transform(pil_image)
            image = image * ts_bbox
        else:
            image = ti
            if not self.keep_background:
                image = ti * ts
            image = image * ts_bbox

        bbox = get_bounding_boxes_from_segmentation(ts_bbox)[0]
        image = crop_image_from_box(image, bbox)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, label

    def load_images_and_labels(self, metadata: pd.DataFrame):
        not_found_files = []
        images = []
        labels = []
        for index, (row_index, img) in tqdm(enumerate(metadata.iterrows()), desc=f'Loading images'):
            image, label = self.load_images_and_labels_at_idx(
                idx=index, metadata=metadata)
            images.append(image)
            labels.append(label)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        print(f"---Data Loader--- Images uploaded: " + str(len(images)))

        print(
            f"Loading complete, some files ({len(not_found_files)}) were not found: {not_found_files}")
        return images, labels
