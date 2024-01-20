from typing import Optional
import torch
import random
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import torchvision.transforms.functional as TF
from albumentations.pytorch import ToTensorV2
import albumentations as A


class StatefulTransform:
    def __init__(self,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 always_rotate: bool = False):
        self.height = height
        self.width = width
        self.always_rotate = always_rotate

        self.image_only_transforms = A.Compose([
            # A.Resize(height=height, width=width),
            # A.Transpose(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
            A.MedianBlur(blur_limit=7, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),

            # A.OpticalDistortion(distort_limit=0.1, p=0.5),
            # A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
            # A.ElasticTransform(alpha=0.5, sigma=25, alpha_affine=25, p=0.5),

            A.CLAHE(clip_limit=4.0, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20,
                                 sat_shift_limit=30, val_shift_limit=20, p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
            #                    rotate_limit=45, p=0.5),
            # A.Cutout(num_holes=8, max_h_size=15,
            #          max_w_size=15, fill_value=0, p=0.5),
            ToTensorV2()
        ])

    def cutout(self, img, seg):
        seg_array = np.array(seg)

        img_width = self.width if self.width is not None else img.size[0]
        img_height = self.height if self.height is not None else img.size[1]
        size = min(img_height, img_width) // 2

        for _ in range(10):
            coverage = random.uniform(0.1, 0.6)
            x = random.randint(0, max(0, img_width - size))
            y = random.randint(0, max(0, img_height - size))
            new_size = int(size * coverage)

            # Check if there is at least one "255" pixel in the cutout area
            if not np.all(seg_array[y:y+new_size, x:x+new_size]) == 0:
                draw_img = ImageDraw.Draw(img)
                draw_seg = ImageDraw.Draw(seg)

                # Draw rectangles on both image and segmentation
                draw_img.rectangle([x, y, x + new_size, y + new_size], fill=0)
                draw_seg.rectangle([x, y, x + new_size, y + new_size], fill=0)
                break  # Break the loop if a valid cutout area is found

        return img, seg

    def __call__(self, img, seg):
        if self.height is not None and self.width is not None:
            # Resize
            img = transforms.Resize((self.height, self.width),
                                    interpolation=Image.BILINEAR)(img)
            seg = transforms.Resize((self.height, self.width),
                                    interpolation=Image.BILINEAR)(seg)

        # Cutout
        if random.random() > 0.7:
            img, seg = self.cutout(img, seg)

        # Horizonal flip
        if random.random() > 0.5:
            img = A.HorizontalFlip(p=1)(img)
            seg = A.HorizontalFlip(p=1)(seg)

        # Vertical flip
        if random.random() > 0.5:
            img = A.VerticalFlip(p=1)(img)
            seg = A.VerticalFlip(p=1)(seg)

        # Random rotation
        if self.always_rotate or random.random() > 0.5:
            angle = random.randint(1, 360)
            ssr = A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=angle, p=1)
            img = ssr(img)
            seg = ssr(seg)

        img = transforms.ToTensor()(img)
        seg = transforms.ToTensor()(seg)

        return img, seg
