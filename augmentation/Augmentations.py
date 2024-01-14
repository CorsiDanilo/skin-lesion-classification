import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple


class MSLANetAugmentation:
    def __init__(self, resize_dim: Tuple[int, int] = (256, 256)):
        self.transform = A.Compose([
            A.Resize(resize_dim),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
            A.MedianBlur(blur_limit=7, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.OpticalDistortion(distort_limit=1.0, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=1., p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20,
                                 sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                               rotate_limit=45, p=0.5),
            A.Cutout(num_holes=8, max_h_size=8,
                     max_w_size=8, fill_value=0, p=0.5),
            ToTensorV2()
        ])
