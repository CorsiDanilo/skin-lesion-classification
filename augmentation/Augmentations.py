import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any, Tuple, Optional


class Augmentations:
    def __init__(self, resize_dim: Optional[Tuple[int, int]] = (256, 256), additional_targets: dict = {}):
        transforms = []
        if resize_dim is not None:
            transforms.append(
                A.Resize(height=resize_dim[0], width=resize_dim[1]))
            transforms.append(
                A.Transpose(p=0.5)
            )
        transforms.extend([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
            A.MedianBlur(blur_limit=7, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),

            A.OpticalDistortion(distort_limit=0.1, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
            A.ElasticTransform(alpha=0.5, sigma=25, alpha_affine=25, p=0.5),

            A.CLAHE(clip_limit=4.0, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20,
                                 sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                               rotate_limit=45, p=0.5),
            A.Cutout(num_holes=1, max_h_size=80,
                     max_w_size=80, fill_value=0, p=0.5),
            ToTensorV2()
        ])
        self.transform = A.Compose(
            transforms, additional_targets=additional_targets)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.transform(*args, **kwargs)
