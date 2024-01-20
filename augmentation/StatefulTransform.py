from typing import Optional
import torch
import random
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import torchvision.transforms.functional as TF


class StatefulTransform:
    def __init__(self,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 always_rotate: bool = False):
        self.height = height
        self.width = width
        self.always_rotate = always_rotate

    def add_gaussian_noise(self, image):
        """
        Add gaussian noise to a PIL Image.
        """
        mean = 0
        stddev = 0.1
        noisy_image = image + torch.randn(image.shape) * stddev + mean
        noisy_image = np.clip(noisy_image, 0., 1.)
        return noisy_image

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

        # Apply the grid distortion
        # if random.random() > 0.5:
        #     # Convert tensors back to PIL Images
        #     # img_pil = to_pil_image(img)
        #     # seg_pil = to_pil_image(seg)

        #     grid_distortion = GridDistortion(p=1)
        #     img = grid_distortion(image=np.array(
        #         img).astype(np.float32))["image"]
        #     seg = grid_distortion(image=np.array(
        #         seg).astype(np.float32))["image"]

        #     img = Image.fromarray(img.astype(np.uint8))
        #     seg = Image.fromarray(seg.astype(np.uint8))

        # Cutout
        if random.random() > 0.7:
            img, seg = self.cutout(img, seg)

        # Horizonal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            seg = TF.hflip(seg)

        # Vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            seg = TF.vflip(seg)

        # Random rotation
        if self.always_rotate or random.random() > 0.5:
            angle = random.randint(1, 360)
            img = TF.rotate(img, angle)
            seg = TF.rotate(seg, angle)

        # if random.random() > 0.5:
        #     elastic_transform = transforms.ElasticTransform()
        #     img = elastic_transform(img)
        #     seg = elastic_transform(seg)

        # if random.random() > 0.5:
        #     color_jitter = ColorJitter(
        #         brightness=0.2, contrast=0.2, saturation=0.2)
        #     img = color_jitter(img)

        img = transforms.ToTensor()(img)
        seg = transforms.ToTensor()(seg)

        # # Add Gaussian noise
        # if random.random() > 0.5:
        #     img = self.add_gaussian_noise(img)

        return img, seg
