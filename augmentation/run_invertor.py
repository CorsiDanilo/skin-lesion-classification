import os
import torch
from torchvision.utils import save_image
from .GAN import Generator
from utils.utils import select_device
from .Invertor import Invertor
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from .gan_config import cfg
from shared.constants import IMAGENET_STATISTICS


def main():
    batch_size = 32
    invertor = Invertor(cfg=cfg)

    fixed_dataloader = ImagesAndSegmentationDataLoader(
        limit=None,
        dynamic_load=True,
        upscale_train=False,
        normalize=False,
        normalization_statistics=IMAGENET_STATISTICS,
        batch_size=batch_size,
        resize_dim=(
            cfg.dataset.resolution,
            cfg.dataset.resolution,
        )
    )
    fixed_data = fixed_dataloader.get_test_dataloader()
    for batch in fixed_data:
        images, _ = batch

        images = images.to(invertor.device)
        first_image = images[0].unsqueeze(0)
        second_image = images[1].unsqueeze(0)
        break

    latent_1 = invertor.embed(first_image, "first_image")
    latent_2 = invertor.embed(second_image, "second_image")

    invertor.style_transfer(latent_1, latent_2)


def offline_style_transfer():
    invertor = Invertor(cfg=cfg)
    latent_path = invertor.latents_dir
    first_latent_path = os.path.join(latent_path, "first_image.pt")
    second_latent_path = os.path.join(latent_path, "second_image.pt")
    latent_1 = torch.load(first_latent_path)
    latent_2 = torch.load(second_latent_path)
    # invertor.style_transfer(latent_1, latent_2)
    image_1 = invertor.generate(latent_1)
    image_2 = invertor.generate(latent_2)
    save_image(image_1, "image_1.png")
    save_image(image_2, "image_2.png")
    invertor.mix_latents(latent_1, latent_2)


def embed_full_dataset():
    pass


if __name__ == '__main__':
    # main()
    offline_style_transfer()
