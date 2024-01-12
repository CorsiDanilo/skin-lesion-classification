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
    invertor = Invertor(cfg=cfg, depth=6)

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
        first_image = images[2].unsqueeze(0)
        break

    invertor.train(first_image)


def manipulate_latent():
    device = select_device()
    latent_1 = torch.load("latents_1.pt")
    latent_2 = torch.load("latents_2.pt")
    latent = (latent_1 + latent_2) / 2
    gen = Generator(num_channels=3,
                    dlatent_size=512,
                    resolution=cfg.dataset.resolution,
                    structure="linear",
                    conditional=False,
                    **cfg.model.gen).to(device)
    gen.load_checkpoints(os.path.join(
        "checkpoints", "stylegan_ffhq_1024_gen.pth"))

    gen.eval()
    gen.g_synthesis.eval()

    g_synthesis = gen.g_synthesis

    img = g_synthesis(dlatents_in=latent, depth=6)

    print(img.shape)

    save_image(img, "manipulated.png")


if __name__ == '__main__':
    main()
    # manipulate_latent()
