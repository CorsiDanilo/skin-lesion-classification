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
        first_image = images[0].unsqueeze(0)
        break

    images = images.to(invertor.device)
    first_image = images[0].unsqueeze(0)

    invertor.train(first_image)


if __name__ == '__main__':
    main()
