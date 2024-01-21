from tqdm import tqdm
from dataloaders.DynamicSegmentationDataLoader import DynamicSegmentationDataLoader
from shared.constants import IMAGENET_STATISTICS
from shared.enums import DynamicSegmentationStrategy


def generate_segmentations():
    dataloder = DynamicSegmentationDataLoader(
        limit=None,
        dynamic_load=True,
        upscale_train=False,
        segmentation_strategy=DynamicSegmentationStrategy.SAM.value,
        normalize=False,
        keep_background=True,
        normalization_statistics=IMAGENET_STATISTICS,
        batch_size=8,
        load_synthetic=True
    )
    train_loader = dataloder.get_train_dataloder()
    for batch in tqdm(train_loader):
        pass
    pass


if __name__ == "__main__":
    generate_segmentations()
