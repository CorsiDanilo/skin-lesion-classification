from typing import Optional
from config import BATCH_SIZE, DATASET_LIMIT, KEEP_BACKGROUND, NORMALIZE
from dataloaders.DataLoader import DataLoader
from dataloaders.DynamicSegmentationDataLoader import DynamicSegmentationDataLoader
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from dataloaders.SegmentedImagesDataLoader import SegmentedImagesDataLoader
from shared.enums import DynamicSegmentationStrategy, SegmentationStrategy


def get_dataloder_from_strategy(strategy: SegmentationStrategy,
                                dynamic_segmentation_strategy: DynamicSegmentationStrategy = DynamicSegmentationStrategy.OPENCV,
                                limit: int = DATASET_LIMIT,
                                dynamic_load: bool = True,
                                upsample_train: bool = True,
                                normalize: bool = NORMALIZE,
                                batch_size: int = BATCH_SIZE,
                                keep_background: Optional[bool] = KEEP_BACKGROUND) -> DataLoader:

    if strategy == SegmentationStrategy.DYNAMIC_SEGMENTATION.value:
        # TODO: Note that this dataloader is broken. I will fix it, I promise.
        dataloader = DynamicSegmentationDataLoader(
            limit=limit,
            dynamic_load=dynamic_load,
            train=True,  # TODO: fix this
            upscale_train=upsample_train,
            segmentation_strategy=dynamic_segmentation_strategy,
            normalize=normalize,
            batch_size=batch_size,
        )
    elif strategy == SegmentationStrategy.SEGMENTATION.value:
        dataloader = SegmentedImagesDataLoader(
            limit=limit,
            dynamic_load=dynamic_load,
            upscale_train=upsample_train,
            normalize=normalize,
            batch_size=batch_size,
            keep_background=keep_background,
        )
    elif strategy == SegmentationStrategy.NO_SEGMENTATION.value:
        dataloader = ImagesAndSegmentationDataLoader(
            limit=limit,
            dynamic_load=dynamic_load,
            upscale_train=upsample_train,
            normalize=normalize,
            batch_size=batch_size,
        )
    else:
        raise NotImplementedError(
            f"Segmentation strategy {strategy} not implemented")
    return dataloader
