from typing import Optional
from config import BATCH_SIZE, DATASET_LIMIT, KEEP_BACKGROUND, NORMALIZE
from dataloaders.DataLoader import DataLoader
from dataloaders.DynamicSegmentationDataLoader import DynamicSegmentationDataLoader
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from dataloaders.DEPRECATED_SegmentedImagesDataLoader import DEPRECTED_SegmentedImagesDataLoader
from shared.enums import DynamicSegmentationStrategy, SegmentationStrategy


def get_dataloder_from_strategy(strategy: SegmentationStrategy,
                                dynamic_segmentation_strategy: DynamicSegmentationStrategy = DynamicSegmentationStrategy.SAM,
                                limit: int = DATASET_LIMIT,
                                dynamic_load: bool = True,
                                upsample_train: bool = True,
                                normalize: bool = NORMALIZE,
                                normalization_statistics: tuple = None,
                                batch_size: int = BATCH_SIZE,
                                keep_background: Optional[bool] = KEEP_BACKGROUND) -> DataLoader:

    if strategy == SegmentationStrategy.DYNAMIC_SEGMENTATION.value:
        dataloader = DynamicSegmentationDataLoader(
            limit=limit,
            dynamic_load=dynamic_load,
            upscale_train=upsample_train,
            segmentation_strategy=dynamic_segmentation_strategy,
            normalize=normalize,
            normalization_statistics=normalization_statistics,
            batch_size=batch_size,
            keep_background=keep_background,
        )
    elif strategy == SegmentationStrategy.SEGMENTATION.value:
        print(f"!------WARNING-----!: SegmentationStrategy doesn't work if the validation set is taken from the test set, since it doesn't have the segmentation!!! ಥ_ಥ")
        dataloader = DEPRECTED_SegmentedImagesDataLoader(
            limit=limit,
            dynamic_load=dynamic_load,
            upscale_train=upsample_train,
            normalize=normalize,
            normalization_statistics=normalization_statistics,
            batch_size=batch_size,
            keep_background=keep_background,

        )
    elif strategy == SegmentationStrategy.NO_SEGMENTATION.value:
        dataloader = ImagesAndSegmentationDataLoader(
            limit=limit,
            dynamic_load=dynamic_load,
            upscale_train=upsample_train,
            normalize=normalize,
            normalization_statistics=normalization_statistics,
            batch_size=batch_size,

        )
    else:
        raise NotImplementedError(
            f"Segmentation strategy {strategy} not implemented")
    return dataloader
