from dataloaders.SegmentedImagesDataLoader import SegmentedImagesDataLoader
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader


def test_segmeneted_images_dataloader():
    dataloder = SegmentedImagesDataLoader(dynamic_load=True)
    train_dataloder, val_dataloder = dataloder.get_train_val_dataloders()
    assert len(train_dataloder) > 0
    assert len(val_dataloder) > 0
    train_batch = next(iter(train_dataloder))
    assert len(train_batch) == 2
    val_batch = next(iter(val_dataloder))
    assert len(val_batch) == 2
    tr_images, tr_labels = train_batch
    val_images, val_labels = val_batch
    assert tr_images.shape == val_images.shape
    assert tr_labels.shape == val_labels.shape


def test_images_and_segmentation_dataloader():
    dataloder = ImagesAndSegmentationDataLoader(dynamic_load=True)
    train_dataloder, val_dataloder = dataloder.get_train_val_dataloders()
    assert len(train_dataloder) > 0
    assert len(val_dataloder) > 0
    train_batch = next(iter(train_dataloder))
    assert len(train_batch) == 3
    val_batch = next(iter(val_dataloder))
    assert len(val_batch) == 3
    tr_images, tr_labels, tr_segmentations = train_batch
    val_images, val_labels, val_segmentations = val_batch
    assert tr_images.shape == val_images.shape
    assert tr_segmentations.shape == val_segmentations.shape
    assert tr_labels.shape == val_labels.shape
