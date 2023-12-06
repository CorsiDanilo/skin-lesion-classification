from enum import Enum


class SegmentationStrategy(Enum):
    DYNAMIC_SEGMENTATION = "dynamic_segmentation"
    SEGMENTATION = "segmentation"
    NO_SEGMENTATION = "no_segmentation"


class DynamicSegmentationStrategy(Enum):
    OPENCV = "opencv"
    SAM = "sam"
