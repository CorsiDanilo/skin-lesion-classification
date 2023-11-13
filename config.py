from enum import Enum
import os


DATA_DIR = 'data'
DATASET_DIR = os.path.join(DATA_DIR, "HAM10000_images")
SEGMENTATION_DIR = os.path.join(
    DATA_DIR, 'HAM10000_segmentations_lesion_tschandl')
METADATA_DIR = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')


BATCH_SIZE = 32
