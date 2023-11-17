from enum import Enum
import os


DATA_DIR = 'data'
DATASET_TRAIN_DIR = os.path.join(DATA_DIR, "HAM10000_images_train")
DATASET_TEST_DIR = os.path.join(DATA_DIR, "HAM10000_images_test")
SEGMENTATION_DIR = os.path.join(
    DATA_DIR, 'HAM10000_segmentations_lesion_tschandl')
METADATA_TRAIN_DIR = os.path.join(DATA_DIR, 'HAM10000_metadata_train.csv')
METADATA_TEST_DIR = os.path.join(DATA_DIR, 'HAM10000_metadata_test.csv')


BATCH_SIZE = 32
INPUT_SIZE = 3
NUM_CLASSES = 7
HIDDEN_SIZE = [32, 64, 128, 256]
N_EPOCHS = 5
LR = 1e-5
LR_DECAY = 0.85
REG = 0.001