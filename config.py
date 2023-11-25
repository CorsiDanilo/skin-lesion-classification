import os


DATA_DIR = 'data'
DATASET_TRAIN_DIR = os.path.join(DATA_DIR, "HAM10000_images_train")
DATASET_TEST_DIR = os.path.join(DATA_DIR, "HAM10000_images_test")
SEGMENTATION_DIR = os.path.join(
    DATA_DIR, 'HAM10000_segmentations_lesion_tschandl')
METADATA_TRAIN_DIR = os.path.join(DATA_DIR, 'HAM10000_metadata_train.csv')
METADATA_TEST_DIR = os.path.join(DATA_DIR, 'HAM10000_metadata_test.csv')

BATCH_SIZE = 512

USE_WANDB = True
# Configurations
INPUT_SIZE = 3
NUM_CLASSES = 7
HIDDEN_SIZE = [32, 64, 128, 256]
N_EPOCHS = 20
LR = 1e-3
LR_DECAY = 0.85
REG = 0.01
SEGMENT = False
CROP_ROI = False
ARCHITECHTURE = "resnet24"
DATASET_LIMIT = None
DROPOUT_P = 0.5
NORMALIZE = True
HISTOGRAM_NORMALIZATION = False
