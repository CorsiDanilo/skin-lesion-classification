import os


DATA_DIR = 'data'
DATASET_TRAIN_DIR = os.path.join(DATA_DIR, "HAM10000_images_train")
DATASET_TEST_DIR = os.path.join(DATA_DIR, "HAM10000_images_test")
SEGMENTATION_DIR = os.path.join(
    DATA_DIR, 'HAM10000_segmentations_lesion_tschandl')
SEGMENTATION_WITH_BOUNDING_BOX_DIR = os.path.join(
    DATA_DIR, 'HAM10000_segmentations_lesion_tschandl_with_bounding_box_450_600')
METADATA_TRAIN_DIR = os.path.join(DATA_DIR, 'HAM10000_metadata_train.csv')
METADATA_NO_DUPLICATES_DIR = os.path.join(
    DATA_DIR, 'HAM10000_metadata_train_no_duplicates.csv')
METADATA_TEST_DIR = os.path.join(DATA_DIR, 'HAM10000_metadata_test.csv')

BATCH_SIZE = 128

USE_WANDB = True
# Configurations
INPUT_SIZE = 3
NUM_CLASSES = 7
HIDDEN_SIZE = [32, 64, 128, 256]
N_EPOCHS = 100
LR = 1e-3
LR_DECAY = 0.85
REG = 0.01
SEGMENT = True
CROP_ROI = True
ARCHITECHTURE = "resnet24"
DATASET_LIMIT = None
DROPOUT_P = 0.3
NORMALIZE = True
# If true, the segmentation is approximated by a squared bounding box.
SEGMENTATION_BOUNDING_BOX = True
BALANCE_UNDERSAMPLING = 0.5
