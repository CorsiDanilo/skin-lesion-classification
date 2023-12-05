import os


DATA_DIR = 'data'
PATH_TO_SAVE_RESULTS = 'results'
DATASET_TRAIN_DIR = os.path.join(DATA_DIR, "HAM10000_images_train")
DATASET_TEST_DIR = os.path.join(DATA_DIR, "HAM10000_images_test")
SEGMENTATION_DIR = os.path.join(
    DATA_DIR, 'HAM10000_segmentations_lesion_tschandl')
SEGMENTATION_WITH_BOUNDING_BOX_DIR = os.path.join(
    DATA_DIR, 'HAM10000_segmentations_lesion_tschandl_with_bounding_box')
METADATA_TRAIN_DIR = os.path.join(DATA_DIR, 'HAM10000_metadata_train.csv')
METADATA_NO_DUPLICATES_DIR = os.path.join(
    DATA_DIR, 'HAM10000_metadata_train_no_duplicates.csv')
METADATA_TEST_DIR = os.path.join(DATA_DIR, 'HAM10000_metadata_test.csv')

BATCH_SIZE = 256

USE_WANDB = True
USE_DML = False #DirectML library for AMD gpu on Windows (set to false if you want to use cpu or standard CUDA)
SAVE_RESULTS = True #Save results in JSON locally
SAVE_MODELS = True #Save models locally

# Configurations
INPUT_SIZE = 3
NUM_CLASSES = 7
HIDDEN_SIZE = [32, 64, 128, 256]
N_EPOCHS = 5
LR = 1e-3
LR_DECAY = 0.85
REG = 0.01
SEGMENT = True
CROP_ROI = True
ARCHITECTURE_CNN = "densenet121" # resnet24, densenet121, inception_v3
ARCHITECTURE_VIT = "standard" #standard, pretrained, efficient
DATASET_LIMIT = 200
DROPOUT_P = 0.3
NORMALIZE = True
SEGMENTATION_BOUNDING_BOX = True # If true, the segmentation is approximated by a squared bounding box.
BALANCE_UNDERSAMPLING = 0.5
USE_DOUBLE_LOSS = True #Use binary loss (benign/malign) and multiclassification loss if true, otherwise use only the multiclassification one

# Transformers configurations
if ARCHITECTURE_CNN == "inception_v3":
    IMAGE_SIZE = (299, 299) # for inception_v3
else:
    IMAGE_SIZE = (224, 224) # for the others
N_HEADS = 1
N_LAYERS = 1
PATCH_SIZE = 16
EMB_SIZE = 800

