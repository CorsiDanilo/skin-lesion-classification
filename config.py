import os

from shared.enums import DynamicSegmentationStrategy, SegmentationStrategy

# ---Dataset Configurations--- #
DATA_DIR = 'data'
PATH_TO_SAVE_RESULTS = 'results'
DATASET_TRAIN_DIR = os.path.join(DATA_DIR, "HAM10000_images_train")
AUGMENTED_IMAGES_DIR = os.path.join(DATA_DIR, "HAM10000_augmented_images")
# DATASET_TEST_DIR = os.path.join(DATA_DIR, "HAM10000_images_test")
SEGMENTATION_DIR = os.path.join(
    DATA_DIR, 'HAM10000_segmentations_lesion_tschandl')
# SEGMENTATION_WITH_BOUNDING_BOX_DIR = os.path.join(
# DATA_DIR, 'HAM10000_segmentations_lesion_tschandl_with_bounding_box_450_600')
METADATA_TRAIN_DIR = os.path.join(
    DATA_DIR, 'HAM10000_metadata_train.csv')
SYNTHETIC_METADATA_TRAIN_DIR = os.path.join(
    DATA_DIR, 'synthetic_metadata_train.csv')
# METADATA_NO_DUPLICATES_DIR = os.path.join(
# DATA_DIR, 'HAM10000_metadata_train.csv')
# METADATA_TEST_DIR = os.path.join(DATA_DIR, 'HAM10000_metadata_test.csv')
DYNAMIC_LOAD = True  # True if you want to load images dynamically, False otherwise

# ---Library Configurations--- #
USE_WANDB = False  # Use wandb for logging
# DirectML library for AMD gpu on Windows (set to false if you want to use cpu or standard CUDA)
USE_DML = True
USE_MPS = False  # Use MPS gpu for MacOS

# ---Train Configurations--- #
RANDOM_SEED = 42  # Random seed
BATCH_SIZE = 16 # Batch size
INPUT_SIZE = 3  # Input size
NUM_CLASSES = 7  # Number of classes for classification
HIDDEN_SIZE = [256, 128]  # Hidden layers configurations
N_EPOCHS = 30  # Number of epochs
LR = 1e-4  # Learning rate
LR_DECAY = 0.85  # Learning rate decay
REG = 0.03  # Weight decay
# Architecture used for training: resnet34, densenet121, inception_v3, standard, pretrained, efficient
ARCHITECTURE = "resnet34"
DATASET_LIMIT = None  # Value (0, dataset_length) used to limit the dataset
DROPOUT_P = 0.3  # Dropout probability
NUM_DROPOUT_LAYERS = 1 # Used in MSLANet to apply several parallel classification layers with a dropout in it. Predictions are averaged to get the final result.
NORMALIZE = True  # True if data must be normalized, False otherwise
OVERSAMPLE_TRAIN = True # True if oversampling (with data augmentation) must be applied, False otherwise
BALANCE_DOWNSAMPLING = 1 # Proporsion used to downsample the majority. Applied only if OVERSAMPLE_TRAIN=True (1=Do not remove any examples from majority class).

# Use binary loss (benign/malign) and multiclassification loss if true, otherwise use only the multiclassification one
USE_MULTIPLE_LOSS = False
# Value used to establish the importance of multiclassification loss over the binary classification loss
MULTIPLE_LOSS_BALANCE = 0.5
# Segmentation approch (NO_segmentation or DYNAMIC_SEGMENTATION to segment the mole with SAM)
SEGMENTATION_STRATEGY = SegmentationStrategy.DYNAMIC_SEGMENTATION.value
# SAM or OPENCV (the latter is deprecated due to low performances)
DYNAMIC_SEGMENTATION_STRATEGY = DynamicSegmentationStrategy.SAM.value
# If true, the background is kept in the segmentation, otherwise it is removed
KEEP_BACKGROUND = True

if ARCHITECTURE == "inception_v3":
    IMAGE_SIZE = (299, 299)  # for inception_v3
else:
    IMAGE_SIZE = (224, 224)  # for the others

# ---Transformers configurations--- #
N_HEADS = 1  # Number of heads for multi-head (self) attention
N_LAYERS = 1  # Number of block layers
PATCH_SIZE = 16  # Patch size
EMB_SIZE = 800  # Final embedding size

# ---General Model and Debug Configurations--- #
SAVE_RESULTS = True  # Save results in JSON locally
SAVE_MODELS = True # Save models locally
PRINT_MODEL_ARCHITECTURE = False  # Print the architecture of the model

# ---Resume Train Configurations--- #
RESUME = False  # True if you have to keep training a model, False if the model must be trained from scratch
# Path of where the model is saved
PATH_MODEL_TO_RESUME = f"resnet34_2023-12-09_09-09-54"
# Resume for epoch (usually the latest one before training was interrupted)
RESUME_EPOCH = 2
