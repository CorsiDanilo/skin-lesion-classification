from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
import os
import wandb
from enum import Enum
from sklearn.metrics import recall_score, accuracy_score
from config import BALANCE_UNDERSAMPLING, BATCH_SIZE, INPUT_SIZE, NUM_CLASSES, HIDDEN_SIZE, N_EPOCHS, LR, REG, SEGMENT, CROP_ROI, ARCHITECHTURE, DATASET_LIMIT, DROPOUT_P, NORMALIZE, SEGMENTATION_BOUNDING_BOX, USE_WANDB, USE_DML, USE_DOUBLE_LOSS, N_HEADS, N_LAYERS, PATCH_SIZE, EMB_SIZE, IMAGE_SIZE
from dataloaders.DynamicSegmentationDataLoader import DynamicSegmentationDataLoader, DynamicSegmentationStrategy
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from dataloaders.SegmentedImagesDataLoader import SegmentedImagesDataLoader
from models.ResNet24Pretrained import ResNet24Pretrained
from models.DenseNetPretrained import DenseNetPretrained
from models.InceptionV3Pretrained import InceptionV3Pretrained

if USE_DML:
    import torch_directml

class SegmentationStrategy(Enum):
    DYNAMIC_SEGMENTATION = "dynamic_segmentation"
    SEGMENTATION = "segmentation"
    NO_SEGMENTATION = "no_segmentation"


SEGMENTATION_STRATEGY = SegmentationStrategy.SEGMENTATION
DYNAMIC_SEGMENTATION_STRATEGY = DynamicSegmentationStrategy.OPENCV if SEGMENTATION_STRATEGY == SegmentationStrategy.DYNAMIC_SEGMENTATION else None
dynamic_load = True

if SEGMENTATION_STRATEGY == SegmentationStrategy.DYNAMIC_SEGMENTATION:
    dataloader = DynamicSegmentationDataLoader(
        limit=DATASET_LIMIT,
        dynamic_load=dynamic_load,
        segmentation_strategy=DYNAMIC_SEGMENTATION_STRATEGY
    )
elif SEGMENTATION_STRATEGY == SegmentationStrategy.SEGMENTATION:
    dataloader = SegmentedImagesDataLoader(
        limit=DATASET_LIMIT,
        dynamic_load=dynamic_load,
    )
elif SEGMENTATION_STRATEGY == SegmentationStrategy.NO_SEGMENTATION:
    dataloader = ImagesAndSegmentationDataLoader(
        limit=DATASET_LIMIT,
        dynamic_load=dynamic_load,
    )
else:
    raise NotImplementedError(
        f"Segmentation strategy {SEGMENTATION_STRATEGY} not implemented")

def select_device():
    if USE_DML:
        return torch_directml.device()
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = select_device()

RESUME = False
FROM_EPOCH = 0

if CROP_ROI:
    assert SEGMENT, f"Crop roi needs segment to be True"

if USE_WANDB:
    # Start a new run
    wandb.init(
        project="melanoma",

        # track hyperparameters and run metadata
        config={
            "learning_rate": LR,
            "architecture": ARCHITECHTURE,
            "epochs": N_EPOCHS,
            'reg': REG,
            'batch_size': BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "dataset": "HAM10K",
            "optimizer": "AdamW",
            "segmentation": SEGMENT,
            "crop_roi": CROP_ROI,
            "dataset_limit": DATASET_LIMIT,
            "dropout_p": DROPOUT_P,
            "normalize": NORMALIZE,
            "resumed": RESUME,
            "from_epoch": FROM_EPOCH,
            "segmentation_bounding_box": SEGMENTATION_BOUNDING_BOX,
            "balance_undersampling": BALANCE_UNDERSAMPLING,
            "initialization": "default",
            "segmentation_strategy": SEGMENTATION_STRATEGY,
            "dynamic_segmentation_strategy": DYNAMIC_SEGMENTATION_STRATEGY,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "patch_size": PATCH_SIZE,
            "emb_size": EMB_SIZE
        },
        resume=RESUME,
    )

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


RANDOM_SEED = 42
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")