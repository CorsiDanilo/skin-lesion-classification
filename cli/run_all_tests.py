# from argparse import ArgumentParser
# import itertools
# import json
# import os

# import torch
# from tqdm import tqdm
# import wandb
# from cli.hparams_tuning import build_dataloaders, get_model
# from models.ResNet34Pretrained import ResNet34Pretrained
# from models.DenseNetPretrained import DenseNetPretrained
# from models.InceptionV3Pretrained import InceptionV3Pretrained
# from models.ViTStandard import ViT_standard
# from models.ViTPretrained import ViT_pretrained
# from models.ViTEfficient import EfficientViT
# from config import BALANCE_DOWNSAMPLING, BATCH_SIZE, DYNAMIC_SEGMENTATION_STRATEGY, EMB_SIZE, IMAGE_SIZE, INPUT_SIZE, N_HEADS, N_LAYERS, NUM_CLASSES, HIDDEN_SIZE, N_EPOCHS, LR, PATCH_SIZE, REG, DATASET_LIMIT, DROPOUT_P, NORMALIZE, PATH_TO_SAVE_RESULTS, RESUME, RESUME_EPOCH, PATH_MODEL_TO_RESUME, RANDOM_SEED, SEGMENTATION_STRATEGY, OVERSAMPLE_TRAIN
# from shared.enums import SegmentationStrategy
# from tests.opencv_segmentation_test import set_seed
# from train_loops.CNN_pretrained import get_normalization_statistics
# from train_loops.train_loop import train_eval_loop
# from utils.dataloader_utils import get_dataloder_from_strategy
# from utils.utils import select_device

# device = select_device()


# def main():
#     architectures = ["densenet121", "standard", "pretrained"]
#     segmentation_strategies = [
#         SegmentationStrategy.DYNAMIC_SEGMENTATION, SegmentationStrategy.NO_SEGMENTATION]
#     keep_backgrounds = [True, False]
#     multiple_loss = [True, False]

#     config = itertools.product(architectures, segmentation_strategies,
#                                keep_backgrounds, multiple_loss)

#     clean_config = []
#     for architecture, segmentation_strategy, keep_background, multiple_loss in config:
#         if segmentation_strategy == SegmentationStrategy.NO_SEGMENTATION and not keep_background:
#             continue
#         clean_config.append(
#             (architecture, segmentation_strategy, keep_background, multiple_loss))

#     print(f"There are {len(clean_config)} configs to run")

#     assert len(clean_config) == 18

#     for architecture, segmentation_strategy, keep_background, multiple_loss in clean_config:
#         config = {
#             "learning_rate": LR,
#             "architecture": architecture,
#             "epochs": 30,
#             'reg': 0.01,
#             'batch_size': 128,
#             "hidden_size": HIDDEN_SIZE if architecture != "pretrained" else [256, 128],
#             "num_classes": NUM_CLASSES,
#             "input_size": INPUT_SIZE,
#             "patch_size": PATCH_SIZE,
#             "emb_size": EMB_SIZE,
#             "image_size": IMAGE_SIZE,
#             "dataset": "HAM10K",
#             "optimizer": "AdamW",
#             "dataset_limit": None,
#             "dropout_p": 0.3,
#             "normalize": NORMALIZE,
#             "resumed": False,
#             "from_epoch": 0,
#             "balance_downsampling": 1,
#             'segmentation_strategy': segmentation_strategy,
#             'dynamic_segmentation_strategy': DYNAMIC_SEGMENTATION_STRATEGY,
#             "oversample_train": True,
#             "multiple_loss": multiple_loss,
#             "use_wandb": True,
#             "keep_background": keep_background,
#             "hparam_tuning": False,
#             "message": None,
#         }
#         train_dataloader, val_dataloader = build_dataloaders(**config)
