import torch
import wandb
from enum import Enum
from config import BALANCE_UNDERSAMPLING, BATCH_SIZE, INPUT_SIZE, NUM_CLASSES, HIDDEN_SIZE, N_EPOCHS, LR, REG, SEGMENT, CROP_ROI, ARCHITECTURE_CNN, DATASET_LIMIT, DROPOUT_P, NORMALIZE, SEGMENTATION_BOUNDING_BOX, PATH_TO_SAVE_RESULTS, RESUME, RESUME_EPOCH, PATH_MODEL_TO_RESUME, RANDOM_SEED
from utils.utils import select_device, set_seed
from train_loops.train_loop import train_eval_loop
from dataloaders.DynamicSegmentationDataLoader import DynamicSegmentationDataLoader, DynamicSegmentationStrategy
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from dataloaders.SegmentedImagesDataLoader import SegmentedImagesDataLoader
from models.ResNet24Pretrained import ResNet24Pretrained
from models.DenseNetPretrained import DenseNetPretrained
from models.InceptionV3Pretrained import InceptionV3Pretrained

def get_model(device):
    if ARCHITECTURE_CNN == "resnet24":
        model = ResNet24Pretrained(
            INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, norm_layer='BN').to(device)
    elif ARCHITECTURE_CNN == "densenet121":
        model = DenseNetPretrained(
            INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, norm_layer='BN').to(device)
    elif ARCHITECTURE_CNN == "inception_v3":
        model = InceptionV3Pretrained(NUM_CLASSES).to(device)
    else:
        raise ValueError(f"Unknown architechture {ARCHITECTURE_CNN}")

    if RESUME:
        model.load_state_dict(torch.load(f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/melanoma_detection_ep{FROM_EPOCH}.pt"))

    for p in model.parameters():
        p.requires_grad = False

    # LAYERS_TO_FINE_TUNE = 20
    # parameters = list(model.parameters())
    # for p in parameters[-LAYERS_TO_FINE_TUNE:]:
    #     p.requires_grad=True

    print(f"--Model-- Using {ARCHITECTURE_CNN} pretrained model")

    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


def main():
    set_seed(RANDOM_SEED)

    class SegmentationStrategy(Enum):
        DYNAMIC_SEGMENTATION = "dynamic_segmentation"
        SEGMENTATION = "segmentation"
        NO_SEGMENTATION = "no_segmentation"

    SEGMENTATION_STRATEGY = SegmentationStrategy.SEGMENTATION.value
    DYNAMIC_SEGMENTATION_STRATEGY = DynamicSegmentationStrategy.OPENCV if SEGMENTATION_STRATEGY == SegmentationStrategy.DYNAMIC_SEGMENTATION else None
    dynamic_load = True

    if SEGMENTATION_STRATEGY == SegmentationStrategy.DYNAMIC_SEGMENTATION.value:
        dataloader = DynamicSegmentationDataLoader(
            limit=DATASET_LIMIT,
            dynamic_load=dynamic_load,
            segmentation_strategy=DYNAMIC_SEGMENTATION_STRATEGY
        )
    elif SEGMENTATION_STRATEGY == SegmentationStrategy.SEGMENTATION.value:
        dataloader = SegmentedImagesDataLoader(
            limit=DATASET_LIMIT,
            dynamic_load=dynamic_load,
        )
    elif SEGMENTATION_STRATEGY == SegmentationStrategy.NO_SEGMENTATION.value:
        dataloader = ImagesAndSegmentationDataLoader(
            limit=DATASET_LIMIT,
            dynamic_load=dynamic_load,
        )
    else:
        raise NotImplementedError(
            f"Segmentation strategy {SEGMENTATION_STRATEGY} not implemented")
    
    device = select_device()

    config = {
        "learning_rate": LR,
        "architecture": ARCHITECTURE_CNN,
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
        "from_epoch": RESUME_EPOCH,
        "segmentation_bounding_box": SEGMENTATION_BOUNDING_BOX,
        "balance_undersampling": BALANCE_UNDERSAMPLING,
        "initialization": "default",
        'segmentation_strategy': SEGMENTATION_STRATEGY,
        'dynamic_segmentation_strategy': DYNAMIC_SEGMENTATION_STRATEGY,
    }
    
    train_loader, val_loader = dataloader.get_train_val_dataloders()
    model = get_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    train_eval_loop(device, train_loader=train_loader, val_loader=val_loader, model=model, config=config, optimizer=optimizer, scheduler=scheduler, resume=RESUME)


if __name__ == "__main__":
    main()
