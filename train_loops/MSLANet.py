import torch
from config import ARCHITECTURE, PRINT_MODEL_ARCHITECTURE, BALANCE_DOWNSAMPLING, BATCH_SIZE, DYNAMIC_SEGMENTATION_STRATEGY, INPUT_SIZE, KEEP_BACKGROUND, NUM_CLASSES, HIDDEN_SIZE, N_EPOCHS, LR, REG, DATASET_LIMIT, DROPOUT_P, NORMALIZE, PATH_TO_SAVE_RESULTS, RESUME, RESUME_EPOCH, PATH_MODEL_TO_RESUME, RANDOM_SEED, SEGMENTATION_STRATEGY, OVERSAMPLE_TRAIN, USE_MULTIPLE_LOSS, USE_WANDB
from dataloaders.MSLANetDataLoader import MSLANetDataLoader
from models.MSLANet import MSLANet
from shared.constants import IMAGENET_STATISTICS, DEFAULT_STATISTICS
from utils.dataloader_utils import get_dataloder_from_strategy
from utils.utils import select_device, set_seed
from train_loops.train_loop import train_eval_loop
from models.ResNet34Pretrained import ResNet34Pretrained
from models.DenseNetPretrained import DenseNetPretrained
from models.InceptionV3Pretrained import InceptionV3Pretrained


def get_model(device):
    if ARCHITECTURE == "resnet34":
        model = ResNet34Pretrained(
            HIDDEN_SIZE, NUM_CLASSES).to(device)
    elif ARCHITECTURE == "densenet121":
        model = DenseNetPretrained(
            HIDDEN_SIZE, NUM_CLASSES).to(device)
    elif ARCHITECTURE == "inception_v3":
        model = InceptionV3Pretrained(
            HIDDEN_SIZE, NUM_CLASSES).to(device)
    else:
        raise ValueError(f"Unknown architechture {ARCHITECTURE}")

    if RESUME:
        model.load_state_dict(torch.load(
            f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/melanoma_detection_{RESUME_EPOCH}.pt"))

    for p in model.parameters():
        p.requires_grad = False

    # LAYERS_TO_FINE_TUNE = 20
    # parameters = list(model.parameters())
    # for p in parameters[-LAYERS_TO_FINE_TUNE:]:
    #     p.requires_grad=True

    print(f"--Model-- Using {ARCHITECTURE} pretrained model")

    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


def get_normalization_statistics():
    image_net_pretrained_models = ["resnet34", "densenet121", "inception_v3"]
    if ARCHITECTURE in image_net_pretrained_models:
        return IMAGENET_STATISTICS
    else:
        return DEFAULT_STATISTICS


def main():
    set_seed(RANDOM_SEED)

    device = select_device()

    model = MSLANet().to(device)
    if PRINT_MODEL_ARCHITECTURE:
        print(f"--Model-- Architecture: {model}")

    config = {
        "learning_rate": LR,
        "architecture": "MSLANet",
        "epochs": N_EPOCHS,
        'reg': REG,
        'batch_size': BATCH_SIZE,
        "num_classes": NUM_CLASSES,
        "dataset": "HAM10K",
        "optimizer": "AdamW",
        "dataset_limit": DATASET_LIMIT,
        "normalize": NORMALIZE,
        "resumed": RESUME,
        "from_epoch": RESUME_EPOCH,
        "balance_downsampling": BALANCE_DOWNSAMPLING,
        "initialization": "default",
        'segmentation_strategy': "Cam Based Crop",
        "oversample_train": OVERSAMPLE_TRAIN,
        "use_wandb": USE_WANDB,
    }

    dataloader = MSLANetDataLoader(
        limit=DATASET_LIMIT,
        dynamic_load=True,
        resize_dim=INPUT_SIZE,
        normalize=NORMALIZE,
        normalization_statistics=get_normalization_statistics(),
        batch_size=BATCH_SIZE,
    )
    train_loader = dataloader.get_train_dataloder()
    val_loader = dataloader.get_val_dataloader()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-4, verbose=True)

    train_eval_loop(device, train_loader=train_loader, val_loader=val_loader, model=model,
                    config=config, optimizer=optimizer, scheduler=scheduler, resume=RESUME)


if __name__ == "__main__":
    main()
