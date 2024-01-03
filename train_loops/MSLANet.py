import torch
from config import ARCHITECTURE, PRINT_MODEL_ARCHITECTURE, BALANCE_DOWNSAMPLING, BATCH_SIZE, DYNAMIC_SEGMENTATION_STRATEGY, INPUT_SIZE, KEEP_BACKGROUND, NUM_CLASSES, HIDDEN_SIZE, N_EPOCHS, LR, REG, DATASET_LIMIT, DROPOUT_P, NORMALIZE, PATH_TO_SAVE_RESULTS, RESUME, RESUME_EPOCH, PATH_MODEL_TO_RESUME, RANDOM_SEED, SEGMENTATION_STRATEGY, OVERSAMPLE_TRAIN, USE_MULTIPLE_LOSS, USE_WANDB
from dataloaders.MSLANetDataLoader import MSLANetDataLoader
from models.MSLANet import MSLANet
from shared.constants import IMAGENET_STATISTICS, DEFAULT_STATISTICS
from utils.utils import select_device, set_seed
from train_loops.mslanet_train_loop import train_eval_loop


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
        'segmentation_strategy': "CAM",
        "oversample_train": OVERSAMPLE_TRAIN,
        "multiple_loss": USE_MULTIPLE_LOSS,
        "use_wandb": USE_WANDB,
    }

    dataloader = MSLANetDataLoader(
        limit=DATASET_LIMIT,
        dynamic_load=True,
        normalize=NORMALIZE,
        normalization_statistics=IMAGENET_STATISTICS,
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
