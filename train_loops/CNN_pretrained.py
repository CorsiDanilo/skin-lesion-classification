import torch
from config import BALANCE_UNDERSAMPLING, BATCH_SIZE, DYNAMIC_SEGMENTATION_STRATEGY, INPUT_SIZE, KEEP_BACKGROUND, NUM_CLASSES, HIDDEN_SIZE, N_EPOCHS, LR, REG, ARCHITECTURE_CNN, DATASET_LIMIT, DROPOUT_P, NORMALIZE, PATH_TO_SAVE_RESULTS, RESUME, RESUME_EPOCH, PATH_MODEL_TO_RESUME, RANDOM_SEED, SEGMENTATION_STRATEGY, UPSAMPLE_TRAIN, USE_DOUBLE_LOSS
from constants import IMAGENET_STATISTICS, DEFAULT_STATISTICS
from utils.dataloader_utils import get_dataloder_from_strategy
from utils.utils import select_device, set_seed
from train_loops.train_loop import train_eval_loop
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
        model.load_state_dict(torch.load(
            f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/melanoma_detection_ep{RESUME_EPOCH}.pt"))

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


def get_normalization_statistics():
    image_net_pretrained_models = ["resnet24", "densenet121", "inception_v3"]
    if ARCHITECTURE_CNN in image_net_pretrained_models:
        return IMAGENET_STATISTICS
    else:
        return DEFAULT_STATISTICS


def main():
    set_seed(RANDOM_SEED)

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
        "dataset_limit": DATASET_LIMIT,
        "dropout_p": DROPOUT_P,
        "normalize": NORMALIZE,
        "resumed": RESUME,
        "from_epoch": RESUME_EPOCH,
        "balance_undersampling": BALANCE_UNDERSAMPLING,
        "initialization": "default",
        'segmentation_strategy': SEGMENTATION_STRATEGY,
        'dynamic_segmentation_strategy': DYNAMIC_SEGMENTATION_STRATEGY,
        "upsample_train": UPSAMPLE_TRAIN,
        "double_loss": USE_DOUBLE_LOSS,
    }

    TAKE_VAL_FROM_TEST = False
    dataloader = get_dataloder_from_strategy(
        strategy=SEGMENTATION_STRATEGY,
        dynamic_segmentation_strategy=DYNAMIC_SEGMENTATION_STRATEGY,
        limit=DATASET_LIMIT,
        dynamic_load=True,
        upsample_train=UPSAMPLE_TRAIN,
        normalize=NORMALIZE,
        normalization_statistics=get_normalization_statistics(),
        batch_size=BATCH_SIZE,
        keep_background=KEEP_BACKGROUND,
        take_val_from_test=TAKE_VAL_FROM_TEST,)
    if not TAKE_VAL_FROM_TEST:
        train_loader, val_loader = dataloader.get_train_dataloder()
    else:
        train_loader = dataloader.get_train_dataloder()
        val_loader, _ = dataloader.get_val_test_dataloader()
    model = get_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5, verbose=True)

    train_eval_loop(device, train_loader=train_loader, val_loader=val_loader, model=model,
                    config=config, optimizer=optimizer, scheduler=scheduler, resume=RESUME)


if __name__ == "__main__":
    main()
