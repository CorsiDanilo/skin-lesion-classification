from argparse import ArgumentParser
import itertools

import torch
from tqdm import tqdm
from models.ResNet24Pretrained import ResNet24Pretrained
from shared.enums import DynamicSegmentationStrategy, SegmentationStrategy
from models.DenseNetPretrained import DenseNetPretrained
from models.InceptionV3Pretrained import InceptionV3Pretrained
from models.ViTEfficient import Conv2d_BN as ViTEfficient, EfficientViTBlock
from models.ViTPretrained import ViT_pretrained as ViTPretrained
from models.ViTStandard import ViT_standard as ViTStandard
from config import ARCHITECTURE, BALANCE_UNDERSAMPLING, BATCH_SIZE, DYNAMIC_SEGMENTATION_STRATEGY, EMB_SIZE, IMAGE_SIZE, INPUT_SIZE, KEEP_BACKGROUND, N_HEADS, N_LAYERS, NUM_CLASSES, HIDDEN_SIZE, N_EPOCHS, LR, PATCH_SIZE, REG, DATASET_LIMIT, DROPOUT_P, NORMALIZE, PATH_TO_SAVE_RESULTS, RESUME, RESUME_EPOCH, PATH_MODEL_TO_RESUME, RANDOM_SEED, SEGMENTATION_STRATEGY, UPSAMPLE_TRAIN, USE_DOUBLE_LOSS, USE_WANDB
from tests.opencv_segmentation_test import set_seed
from train_loops.CNN_pretrained import get_normalization_statistics
from train_loops.train_loop import train_eval_loop
from utils.dataloader_utils import get_dataloder_from_strategy
from utils.utils import select_device

# TODO: work in progress

hparams_space = {}

device = select_device()


def init_with_parsed_arguments():
    parser = ArgumentParser()
    parser.add_argument("--segmentation-strategy",
                        type=str, default="no_segmentation")
    parser.add_argument("--architecture", type=str)
    parser.add_argument("--dataset-limit", type=int, default=DATASET_LIMIT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--reg", type=float, default=None)
    parser.add_argument("--dropout-p", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    # parser.add_argument("--normalize", action="store_true" if NORMALIZE else "store_false")
    parser.add_argument(
        "--resumed", action="store_true", default=False)
    parser.add_argument("--from-epoch", type=int, default=RESUME_EPOCH)
    # parser.add_argument("--balance-undersampling", action="store_true" if BALANCE_UNDERSAMPLING else "store_false")
    # parser.add_argument("--upsample-train", action="store_true" if UPSAMPLE_TRAIN else "store_false")
    parser.add_argument("--double-loss", action="store_true", default=False)
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--dynamic-load", action="store_true", default=False)

    args = parser.parse_args()
    assert args.architecture is not None, "You must specify an architecture"

    global dynamic_load
    dynamic_load = args.dynamic_load

    print(f"Parsed arguments are {args}")
    print(f"Vars parsed arguments are {vars(args)}")

    if args.reg is not None and args.dropout_p is not None:
        print(f"----REG AND DROPOUT_P ARE NOT NONE, NOT DOING HPARAMS TUNING----")
        train_loader, val_loader = build_dataloaders()
        init_run(train_loader=train_loader,
                 val_loader=val_loader,
                 **vars(args))
    else:
        hparams_tuning(**vars(args))


def hparams_tuning(**hparams):
    train_loader, val_loader = build_dataloaders()
    hparams_space = {
        "reg": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "dropout_p": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    combinations = list(itertools.product(*hparams_space.values()))
    print(f"Combinations are {combinations}")
    for combination in tqdm(combinations, "Hparams tuning"):
        hparams.update(dict(zip(hparams_space.keys(), combination)))
        # print(f"Hparams are {hparams}")
        init_run(train_loader=train_loader,
                 val_loader=val_loader,
                 **hparams)


def init_run(train_loader, val_loader, **kwargs):
    model = get_model(kwargs.get("architecture"))

    run_train_eval_loop(model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        **kwargs)


def get_model(architecture: str):
    if architecture == "resnet24":
        model = ResNet24Pretrained(
            HIDDEN_SIZE, NUM_CLASSES).to(device)
    elif architecture == "densenet121":
        model = DenseNetPretrained(
            HIDDEN_SIZE, NUM_CLASSES).to(device)
    elif architecture == "inception_v3":
        model = InceptionV3Pretrained(NUM_CLASSES).to(device)
    elif architecture == "pretrained":
        model = ViTPretrained(NUM_CLASSES, pretrained=True).to(device)
    elif architecture == "standard":
        model = ViTStandard(in_channels=INPUT_SIZE, patch_size=PATCH_SIZE, d_model=EMB_SIZE,
                            img_size=IMAGE_SIZE, n_classes=NUM_CLASSES, n_head=N_HEADS, n_layers=N_LAYERS).to(device)
    elif architecture == "efficient":
        model = EfficientViTBlock(img_size=224, patch_size=16, in_chans=INPUT_SIZE, stages=['s', 's', 's'], embed_dim=[
            64, 128, 192], key_dim=[16, 16, 16], depth=[1, 2, 3], window_size=[7, 7, 7], kernels=[5, 5, 5, 5])
    else:
        raise ValueError(f"Unknown architechture {ARCHITECTURE}")

    if RESUME:
        model.load_state_dict(torch.load(
            f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/melanoma_detection_{RESUME_EPOCH}.pt"))

    for p in model.parameters():
        p.requires_grad = False

    print(f"--Model-- Using {ARCHITECTURE} pretrained model")

    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


def build_dataloaders():
    dataloader = get_dataloder_from_strategy(
        strategy=SEGMENTATION_STRATEGY,
        dynamic_segmentation_strategy=DYNAMIC_SEGMENTATION_STRATEGY,
        limit=DATASET_LIMIT,
        dynamic_load=dynamic_load,
        upsample_train=UPSAMPLE_TRAIN,
        normalize=NORMALIZE,
        normalization_statistics=get_normalization_statistics(),
        batch_size=BATCH_SIZE,
        keep_background=KEEP_BACKGROUND,)
    train_loader = dataloader.get_train_dataloder()
    val_loader = dataloader.get_val_dataloader()
    return train_loader, val_loader


def run_train_eval_loop(model, train_loader, val_loader, **kwargs):
    set_seed(RANDOM_SEED)

    config = {
        "learning_rate": LR if kwargs.get("lr") is None else kwargs.get("lr"),
        "architecture": kwargs.get("architecture"),
        "epochs": N_EPOCHS if kwargs.get("epochs") is None else kwargs.get("epochs"),
        'reg': REG if kwargs.get("reg") is None else kwargs.get("reg"),
        'batch_size': BATCH_SIZE if kwargs.get("batch_size") is None else kwargs.get("batch_size"),
        "input_size": INPUT_SIZE if kwargs.get("input_size") is None else kwargs.get("input_size"),
        "hidden_size": HIDDEN_SIZE if kwargs.get("hidden_size") is None else kwargs.get("hidden_size"),
        "num_classes": NUM_CLASSES if kwargs.get("num_classes") is None else kwargs.get("num_classes"),
        "dataset": "HAM10K",
        "optimizer": "AdamW",
        "dataset_limit": DATASET_LIMIT if kwargs.get("dataset_limit") is None else kwargs.get("dataset_limit"),
        "dropout_p": DROPOUT_P if kwargs.get("dropout_p") is None else kwargs.get("dropout_p"),
        "normalize": NORMALIZE,
        "resumed": RESUME if kwargs.get("resumed") is None else kwargs.get("resumed"),
        "from_epoch": RESUME_EPOCH if kwargs.get("from_epoch") is None else kwargs.get("from_epoch"),
        "balance_undersampling": BALANCE_UNDERSAMPLING,
        "initialization": "default",
        'segmentation_strategy': SEGMENTATION_STRATEGY if kwargs.get("segmentation_strategy") is None else kwargs.get("segmentation_strategy"),
        'dynamic_segmentation_strategy': DYNAMIC_SEGMENTATION_STRATEGY if kwargs.get("dynamic_segmentation_strategy") is None else kwargs.get("dynamic_segmentation_strategy"),
        "upsample_train": UPSAMPLE_TRAIN,
        "double_loss": kwargs.get("double_loss"),
        "use_wandb": kwargs.get("use_wandb"),
    }

    print(f"---CURRENT CONFIGURATION---\n{config}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5, verbose=True)

    train_eval_loop(device, train_loader=train_loader, val_loader=val_loader, model=model,
                    config=config, optimizer=optimizer, scheduler=scheduler, resume=RESUME)


if __name__ == "__main__":
    init_with_parsed_arguments()
