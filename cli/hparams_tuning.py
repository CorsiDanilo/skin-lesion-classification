from argparse import ArgumentParser
import itertools
import json
import os

import torch
from tqdm import tqdm
import wandb
from models.ResNet24Pretrained import ResNet24Pretrained
from models.DenseNetPretrained import DenseNetPretrained
from models.InceptionV3Pretrained import InceptionV3Pretrained
from models.ViTStandard import ViT_standard
from models.ViTPretrained import ViT_pretrained
from models.ViTEfficient import EfficientViT
from config import ARCHITECTURE, BALANCE_UNDERSAMPLING, BATCH_SIZE, DYNAMIC_SEGMENTATION_STRATEGY, EMB_SIZE, IMAGE_SIZE, INPUT_SIZE, N_HEADS, N_LAYERS, NUM_CLASSES, HIDDEN_SIZE, N_EPOCHS, LR, PATCH_SIZE, REG, DATASET_LIMIT, DROPOUT_P, NORMALIZE, PATH_TO_SAVE_RESULTS, RESUME, RESUME_EPOCH, PATH_MODEL_TO_RESUME, RANDOM_SEED, SEGMENTATION_STRATEGY, UPSAMPLE_TRAIN, USE_DOUBLE_LOSS, USE_WANDB
from tests.opencv_segmentation_test import set_seed
from train_loops.CNN_pretrained import get_normalization_statistics
from train_loops.train_loop import train_eval_loop
from utils.dataloader_utils import get_dataloder_from_strategy
from utils.utils import select_device

# TODO: work in progress

hparams_space = {}

device = select_device()


def init_with_parsed_arguments():
    set_seed(RANDOM_SEED)
    parser = ArgumentParser()
    parser.add_argument("--segmentation-strategy",
                        type=str, default="dynamic_segmentation")

    # REQUIRED: resnet24, densenet121, inception_v3, standard, pretrained, efficient
    parser.add_argument("--architecture", type=str)
    parser.add_argument("--dataset-limit", type=int, default=DATASET_LIMIT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--reg", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--balance_undersampling",
                        type=float, default=BALANCE_UNDERSAMPLING)

    # If True, it will not use the double loss
    parser.add_argument("--no-double-loss", action="store_true", default=False)

    # If True, it will not use wandb for logging
    parser.add_argument("--no-wandb", action="store_true", default=False)

    # If True, it uses dynamic load instead of loading the full dataset in memory (this is way slower)
    parser.add_argument("--dynamic-load", action="store_true", default=False)

    # If True, it will remove the background from the images (only for dynamic segmentation)
    parser.add_argument("--no-background",
                        action="store_true", default=False)

    # If True, will reset the combinations tried for the current architecture
    parser.add_argument("--force-reset", action="store_true", default=False)

    args = parser.parse_args()
    assert args.architecture is not None, "You must specify an architecture"

    global dynamic_load
    dynamic_load = args.dynamic_load

    print(f"Parsed arguments are {args}")
    print(f"Vars parsed arguments are {vars(args)}")

    kwargs = vars(args)

    config = {
        "learning_rate": LR if kwargs.get("lr") is None else kwargs.get("lr"),
        "architecture": kwargs.get("architecture"),
        "epochs": N_EPOCHS if kwargs.get("epochs") is None else kwargs.get("epochs"),
        'reg': REG if kwargs.get("reg") is None else kwargs.get("reg"),
        'batch_size': BATCH_SIZE if kwargs.get("batch_size") is None else kwargs.get("batch_size"),
        "hidden_size": HIDDEN_SIZE if kwargs.get("architecture") != "pretrained" else [256, 128],
        "num_classes": NUM_CLASSES if kwargs.get("num_classes") is None else kwargs.get("num_classes"),
        "input_size": INPUT_SIZE if kwargs.get("input_size") is None else kwargs.get("input_size"),
        "patch_size": PATCH_SIZE if kwargs.get("patch_size") is None else kwargs.get("patch_size"),
        "emb_size": EMB_SIZE if kwargs.get("emb_size") is None else kwargs.get("emb_size"),
        "image_size": IMAGE_SIZE if kwargs.get("image_size") is None else kwargs.get("image_size"),
        "dataset": "HAM10K",
        "optimizer": "AdamW",
        "dataset_limit": DATASET_LIMIT if kwargs.get("dataset_limit") is None else kwargs.get("dataset_limit"),
        "dropout_p": DROPOUT_P if kwargs.get("dropout") is None else kwargs.get("dropout"),
        "normalize": NORMALIZE,
        "resumed": False,
        "from_epoch": 0,
        "balance_undersampling": BALANCE_UNDERSAMPLING if kwargs.get("balance_undersampling") is None else kwargs.get("balance_undersampling"),
        # "initialization": "default",
        'segmentation_strategy': SEGMENTATION_STRATEGY if kwargs.get("segmentation_strategy") is None else kwargs.get("segmentation_strategy"),
        'dynamic_segmentation_strategy': DYNAMIC_SEGMENTATION_STRATEGY if kwargs.get("dynamic_segmentation_strategy") is None else kwargs.get("dynamic_segmentation_strategy"),
        "upsample_train": UPSAMPLE_TRAIN,
        "double_loss": not kwargs.get("no_double_loss"),
        "use_wandb": not kwargs.get("no_wandb"),
        "keep_background": not kwargs.get("no_background"),
        "hparam_tuning": True if (kwargs.get("reg") is None and kwargs.get("dropout") is None) else False,
        "force_reset": kwargs.get("force_reset"),
    }

    train_loader, val_loader = build_dataloaders(**config)
    if args.reg is not None and args.dropout is not None:
        print(f"----REG AND DROPOUT_P ARE NOT NONE, NOT DOING HPARAMS TUNING----")
        init_run(train_loader=train_loader,
                 val_loader=val_loader,
                 **config)
    else:
        hparams_tuning(train_loader=train_loader,
                       val_loader=val_loader,
                       **config)


def hparams_tuning(train_loader, val_loader, **hparams):
    hparams_space = {
        "reg": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "dropout_p": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    }
    combinations = list(itertools.product(*hparams_space.values()))

    if hparams["force_reset"]:
        combinations_tried = {}
    else:
        # Filter combinations that have already been tried
        if os.path.exists("./results/combinations.json"):
            with open("./results/combinations.json", "r") as f:
                combinations_tried = json.load(f)
        else:
            combinations_tried = {}

        print(f"COMBINATIONS TRIED: {combinations_tried}")

        curr_architecture = f"{hparams['architecture']}_{hparams['segmentation_strategy']}_{hparams['double_loss']}_{hparams['keep_background']}"
        if curr_architecture in combinations_tried:
            combinations = [
                combination for combination in combinations if list(combination) not in combinations_tried[curr_architecture]]
            print(
                f"----Found {len(combinations_tried[curr_architecture])} combinations already tried for {curr_architecture}, excluding them from the run! ----")
        else:
            combinations_tried[curr_architecture] = []

    print(f"Combinations are {combinations}")
    for combination in tqdm(combinations, "Hparams tuning"):
        hparams.update(dict(zip(hparams_space.keys(), combination)))
        # print(f"Hparams are {hparams}")
        init_run(train_loader=train_loader,
                 val_loader=val_loader,
                 **hparams)

        combinations_tried[curr_architecture].append(combination)
        with open("./results/combinations.json", "w") as f:
            json.dump(combinations_tried, f)


def init_run(train_loader, val_loader, **kwargs):
    model = get_model(**kwargs)

    if kwargs["use_wandb"]:
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project="melanoma",
            config=kwargs,  # Track hyperparameters and run metadata
            resume=False,
        )

    run_train_eval_loop(model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        **kwargs)


def get_model(**kwargs):
    architecture = kwargs.get("architecture")
    dropout_p = kwargs.get("dropout_p")
    if architecture == "resnet24":
        model = ResNet24Pretrained(
            hidden_layers=HIDDEN_SIZE,
            num_classes=NUM_CLASSES,
            dropout_p=dropout_p).to(device)
    elif architecture == "densenet121":
        model = DenseNetPretrained(
            hidden_layers=HIDDEN_SIZE,
            num_classes=NUM_CLASSES,
            dropout_p=dropout_p).to(device)
    elif architecture == "inception_v3":
        model = InceptionV3Pretrained(
            num_classes=NUM_CLASSES,
            dropout_p=dropout_p).to(device)
    elif architecture == "pretrained":
        hidden_size = [256, 128]
        model = ViT_pretrained(
            hidden_layers=hidden_size,
            num_classes=NUM_CLASSES,
            pretrained=True,
            dropout=dropout_p).to(device)
    elif architecture == "standard":
        model = ViT_standard(in_channels=INPUT_SIZE, patch_size=PATCH_SIZE, d_model=EMB_SIZE,
                             img_size=IMAGE_SIZE, n_classes=NUM_CLASSES, n_head=N_HEADS, n_layers=N_LAYERS,
                             dropout=dropout_p).to(device)
    elif architecture == "efficient":
        # TODO: dropout not implemented, add it later
        model = EfficientViT(img_size=224, patch_size=16, in_chans=INPUT_SIZE, stages=['s', 's', 's'], embed_dim=[
            64, 128, 192], key_dim=[16, 16, 16], depth=[1, 2, 3], window_size=[7, 7, 7], kernels=[5, 5, 5, 5])
    else:
        raise ValueError(f"Unknown architechture {architecture}")

    if RESUME:
        model.load_state_dict(torch.load(
            f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/melanoma_detection_{RESUME_EPOCH}.pt"))

    if architecture in ["resnet24", "densenet121", "inception_v3"]:
        for p in model.parameters():
            p.requires_grad = False

        print(f"--Model-- Using {architecture} pretrained model")

        for p in model.classifier.parameters():
            p.requires_grad = True

    return model


def build_dataloaders(**args):

    dataloader = get_dataloder_from_strategy(
        strategy=args["segmentation_strategy"],
        dynamic_segmentation_strategy=args["dynamic_segmentation_strategy"],
        limit=args["dataset_limit"],
        dynamic_load=dynamic_load,
        upsample_train=args["upsample_train"],
        normalize=args["normalize"],
        normalization_statistics=get_normalization_statistics(),
        batch_size=args["batch_size"],
        keep_background=args["keep_background"],)

    train_loader = dataloader.get_train_dataloder()
    val_loader = dataloader.get_val_dataloader()
    return train_loader, val_loader


def run_train_eval_loop(model, train_loader, val_loader, **kwargs):
    print(f"---CURRENT CONFIGURATION---\n{kwargs}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=kwargs["learning_rate"], weight_decay=kwargs["reg"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=kwargs["epochs"], eta_min=1e-5, verbose=True)

    train_eval_loop(device, train_loader=train_loader, val_loader=val_loader, model=model,
                    config=kwargs, optimizer=optimizer, scheduler=scheduler, resume=RESUME)


if __name__ == "__main__":
    init_with_parsed_arguments()
