from typing import Dict
import torch
import numpy as np
import random
import os
import wandb
from models.SAM import SAM
from tqdm import tqdm
from utils.plot_utils import plot_segmentations_batch
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import monai

from utils.utils import resize_images, resize_segmentations

#### Utility functions ####


def preprocess_images(images: torch.Tensor, params: Dict[str, float]):
    import torchvision.transforms.functional as TF
    from PIL import Image
    adjust_contrast = params.get("adjust_contrast", None)
    adjust_brightness = params.get("adjust_brightness", None)
    adjust_saturation = params.get("adjust_saturation", None)
    adjust_gamma = params.get("adjust_gamma", None)
    adjust_hue = params.get("adjust_hue", None)
    adjust_sharpness = params.get("adjust_sharpness", None)
    gaussian_blur = params.get("gaussian_blur", None)

    preprocessed_images = []
    for image in images:
        image = Image.fromarray((image * 255).permute(
            1, 2, 0).cpu().numpy().astype(np.uint8))
        if adjust_contrast is not None:
            image = TF.adjust_contrast(image, adjust_contrast)
        if adjust_brightness is not None:
            image = TF.adjust_brightness(image, adjust_brightness)
        if adjust_saturation is not None:
            image = TF.adjust_saturation(image, adjust_saturation)
        if adjust_gamma is not None:
            image = TF.adjust_gamma(image, adjust_gamma)
        if adjust_hue is not None:
            image = TF.adjust_hue(image, adjust_hue)
        if adjust_sharpness is not None:
            image = TF.adjust_sharpness(image, adjust_sharpness)
        if gaussian_blur is not None:
            image = TF.gaussian_blur(image, gaussian_blur)

        image = TF.to_tensor(image)
        preprocessed_images.append(image)
    result = torch.stack(preprocessed_images, dim=0).to(torch.float32)
    return result

#######################


# Configurations
USE_WANDB = False
N_EPOCHS = 40
LR = 1e-5
LR_DECAY = 0.7
ARCHITECHTURE = "SAM"
DATASET_LIMIT = None
NORMALIZE = False  # NOTE: the normalization is done by SAM
BATCH_SIZE = 64
IMG_SIZE = 128
THRESHOLD = 0.5
UPSCALE_TRAIN = False
PATIENCE = 40
preprocess_params = {
    'adjust_contrast': 1.5,
    'adjust_brightness': 1.2,
    'adjust_saturation': 2,
    'adjust_gamma': 1.5,
    'gaussian_blur': 5}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)


RESUME = False
FROM_EPOCH = 0


if USE_WANDB:
    # Start a new run
    wandb.init(
        project="melanoma",

        # track hyperparameters and run metadata
        config={
            "task": "segmentation",
            "learning_rate": LR,
            "architecture": ARCHITECHTURE,
            "epochs": N_EPOCHS,
            "dataset": "HAM10K",
            "optimizer": "Adam",
            "dataset_limit": DATASET_LIMIT,
            "normalize": NORMALIZE,
            "resumed": RESUME,
            "from_epoch": FROM_EPOCH,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "loss_type": "monai.DiceCELoss",
            "preprocess_params": preprocess_params,
            "upscale_train": UPSCALE_TRAIN,
            "scheduler_patience": PATIENCE,
            "lr_decay": LR_DECAY
        },
        resume=RESUME,
    )

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


def save_model(model, model_name, epoch):
    torch.save(model.state_dict(), f"{model_name}_{epoch}.pt")


# def find_best_preprocessing_pars():
#     dataloader = ImagesAndSegmentationDataLoader(
#         dynamic_load=True,
#         normalize=NORMALIZE,
#         upscale_train=False,
#         batch_size=128)
#     train_loader, _ = dataloader.get_train_val_dataloders()
#     model = get_model().to(device)
#     img_size = (model.get_img_size(), model.get_img_size())

#     def intersection_over_union(pred, target):
#         intersection = (pred * target).sum((1, 2))
#         union = pred.sum((1, 2)) + target.sum((1, 2)) - \
#             intersection

#         iou = (intersection + 1e-6) / (union + 1e-6)
#         return iou.mean()
#     params_space = {
#         "adjust_contrast": [1.5, 2, 2.5],
#         "adjust_brightness": [None, 0.5, 0.8, 1, 1.2, 1.5],
#         "adjust_saturation": [None, 0.5, 1, 1.5, 2],
#         "adjust_gamma": [None, 0.5, 1, 1.5],
#         "adjust_hue": [None, -0.5, 0.5],
#         # "adjust_sharpness": [None, 0.5, 1, 1.5, 2],
#         "gaussian_blur": [5, 7],
#     }

#     import itertools
#     params_keys = list(params_space.keys())
#     params_values = list(params_space.values())
#     print(f"Params space is {params_space}")
#     all_combinations = list(itertools.product(*params_values))
#     runs = dict()
#     batch = next(iter(train_loader))
#     model.model.eval()
#     pbar = tqdm(enumerate(train_loader), total=len(
#         all_combinations), desc=f"Preprocessing params")
#     max_iou = -torch.inf
#     failed = []
#     with torch.no_grad():
#         # Step 1213/2160: Best iou is 0.8683 achieved with {'adjust_contrast': 1.5, 'adjust_brightness': 1.2, 'adjust_saturation': 2, 'adjust_gamma': 1.5, 'adjust_hue': None, 'gaussian_blur': 5}
#         for combination in all_combinations:
#             params = dict(zip(params_keys, combination))
#             pbar.set_description(
#                 f"Preprocessing params {params}")
#             pbar.update(1)
#             tr_images, _, tr_segmentation = batch
#             tr_images = resize_images(tr_images, new_size=img_size)
#             tr_segmentation = resize_segmentations(
#                 tr_segmentation, new_size=(img_size))
#             try:
#                 tr_images = preprocess_images(tr_images, params)
#             except:
#                 failed.append(params)
#                 continue

#             tr_images = tr_images.to(device)
#             tr_segmentation = tr_segmentation.to(device)

#             box_torch = torch.stack([get_bounding_boxes_from_segmentation(
#                 box)[0] for box in tr_segmentation.permute(0, 1, 3, 2)]).to(device)

#             tr_segmentation = normalize(
#                 tr_segmentation).to(device)

#             upscaled_masks = model(tr_images, box_torch)
#             with torch.no_grad():
#                 print(
#                     f"Min and max normalized upscaled mask are {torch.min(normalize_0_1(upscaled_mask)), torch.max(normalize_0_1(upscaled_mask))}")
#             binary_mask = threshold(normalize_0_1(
#                 upscaled_masks), THRESHOLD, 0).to(device)

#             assert binary_mask.shape == tr_segmentation.shape

#             iou = intersection_over_union(binary_mask, tr_segmentation)
#             if iou > max_iou:
#                 max_iou = iou
#                 best_params = params
#             runs[tuple(params.items())] = iou
#             print(f"Current iou is {iou:.4f} achieved with {params}")
#             print(f"Best iou is {max_iou:.4f} achieved with {best_params}")

#     with open("best_pars.txt", "w") as f:
#         f.write(f"Best params are: {best_params}")
#         f.write("\n------------------\n")
#         f.write(f"Best iou is: {max_iou}")
#         f.write("\n------------------\n")
#         f.write(
#             f"There are {len(failed)} failed runs \n Failed params are: {failed}")
#     with open("runs.txt", "w") as f:
#         for i, run in enumerate(runs):
#             f.write(f"RUN {i}: \n")
#             f.write(f"  {run}\n")

#     return best_params, max_iou, runs, failed


def train_eval_loop():
    dataloader = ImagesAndSegmentationDataLoader(
        dynamic_load=True,
        normalize=NORMALIZE,
        upscale_train=UPSCALE_TRAIN,
        batch_size=BATCH_SIZE)
    # TODO: split the train in train and val, in order to not use the other validation set and maybe introduce a bias
    train_loader = dataloader.get_train_dataloder()
    val_loader = dataloader.get_val_dataloader()
    model = get_model().to(device)
    img_size = (model.get_img_size(), model.get_img_size())

    def intersection_over_union(pred, target):
        pred = torch.sigmoid(pred)
        pred = (pred > THRESHOLD).float()
        intersection = (pred * target).sum((1, 2))
        union = pred.sum((1, 2)) + target.sum((1, 2)) - \
            intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()

    loss_function = monai.losses.DiceCELoss(
        sigmoid=True, squared_pred=True, reduction='mean')

    if model.custom_size:
        optimizer = torch.optim.Adam(
            [*model.model.mask_decoder.parameters(), *model.model.vision_encoder.parameters()], lr=LR)
    else:
        optimizer = torch.optim.Adam(
            model.model.mask_decoder.parameters(), lr=LR)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_DECAY, patience=PATIENCE, verbose=True)

    best_eval_accuracy = -torch.inf

    for epoch in range(FROM_EPOCH if RESUME else 0, N_EPOCHS):
        model.model.train()
        pbar = tqdm(enumerate(train_loader), total=len(
            train_loader), desc=f"TRAINING | Epoch {epoch}")
        for tr_i, (tr_images, _, tr_segmentation) in enumerate(train_loader):
            with torch.no_grad():
                # tr_images = tr_images.to(torch.float32)
                tr_images = resize_images(tr_images, new_size=img_size)
                tr_segmentation = resize_segmentations(
                    tr_segmentation, new_size=(img_size))
                tr_images = preprocess_images(
                    tr_images, params=preprocess_params)

                tr_images = tr_images.to(device)
                tr_segmentation = tr_segmentation.to(device)

            upscaled_masks = model(tr_images, None)

            assert upscaled_masks.shape == tr_segmentation.shape

            loss = loss_function(upscaled_masks, tr_segmentation)
            with torch.no_grad():
                iou = intersection_over_union(
                    upscaled_masks, tr_segmentation)
            if USE_WANDB:
                wandb.log(
                    {"train_loss": loss.item()})
                wandb.log(
                    {"train_iou": iou.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            pbar.set_description(
                f"TRAINING | Epoch {epoch} | Loss {loss.item():.4f} | Iou {iou:.4f}")
            pbar.update(1)

            if tr_i % 30 == 0:
                with torch.no_grad():
                    binary_mask = torch.sigmoid(upscaled_masks)
                    binary_mask = (binary_mask > THRESHOLD).float()
                    plot_segmentations_batch(
                        epoch=epoch,
                        tr_i=tr_i,
                        pred_bin_mask=binary_mask[:10],
                        gt_mask=tr_segmentation[:10],
                        boxes=None,
                        images=tr_images[:10],
                        upscaled_mask=upscaled_masks[:10],
                        name="results")
        with torch.no_grad():
            pbar.close()
            save_model(model.model, ARCHITECHTURE, epoch)
            binary_mask = torch.sigmoid(upscaled_masks)
            binary_mask = (binary_mask > THRESHOLD).float()
            plot_segmentations_batch(
                epoch=epoch,
                tr_i=tr_i,
                pred_bin_mask=binary_mask[:10],
                gt_mask=tr_segmentation[:10],
                boxes=None,
                images=tr_images[:10],
                upscaled_mask=upscaled_masks[:10],
                name="results")
            # Evaluation (ignored for now)
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            for val_i, (val_images, _, val_segmentations) in tqdm(enumerate(val_loader), f"Evaluation"):
                # val_images = val_images.to(torch.float32)
                val_images = resize_images(val_images, new_size=img_size)
                val_segmentations = resize_segmentations(
                    val_segmentations, new_size=(img_size))
                val_images = preprocess_images(
                    val_images, params=preprocess_params)
                # print(f"tr_semgnentation shape is {val_segmentations.shape}")
                val_segmentations = val_segmentations.to(device)
                val_images = val_images.to(device)

                upscaled_masks = model(val_images, None)

                loss = loss_function(upscaled_masks, val_segmentations)
                iou = intersection_over_union(
                    upscaled_masks, val_segmentations)
                if USE_WANDB:
                    wandb.log(
                        {"val_loss": loss.item()})
                    wandb.log(
                        {"val_iou": iou.item()})
                loss_sum += loss.item()

                if iou > best_eval_accuracy:
                    best_eval_accuracy = iou
                    save_model(model.model, f"{ARCHITECHTURE}_best", epoch)
                    print(
                        f"New best iou is {best_eval_accuracy:.4f} at epoch {epoch}")
            mean_loss = loss_sum / len(val_loader)
            print(f"Validation avg loss at epoch {epoch}: {mean_loss}")
            binary_mask = torch.sigmoid(upscaled_masks)
            binary_mask = (binary_mask > THRESHOLD).float()
            plot_segmentations_batch(
                epoch=epoch,
                tr_i=tr_i,
                pred_bin_mask=binary_mask[:10],
                gt_mask=val_segmentations[:10],
                boxes=None,
                images=val_images[:10],
                upscaled_mask=upscaled_masks[:10],
                name="eval_results")


def get_model():
    model = SAM(custom_size=True, img_size=IMG_SIZE).to(device)

    if RESUME:
        model.model.load_state_dict(torch.load(
            f"{ARCHITECHTURE}_{FROM_EPOCH-1}.pt"))

    return model


if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    train_eval_loop()
