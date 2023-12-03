import cv2
import torch
import torch.nn as nn
import numpy as np
import random
import os
import wandb
from models.SAM import SAM
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from sklearn.metrics import recall_score, accuracy_score
from utils.plot_utils import plot_segmentations_batch, plot_segmentations_single_sample
from utils.utils import get_bounding_boxes_from_segmentation
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
import torch.nn.functional as F

# Configurations
USE_WANDB = False
N_EPOCHS = 100
LR = 1e-3
LR_DECAY = 0.85
REG = 0.01
ARCHITECHTURE = "sam"
DATASET_LIMIT = None
NORMALIZE = False
BALANCE_UNDERSAMPLING = 1
BATCH_SIZE = 32

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
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
            'reg': REG,
            "dataset": "HAM10K",
            "optimizer": "AdamW",
            "dataset_limit": DATASET_LIMIT,
            "normalize": NORMALIZE,
            "resumed": RESUME,
            "from_epoch": FROM_EPOCH,
            "balance_undersampling": BALANCE_UNDERSAMPLING
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


def train_eval_loop():
    dataloader = ImagesAndSegmentationDataLoader(
        dynamic_load=True,
        normalize=False,
        upscale_train=False,
        batch_size=BATCH_SIZE)
    train_loader, val_loader = dataloader.get_train_val_dataloders()
    model = get_model()
    img_size = (model.get_img_size(), model.get_img_size())
    model = model.model

    def iou_loss(pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou

    # loss_function = nn.MSELoss()
    loss_function = iou_loss
    # loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.mask_decoder.parameters(), lr=LR, weight_decay=REG)

    best_eval_accuracy = -torch.inf
    for epoch in range(FROM_EPOCH if RESUME else 0, N_EPOCHS):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(
            train_loader), desc=f"TRAINING | Epoch {epoch}")
        for tr_i, (tr_images, _, tr_segmentation) in enumerate(train_loader):
            tr_images = tr_images.to(torch.float32)

            # TODO: add these two functions to utils
            def resize_images(images, new_size=(800, 800)):
                return torch.stack([torch.from_numpy(cv2.resize(
                    image.permute(1, 2, 0).numpy(), new_size)) for image in images]).permute(0, 3, 1, 2)

            def resize_segmentations(segmentation, new_size=(800, 800)):
                return torch.stack([torch.from_numpy(cv2.resize(
                    image.permute(1, 2, 0).numpy(), new_size)) for image in segmentation]).unsqueeze(0).permute(1, 0, 2, 3)

            tr_images = resize_images(tr_images, new_size=img_size)
            tr_segmentation = resize_segmentations(
                tr_segmentation, new_size=(img_size))
            # print(f"tr_semgnentation shape is {tr_segmentation.shape}")
            tr_images = tr_images.to(device)
            with torch.no_grad():
                image_embedding = model.image_encoder(tr_images)
                box_torch = torch.stack([get_bounding_boxes_from_segmentation(
                    box)[0] for box in tr_segmentation]).to(device)
                # print(f"Box torch shape is {box_torch.shape}")
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # print(f"Low res masks shape is {low_res_masks.shape}")
            # NOTE: this upscaling gives a strange gradient-like image, that's why I'm using the F.interpolate
            # upscaled_masks = model.postprocess_masks(low_res_masks, (32, 32), img_size).to(device)
            upscaled_masks = F.interpolate(
                low_res_masks, scale_factor=4, mode='bicubic', align_corners=False)

            # print(f"Upscaled masks shape is {upscaled_masks.shape}")
            # print(f"tr_segmentation shape is {tr_segmentation.shape}")

            binary_mask = normalize(
                threshold(upscaled_masks, 0.0, 0)).to(device)
            if tr_i % 10 == 0:
                # plot_segmentations_batch(
                # epoch=epoch, tr_i=tr_i, pred_mask=low_res_masks, gt_mask=upscaled_masks, name="low_res_masks_vs_upscaled_masks")
                plot_segmentations_batch(
                    epoch=epoch, tr_i=tr_i, pred_mask=binary_mask, gt_mask=tr_segmentation, name="binary_mask_vs_tr_segmentation")

            loss: torch.Tensor = loss_function(binary_mask, tr_segmentation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f"TRAINING | Epoch {epoch} | Loss {loss.item():.4f} | Iou {torch.mean(iou_predictions):.4f}")
            pbar.update(1)

        # Evaluation
        model.eval()
        loss_sum = 0
        with torch.no_grad():
            for val_i, (val_images, _, val_segmentations) in tqdm(enumerate(val_loader), f"Evaluation"):
                val_images = val_images.to(torch.float32)

                # TODO: add these two functions to utils
                def resize_images(images, new_size=(800, 800)):
                    return torch.stack([torch.from_numpy(cv2.resize(
                        image.permute(1, 2, 0).numpy(), new_size)) for image in images]).permute(0, 3, 1, 2)

                def resize_segmentations(segmentation, new_size=(800, 800)):
                    return torch.stack([torch.from_numpy(cv2.resize(
                        image.permute(1, 2, 0).numpy(), new_size)) for image in segmentation]).unsqueeze(0).permute(1, 0, 2, 3)

                val_images = resize_images(val_images, new_size=img_size)
                val_segmentations = resize_segmentations(
                    val_segmentations, new_size=(img_size))
                # print(f"tr_semgnentation shape is {val_segmentations.shape}")
                val_images = val_images.to(device)

                image_embedding = model.image_encoder(val_images)
                box_torch = torch.stack([get_bounding_boxes_from_segmentation(
                    box)[0] for box in val_segmentations]).to(device)
                # print(f"Box torch shape is {box_torch.shape}")
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                upscaled_masks = F.interpolate(
                    low_res_masks, scale_factor=4, mode='bicubic', align_corners=False)

                binary_mask = normalize(
                    threshold(upscaled_masks, 0.0, 0)).to(device)
                if val_i % 10 == 0:
                    plot_segmentations_batch(
                        epoch=epoch, tr_i=val_i, pred_mask=binary_mask, gt_mask=val_segmentations, name="eval_binary_mask_vs_val_segmentation")

                loss = loss_function(binary_mask, val_segmentations)
                loss_sum += loss.item()
        mean_loss = loss_sum / len(val_loader)
        print(f"Validation avg loss at epoch {epoch}: {mean_loss}")


def get_model():
    img_size = 128
    model = SAM(img_size=img_size).to(device)

    if RESUME:
        model.load_state_dict(torch.load(
            f"{ARCHITECHTURE}_{FROM_EPOCH-1}.pt"))

    return model


def main():
    set_seed(RANDOM_SEED)
    train_eval_loop()


if __name__ == "__main__":
    main()
