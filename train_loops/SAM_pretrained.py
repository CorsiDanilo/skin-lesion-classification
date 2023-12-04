import cv2
import torch
import numpy as np
import random
import os
import wandb
from models.SAM import SAM
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from utils.plot_utils import plot_segmentations_batch
from utils.utils import get_bounding_boxes_from_segmentation
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
import torch.nn.functional as F
import torch.nn as nn
import monai

# Configurations
USE_WANDB = True
N_EPOCHS = 100
LR = 0.001
LR_DECAY = 0.85
ARCHITECHTURE = "SAM"
DATASET_LIMIT = None
NORMALIZE = False
BATCH_SIZE = 256
MOMENTUM = 0.9

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
            "momentum": MOMENTUM,
            "architecture": ARCHITECHTURE,
            "epochs": N_EPOCHS,
            "dataset": "HAM10K",
            "optimizer": "Adam",
            "dataset_limit": DATASET_LIMIT,
            "normalize": NORMALIZE,
            "resumed": RESUME,
            "from_epoch": FROM_EPOCH,
            "loss_type": "monai.DiceCELoss",
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
        normalize=NORMALIZE,
        upscale_train=False,
        batch_size=BATCH_SIZE)
    train_loader, val_loader = dataloader.get_train_val_dataloders()
    model = get_model()
    img_size = (model.get_img_size(), model.get_img_size())
    model = (model.model).to(device)

    def intersection_over_union(pred, target):
        intersection = (pred * target).sum((1, 2))
        union = pred.sum((1, 2)) + target.sum((1, 2)) - \
            intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()

    # def loss_function(pred, target):
        # return 1 - intersection_over_union(pred, target)

    loss_function = monai.losses.DiceCELoss(
        sigmoid=True, squared_pred=True, reduction='mean')

    optimizer = torch.optim.Adam(
        model.mask_decoder.parameters(), lr=LR, weight_decay=0)

    best_eval_accuracy = -torch.inf
    for epoch in range(FROM_EPOCH if RESUME else 0, N_EPOCHS):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(
            train_loader), desc=f"TRAINING | Epoch {epoch}")
        for tr_i, (tr_images, _, tr_segmentation) in enumerate(train_loader):

            # TODO: add these two functions to utils
            def resize_images(images, new_size=(800, 800)):
                return torch.stack([torch.from_numpy(cv2.resize(
                    image.permute(1, 2, 0).numpy(), new_size)) for image in images]).permute(0, 3, 1, 2)

            def resize_segmentations(segmentation, new_size=(800, 800)):
                return torch.stack([torch.from_numpy(cv2.resize(
                    image.permute(1, 2, 0).numpy(), new_size)) for image in segmentation]).unsqueeze(0).permute(1, 0, 2, 3)

            def take_points_from_segmentations(segmentations):
                import matplotlib.pyplot as plt
                segmentations = segmentations.squeeze()
                print(f"Segmentations shape is {segmentations.shape}")
                points = []
                labels = []

                # Loop over each image in the batch
                for i in range(segmentations.shape[0]):
                    segmentation = segmentations[i]

                    # Get the indices of all white (1) and black (0) pixels
                    white_indices = torch.nonzero(
                        segmentation == 1, as_tuple=False)
                    black_indices = torch.nonzero(
                        segmentation == 0, as_tuple=False)

                    center = torch.tensor(
                        [segmentation.shape[0] // 2, segmentation.shape[1] // 2])

                    # Calculate the Euclidean distance of each white point from the center
                    distances = torch.sqrt(
                        ((white_indices - center) ** 2).sum(dim=1))

                    # Select the white point with the smallest distance from the center
                    white_point = white_indices[distances.argmin()].flatten()
                    black_point = black_indices[torch.randint(
                        low=0, high=len(black_indices), size=(1,))].flatten()

                    points.append(torch.stack([white_point, black_point]))
                    labels.append(torch.tensor([1, 0]))

                    # plt.subplot(segmentations.shape[0] // 2, 2, i + 1)
                    # plt.imshow(segmentation, cmap='gray')
                    # plt.scatter([white_point[1], black_point[1]], [
                    #             white_point[0], black_point[0]], c=['g', 'r'])
                    # plt.axis('off')

                # plt.savefig(f"points_{epoch}_{tr_i}.png")
                points = torch.stack(points).squeeze()
                labels = torch.stack(labels)

                # print(f"Points shape is {points.shape}")
                # print(f"Labels shape is {labels.shape}")

                # print(f"Points and labels are {points} and {labels}")

                return points, labels

            with torch.no_grad():
                tr_images = tr_images.to(torch.float32)
                tr_images = resize_images(tr_images, new_size=img_size)
                tr_segmentation = resize_segmentations(
                    tr_segmentation, new_size=(img_size))

                tr_images = tr_images.to(device)
                tr_segmentation = tr_segmentation.to(device)

                box_torch = torch.stack([get_bounding_boxes_from_segmentation(
                    box)[0] for box in tr_segmentation]).to(device)

                tr_segmentation = normalize(threshold(
                    tr_segmentation, 0.8, 0)).to(device)

                # points = take_points_from_segmentations(tr_segmentation)
                # masks = tr_segmentation
                points = None
                masks = None

                image_embedding = model.image_encoder(tr_images)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=points,
                    boxes=box_torch,
                    masks=masks,
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
            # print(f"Tr segmentation shape is {tr_segmentation.shape}")

            binary_mask = normalize(
                threshold(upscaled_masks, 0.8, 0)).to(device)

            assert binary_mask.shape == tr_segmentation.shape

            loss = loss_function(binary_mask, tr_segmentation)
            iou = intersection_over_union(binary_mask, tr_segmentation)
            if USE_WANDB:
                wandb.log(
                    {"train_loss": loss.item()})
                wandb.log(
                    {"train_iou": iou.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f"TRAINING | Epoch {epoch} | Loss {loss.item():.4f} | Iou {iou:.4f}")
            pbar.update(1)

            if tr_i % 10 == 0:
                plot_segmentations_batch(
                    epoch=epoch, tr_i=tr_i, pred_mask=upscaled_masks[:10], gt_mask=tr_segmentation[:10], name="upscaled_masks_vs_tr_segmentation")
                plot_segmentations_batch(
                    epoch=epoch, tr_i=tr_i, pred_mask=binary_mask[:10], gt_mask=tr_segmentation[:10], name="binary_mask_vs_tr_segmentation")
        pbar.close()
        plot_segmentations_batch(
            epoch=epoch, tr_i=tr_i, pred_mask=binary_mask, gt_mask=tr_segmentation, name="binary_mask_vs_tr_segmentation")
        # Evaluation
        continue
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

                val_segmentations = normalize(threshold(
                    val_segmentations, 0.0, 0)).to(device)
                loss = loss_function(binary_mask, val_segmentations)
                iou = intersection_over_union(binary_mask, val_segmentations)
                if USE_WANDB:
                    wandb.log(
                        {"val_loss": loss.item()})
                    wandb.log(
                        {"val_iou": iou.item()})
                loss_sum += loss.item()
        mean_loss = loss_sum / len(val_loader)
        print(f"Validation avg loss at epoch {epoch}: {mean_loss}")


def get_model():
    img_size = 64
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
