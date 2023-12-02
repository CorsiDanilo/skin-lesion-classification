from typing import Dict
import cv2
import torch
import torch.nn as nn
import numpy as np
import random
import os
import wandb
from models.SAM import SAM
from dataloaders.old_dataloaders import create_dataloaders
from tqdm import tqdm
from torch.nn.functional import threshold, normalize

from sklearn.metrics import recall_score, accuracy_score

from utils.utils import get_bounding_boxes_from_segmentation

USE_WANDB = False
# Configurations
N_EPOCHS = 100
LR = 1e-3
LR_DECAY = 0.85
REG = 0.01
ARCHITECHTURE = "sam"
DATASET_LIMIT = None
NORMALIZE = True
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
resnet_mean = torch.tensor([0.485, 0.456, 0.406])
resnet_std = torch.tensor([0.229, 0.224, 0.225])


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


def create_loaders():
    # Load the dataset
    resnet_mean = torch.tensor([0.485, 0.456, 0.406])
    resnet_std = torch.tensor([0.229, 0.224, 0.225])

    train_loader, val_loader, test_loader = create_dataloaders(
        mean=resnet_mean,
        std=resnet_std,
        normalize=NORMALIZE,
        limit=DATASET_LIMIT,
        batch_size=BATCH_SIZE,
        dynamic_load=True)
    return train_loader, val_loader, test_loader


def save_model(model, model_name, epoch):
    torch.save(model.state_dict(), f"{model_name}_{epoch}.pt")


def train_eval_loop():
    train_loader, val_loader, _ = create_loaders()
    model = get_model().model
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.mask_decoder.parameters())

    total_step = len(train_loader)
    best_accuracy = -torch.inf
    for epoch in range(FROM_EPOCH if RESUME else 0, N_EPOCHS):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(
            train_loader), desc=f"TRAINING | Epoch {epoch}")
        for tr_i, (tr_images, tr_labels, tr_segmentation) in enumerate(train_loader):
            tr_images = tr_images.to(torch.float32)

            def resize_images(images, new_size=(800, 800)):
                return torch.stack([torch.from_numpy(cv2.resize(
                    image.permute(1, 2, 0).numpy(), new_size)) for image in images]).permute(0, 3, 1, 2)

            def resize_segmentations(segmentation, new_size=(800, 800)):
                return torch.stack([torch.from_numpy(cv2.resize(
                    image.permute(1, 2, 0).numpy(), new_size)) for image in segmentation]).unsqueeze(0).permute(1, 0, 2, 3)

            tr_images = resize_images(tr_images, new_size=(64, 64))
            tr_segmentation = resize_segmentations(
                tr_segmentation, new_size=(64, 64))
            # print(f"tr_semgnentation shape is {tr_segmentation.shape}")
            tr_images = tr_images.to(device)
            tr_labels = tr_labels.to(device)
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
            upscaled_masks = model.postprocess_masks(
                low_res_masks, (64, 64), (64, 64)).to(device)

            binary_mask = normalize(
                threshold(upscaled_masks, 0.0, 0)).to(device)

            loss = loss_function(binary_mask, tr_segmentation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f"TRAINING | Epoch {epoch} | Loss {loss.item():.4f} | Iou {torch.mean(iou_predictions):.4f}")
            pbar.update(1)
            continue
        return
        model.eval()
        with torch.no_grad():
            val_loss_iter = 0
            epoch_val_preds = torch.tensor([]).to(device)
            epoch_val_labels = torch.tensor([]).to(device)
            for val_i, (val_images, val_labels) in enumerate(val_loader):

                val_images = val_images.to(torch.float32)
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                validation_preds = torch.argmax(val_outputs, -1).detach()
                epoch_val_preds = torch.cat(
                    (epoch_val_preds, validation_preds), 0)
                epoch_val_labels = torch.cat(
                    (epoch_val_labels, val_labels), 0)

                val_loss = loss_function(val_outputs, val_labels)
                if USE_WANDB:
                    wandb.log({"Validation Loss": val_loss.item()})
                val_loss_iter += val_loss.item()

            val_accuracy = accuracy_score(
                epoch_val_labels.cpu().numpy(), epoch_val_preds.cpu().numpy()) * 100
            val_recall = recall_score(
                epoch_val_labels.cpu().numpy(), epoch_val_preds.cpu().numpy(), average='macro') * 100
            val_f1 = 2 * (val_accuracy * val_recall) / \
                (val_accuracy + val_recall)

            # if val_accuracy > best_accuracy:
            #     best_accuracy = val_accuracy
            #     save_model(model, ARCHITECHTURE, epoch)

            if USE_WANDB:
                wandb.log({"Validation Accuracy": val_accuracy})
                wandb.log({"Validation Recall": val_recall})
                wandb.log({"Validation F1": val_f1})

            print(
                'Validation -> Validation accuracy for epoch {} is: {:.4f}%'.format(epoch+1, val_accuracy))
            print(
                'Validation -> Validation recall for epoch {} is: {:.4f}%'.format(epoch+1, val_recall))
            print(
                'Validation -> Validation f1 for epoch {} is: {:.4f}%'.format(epoch+1, val_f1))
            print(
                'Validation -> Validation loss for epoch {} is: {:.4f}'.format(epoch+1, val_loss.item()))


def get_model():
    model = SAM().to(device)

    if RESUME:
        model.load_state_dict(torch.load(
            f"{ARCHITECHTURE}_{FROM_EPOCH-1}.pt"))

    return model


def main():
    set_seed(RANDOM_SEED)
    train_eval_loop()


if __name__ == "__main__":
    main()
