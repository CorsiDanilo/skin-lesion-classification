from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import random
import os
import wandb
import yolov5

from detection_dataloaders import create_dataloaders
from tqdm import tqdm

from sklearn.metrics import recall_score, accuracy_score

from utilities import get_bounding_boxes_from_segmentation, zoom_out
import numpy as np
import matplotlib.pyplot as plt

USE_WANDB = False
# Configurations
N_EPOCHS = 100
LR = 1e-3
LR_DECAY = 0.85
REG = 0.01
ARCHITECHTURE = "rcnn"
DATASET_LIMIT = None
NORMALIZE = False
BALANCE_UNDERSAMPLING = 1
BATCH_SIZE = 4

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


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    model = get_model()
    loss_function = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    total_step = len(train_loader)
    best_accuracy = -torch.inf
    for epoch in range(FROM_EPOCH if RESUME else 0, N_EPOCHS):
        model.train()
        epoch_tr_preds = torch.tensor([]).to(device)
        epoch_tr_labels = torch.tensor([]).to(device)
        for tr_i, (tr_images, tr_labels, tr_segmentations) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            tr_images = tr_images.to(torch.float32)
            tr_images = tr_images.to(device)
            tr_images = torch.stack([zoom_out(img)
                                    for img in tr_images], dim=0)
            print(f"Tr_images shape is {tr_images.shape}")
            tr_outputs = model(tr_images)
            # Select the predictions for the first image in the batch
            preds = [tensor[0] for tensor in tr_outputs]
            print(
                f"Preds has lentgh {len(preds)} with shapes {[pred.shape for pred in preds]}")
            # Concatenate the predictions from all scales
            preds = torch.cat([pred.view(-1, 85) for pred in preds], dim=0)

            # Select only the bounding box coordinates
            boxes = preds[..., :4]

            images = [img.permute(2, 1, 0).numpy()
                      for img in tr_images]
            print(f"Images shape is {np.stack(images).shape}")
            print(f"Boxes are {boxes} with shape {boxes.shape}")
            # Tr output is
            # {'loss_classifier': tensor(0.6931),
            # 'loss_box_reg': tensor(0., grad_fn=<DivBackward0>),
            # 'loss_objectness': tensor(3.3467),
            # 'loss_rpn_box_reg': tensor(0.5923)}
            # print(
            # f"Tr output at step {tr_i} is {tr_outputs[0]} with length {tr_outputs[0].shape}")
            raise NotImplementedError("Debug time!")
            tr_loss = tr_outputs["loss_box_reg"]
            if USE_WANDB:
                wandb.log({"Training Loss": tr_loss.item()})

            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

        #     with torch.no_grad():
        #         training_preds = torch.argmax(tr_outputs, -1).detach()
        #         epoch_tr_preds = torch.cat(
        #             (epoch_tr_preds, training_preds), 0)
        #         epoch_tr_labels = torch.cat(
        #             (epoch_tr_labels, tr_labels), 0)

        # tr_accuracy = accuracy_score(
        #     epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy()) * 100
        # tr_recall = recall_score(
        #     epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy(), average='macro') * 100
        # tr_f1 = 2 * (tr_accuracy * tr_recall) / (tr_accuracy + tr_recall)
        # if USE_WANDB:
        #     wandb.log({"Training Accuracy": tr_accuracy})
        #     wandb.log({"Training Recall": tr_recall})
        #     wandb.log({"Training F1": tr_f1})

        # print('Training -> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%'
        #       .format(epoch+1, N_EPOCHS, tr_i+1, total_step, tr_loss.item, tr_accuracy, tr_recall))
        continue
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


def parse_target(segmentation: torch.Tensor) -> Dict[torch.Tensor, torch.Tensor]:
    target = {}
    # Replace with your function to get the bounding boxes
    target['boxes'] = get_bounding_boxes_from_segmentation(segmentation)
    # Replace with your function to get the labels
    target['labels'] = [0 for _ in range(len(target['boxes']))]
    target["boxes"] = target["boxes"].to(device)
    target["labels"] = torch.tensor(
        target["labels"]).to(torch.int64).to(device)
    return target


def get_model():
    model = yolov5.load("yolov5s.pt", device=device)

    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    if RESUME:
        model.load_state_dict(torch.load(
            f"{ARCHITECHTURE}_{FROM_EPOCH-1}.pt"))

    return model


def main():
    set_seed(RANDOM_SEED)
    train_eval_loop()


if __name__ == "__main__":
    main()
