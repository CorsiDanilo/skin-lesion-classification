from sklearn.metrics import recall_score, accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
import os
import wandb
from enum import Enum
from datetime import datetime
import copy
from config import BALANCE_UNDERSAMPLING, BATCH_SIZE, INPUT_SIZE, NUM_CLASSES, HIDDEN_SIZE, N_EPOCHS, LR, REG, SEGMENT, CROP_ROI, ARCHITECTURE_CNN, DATASET_LIMIT, DROPOUT_P, NORMALIZE, SEGMENTATION_BOUNDING_BOX, USE_WANDB, PATH_TO_SAVE_RESULTS, SAVE_MODELS, SAVE_RESULTS, USE_DOUBLE_LOSS
from utils.utils import select_device, save_results, save_model
from dataloaders.DynamicSegmentationDataLoader import DynamicSegmentationDataLoader, DynamicSegmentationStrategy
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from dataloaders.SegmentedImagesDataLoader import SegmentedImagesDataLoader
from models.ResNet24Pretrained import ResNet24Pretrained
from models.DenseNetPretrained import DenseNetPretrained
from models.InceptionV3Pretrained import InceptionV3Pretrained


class SegmentationStrategy(Enum):
    DYNAMIC_SEGMENTATION = "dynamic_segmentation"
    SEGMENTATION = "segmentation"
    NO_SEGMENTATION = "no_segmentation"


SEGMENTATION_STRATEGY = SegmentationStrategy.SEGMENTATION
DYNAMIC_SEGMENTATION_STRATEGY = DynamicSegmentationStrategy.OPENCV if SEGMENTATION_STRATEGY == SegmentationStrategy.DYNAMIC_SEGMENTATION else None
dynamic_load = True

if SEGMENTATION_STRATEGY == SegmentationStrategy.DYNAMIC_SEGMENTATION:
    dataloader = DynamicSegmentationDataLoader(
        limit=DATASET_LIMIT,
        dynamic_load=dynamic_load,
        segmentation_strategy=DYNAMIC_SEGMENTATION_STRATEGY
    )
elif SEGMENTATION_STRATEGY == SegmentationStrategy.SEGMENTATION:
    dataloader = SegmentedImagesDataLoader(
        limit=DATASET_LIMIT,
        dynamic_load=dynamic_load,
    )
elif SEGMENTATION_STRATEGY == SegmentationStrategy.NO_SEGMENTATION:
    dataloader = ImagesAndSegmentationDataLoader(
        limit=DATASET_LIMIT,
        dynamic_load=dynamic_load,
    )
else:
    raise NotImplementedError(
        f"Segmentation strategy {SEGMENTATION_STRATEGY} not implemented")

device = select_device()

RESUME = False
PATH_MODEL_TO_RESUME = f"CNN_resnet24_2023-12-04_17-26-22"
FROM_EPOCH = 2

if CROP_ROI:
    assert SEGMENT, f"Crop roi needs segment to be True"

if USE_WANDB:
    # Start a new run
    wandb.init(
        project="melanoma",

        # track hyperparameters and run metadata
        config={
            "learning_rate": LR,
            "architecture": ARCHITECTURE_CNN,
            "epochs": N_EPOCHS,
            'reg': REG,
            'batch_size': BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "dataset": "HAM10K",
            "optimizer": "AdamW",
            "segmentation": SEGMENT,
            "crop_roi": CROP_ROI,
            "dataset_limit": DATASET_LIMIT,
            "dropout_p": DROPOUT_P,
            "normalize": NORMALIZE,
            "resumed": RESUME,
            "from_epoch": FROM_EPOCH,
            "segmentation_bounding_box": SEGMENTATION_BOUNDING_BOX,
            "balance_undersampling": BALANCE_UNDERSAMPLING,
            "initialization": "default",
            'segmentation_strategy': SEGMENTATION_STRATEGY,
            'dynamic_segmentation_strategy': DYNAMIC_SEGMENTATION_STRATEGY,
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


def train_eval_loop():
    train_loader, val_loader = dataloader.get_train_val_dataloders()
    model = get_model()
    loss_function_multiclass = nn.CrossEntropyLoss()
    if USE_DOUBLE_LOSS:
        loss_function_binary = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    if RESUME:
        data_name = PATH_MODEL_TO_RESUME
    else:
        # Creation of folders where to save data (plots and models)
        current_datetime = datetime.now()
        current_datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        data_name = f"CNN_{ARCHITECTURE_CNN}_{current_datetime_str}"

    total_step = len(train_loader)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_recalls = []
    val_recalls = []
    best_model = None
    best_loss = None
    for epoch in range(FROM_EPOCH if RESUME else 0, N_EPOCHS):
        model.train()
        tr_loss_iter = 0
        epoch_tr_preds = torch.tensor([]).to(device)
        epoch_tr_labels = torch.tensor([]).to(device)
        for tr_i, (tr_images, tr_labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            tr_images = tr_images.to(device)
            tr_labels = tr_labels.to(device)

            tr_outputs = model(tr_images) #Prediction
            
            #First loss: Multiclassification loss considering all classes
            tr_epoch_loss_multiclass = loss_function_multiclass(tr_outputs, tr_labels)
            tr_epoch_loss = tr_epoch_loss_multiclass

            if USE_DOUBLE_LOSS:
                tr_labels_binary = torch.zeros_like(tr_labels, dtype=torch.long).to(device)
                tr_labels_binary[(tr_labels == 0) | (tr_labels == 1) | (tr_labels == 6)] = 1  # Set ground-truth to 1 for classes 0, 1, and 6 (the malignant classes)

                #Second loss: Binary loss considering only benign/malignant classes
                tr_outputs_binary = torch.zeros_like(tr_outputs[:, :2]).to(device)
                tr_outputs_binary[:, 1] = torch.sum(tr_outputs[:, [0, 1, 6]], dim=1)
                tr_outputs_binary[:, 0] = 1 - tr_outputs_binary[:, 1]

                tr_epoch_loss_binary = loss_function_binary(tr_outputs_binary, tr_labels_binary)

                #Sum of the losses
                tr_epoch_loss += tr_epoch_loss_binary
            if USE_WANDB:
                wandb.log({"Training Loss": tr_epoch_loss.item()})

            optimizer.zero_grad()
            tr_epoch_loss.backward()
            optimizer.step()

            with torch.no_grad():
                training_preds = torch.argmax(tr_outputs, -1).detach()
                epoch_tr_preds = torch.cat((epoch_tr_preds, training_preds), 0)
                epoch_tr_labels = torch.cat((epoch_tr_labels, tr_labels), 0)

                tr_loss_iter += tr_epoch_loss.item()
                tr_accuracy = accuracy_score(epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy()) * 100
                train_accuracies.append(tr_accuracy)
                tr_recall = recall_score(epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy(), average='macro', zero_division=0) * 100
                train_recalls.append(tr_recall)
                tr_loss = tr_loss_iter/(len(train_loader)*BATCH_SIZE)      
                train_losses.append(tr_loss)
                
                if (tr_i+1) % 50 == 0:
                    print ('Training -> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%'
                        .format(epoch+1, N_EPOCHS, tr_i+1, total_step, tr_loss, tr_accuracy, tr_recall))

        model.eval()
        with torch.no_grad():
            val_loss_iter = val_batch_iter = 0
            epoch_val_preds = torch.tensor([]).to(device)
            epoch_val_labels = torch.tensor([]).to(device)
            for val_i, (val_images, val_labels) in enumerate(val_loader):
                val_batch_iter += 1
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                
                val_outputs = model(val_images).to(device)
                val_preds = torch.argmax(val_outputs, -1).detach()
                epoch_val_preds = torch.cat((epoch_val_preds, val_preds), 0)
                epoch_val_labels = torch.cat((epoch_val_labels, val_labels), 0)  
                
                #First loss: Multiclassification loss considering all classes
                val_epoch_loss_multiclass = loss_function_multiclass(val_outputs, val_labels)
                val_epoch_loss = val_epoch_loss_multiclass

                if USE_DOUBLE_LOSS:
                    val_labels_binary = torch.zeros_like(val_labels, dtype=torch.long).to(device)
                    val_labels_binary[(val_labels == 0) | (val_labels == 1) | (val_labels == 6)] = 1  # Set ground-truth to 1 for classes 0, 1, and 6 (the malignant classes)

                    #Second loss: Binary loss considering only benign/malignant classes
                    val_outputs_binary = torch.zeros_like(val_outputs[:, :2]).to(device)
                    val_outputs_binary[:, 1] = torch.sum(val_outputs[:, [0, 1, 6]], dim=1)
                    val_outputs_binary[:, 0] = 1 - val_outputs_binary[:, 1]

                    val_epoch_loss_binary = loss_function_binary(val_outputs_binary, val_labels_binary)

                    #Sum of the losses
                    val_epoch_loss += val_epoch_loss_binary
                    
                val_loss_iter += val_epoch_loss.item()
                avg_val_loss = val_loss_iter / val_batch_iter #Calculate the average validation loss
                #scheduler.step(avg_val_loss) #Step the scheduler based on the validation loss
            scheduler.step() #Activate the scheduler
            if USE_WANDB:
                wandb.log({"Validation Loss": val_epoch_loss.item()})
            val_loss = val_loss_iter/(len(val_loader)*BATCH_SIZE)
            val_losses.append(val_loss)
            val_accuracy = accuracy_score(epoch_val_labels.cpu().numpy(), epoch_val_preds.cpu().numpy()) * 100
            val_accuracies.append(val_accuracy)
            val_recall = recall_score(epoch_val_labels.cpu().numpy(), epoch_val_preds.cpu().numpy(), average='macro', zero_division=0) * 100
            val_recalls.append(val_recall)
            if USE_WANDB:
                wandb.log({"Validation Accuracy": val_accuracy})
                wandb.log({"Validation Recall": val_recall})
            print ('Validation -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%'
                        .format(epoch+1, N_EPOCHS, val_loss, val_accuracy, val_recall))

            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model)
            current_results = {
                'epoch': epoch+1,
                'validation_loss': val_loss,
                'training_loss': tr_loss,
                'validation_accuracy': val_accuracy,
                'training_accuracy': tr_accuracy,
                'validation_recall': val_recall,
                'training_recall': tr_recall
            }
            if SAVE_RESULTS:
                save_results(data_name, current_results)
            if SAVE_MODELS:
                save_model(data_name, model, epoch)
            if epoch == N_EPOCHS-1 and SAVE_MODELS:
                save_model(data_name, best_model, epoch=None, is_best=True)


def get_model():
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
        model.load_state_dict(torch.load(f"{PATH_TO_SAVE_RESULTS}/{PATH_MODEL_TO_RESUME}/models/melanoma_detection_ep{FROM_EPOCH}.pt"))

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


def main():
    set_seed(RANDOM_SEED)
    train_eval_loop()


if __name__ == "__main__":
    main()
