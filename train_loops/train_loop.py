from typing import Any, Dict
from sklearn.metrics import recall_score, accuracy_score
from utils.utils import save_results, save_model, save_configurations
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
from datetime import datetime
import copy
from config import SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH


def train_eval_loop(device,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    model,
                    config,
                    optimizer,
                    scheduler,
                    resume=False):

    if config["use_wandb"] and "hparam_tuning" not in config:
        # Start a new run
        wandb.init(
            project="melanoma",
            config=config,  # Track hyperparameters and run metadata
            resume=resume,
        )

    loss_function_multiclass = nn.CrossEntropyLoss()
    if config["double_loss"]:
        loss_function_binary = nn.CrossEntropyLoss()

    if resume:
        data_name = PATH_MODEL_TO_RESUME
    else:
        # Creation of folders where to save data (plots and models)
        current_datetime = datetime.now()
        current_datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        data_name = f"{config['architecture']}_{current_datetime_str}"

        if SAVE_RESULTS:
            # Save configurations in JSON
            save_configurations(data_name, config)

    total_step = len(train_loader)
    best_model = None
    best_accuracy = None
    for epoch in range(RESUME_EPOCH if resume else 0, config["epochs"]):
        model.train()
        epoch_tr_preds = torch.tensor([]).to(device)
        epoch_tr_labels = torch.tensor([]).to(device)
        for tr_i, tr_batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            if len(tr_batch) == 3:
                tr_images, tr_labels, _ = tr_batch
            else:
                tr_images, tr_labels = tr_batch
            tr_images = tr_images.to(device)
            tr_labels = tr_labels.to(device)

            tr_outputs = model(tr_images)  # Prediction

            # First loss: Multiclassification loss considering all classes
            tr_epoch_loss_multiclass = loss_function_multiclass(
                tr_outputs, tr_labels)
            tr_epoch_loss = tr_epoch_loss_multiclass

            if config["double_loss"]:
                tr_labels_binary = torch.zeros_like(
                    tr_labels, dtype=torch.long).to(device)
                # Set ground-truth to 1 for classes 2, 3, and 4 (the malignant classes)
                tr_labels_binary[(tr_labels == 2) | (
                    tr_labels == 3) | (tr_labels == 4)] = 1

                # Second loss: Binary loss considering only benign/malignant classes
                tr_outputs_binary = torch.zeros_like(
                    tr_outputs[:, :2]).to(device)
                tr_outputs_binary[:, 1] = torch.sum(
                    tr_outputs[:, [2, 3, 4]], dim=1)
                tr_outputs_binary[:, 0] = 1 - tr_outputs_binary[:, 1]

                tr_epoch_loss_binary = loss_function_binary(
                    tr_outputs_binary, tr_labels_binary)

                # Sum of the losses
                tr_epoch_loss += tr_epoch_loss_binary

            optimizer.zero_grad()
            tr_epoch_loss.backward()
            optimizer.step()

            with torch.no_grad():
                tr_preds = torch.argmax(tr_outputs, -1).detach()
                epoch_tr_preds = torch.cat((epoch_tr_preds, tr_preds), 0)
                epoch_tr_labels = torch.cat((epoch_tr_labels, tr_labels), 0)

        with torch.no_grad():
            tr_accuracy = accuracy_score(
                epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy()) * 100
            tr_recall = recall_score(
                epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy(), average='macro', zero_division=0) * 100

            if config["use_wandb"]:
                wandb.log({"Training Loss": tr_epoch_loss.item()})
                wandb.log({"Training Accuracy": tr_accuracy})
                wandb.log({"Training Recall": tr_recall})
            if (tr_i+1) % 50 == 0:
                print('Training -> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%'
                      .format(epoch+1, config["epochs"], tr_i+1, total_step, tr_epoch_loss, tr_accuracy, tr_recall))
        model.eval()
        with torch.no_grad():
            epoch_val_preds = torch.tensor([]).to(device)
            epoch_val_labels = torch.tensor([]).to(device)
            for val_i, val_batch in enumerate(val_loader):
                if len(val_batch) == 3:
                    val_images, val_labels, _ = val_batch
                else:
                    val_images, val_labels = val_batch
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images).to(device)
                val_preds = torch.argmax(val_outputs, -1).detach()
                epoch_val_preds = torch.cat((epoch_val_preds, val_preds), 0)
                epoch_val_labels = torch.cat((epoch_val_labels, val_labels), 0)

                # First loss: Multiclassification loss considering all classes
                val_epoch_loss_multiclass = loss_function_multiclass(
                    val_outputs, val_labels)
                val_epoch_loss = val_epoch_loss_multiclass

                if config["double_loss"]:
                    val_labels_binary = torch.zeros_like(
                        val_labels, dtype=torch.long).to(device)
                    # Set ground-truth to 1 for classes 2, 3, and 4 (the malignant classes)
                    val_labels_binary[(val_labels == 2) | (
                        val_labels == 3) | (val_labels == 4)] = 1

                    # Second loss: Binary loss considering only benign/malignant classes
                    val_outputs_binary = torch.zeros_like(
                        val_outputs[:, :2]).to(device)
                    val_outputs_binary[:, 1] = torch.sum(
                        val_outputs[:, [2, 3, 4]], dim=1)
                    val_outputs_binary[:, 0] = 1 - val_outputs_binary[:, 1]

                    val_epoch_loss_binary = loss_function_binary(
                        val_outputs_binary, val_labels_binary)

                    # Sum of the losses
                    val_epoch_loss += val_epoch_loss_binary

            val_accuracy = accuracy_score(
                epoch_val_labels.cpu().numpy(), epoch_val_preds.cpu().numpy()) * 100
            val_recall = recall_score(epoch_val_labels.cpu().numpy(
            ), epoch_val_preds.cpu().numpy(), average='macro', zero_division=0) * 100
            if config["use_wandb"]:
                wandb.log({"Validation Loss": val_epoch_loss.item()})
                wandb.log({"Validation Accuracy": val_accuracy})
                wandb.log({"Validation Recall": val_recall})
            print('Validation -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%'
                  .format(epoch+1, config["epochs"], val_epoch_loss, val_accuracy, val_recall))

            if best_accuracy is None or val_accuracy < best_accuracy:
                best_accuracy = val_accuracy
                best_model = copy.deepcopy(model)
            current_results = {
                'epoch': epoch+1,
                'validation_loss': val_epoch_loss.item(),
                'training_loss': tr_epoch_loss.item(),
                'validation_accuracy': val_accuracy,
                'training_accuracy': tr_accuracy,
                'validation_recall': val_recall,
                'training_recall': tr_recall
            }
            if SAVE_RESULTS:
                save_results(data_name, current_results)
            if SAVE_MODELS:
                save_model(data_name, model, epoch)
            if epoch == config["epochs"]-1 and SAVE_MODELS:
                save_model(data_name, best_model, epoch=None, is_best=True)

        scheduler.step()
