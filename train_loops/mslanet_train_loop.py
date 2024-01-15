from typing import Any, Dict
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, roc_auc_score
from utils.utils import save_results, save_model, save_configurations
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
from datetime import datetime
import copy
from config import NUM_CLASSES, SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, USE_MULTIPLE_LOSS, MULTIPLE_LOSS_BALANCE


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

    criterion = nn.CrossEntropyLoss()  # Loss function

    if resume:
        data_name = PATH_MODEL_TO_RESUME
    else:
        # Definition of the parameters to create folders where to save data (plots and models)
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
        for tr_i, ((tr_image_ori, tr_image_low, tr_image_high), tr_labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            tr_image_ori = tr_image_ori.to(device)
            tr_image_low = tr_image_low.to(device)
            tr_image_high = tr_image_high.to(device)
            tr_labels = tr_labels.to(device)

            tr_output_ori = model(tr_image_ori)  # Prediction
            tr_output_low = model(tr_image_low)  # Prediction
            tr_output_high = model(tr_image_high)  # Prediction

            tr_outputs = (tr_output_ori + tr_output_low + tr_output_high) / 3

            # Multiclassification loss considering all classes
            tr_epoch_loss = criterion(tr_outputs, tr_labels)

            optimizer.zero_grad()
            tr_epoch_loss.backward()
            optimizer.step()

            with torch.no_grad():
                tr_preds = torch.argmax(tr_outputs, -1).detach()
                epoch_tr_preds = torch.cat((epoch_tr_preds, tr_preds), 0)
                epoch_tr_labels = torch.cat((epoch_tr_labels, tr_labels), 0)

                tr_accuracy = accuracy_score(
                    epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy()) * 100
                tr_sensitivity = recall_score(
                    epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy(), average='macro', zero_division=0) * 100
                conf_matrix = confusion_matrix(epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy())
                tr_specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) * 100

                if (tr_i+1) % 5 == 0:
                    print('Training -> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Sensitivity (Recall): {:.4f}%, Specificity: {:.4f}%'
                          .format(epoch+1, config["epochs"], tr_i+1, total_step, tr_epoch_loss, tr_accuracy, tr_sensitivity, tr_specificity))

                for class_label in range(NUM_CLASSES):
                    #class_indices = epoch_tr_labels == class_label
                    #class_preds = epoch_tr_preds[class_indices]
                    #class_labels = epoch_tr_labels[class_indices]
                    class_labels_binary = torch.zeros_like(epoch_tr_labels, dtype=torch.long).to(device)
                    class_labels_binary[(epoch_tr_labels == class_label)] = 1

                    class_preds_binary = torch.zeros_like(epoch_tr_preds, dtype=torch.long).to(device)
                    class_preds_binary[(epoch_tr_preds == class_label)] = 1

                    if len(class_preds_binary) > 0:
                        class_accuracy = accuracy_score(
                            class_labels_binary.cpu().numpy(), class_preds_binary.cpu().numpy()) * 100
                        class_sensitivity = recall_score(
                            class_labels_binary.cpu().numpy(), class_preds_binary.cpu().numpy(), average='binary', pos_label=1, zero_division=0) * 100
                        class_conf_matrix = confusion_matrix(
                            class_labels_binary.cpu().numpy(), class_preds_binary.cpu().numpy())
                        if (class_conf_matrix[0, 0] + class_conf_matrix[0, 1]) > 0:
                            class_specificity = class_conf_matrix[0, 0] / (class_conf_matrix[0, 0] + class_conf_matrix[0, 1]) * 100
                        else:
                            class_specificity = 0
                        if len(set(class_labels_binary.cpu().numpy())) > 1:
                            class_auc = roc_auc_score(
                                class_labels_binary.cpu().numpy(), class_preds_binary.cpu().numpy()) * 100
                        else:
                            class_auc = 0
                        
                        if (tr_i+1) % 5 == 0:
                            print(f'Class {class_label} - Accuracy: {class_accuracy:.2f}%, Sensitivity: {class_sensitivity:.2f}%, Specificity: {class_specificity:.2f}%, AUC: {class_auc:.2f}%')
                

        if config["use_wandb"]:
            wandb.log({"Training Loss": tr_epoch_loss.item()})
            wandb.log({"Training Accuracy": tr_accuracy})
            wandb.log({"Training Sensitivity": tr_sensitivity})
            wandb.log({"Training Specificity": tr_specificity})

        model.eval()
        with torch.no_grad():
            epoch_val_preds = torch.tensor([]).to(device)
            epoch_val_labels = torch.tensor([]).to(device)
            for val_i, ((val_image_ori, val_image_low, val_image_high), val_labels) in enumerate(val_loader):
                val_image_ori = val_image_ori.to(device)
                val_image_low = val_image_low.to(device)
                val_image_high = val_image_high.to(device)
                val_labels = val_labels.to(device)

                val_output_ori = model(val_image_ori)  # Prediction
                val_output_low = model(val_image_low)  # Prediction
                val_output_high = model(val_image_high)  # Prediction

                val_outputs = (
                    val_output_ori + val_output_low + val_output_high) / 3

                val_preds = torch.argmax(val_outputs, -1).detach()
                epoch_val_preds = torch.cat((epoch_val_preds, val_preds), 0)
                epoch_val_labels = torch.cat((epoch_val_labels, val_labels), 0)

                # First loss: Multiclassification loss considering all classes
                val_epoch_loss_multiclass = criterion(
                    val_outputs, val_labels)
                val_epoch_loss = val_epoch_loss_multiclass

                if config["multiple_loss"]:
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

                    val_epoch_loss_binary = criterion(
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
                'validation_accuracy': val_accuracy,
                'validation_recall': val_recall,
                'training_loss': tr_epoch_loss.item(),
                'training_accuracy': tr_accuracy,
                'training_specificity': tr_specificity,
                'training_sensitivity': tr_sensitivity
            }
            if SAVE_RESULTS:
                save_results(data_name, current_results)
            if SAVE_MODELS:
                save_model(data_name, model, epoch)
            if epoch == config["epochs"]-1 and SAVE_MODELS:
                save_model(data_name, best_model, epoch=None, is_best=True)
