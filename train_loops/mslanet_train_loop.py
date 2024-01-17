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

                if (tr_i+1) % 5 == 0:
                    print('Training -> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Sensitivity (Recall): {:.4f}%'
                          .format(epoch+1, config["epochs"], tr_i+1, total_step, tr_epoch_loss, tr_accuracy, tr_sensitivity))

                tr_classes_metrics = {}
                for class_label in range(NUM_CLASSES):
                    # class_indices = epoch_tr_labels == class_label
                    # class_preds = epoch_tr_preds[class_indices]
                    # class_labels = epoch_tr_labels[class_indices]
                    tr_class_labels_binary = torch.zeros_like(
                        epoch_tr_labels, dtype=torch.long).to(device)
                    tr_class_labels_binary[(
                        epoch_tr_labels == class_label)] = 1

                    tr_class_preds_binary = torch.zeros_like(
                        epoch_tr_preds, dtype=torch.long).to(device)
                    tr_class_preds_binary[(epoch_tr_preds == class_label)] = 1

                    if len(tr_class_preds_binary) > 0:
                        tr_class_accuracy = accuracy_score(
                            tr_class_labels_binary.cpu().numpy(), tr_class_preds_binary.cpu().numpy()) * 100
                        tr_class_sensitivity = recall_score(
                            tr_class_labels_binary.cpu().numpy(), tr_class_preds_binary.cpu().numpy(), average='binary', pos_label=1, zero_division=0) * 100
                        tr_class_conf_matrix = confusion_matrix(
                            tr_class_labels_binary.cpu().numpy(), tr_class_preds_binary.cpu().numpy())
                        if tr_class_conf_matrix.shape == (2, 2) and (tr_class_conf_matrix[0, 0] + tr_class_conf_matrix[0, 1]) > 0:
                            tr_class_specificity = tr_class_conf_matrix[0, 0] / (
                                tr_class_conf_matrix[0, 0] + tr_class_conf_matrix[0, 1]) * 100
                        else:
                            tr_class_specificity = 0
                        if len(set(tr_class_labels_binary.cpu().numpy())) > 1:
                            tr_class_auc = roc_auc_score(
                                tr_class_labels_binary.cpu().numpy(), tr_class_preds_binary.cpu().numpy()) * 100
                        else:
                            tr_class_auc = 0

                        tr_class_metrics = {"accuracy": tr_class_accuracy, "sensitivity": tr_class_sensitivity,
                                            "specificity": tr_class_specificity, "auc": tr_class_auc}
                        tr_classes_metrics[class_label] = tr_class_metrics

                        if (tr_i+1) % 5 == 0:
                            print(
                                f'Class {class_label} - Accuracy: {tr_class_accuracy:.2f}%, Sensitivity: {tr_class_sensitivity:.2f}%, Specificity: {tr_class_specificity:.2f}%, AUC: {tr_class_auc:.2f}%')

        if config["use_wandb"]:
            wandb.log({"Training Loss": tr_epoch_loss.item()})
            wandb.log({"Training Accuracy": tr_accuracy})
            wandb.log({"Training Sensitivity": tr_sensitivity})
            wandb.log({"Training Classes Metrics": tr_classes_metrics})

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

            val_accuracy = accuracy_score(
                epoch_val_labels.cpu().numpy(), epoch_val_preds.cpu().numpy()) * 100
            val_sensitivity = recall_score(
                epoch_val_labels.cpu().numpy(), epoch_val_preds.cpu().numpy(), average='macro', zero_division=0) * 100

            print('Validation -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Sensitivity (Recall): {:.4f}%'
                  .format(epoch+1, config["epochs"], val_epoch_loss, val_accuracy, val_sensitivity))

            val_classes_metrics = {}
            for class_label in range(NUM_CLASSES):
                val_class_labels_binary = torch.zeros_like(
                    epoch_val_labels, dtype=torch.long).to(device)
                val_class_labels_binary[(epoch_val_labels == class_label)] = 1

                val_class_preds_binary = torch.zeros_like(
                    epoch_val_preds, dtype=torch.long).to(device)
                val_class_preds_binary[(epoch_val_preds == class_label)] = 1

                if len(val_class_preds_binary) > 0:
                    val_class_accuracy = accuracy_score(
                        val_class_labels_binary.cpu().numpy(), val_class_preds_binary.cpu().numpy()) * 100
                    val_class_sensitivity = recall_score(
                        val_class_labels_binary.cpu().numpy(), val_class_preds_binary.cpu().numpy(), average='binary', pos_label=1, zero_division=0) * 100
                    val_class_conf_matrix = confusion_matrix(
                        val_class_labels_binary.cpu().numpy(), val_class_preds_binary.cpu().numpy())
                    if val_class_conf_matrix.shape == (2, 2) and (val_class_conf_matrix[0, 0] + val_class_conf_matrix[0, 1]) > 0:
                        val_class_specificity = val_class_conf_matrix[0, 0] / (
                            val_class_conf_matrix[0, 0] + val_class_conf_matrix[0, 1]) * 100
                    else:
                        val_class_specificity = 0
                    if len(set(val_class_labels_binary.cpu().numpy())) > 1:
                        val_class_auc = roc_auc_score(
                            val_class_labels_binary.cpu().numpy(), val_class_preds_binary.cpu().numpy()) * 100
                    else:
                        val_class_auc = 0

                    val_class_metrics = {"accuracy": val_class_accuracy, "sensitivity": val_class_sensitivity,
                                         "specificity": val_class_specificity, "auc": val_class_auc}
                    val_classes_metrics[class_label] = val_class_metrics

                    print(f'Class {class_label} - Accuracy: {val_class_accuracy:.2f}%, Sensitivity: {val_class_sensitivity:.2f}%, Specificity: {val_class_specificity:.2f}%, AUC: {val_class_auc:.2f}%')

            if config["use_wandb"]:
                wandb.log({"Validation Loss": val_epoch_loss.item()})
                wandb.log({"Validation Accuracy": val_accuracy})
                wandb.log({"Validation Sensitivity": val_sensitivity})
                wandb.log({"Validation Classes Metrics": val_classes_metrics})

            if best_accuracy is None or val_accuracy < best_accuracy:
                best_accuracy = val_accuracy
                best_model = copy.deepcopy(model)
            current_results = {
                'epoch': epoch+1,
                'validation_loss': val_epoch_loss.item(),
                'validation_accuracy': val_accuracy,
                'validation_sensitivity': val_sensitivity,
                'validation_classes_metrics': val_classes_metrics,
                'training_loss': tr_epoch_loss.item(),
                'training_accuracy': tr_accuracy,
                'training_sensitivity': tr_sensitivity,
                'training_classes_metrics': tr_classes_metrics
            }
            if SAVE_RESULTS:
                save_results(data_name, current_results)
            if SAVE_MODELS:
                save_model(data_name, model, epoch)
            if epoch == config["epochs"]-1 and SAVE_MODELS:
                save_model(data_name, best_model, epoch=None, is_best=True)
