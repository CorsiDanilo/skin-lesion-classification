import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, roc_auc_score
from tqdm import tqdm
import wandb
from models.MSLANet import MSLANet
from train_loops.CNN_pretrained import get_normalization_statistics
from utils.utils import save_results, set_seed, select_device
from utils.dataloader_utils import get_dataloder_from_strategy
from config import  DYNAMIC_SEGMENTATION_STRATEGY, KEEP_BACKGROUND, LOAD_SYNTHETIC, NUM_DROPOUT_LAYERS, OVERSAMPLE_TRAIN, SAVE_RESULTS, DATASET_LIMIT, NORMALIZE, RANDOM_SEED, PATH_TO_SAVE_RESULTS, NUM_CLASSES, DROPOUT_P, BATCH_SIZE, SEGMENTATION_STRATEGY, USE_WANDB

def test(test_model, test_loader, device, data_name):
    criterion = nn.CrossEntropyLoss()
    test_model.eval()
    test_loss_iter = 0

    with torch.no_grad():
        epoch_test_preds = torch.tensor([]).to(device)
        epoch_test_labels = torch.tensor([]).to(device)
        epoch_test_scores = torch.tensor([]).to(device)
        for _, test_batch in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            if len(test_batch) == 3:
                test_images, test_labels, _ = test_batch
            else:
                test_images, test_labels = test_batch

            #test_image_ori = test_image_ori.to(device)
            #test_image_low = test_image_low.to(device)
            #test_image_high = test_image_high.to(device)
            test_labels = test_labels.to(device)

            #test_output_ori = test_model(test_image_ori)  # Prediction
            #test_output_low = test_model(test_image_low)  # Prediction
            #test_output_high = test_model(test_image_high)  # Prediction

            test_outputs = test_model(test_images)

            #test_outputs = (test_output_ori + test_output_low + test_output_high) / 3

            # Multiclassification loss considering all classes
            test_epoch_loss = criterion(test_outputs, test_labels)
            test_loss_iter += test_epoch_loss.item()

            test_preds = torch.argmax(test_outputs, -1).detach()
            epoch_test_preds = torch.cat((epoch_test_preds, test_preds), 0)
            epoch_test_labels = torch.cat((epoch_test_labels, test_labels), 0)

            test_outputs = test_outputs.t()
            epoch_test_scores = torch.cat((epoch_test_scores, test_outputs), 1)

        test_loss = test_loss_iter / (len(test_loader) * BATCH_SIZE)
        test_accuracy = accuracy_score(
            epoch_test_labels.cpu().numpy(), epoch_test_preds.cpu().numpy()) * 100
        test_sensitivity = recall_score(
            epoch_test_labels.cpu().numpy(), epoch_test_preds.cpu().numpy(), average='macro', zero_division=0) * 100

        print('Test -> Loss: {:.4f}, Accuracy: {:.4f}%, Sensitivity (Recall): {:.4f}%'.format(
            test_loss, test_accuracy, test_sensitivity))
        
        test_classes_metrics = {}
        for class_label in range(NUM_CLASSES):
            test_class_labels_binary = torch.zeros_like(
                epoch_test_labels, dtype=torch.long).to(device)
            test_class_labels_binary[(epoch_test_labels == class_label)] = 1

            test_class_preds_binary = torch.zeros_like(
                epoch_test_preds, dtype=torch.long).to(device)
            test_class_preds_binary[(epoch_test_preds == class_label)] = 1

            test_scores_class = epoch_test_scores[class_label]

            if len(test_class_preds_binary) > 0:
                test_class_accuracy = accuracy_score(
                    test_class_labels_binary.cpu().numpy(), test_class_preds_binary.cpu().numpy()) * 100
                test_class_sensitivity = recall_score(
                    test_class_labels_binary.cpu().numpy(), test_class_preds_binary.cpu().numpy(), average='binary', pos_label=1, zero_division=0) * 100
                test_class_conf_matrix = confusion_matrix(
                    test_class_labels_binary.cpu().numpy(), test_class_preds_binary.cpu().numpy())
                if test_class_conf_matrix.shape == (2, 2) and (test_class_conf_matrix[0, 0] + test_class_conf_matrix[0, 1]) > 0:
                    test_class_specificity = test_class_conf_matrix[0, 0] / (
                        test_class_conf_matrix[0, 0] + test_class_conf_matrix[0, 1]) * 100
                else:
                    test_class_specificity = 0
                if len(set(test_class_labels_binary.cpu().numpy())) > 1:
                    test_class_auc = roc_auc_score(
                        test_class_labels_binary.cpu().numpy(), test_scores_class.cpu().numpy()) * 100
                else:
                    test_class_auc = 0

                test_class_metrics = {"accuracy": test_class_accuracy, "sensitivity": test_class_sensitivity,
                                        "specificity": test_class_specificity, "auc": test_class_auc}
                test_classes_metrics[class_label] = test_class_metrics

                print(f'Class {class_label} - Accuracy: {test_class_accuracy:.2f}%, Sensitivity (Recall): {test_class_sensitivity:.2f}%, Specificity: {test_class_specificity:.2f}%, AUC: {test_class_auc:.2f}%')

        test_results = {
            'test_accuracy': test_accuracy,
            'test_recall': test_sensitivity,
            'test_loss': test_loss
        }
        if SAVE_RESULTS:
            save_results(data_name, test_results, test=True)
            if USE_WANDB:
                wandb.log({"Testing Loss": test_loss.item()})
                wandb.log({"Testing Accuracy": test_accuracy})
                wandb.log({"Testing Sensitivity": test_sensitivity})
                wandb.log({"Testing Classes Metrics": test_classes_metrics})

def load_test_model(model, model_path, epoch, device):
    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/melanoma_detection_{epoch}.pt", map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main(model_path, epoch):
    set_seed(RANDOM_SEED)
    device = select_device()
    model = MSLANet(num_classes=NUM_CLASSES, dropout_num=NUM_DROPOUT_LAYERS, dropout_p=DROPOUT_P).to(device)
    model = load_test_model(model, model_path, epoch, device)

    dataloader = get_dataloder_from_strategy(
        strategy=SEGMENTATION_STRATEGY,
        dynamic_segmentation_strategy=DYNAMIC_SEGMENTATION_STRATEGY,
        limit=DATASET_LIMIT,
        dynamic_load=False,
        oversample_train=OVERSAMPLE_TRAIN,
        normalize=NORMALIZE,
        normalization_statistics=get_normalization_statistics(),
        batch_size=BATCH_SIZE,
        keep_background=KEEP_BACKGROUND,
        load_synthetic=LOAD_SYNTHETIC
    )
    test_dataloader = dataloader.get_test_dataloader()
    test(model, test_dataloader, device, model_path)

if __name__ == "__main__":
    # Name of the sub-folder into "results" folder in which to find the model to test (e.g. "resnet34_2023-12-10_12-29-49")
    model_path = "MSLANet_2024-01-21_13-42-54"
    # Specify the epoch number (e.g. 2) or "best" to get best model
    epoch = "1"

    main(model_path, epoch)
