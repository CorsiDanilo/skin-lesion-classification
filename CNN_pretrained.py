import torch
import torch.nn as nn
import numpy as np
import random
import os
import wandb
from config import BALANCE_UNDERSAMPLING, BATCH_SIZE, INPUT_SIZE, NUM_CLASSES, HIDDEN_SIZE, N_EPOCHS, LR, REG, SEGMENT, CROP_ROI, ARCHITECHTURE, DATASET_LIMIT, DROPOUT_P, NORMALIZE, SEGMENTATION_BOUNDING_BOX, USE_WANDB
from models.ResNet24Pretrained import ResNet24Pretrained
from models.DenseNetPretrained import DenseNetPretrained
from models.InceptionV3Pretrained import InceptionV3Pretrained

from dataloaders.segmentation_dataloaders import create_dataloaders
from tqdm import tqdm

from sklearn.metrics import recall_score, accuracy_score

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps")
print('Using device: %s' % device)

RESUME = False
FROM_EPOCH = 0

if CROP_ROI:
    assert SEGMENT, f"Crop roi needs segment to be True"

if USE_WANDB:
    # Start a new run
    wandb.init(
        project="melanoma",

        # track hyperparameters and run metadata
        config={
            "learning_rate": LR,
            "architecture": ARCHITECHTURE,
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
            "dynamic_segmentation": False,
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
        size=(224, 224),
        dynamic_load=True)
    return train_loader, val_loader, test_loader


def save_model(model, model_name, epoch):
    torch.save(model.state_dict(), f"{model_name}_{epoch}.pt")


def train_eval_loop():
    train_loader, val_loader, _ = create_loaders()
    model = get_model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    total_step = len(train_loader)
    train_losses = []
    val_losses = []
    best_accuracy = -torch.inf
    val_accuracies = []
    # best_model = type(model)(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, norm_layer='BN') # get a new instance
    for epoch in range(FROM_EPOCH if RESUME else 0, N_EPOCHS):
        model.train()
        tr_loss_iter = 0
        epoch_tr_preds = torch.tensor([]).to(device)
        epoch_tr_labels = torch.tensor([]).to(device)
        for tr_i, (tr_images, tr_labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            tr_images = tr_images.to(torch.float32)
            tr_images = tr_images.to(device)
            tr_labels = tr_labels.to(device)

            tr_outputs = model(tr_images)
            tr_loss = loss_function(tr_outputs, tr_labels)
            if USE_WANDB:
                wandb.log({"Training Loss": tr_loss.item()})

            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

            with torch.no_grad():
                training_preds = torch.argmax(tr_outputs, -1).detach()
                epoch_tr_preds = torch.cat(
                    (epoch_tr_preds, training_preds), 0)
                epoch_tr_labels = torch.cat(
                    (epoch_tr_labels, tr_labels), 0)

            tr_loss_iter += tr_loss.item()
        tr_accuracy = accuracy_score(
            epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy()) * 100
        tr_recall = recall_score(
            epoch_tr_labels.cpu().numpy(), epoch_tr_preds.cpu().numpy(), average='macro') * 100
        tr_f1 = 2 * (tr_accuracy * tr_recall) / (tr_accuracy + tr_recall)
        if USE_WANDB:
            wandb.log({"Training Accuracy": tr_accuracy})
            wandb.log({"Training Recall": tr_recall})
            wandb.log({"Training F1": tr_f1})

        print('Training -> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%'
              .format(epoch+1, N_EPOCHS, tr_i+1, total_step, tr_loss.item(), tr_accuracy, tr_recall))

        train_losses.append(tr_loss_iter/(len(train_loader)*BATCH_SIZE))

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

            val_losses.append(val_loss_iter/(len(val_loader)*BATCH_SIZE))

            val_accuracy = accuracy_score(
                epoch_val_labels.cpu().numpy(), epoch_val_preds.cpu().numpy()) * 100
            val_recall = recall_score(
                epoch_val_labels.cpu().numpy(), epoch_val_preds.cpu().numpy(), average='macro') * 100
            val_f1 = 2 * (val_accuracy * val_recall) / \
                (val_accuracy + val_recall)

            # if val_accuracy > best_accuracy:
            #     best_accuracy = val_accuracy
            #     save_model(model, ARCHITECHTURE, epoch)

            val_accuracies.append(val_accuracy)
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
    if ARCHITECHTURE == "resnet24":
        model = ResNet24Pretrained(
            INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, norm_layer='BN').to(device)
    elif ARCHITECHTURE == "densenet121":
        model = DenseNetPretrained(
            INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, norm_layer='BN').to(device)
    elif ARCHITECHTURE == "inception_v3":
        model = InceptionV3Pretrained(NUM_CLASSES).to(device)
    else:
        raise ValueError(f"Unknown architechture {ARCHITECHTURE}")

    if RESUME:
        model.load_state_dict(torch.load(
            f"{ARCHITECHTURE}_{FROM_EPOCH-1}.pt"))

    for p in model.parameters():
        p.requires_grad = False

    # LAYERS_TO_FINE_TUNE = 20
    # parameters = list(model.parameters())
    # for p in parameters[-LAYERS_TO_FINE_TUNE:]:
    #     p.requires_grad=True

    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


def main():
    set_seed(RANDOM_SEED)
    train_eval_loop()


if __name__ == "__main__":
    main()
