from torchvision.models import ResNet34_Weights
from torchvision.models import Inception_V3_Weights
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random
import os
import wandb
from config import BATCH_SIZE

import dataloaders
import utils
from tqdm import tqdm

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps")
print('Using device: %s' % device)

USE_WANDB = True
# Configurations
INPUT_SIZE = 3
NUM_CLASSES = 7
HIDDEN_SIZE = [32, 64, 128, 256]
N_EPOCHS = 20
LR = 1e-3
LR_DECAY = 0.85
REG = 0.01
SEGMENT = True
CROP_ROI = True
ARCHITECHTURE = "resnet24"
DATASET_LIMIT = None
DROPOUT_P = 0.1
NORMALIZE = True
HISTOGRAM_NORMALIZATION = False

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
            "histogram_normalization": HISTOGRAM_NORMALIZATION

        }
    )


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    set_seed(RANDOM_SEED)


def create_loaders():
    # Load the dataset
    resnet_mean = torch.tensor([0.485, 0.456, 0.406])
    resnet_std = torch.tensor([0.229, 0.224, 0.225])

    train_loader, val_loader, test_loader = dataloaders.create_dataloaders(
        mean=resnet_mean,
        std=resnet_std,
        normalize=NORMALIZE,
        limit=DATASET_LIMIT,
        size=(224, 224))
    return train_loader, val_loader, test_loader


class ResNet24Pretrained(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
        super(ResNet24Pretrained, self).__init__()
        self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(self.model.fc.in_features, 256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(p=DROPOUT_P),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, num_classes, bias=False),
            nn.BatchNorm1d(num_classes),

        )
        self.model.fc = self.classifier

        model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model has {params} trainable params.')

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DenseNetPretrained(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
        super(DenseNetPretrained, self).__init__()
        self.model = models.densenet121(pretrained=True)

        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_P),

            nn.Linear(self.model.classifier.in_features, 256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, num_classes, bias=False),
            nn.BatchNorm1d(num_classes),
        )

        self.model.classifier = self.classifier

        model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model has {params} trainable params.')

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)

        print(f"In features are: {self.model.fc.in_features}")
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_P),

            nn.Linear(self.model.fc.in_features, 1024, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 64, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, num_classes, bias=False),
            nn.BatchNorm1d(num_classes),
        )

        self.model.fc = self.classifier

        model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model has {params} trainable params.')

    def forward(self, x):
        if self.model.training:
            return self.model(x).logits
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def train_eval_loop():
    train_loader, val_loader, test_loader = create_loaders()
    model = get_model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    total_step = len(train_loader)
    train_losses = []
    val_losses = []
    best_accuracy = None
    val_accuracies = []
    # best_model = type(model)(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, norm_layer='BN') # get a new instance
    for epoch in range(N_EPOCHS):
        model.train()
        tr_loss_iter = 0
        training_count = 0
        training_correct_preds = 0
        for tr_i, (tr_images, tr_labels, segmentations) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            if SEGMENT:
                # Apply segmentation
                tr_images = torch.mul(tr_images, segmentations)
                if CROP_ROI:
                    tr_images = utils.crop_roi(tr_images, size=(224, 224))
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
                training_count += len(tr_labels)
                training_correct_preds += (training_preds == tr_labels).sum()

            tr_loss_iter += tr_loss.item()

        current_train_accuracy = 100 * (training_correct_preds/training_count)
        if USE_WANDB:
            wandb.log({"Training Accuracy": current_train_accuracy})
        print('Training -> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%'
              .format(epoch+1, N_EPOCHS, tr_i+1, total_step, tr_loss.item(), current_train_accuracy))

        train_losses.append(tr_loss_iter/(len(train_loader)*BATCH_SIZE))

        # LR *= LR_DECAY
        # update_lr(optimizer, LR)

        model.eval()
        with torch.no_grad():
            validation_correct_preds = 0
            validation_count = 0
            val_loss_iter = 0
            for val_i, (val_images, val_labels, segmentations) in enumerate(val_loader):
                if SEGMENT:
                    # Apply segmentation
                    val_images = torch.mul(val_images, segmentations)
                    if CROP_ROI:
                        val_images = utils.crop_roi(
                            val_images, size=(224, 224))
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                validation_preds = torch.argmax(val_outputs, -1).detach()
                validation_count += len(val_labels)
                validation_correct_preds += (validation_preds ==
                                             val_labels).sum()
                val_loss = loss_function(val_outputs, val_labels)
                if USE_WANDB:
                    wandb.log({"Validation Loss": val_loss.item()})
                val_loss_iter += val_loss.item()

            val_losses.append(val_loss_iter/(len(val_loader)*BATCH_SIZE))

            val_accuracy = 100 * (validation_correct_preds / validation_count)
            val_accuracies.append(val_accuracy)
            if USE_WANDB:
                wandb.log({"Validation Accuracy": val_accuracy})

            print(
                'Validation -> Validation accuracy for epoch {} is: {:.4f}%'.format(epoch+1, val_accuracy))
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
        model = InceptionV3(NUM_CLASSES).to(device)
    else:
        raise ValueError(f"Unknown architechture {ARCHITECHTURE}")

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
    train_eval_loop()


if __name__ == "__main__":
    main()
