import numpy as np
import torch
from torch import nn
import models
from torchvision import models
from torchvision.models import ResNet50_Weights
from utils.utils import select_device
from config import DROPOUT_P, HIDDEN_SIZE


class MSLANet_v2(nn.Module):
    def __init__(self, num_classes, hidden_layers=HIDDEN_SIZE, dropout_num=1, dropout_p=DROPOUT_P):
        super(MSLANet_v2, self).__init__()
        self.device = select_device()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout_num = dropout_num
        self.dropout_p = dropout_p
        self.hidden_layers = hidden_layers
        self.dropout = nn.Dropout(p=dropout_p)

        # Flag not to train the ResNet50 parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.layers = []
        if len(hidden_layers) == 0:
            self.layers.append(self.dropout)
            self.layers.append(
                nn.Linear(self.model.fc.in_features, num_classes, bias=False))
        else:
            self.layers.append(self.dropout)
            for i in range(len(hidden_layers)):
                if i == 0:
                    self.layers.append(
                        nn.Linear(self.model.fc.in_features, hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
                else:
                    self.layers.append(
                        nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=False))
                    self.layers.append(self.relu)
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
            self.layers.append(
                nn.Linear(hidden_layers[-1], num_classes, bias=False))
            self.layers.append(nn.BatchNorm1d(num_classes))

        self.classifier = nn.Sequential(*self.layers)
        self.model.fc = self.classifier

    def forward(self, x):
        dropout_preds = [] # Dropout predictions
        for _ in range(self.dropout_num): 
            resnet50_output = self.model(x).to(self.device)
            dropout_preds.append(resnet50_output)
        
        average_dropout_layers = torch.stack(dropout_preds).mean(dim=0)
        return average_dropout_layers