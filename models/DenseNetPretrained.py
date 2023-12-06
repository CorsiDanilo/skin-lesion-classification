from torch import nn
from torchvision import models
from torchvision.models import DenseNet121_Weights
from config import DROPOUT_P
import numpy as np


class DenseNetPretrained(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
        super(DenseNetPretrained, self).__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_P),

            nn.Linear(self.model.classifier.in_features, 1024, bias=False),
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
