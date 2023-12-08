from torch import nn
from torchvision import models
from torchvision.models import Inception_V3_Weights
from config import DROPOUT_P
import numpy as np


class InceptionV3Pretrained(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3Pretrained, self).__init__()
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
