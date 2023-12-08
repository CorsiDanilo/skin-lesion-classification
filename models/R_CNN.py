from torch import nn
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from config import DROPOUT_P
import numpy as np


class BoxOnlyPredictor(FastRCNNPredictor):
    def __init__(self, in_channels):
        # num_classes is 2 because we only care about background and object
        super().__init__(in_channels, num_classes=4)

    def forward(self, x):
        box_regression = super().forward(x)[0]
        print(f"Box regression shape: {box_regression.shape}")
        class_logits = torch.zeros(
            (box_regression.shape[0], 2), device=box_regression.device)
        return class_logits, box_regression


class R_CNN(nn.Module):
    def __init__(self):
        super(R_CNN, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = BoxOnlyPredictor(in_features)

    def forward(self, inputs, targets=None):
        # output = self.model(inputs, targets)
        self.model.eval()
        output = self.model(inputs)
        print(f"Model output is {output}")
        return output

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
