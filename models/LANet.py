from typing import List
import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet34_Weights
from models.GradCAM import GradCAM
from config import DROPOUT_P, HIDDEN_SIZE, NUM_CLASSES, INPUT_SIZE
from utils.utils import select_device

"""
class LANet(nn.Module):
    def __init__(self):
        super(LANet, self).__init__()
        self.cnn_block = models.resnet50(pretrained=True)
        self.cnn_block_features = nn.Sequential(*list(self.cnn_block.children())[:-2])
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_channels = self.cnn_block.layer4[-1].conv3.out_channels
        self.conv1x1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

    def mixed_sigmoid(self, x):
        return torch.sigmoid(x) * x
    
    def forward(self, x):
        cnn_block_feature_map = self.cnn_block_features(x)
        print(cnn_block_feature_map.shape)
        downsamples_cnn_block_feature_map = self.adaptive_avg_pool(cnn_block_feature_map)
        print(downsamples_cnn_block_feature_map.shape)
        final_layer = self.conv1x1(cnn_block_feature_map)
        final_layer = self.mixed_sigmoid(final_layer)
        out = torch.cat([downsamples_cnn_block_feature_map * final_layer], dim=1)
        return out
"""


class LANet(nn.Module):
    def __init__(self, hidden_layers, num_classes, dropout=DROPOUT_P):
        super(LANet, self).__init__()
        self.device = select_device()
        self.model = models.resnet50(pretrained=True).to(self.device)

        self.model_features = nn.Sequential(*list(self.model.children())[:-2])
        self.in_channels = self.model.layer4[-1].conv3.out_channels
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(7)
        self.conv1x1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        '''
        self.layers = []
        if len(hidden_layers) == 0:
            self.layers.append(self.dropout)
            self.layers.append(nn.Linear(self.in_channels * self.num_classes * self.num_classes, self.num_classes, bias=False))
        else:
            self.layers.append(self.dropout)
            for i in range(len(hidden_layers)):
                if i == 0:
                    self.layers.append(nn.Linear(self.in_channels * self.num_classes * self.num_classes, hidden_layers[i], bias=False))
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
                    self.layers.append(self.relu)
                    self.layers.append(self.dropout)
                else:
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=False))
                    self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
                    self.layers.append(self.relu)
                    self.layers.append(self.dropout)
            self.layers.append(nn.Linear(hidden_layers[-1], self.num_classes, bias=False))
            self.layers.append(nn.BatchNorm1d(self.num_classes)) 
        self.classifier = nn.Sequential(*self.layers)
        '''
        self.model.fc = nn.Identity()  # Remove the final fully connected layer from the ResNet model

        # Freeze all the layers except the fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.cnn_blocks = self.extract_cnn_blocks()

    def conv1d(self, out_channels): return nn.Conv2d(
        in_channels=2048, out_channels=out_channels, kernel_size=1).to(self.device)

    def mixed_sigmoid(self, x):
        return torch.sigmoid(x) * x

    def extract_cnn_blocks(self) -> List[nn.Module]:
        """"
        Extracts the list of CNN blocks from the ResNet model
        """
        blocks = [
            nn.Sequential(
                self.model.conv1,
                self.model.bn1,
                self.model.relu,
                self.model.maxpool,
                self.model.layer1
            )
        ]
        for index, child in enumerate(list(self.model.children())[5:]):
            if isinstance(child, nn.Sequential):
                for layer in child.children():
                    blocks.append(layer)
        return blocks

    def forward(self, x):
        resnet_feature_map = self.model_features(x)
        cat_output = resnet_feature_map.detach()
        C_x, H_x, W_x = x.shape[1:]
        for index, cnn_block in enumerate(self.cnn_blocks):
            if index == 0:
                curr_activation_map = x
            curr_activation_map = cnn_block(curr_activation_map)
            C_o, H_o, W_o = curr_activation_map.shape[1:]
            # print(f"C_o: {C_o}, H_o: {H_o}, W_o: {W_o}")

            avg_pool_feature_map = self.adaptive_avg_pool(
                curr_activation_map)

            conv1d_feature_map = self.conv1d(
                avg_pool_feature_map.shape[1])(resnet_feature_map)
            conv_1d_feature_map = self.mixed_sigmoid(conv1d_feature_map)
            output = avg_pool_feature_map * conv_1d_feature_map
            # print(f"Output shape for CNN block {index} is {output.shape}")
            cat_output = torch.cat((cat_output, output), dim=1)
        # print(f"Cat output shape is {cat_output.shape}")
        return cat_output


if __name__ == "__main__":
    cam_instance = GradCAM()
    lanet_model = LANet(INPUT_SIZE, NUM_CLASSES)
    image_path = "/Users/dov/Library/Mobile Documents/com~apple~CloudDocs/dovsync/Documenti Universita/Advanced Machine Learning/AML Project.nosync/melanoma-detection/data/HAM10000_images_test/ISIC_0034524.jpg"
    thresholds = [120]
    for t in thresholds:
        _, cropped_img, _ = cam_instance.generate_cam(image_path, t)
        cropped_img = cropped_img.unsqueeze(0)
        output = lanet_model(cropped_img)
        print(output.shape)
