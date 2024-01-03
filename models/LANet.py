from typing import List
import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from models.GradCAM import GradCAM
from config import DROPOUT_P, HIDDEN_SIZE, NUM_CLASSES, INPUT_SIZE
from utils.utils import select_device


class LANet(nn.Module):
    def __init__(self, hidden_layers, num_classes, dropout=DROPOUT_P):
        super(LANet, self).__init__()
        self.device = select_device()
        self.model = models.resnet50(
            weights=ResNet50_Weights.DEFAULT).to(self.device)

        self.model_features = nn.Sequential(*list(self.model.children())[:-2])
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(NUM_CLASSES)
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        out_features = [256,
                        512,
                        512,
                        512,
                        512,
                        1024,
                        1024,
                        1024,
                        1024,
                        1024,
                        1024,
                        2048,
                        2048,
                        2048]

        self.conv1x1_layers = nn.ModuleList(
            [nn.Conv2d(in_channels=2048, out_channels=i, kernel_size=1).to(self.device) for i in out_features])

        self.cnn_blocks = self.extract_cnn_blocks()

        # Remove the final fully connected layer from the ResNet model
        self.model.fc = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = False
        # for param in self.model.fc.parameters():
        #     param.requires_grad = True

    def mixed_sigmoid(self, Y):
        M = self.sigmoid(Y) * Y
        return M

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
        cat_output = resnet_feature_map
        C_x, H_x, W_x = x.shape[1:]
        for index, cnn_block in enumerate(self.cnn_blocks):
            if index == 0:
                curr_activation_map = x
            curr_activation_map = cnn_block(curr_activation_map)
            # NOTE: just for speed, skip some cnn blocks
            if index > 5:
                continue
            C_o, H_o, W_o = curr_activation_map.shape[1:]
            # print(f"C_o: {C_o}, H_o: {H_o}, W_o: {W_o}")

            avg_pool_feature_map = self.adaptive_avg_pool(
                curr_activation_map)
            conv1d_feature_map = self.conv1x1_layers[index](
                resnet_feature_map)
            conv_1d_feature_map = self.mixed_sigmoid(conv1d_feature_map)
            output = avg_pool_feature_map * conv_1d_feature_map
            cat_output = torch.cat((cat_output, output), dim=1)
        # print(f"Cat output shape is {cat_output.shape}")
        return cat_output


if __name__ == "__main__":
    cam_instance = GradCAM()
    lanet_model = LANet(HIDDEN_SIZE, NUM_CLASSES)
    image_path = "/Users/dov/Library/Mobile Documents/com~apple~CloudDocs/dovsync/Documenti Universita/Advanced Machine Learning/AML Project.nosync/melanoma-detection/data/HAM10000_images_test/ISIC_0034524.jpg"
    thresholds = [120]
    for t in thresholds:
        _, cropped_img, _ = cam_instance.generate_cam(image_path, t)
        cropped_img = cropped_img.unsqueeze(0)
        output = lanet_model(cropped_img)
        print(output.shape)
