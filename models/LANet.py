import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet34_Weights
from models.GradCAM import GradCAM
from config import DROPOUT_P, HIDDEN_SIZE, NUM_CLASSES, INPUT_SIZE

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
        self.model = models.resnet50(pretrained=True)
        self.model_features = nn.Sequential(*list(self.model.children())[:-2])
        self.in_channels = self.model.layer4[-1].conv3.out_channels
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(7)
        self.conv1x1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)
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
        self.model.fc = nn.Identity() # Remove the final fully connected layer from the ResNet model

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )


    def mixed_sigmoid(self, x):
        return torch.sigmoid(x) * x
    
    def forward(self, x):
        resnet_feature_map = self.model_features(x)  # Compute the feature map using the ResNet50 pretrained model
        conv1_output = self.conv_layers[:3](resnet_feature_map) # Apply the first ConvBlock
        print(conv1_output.shape)
        avg_pool_feature_map = self.adaptive_avg_pool(conv1_output) # Apply 
        last_layer = self.conv_layers[3:](conv1_output)
        print(avg_pool_feature_map.shape)
        attention_mask = self.mixed_sigmoid(self.conv1x1(last_layer))
        print(attention_mask.shape)
        output = last_layer + (avg_pool_feature_map * attention_mask)
        print(output.shape)
        return last_layer

if __name__ == "__main__":
    cam_instance = GradCAM()
    lanet_model = LANet(INPUT_SIZE, NUM_CLASSES)
    image_path = 'C:/Users/aless/OneDrive/Desktop/Mole_images/20231213_192133.jpg'
    thresholds = [120]
    for t in thresholds:
        _, cropped_img, _ = cam_instance.generate_cam(image_path, t)
        cropped_img = cropped_img.unsqueeze(0)
        output = lanet_model(cropped_img)
        print(output.shape)
    

