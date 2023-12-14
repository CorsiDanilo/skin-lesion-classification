import torch
from torch import nn
from torchvision import models

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
    
    # TO DO: There is something wrong with dimensions
    def forward(self, x):
        cnn_block_feature_map = self.cnn_block_features(x)
        print(cnn_block_feature_map.shape)
        downsamples_cnn_block_feature_map = self.adaptive_avg_pool(cnn_block_feature_map)
        print(downsamples_cnn_block_feature_map.shape)
        final_layer = self.conv1x1(cnn_block_feature_map)
        final_layer = self.mixed_sigmoid(final_layer)
        out = torch.cat([downsamples_cnn_block_feature_map * final_layer], dim=1)
        return out

# Example usage
lanet_model = LANet()
input_data = torch.randn(1, 3, 224, 224)  # Replace with your input dimensions
output = lanet_model(input_data)
print(output.shape)

