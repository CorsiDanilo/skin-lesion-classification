import torch
from torch import nn
from torchvision import models

class MSLANet(nn.Module):
    def __init__(self):
        super(MSLANet, self).__init__()

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

# Example usage
lanet_model = MSLANet()
input_data = torch.randn(1, 3, 224, 224)  # Replace with your input dimensions
output = lanet_model(input_data)
print(output.shape)

