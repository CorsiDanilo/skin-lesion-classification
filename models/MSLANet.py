import torch
from torch import nn
from torchvision import models
from models.GradCAM import GradCAM

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

if __name__ == "__main__":
    cam_instance = GradCAM()
    lanet_model = MSLANet()
    image_path = 'C:/Users/aless/OneDrive/Desktop/Mole_images/20231213_192133.jpg'
    thresholds = [120]
    for t in thresholds:
        _, cropped_img, _ = cam_instance.generate_cam(image_path, t)
        output = lanet_model(cropped_img)
        print(output.shape)
