import torch
import torch.nn as nn
import torchvision.models as models
import timm

class ViT_pretrained(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super (ViT_pretrained, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x