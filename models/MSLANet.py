import torch
from torch import nn
from models.LANet import LANet
from models.GradCAM import GradCAM
from utils.utils import select_device
from config import DROPOUT_P, NUM_CLASSES


class MSLANet(nn.Module):
    def __init__(self, num_classes, dropout_num=1, dropout_p=0.5):
        super(MSLANet, self).__init__()
        self.device = select_device()
        self.lanet_model = LANet().to(self.device)
        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout_num = dropout_num
        self.dropout_p = dropout_p
        self.fc_layers = [nn.Linear(1, num_classes).to(self.device) for _ in range(dropout_num)]

    def forward(self, x):
        cropped_img = x.to(self.device)
        lanet_output = self.lanet_model(cropped_img).to(self.device)
        # NOTE: x is the cropped image
        five_dropout_preds = []
        # Five dropout predictions
        for i in range(self.dropout_num):
            if self.fc_layers[i].in_features == 1:
                lanet_shape = lanet_output.shape
                input_size = lanet_shape[1] * lanet_shape[2] * lanet_shape[3]
                self.fc_layers[i] = nn.Linear(input_size, NUM_CLASSES).to(self.device)
            
            dropout_layer = self.flatten(lanet_output)
            dropout_layer = self.dropout(dropout_layer)
            dropout_layer = self.relu(dropout_layer)
            dropout_layer = self.fc_layers[i](dropout_layer).to(self.device)
            five_dropout_preds.append(dropout_layer)
        
        average_dropout_layers = torch.stack(five_dropout_preds).mean(dim=0)
        return average_dropout_layers

if __name__ == "__main__":
    cam_instance = GradCAM()
    lanet_model = MSLANet()
    thresh = 120
    image_path = "/Users/dov/Library/Mobile Documents/com~apple~CloudDocs/dovsync/Documenti Universita/Advanced Machine Learning/AML Project.nosync/melanoma-detection/data/HAM10000_images_test/ISIC_0034524.jpg"
    _, cropped_img, _ = cam_instance.generate_cam(image_path, thresh)
    # Add batch dimension since we are using just an image
    cropped_img = cropped_img.unsqueeze(0)
    output = lanet_model(cropped_img)
    print(output.shape)
    print(output.view(-1))
