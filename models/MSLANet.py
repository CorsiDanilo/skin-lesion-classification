import torch
from torch import nn
from models.LANet import LANet
from models.GradCAM import GradCAM
from utils.utils import select_device
from config import DROPOUT_P


class MSLANet(nn.Module):
    def __init__(self):
        super(MSLANet, self).__init__()
        self.device = select_device()
        self.lanet_model = LANet().to(self.device)
        self.fc1 = nn.Linear(263424, 7).to(self.device)
        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # NOTE: x is the cropped image
        five_dropout_preds = []
        # Five dropout predictions
        NUM_OF_DROPOUTS = 1
        for _ in range(NUM_OF_DROPOUTS):
            cropped_img = x.to(self.device)
            lanet_output = self.lanet_model(cropped_img).to(self.device)
            lanet_output = self.flatten(lanet_output)
            lanet_output = self.dropout(lanet_output)
            lanet_output = self.relu(lanet_output)
            lanet_output = self.fc1(lanet_output).to(self.device)
            five_dropout_preds.append(lanet_output)
        average_lanet_output = torch.stack(five_dropout_preds).mean(dim=0)
        # lanet_output = nn.Softmax(dim=-1)(average_lanet_output)
        return average_lanet_output


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
