import torch
from torch import nn
from torchvision import models
from config import INPUT_SIZE, NUM_CLASSES
from models.LANet import LANet
from models.GradCAM import GradCAM
from utils.utils import select_device


class MSLANet(nn.Module):
    def __init__(self):
        super(MSLANet, self).__init__()
        self.device = select_device()
        self.lanet_model = LANet(INPUT_SIZE, NUM_CLASSES).to(self.device)
        self.thresholds = [120]
        self.image_path = "/Users/dov/Library/Mobile Documents/com~apple~CloudDocs/dovsync/Documenti Universita/Advanced Machine Learning/AML Project.nosync/melanoma-detection/data/HAM10000_images_test/ISIC_0034524.jpg"

    def forward(self, x):
        # NOTE: x is the cropped image
        thresh_preds = []
        for thresh in self.thresholds:
            five_dropout_preds = []
            # Five dropout predictions
            for _ in range(5):
                # _, cropped_img, _ = cam_instance.generate_cam(
                #     self.image_path, thresh)
                # # Add batch dimension since we are using just an image
                # cropped_img = cropped_img.unsqueeze(0)
                cropped_img = x.to(self.device)
                batch_size = cropped_img.shape[0]
                lanet_output = self.lanet_model(cropped_img).to(self.device)
                lanet_output = nn.Dropout(p=0.5)(lanet_output).to(self.device)
                lanet_output = lanet_output.view(
                    batch_size, -1).to(self.device)
                lanet_output = nn.Linear(
                    lanet_output.shape[-1], 7).to(self.device)(lanet_output).to(self.device)
                five_dropout_preds.append(lanet_output)
            average_lanet_output = torch.stack(five_dropout_preds).mean(dim=0)
            lanet_output = nn.Softmax(dim=1)(average_lanet_output)
            thresh_preds.append(lanet_output)
        return torch.stack(thresh_preds).mean(dim=0)


if __name__ == "__main__":
    cam_instance = GradCAM()
    lanet_model = MSLANet()
    output = lanet_model()
    print(output.shape)
    print(output.view(-1))
