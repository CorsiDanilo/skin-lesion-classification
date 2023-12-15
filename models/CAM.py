import cv2
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch import nn
import numpy as np
from PIL import Image

class CAM(nn.Module):
    def __init__(self, image_path, threshold=0.5):
        super(CAM, self).__init__()
        self.image_path = image_path
        self.threshold = threshold
        self.original_image = cv2.imread(image_path)
        self.model = models.resnet50()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        self.model.eval()

    def preprocess_image(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)  # Convert NumPy array to PIL Image
        img_tensor = transform(img_pil)
        img_tensor = Variable(img_tensor.unsqueeze(0), requires_grad=True)
        return img_tensor

    def generate_cam(self):
        img = self.preprocess_image(self.image_path)

        # Forward pass
        features = self.model(img)
        output = features.mean(dim=(2, 3))
        class_idx = torch.argmax(output) # Get the class index with the highest score
        feature_map = features[0, class_idx].data.numpy() # Get the feature map from the last convolutional layer
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) # Normalize the feature map
        
        feature_map_original_size = cv2.resize(feature_map, (self.original_image.shape[1], self.original_image.shape[0])) # Resize the feature map to match the original image size
        heatmap = cv2.applyColorMap(np.uint8(255 * feature_map_original_size), cv2.COLORMAP_JET) # Create a heatmap
        cam_image = heatmap * 0.7 + self.original_image * 0.3 # Superimpose the heatmap on the original image
        cam_image = np.uint8(cam_image)
        cv2.imwrite('cam_image.jpg', cam_image)

        print(feature_map_original_size.shape)


        # Binarize the heatmap based on the threshold
        binary_heatmap = (feature_map_original_size > self.threshold).astype(np.uint8)
        cv2.imwrite('cam_image3.jpg', binary_heatmap)

        # Find connected components
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_heatmap)

        # Draw rectangles on the original image
        for i in range(1, stats.shape[0]):
            x, y, w, h = stats[i][:4]
            cv2.rectangle(self.original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite('cam_image_with_rectangles.jpg', self.original_image)


# Example usage:
image_path = 'C:/Users/aless/Desktop/ISIC_0024306.jpg'
cam_instance = CAM(image_path, threshold=130)
cam_instance.generate_cam()