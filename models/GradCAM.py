import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

class GradCAM(nn.Module):
    def __init__(self):
        super(GradCAM, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.target_layer = "layer4"
        self.model.eval()
        self.feature_maps = None
        self.grads = None

        # Register hooks for feature extraction and gradient computation
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.grads = grad_output[0]

        target_layer = self.model._modules[self.target_layer]
        self.forward_hook = target_layer.register_forward_hook(forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_cam(self, image_path, threshold):
        if threshold < 0 or threshold > 255:
            return ValueError("Threshold must be a value between 0 and 255.")

        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_img = transform(img)
        input_img = input_img.unsqueeze(0)

        # Get model prediction and compute gradients
        self.model.zero_grad()
        output = self.model(input_img)
        output[0, output.argmax()].backward()

        # Compute CAM
        weights = torch.mean(self.grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Upsample CAM to the original image size
        cam = torch.nn.functional.interpolate(cam, size=(img.size[1], img.size[0]), mode='bilinear', align_corners=False)

        # Normalize CAM values between 0 and 1
        cam -= cam.min()
        cam /= cam.max()

        cam_np = (cam.detach().numpy()[0, 0, :, :])*255 # Convert CAM to a numpy array
        binary_mask = (cam_np > threshold).astype(np.uint8)*255 # Apply thresholding
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the binary mask
        bounding_rect = cv2.boundingRect(np.vstack(contours)) # Find the minimum bounding rectangle around all contours
        
        # Apply rectangle to the original image to show the cropping area
        img_with_rect = np.array(img)
        cv2.rectangle(img_with_rect, (bounding_rect[0], bounding_rect[1]),
                      (bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]),
                      (0, 255, 0), 2)
        
        # Crop the image with the found bounding box
        cropped_img = img.crop((bounding_rect[0], bounding_rect[1],
                               bounding_rect[0] + bounding_rect[2],
                               bounding_rect[1] + bounding_rect[3]))
        cropped_img = transform(cropped_img)

        # Apply color map to the orginal image to show the focus areas
        img_np = np.array(img)
        heatmap = cv2.applyColorMap(np.uint8(cam_np), cv2.COLORMAP_JET) # Apply colormap to the CAM
        cam_img = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0) # Superimpose the heatmap on the original image

        return img_with_rect, cropped_img, cam_img
    
if __name__ == "__main__":
    cam_instance = GradCAM()
    image_path = 'C:/Users/aless/OneDrive/Desktop/Mole_images/20231213_192133.jpg'
    thresholds = [120, 200]
    for t in thresholds:
        _, cropped_img, _ = cam_instance.generate_cam(image_path, t)
        cropped_img.show()