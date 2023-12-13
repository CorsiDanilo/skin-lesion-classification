import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

from models.ResNet24Pretrained import ResNet24Pretrained
from models.SAM import SAM
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import os
from config import PATH_TO_SAVE_RESULTS, HIDDEN_SIZE, NUM_CLASSES, IMAGE_SIZE
from train_loops.SAM_pretrained import preprocess_images
from utils.utils import approximate_bounding_box_to_square, crop_image_from_box, get_bounding_boxes_from_segmentation, resize_images, resize_segmentations, select_device
import json

device = select_device()
# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
preprocess_params = {
    'adjust_contrast': 1.5,
    'adjust_brightness': 1.2,
    'adjust_saturation': 2,
    'adjust_gamma': 1.5,
    'gaussian_blur': 5}
SAM_IMG_SIZE = 128
KEEP_BACKGROUND = False

def get_model(model_path, epoch, sam_checkpoint_path):
    # Load configuration
    conf_path = PATH_TO_SAVE_RESULTS + f"/{model_path}/configurations.json"
    configurations = None
    if os.path.exists(conf_path):
        print(
            "--Model-- Old configurations found. Using those configurations for the test.")
        with open(conf_path, 'r') as json_file:
            configurations = json.load(json_file)
    else:
        print("--Model-- Old configurations NOT found. Using configurations in the config for test.")

    model = ResNet24Pretrained(
            HIDDEN_SIZE if configurations is None else configurations["hidden_size"], NUM_CLASSES if configurations is None else configurations["num_classes"]).to(device)

    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/melanoma_detection_{epoch}.pt")
    model.load_state_dict(state_dict)

    sam_model = SAM(
                custom_size=True,
                img_size=SAM_IMG_SIZE,
                checkpoint_path=sam_checkpoint_path).to(device)
    
    return model, sam_model

def crop_to_background(images: torch.Tensor,
                           segmentations: torch.Tensor,
                           resize: bool = True):
        bboxes = [get_bounding_boxes_from_segmentation(
            mask)[0] for mask in segmentations]

        cropped_images = []
        for image, bbox in zip(images, bboxes):
            bbox = approximate_bounding_box_to_square(bbox)
            cropped_image = crop_image_from_box(
                image, bbox, size=IMAGE_SIZE if resize else None)
            cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1)
            cropped_images.append(cropped_image)

        cropped_images = torch.stack(cropped_images)

        return cropped_images

def sam_segmentation_pipeline(sam_model, images):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        THRESHOLD = 0.5
        resized_images = resize_images(images, new_size=(
            sam_model.get_img_size(), sam_model.get_img_size())).to(device)

        resized_images = preprocess_images(
            resized_images, params=preprocess_params)

        upscaled_masks = sam_model(resized_images)
        binary_masks = torch.sigmoid(upscaled_masks)
        binary_masks = (binary_masks > THRESHOLD).float()
        binary_masks = resize_segmentations(
            binary_masks, new_size=(450, 450)).to(device)

        images = resize_images(images, new_size=(450, 450)).to(device)
        if not KEEP_BACKGROUND:
            images = binary_masks * images

        cropped_images = crop_to_background(images, binary_masks)
        cropped_images = cropped_images.to(device)
        binary_masks = binary_masks.to(device)
        return cropped_images, binary_masks

def decode_prediction(pred):
    if pred == 0:
        return "Melanocytic nevi (Begign)"
    elif pred == 1:
        return "Benign lesions of the keratosis (Begign)"
    elif pred == 2:
        return "Melanoma (Malignant)"
    elif pred == 3:
        return "Actinic keratoses and intraepithelial carcinoma (Malignant)"
    elif pred == 4:
        return "Basal cell carcinoma (Malignant)"
    elif pred == 5:
        return "Dermatofibroma (Begign)"
    else:
        return "Vascular lesion (Begign)"

def process_image(image_path):
    model, sam_model = get_model("resnet24_2023-12-13_17-55-55", 1, "checkpoints/sam_checkpoint.pt")
    model = model.to(device)
    sam_model = sam_model.to(device)
    model.eval()
    sam_model.model.eval()

    # Open and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Make a prediction using the model
    with torch.no_grad():
        output = model(image)

    # You need to adapt this part based on your model's output
    # For example, if your model outputs probabilities, you can use torch.softmax
    probabilities = torch.softmax(output, dim=1)[0]

    # Get the predicted class index
    predicted_class = torch.argmax(probabilities).item()

    segmented_image, binary_mask = sam_segmentation_pipeline(sam_model, image)
    segmented_image = segmented_image.squeeze(0)
    binary_mask = binary_mask.squeeze(0)

    return image, segmented_image, binary_mask, predicted_class

def open_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if file_path:
        processed_image, segmented_image, binary_mask, predicted_class = process_image(file_path)

        # Display the selected image and the predicted class
        display_image(processed_image, segmented_image, binary_mask, predicted_class)

def display_image(image, segmented_image, binary_mask, predicted_class):
    # Convert the PyTorch tensor to a NumPy array
    image_one = image.cpu().squeeze(0).numpy().transpose((1, 2, 0))
    image_two = segmented_image.cpu().squeeze(0).numpy().transpose((1, 2, 0))

    # Convert the NumPy arrays to PIL Images
    image_one_pil = Image.fromarray((image_one * 255).astype('uint8'))
    image_two_pil = Image.fromarray((image_two * 255).astype('uint8'))

    # Create the PhotoImage objects
    image_one = ImageTk.PhotoImage(image_one_pil)
    image_two = ImageTk.PhotoImage(image_two_pil)

    # Update the first panel
    panel1.config(image=image_one)
    panel1.image = image_one

    # Update the second panel
    panel2.config(image=image_two)
    panel2.image = image_two

    # Update the result label with the predicted class
    result_text.set(f"{decode_prediction(predicted_class)}")

def set_window_size():
    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set the window size (80% of the screen width and height)
    window_width = int(0.3 * screen_width)
    window_height = int(0.3 * screen_height)

    # Set the window geometry
    root.geometry(f"{window_width}x{window_height}+{int((screen_width - window_width) / 2)}+{int((screen_height - window_height) / 2)}")

# Create the main window
root = tk.Tk()
root.title("Image Prediction GUI")

# Set window size based on desktop dimensions
set_window_size()

# Create a button to open an image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

# Create a label to display the result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 12))
result_label.pack(pady=10)

# Create three panels with a small border between them
panel1 = tk.Label(root)
panel1.pack(side="left", padx=5)

panel2 = tk.Label(root)
panel2.pack(side="left", padx=5)

# Start the GUI event loop
root.mainloop()