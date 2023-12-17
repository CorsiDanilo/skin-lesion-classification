import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import os
from config import PATH_TO_SAVE_RESULTS, HIDDEN_SIZE, NUM_CLASSES, IMAGE_SIZE, DROPOUT_P, INPUT_SIZE, EMB_SIZE, PATCH_SIZE, N_HEADS, N_LAYERS, HIDDEN_SIZE
from constants import DEFAULT_STATISTICS, IMAGENET_STATISTICS
from train_loops.SAM_pretrained import preprocess_images
from utils.utils import approximate_bounding_box_to_square, crop_image_from_box, get_bounding_boxes_from_segmentation, resize_images, resize_segmentations, select_device
from models.SAM import SAM
from models.ResNet24Pretrained import ResNet24Pretrained
from models.DenseNetPretrained import DenseNetPretrained
from models.InceptionV3Pretrained import InceptionV3Pretrained
from models.ViTStandard import ViT_standard
from models.ViTPretrained import ViT_pretrained
from models.ViTEfficient import EfficientViT

device = select_device()
SAM_IMG_SIZE = 128
KEEP_BACKGROUND = False
DEMO_MODEL_PATH = "pretrained_2023-12-16_07-54-16"
DEMO_MODEL_EPOCH = "best"

def get_model(model_path, epoch, sam_checkpoint_path="checkpoints/sam_checkpoint.pt"):
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

    type = model_path.split('_')[0]
    if type == "resnet24":
        model = ResNet24Pretrained(
            HIDDEN_SIZE if configurations is None else configurations["hidden_size"], NUM_CLASSES if configurations is None else configurations["num_classes"]).to(device)
        normalization_stats = IMAGENET_STATISTICS
    elif type == "densenet121":
        model = DenseNetPretrained(
            HIDDEN_SIZE if configurations is None else configurations["hidden_size"], NUM_CLASSES if configurations is None else configurations["num_classes"]).to(device)
        normalization_stats = IMAGENET_STATISTICS
    elif type == "inception_v3":
        model = InceptionV3Pretrained(
            HIDDEN_SIZE if configurations is None else configurations["hidden_size"], NUM_CLASSES if configurations is None else configurations["num_classes"]).to(device)
        normalization_stats = IMAGENET_STATISTICS
    elif type == "standard":
        model = ViT_standard(in_channels=INPUT_SIZE if configurations is None else configurations["input_size"],
                             patch_size=PATCH_SIZE if configurations is None else configurations[
                                 "patch_size"],
                             d_model=EMB_SIZE if configurations is None else configurations[
                                 "emb_size"],
                             img_size=IMAGE_SIZE if configurations is None else configurations[
                                 "image_size"],
                             n_classes=NUM_CLASSES if configurations is None else configurations[
                                 "num_classes"],
                             n_head=N_HEADS if configurations is None else configurations["n_heads"],
                             n_layers=N_LAYERS if configurations is None else configurations["n_layers"],
                             dropout=DROPOUT_P).to(device)
        normalization_stats = DEFAULT_STATISTICS
    elif type == "pretrained":
        model = ViT_pretrained(
            HIDDEN_SIZE if configurations is None else configurations["hidden_size"], NUM_CLASSES if configurations is None else configurations["num_classes"], pretrained=True, dropout=DROPOUT_P).to(device)
        normalization_stats = IMAGENET_STATISTICS
    elif type == "efficient":
        model = EfficientViT(img_size=224, patch_size=16, in_chans=INPUT_SIZE if configurations is None else configurations["input_size"], stages=['s', 's', 's'],
                             embed_dim=[64, 128, 192], key_dim=[16, 16, 16], depth=[1, 2, 3], window_size=[7, 7, 7], kernels=[5, 5, 5, 5])
        normalization_stats = DEFAULT_STATISTICS
    else:
        raise ValueError(f"Unknown architecture {type}")

    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/melanoma_detection_{epoch}.pt")
    model.load_state_dict(state_dict)

    sam_model = SAM(
                custom_size=True,
                img_size=SAM_IMG_SIZE,
                checkpoint_path=sam_checkpoint_path).to(device)
    
    return model, sam_model, normalization_stats

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
        
        preprocess_params = {
            'adjust_contrast': 1.5,
            'adjust_brightness': 1.2,
            'adjust_saturation': 2,
            'adjust_gamma': 1.5,
            'gaussian_blur': 5
        }

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
        return tuple(("Melanocytic nevi", 0))
    elif pred == 1:
        return tuple(("Benign lesions of the keratosis", 0))
    elif pred == 2:
        return tuple(("Melanoma", 1))
    elif pred == 3:
        return tuple(("Actinic keratoses and intraepithelial carcinoma", 1))
    elif pred == 4:
        return tuple(("Basal cell carcinoma", 1))
    elif pred == 5:
        return tuple(("Dermatofibroma", 0))
    else:
        return tuple(("Vascular lesion", 0))

def process_image(image_path, model_path=DEMO_MODEL_PATH, epoch=DEMO_MODEL_EPOCH):
    model, sam_model, normalization_stats = get_model(model_path, epoch)
    model = model.to(device)
    sam_model = sam_model.to(device)
    model.eval()
    sam_model.model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

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

    # Show the the texts once the prediction is done
    text_label_panel1.grid(row=3, column=0)
    text_label_panel2.grid(row=3, column=1)
    text_label_result.grid(row=5, column=0, columnspan=2)

    # Update the result label with the predicted class
    pred_text = decode_prediction(predicted_class)
    result_text.set(f"{pred_text[0]} ({'Benign' if pred_text[1] == 0 else 'Malignant'})")
    if pred_text[1] == 0:
        result_label.config(fg="green")
    else:
        result_label.config(fg="red")

def set_window_size():
    # Get the screen width and height
    #screen_width = root.winfo_screenwidth()
    #screen_height = root.winfo_screenheight()

    # Set the window size
    window_width = 480  # Adjust the width as needed
    window_height = 450  # Adjust the height as needed

    # Set the window size (80% of the screen width and height)
    #window_width = int(0.3 * screen_width)
    #window_height = int(0.3 * screen_height)

    # Set the window geometry
    root.geometry(f"{window_width}x{window_height}+{int((root.winfo_screenwidth() - window_width) / 2)}+{int((root.winfo_screenheight() - window_height) / 2)}")
    
    # Disable window resizing
    root.resizable(False, False)

# Create the main window
root = tk.Tk()
root.title("Melanoma Detection Demo")

# Set window size based on desktop dimensions
set_window_size()

# Create a title label
title_label = tk.Label(root, text="Melanoma Detection", font=("Helvetica", 16, "underline"), bg="pink")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Create a description
description_label = tk.Label(root, text="Please, upload an image with a mole to make the diagnosis:", font=("Helvetica", 12))
description_label.grid(row=1, column=0, columnspan=2, pady=10)

# Create a button to open an image
open_button = tk.Button(root, text="Upload Image", command=open_image)
open_button.grid(row=2, column=0, columnspan=2)

# Create two panels with a small border between them
panel1 = tk.Label(root)
panel2 = tk.Label(root)

# Create a label for text above panel 1
text_label_panel1 = tk.Label(root, text="Uploaded image", font=("Helvetica", 12))
text_label_panel1.grid_remove()

# Create a label for text above panel 1
text_label_panel2 = tk.Label(root, text="Segmented mole", font=("Helvetica", 12))
text_label_panel2.grid_remove()

# Center panels in the window
panel1.grid(row=4, column=0, padx=5)
panel2.grid(row=4, column=1, padx=5)

# Create a label to display the result
text_label_result = tk.Label(root, text="Result of the diagnosis:", font=("Helvetica", 12, "underline"))
text_label_result.grid_remove()
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 12))
result_label.grid(row=6, column=0, columnspan=2)

# Assign the panels a mock image
grey_image_array = np.ones((224, 224, 3), dtype=np.uint8) * (240, 240, 240)
grey_image_pil = Image.fromarray(grey_image_array.astype('uint8'))
grey_image = ImageTk.PhotoImage(grey_image_pil)
panel1.config(image=grey_image)
panel1.image = grey_image
panel2.config(image=grey_image)
panel2.image = grey_image

# Start the GUI event loop
root.mainloop()