import os
from PIL import Image

metadata_file = 'metadata_train.csv'
base_data_folder = 'C:/Users/aless/OneDrive/Desktop/Projects/melanoma-detection/data/'

NOT_FOUND_IMAGES = []
UNSUPPORTED_IMAGES = []

def check_image_existence(image_id):
    folders_to_check = [
        'offline_computed_dataset_no_synthetic/offline_images/train/',
        'offline_computed_dataset_no_synthetic/gradcam_70/train/',
        'offline_computed_dataset_no_synthetic/gradcam_110/train/'
    ]

    checked_folders = 0
    for folder in folders_to_check:
        image_path = os.path.join(base_data_folder, folder, f'{image_id}.png')
        if os.path.exists(image_path):
            checked_folders += 1
        
        try:
            img = Image.open(image_path)
            img.verify()  # Verify if the file is a valid image
            img.close()
        except (IOError, SyntaxError):
            UNSUPPORTED_IMAGES.append(image_id)
            print(f"Image {image_id} is not a supported image type.")
            return False

    if checked_folders == len(folders_to_check):
        return True
    return False

def main():
    with open(metadata_file, 'r') as file:
        lines = file.readlines()

    for line in lines[1:]:  # Skip header line
        _, _, image_id = line.strip().split(',')
        if check_image_existence(image_id):
            pass
            #print(f"Image {image_id} exists in the specified directories.")
        else:
            NOT_FOUND_IMAGES.append(image_id)
            print(f"Image {image_id} does not exist in the specified directories.")
    print(f"Total images not found: {len(NOT_FOUND_IMAGES)}")

if __name__ == "__main__":
    main()