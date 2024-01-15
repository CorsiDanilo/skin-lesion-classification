import os
import pickle
from typing import List
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dataloaders.StyleGANDataLoader import StyleGANDataLoader
from .Invertor import Invertor
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from .gan_config import cfg
from shared.constants import DEFAULT_STATISTICS


def embed_everything():
    """
    This function is used to embed all images from the train dataset.
    It will save all the files inside the augmentation/invertor_results/latents directory.

    Since Images2Stylegan++ optimized both the latent vector w and the noise, for each image it saves two files:

    - [image_name]_w.pt
    - [image_name]_noise.pt

    If the image is augmented, it will save the files with the suffix "_augmented".
    """
    create_computed_latents_set()
    batch_size = 16
    invertor = Invertor(cfg=cfg)

    dataloader = StyleGANDataLoader(
        dynamic_load=True,
        batch_size=batch_size,
        resize_dim=(
            cfg.dataset.resolution,
            cfg.dataset.resolution
        )
    )
    train_dataloders = dataloader.get_train_dataloder()
    computed_latents_set = pickle.load(
        open("computed_latents_set.pkl", "rb"))
    print(f"Computed latents set length is {len(computed_latents_set)}")
    # Skip label 0 since is the majority class
    for dataloader_label in tqdm(range(1, 8)):
        processed_images = 0

        for batch in tqdm(train_dataloders[dataloader_label], desc="Checking images already embedded"):
            images, labels, image_paths, augmented_list = batch
            for image_path in image_paths:
                clean_path = image_path.split("/")[-1].replace(".jpg", "")
                if clean_path in computed_latents_set:
                    processed_images += 1

        for index, batch in enumerate(tqdm(train_dataloders[dataloader_label])):
            print(f"Currently processed {processed_images} images")
            if processed_images >= 100:
                break
            images, labels, image_paths, augmented_list = batch
            clean_paths = [image_path.split(
                "/")[-1].replace(".jpg", "") for image_path in image_paths]
            clean_paths = [clean_path + "_augmented" if augmented else clean_path for clean_path,
                           augmented in zip(clean_paths, augmented_list)]

            new_clean_paths = []
            new_images = []
            for image, clean_path in zip(images, clean_paths):
                if clean_path not in computed_latents_set:
                    new_images.append(image)
                    new_clean_paths.append(clean_path)

            clean_paths = new_clean_paths
            images = torch.stack(new_images)
            print(f"images shape is {images.shape}")
            if clean_paths == []:
                continue

            print(f"Currently processing {len(clean_paths)} images")
            latent, noise_list = invertor.embed(
                images=images,
                names=clean_paths,
                save_images=False,
                w_epochs=500,
                n_epochs=500,
                verbose=False,
                show_pbar=True)
            processed_images += batch_size


def test_noise_saving():
    invertor = Invertor(cfg=cfg)
    noise = torch.load(os.path.join(
        invertor.latents_dir, "ISIC_0024877_noise.pt"), map_location=invertor.device)
    print(f"Noise type is {type(noise)}")
    print(f"Noise length is {len(noise)}")
    for n in noise:
        print(f"Noise shape is {n.shape}")
        print(f"Noise dtype is {n.dtype}")


def recover_embeddings_state():
    """
    Function to recover which images have been already embedded, in case the embedding process is interrupted.
    """
    raise NotImplementedError("This function is not implemented yet.")


def generate_augmented_images(generation_ratio: int = 8):
    """
    Function to generate augmented images from the images' embeddings. 
    It will also generate the metadata.csv file. 

    :params generation_ratio: the number of augmented images to generate for each image.
    """
    invertor = Invertor(cfg=cfg)

    augmented_image_path = os.path.join(
        invertor.results_dir, "augmented_images")
    os.makedirs(augmented_image_path, exist_ok=True)

    latent_noise_dict = {}
    for filename in tqdm(os.listdir(invertor.latents_dir), desc="Loading latents and noise"):
        if filename.endswith("_w.pt"):
            image_name = filename.replace("_w.pt", "")
            noise_name = os.path.join(
                invertor.latents_dir, image_name + "_noise.pt")
            latent_name = os.path.join(invertor.latents_dir, filename)
            latent_noise_dict[image_name] = {
                "w": latent_name, "noise": noise_name}

    for image_name, latent_noise in tqdm(latent_noise_dict.items(), desc="Generating augmented images"):
        latent = torch.load(latent_noise["w"], map_location=invertor.device)
        noise = torch.load(latent_noise["noise"], map_location=invertor.device)
        latent = latent.unsqueeze(0)
        invertor.update_noise(noise)
        for i in range(generation_ratio):
            latent_variation = latent.clone() + torch.randn_like(latent) * 0.15
            image = invertor.generate(latent_variation)
            augmented_image_name = f"{image_name}_augmented_{i}.png"
            save_image(image, os.path.join(
                augmented_image_path, augmented_image_name))


def create_computed_latents_set():
    invertor = Invertor(cfg=cfg)
    set_name = "computed_latents_set.pkl"

    if os.path.exists(set_name):
        _set = pickle.load(open(set_name, "rb"))
        print(f"Loaded set from {set_name}, it has length {len(_set)}")
    else:
        _set = set()
    for filename in os.listdir(invertor.latents_dir):
        if filename.endswith("_w.pt"):
            image_name = filename.replace("_w.pt", "")
            _set.add(image_name)
    print(f"Set length is {len(_set)}")
    print(f"Set is {_set}")
    pickle.dump(_set, open("computed_latents_set.pkl", "wb"))


def generate_metadata():
    """
    Function to generate the metadata csv file from the augmented images.
    """
    raise NotImplementedError("This function is not implemented yet.")


def generate_similar_images():
    num_images_to_generate = 20
    batch_size = 16
    invertor = Invertor(cfg=cfg)

    fixed_dataloader = ImagesAndSegmentationDataLoader(
        limit=None,
        dynamic_load=True,
        upscale_train=False,
        normalize=False,
        normalization_statistics=DEFAULT_STATISTICS,
        batch_size=batch_size,
        resize_dim=(
            cfg.dataset.resolution,
            cfg.dataset.resolution,
        )
    )
    fixed_data = fixed_dataloader.get_test_dataloader()
    for batch in fixed_data:
        images, _ = batch

        images = images.to(invertor.device)
        image = images[10].unsqueeze(0)
        break
    latent_1, noise_list_1 = invertor.embed(
        image, "image", w_epochs=800, n_epochs=800)
    invertor.update_noise(noise_list_1)
    for i in range(num_images_to_generate):
        variant_latent = latent_1.clone() + torch.randn_like(latent_1) * 0.15
        image = invertor.generate(variant_latent)
        save_image(
            image, f"augmentation/invertor_results/resample_results/resampled_image_{i}.png")


def embed_and_style_transfer():
    """
    This is a function that is used to see the quality of the embedded image and the quality of the style transfer.
    """
    batch_size = 16
    invertor = Invertor(cfg=cfg)

    fixed_dataloader = ImagesAndSegmentationDataLoader(
        limit=None,
        dynamic_load=True,
        upscale_train=False,
        normalize=False,
        normalization_statistics=DEFAULT_STATISTICS,
        batch_size=batch_size,
        resize_dim=(
            cfg.dataset.resolution,
            cfg.dataset.resolution,
        )
    )
    fixed_data = fixed_dataloader.get_test_dataloader()
    for batch in fixed_data:
        images, labels = batch

        images = images.to(invertor.device)
        first_image = images[1].unsqueeze(0)
        second_image = [image for index, (image, label) in enumerate(zip(
            images, labels)) if index != 1 and label == labels[1]][0].unsqueeze(0)
        break

    latent_1, noise_list_1 = invertor.embed(first_image, "first_image")
    latent_2, noise_list_2 = invertor.embed(second_image, "second_image")

    # NOTE: trying to invert the noise
    invertor.style_transfer(
        latent_1, latent_2, noise_list_2, noise_list_1)

############################
# OFFLINE TEST OPERATIONS  #
############################


def offline_mix_latents():
    invertor = Invertor(cfg=cfg)
    latent_path = invertor.latents_dir
    first_latent_path = os.path.join(latent_path, "first_image_w.pt")
    second_latent_path = os.path.join(latent_path, "second_image_w.pt")
    first_noise_path = os.path.join(latent_path, "first_image_noise.pt")
    second_noise_path = os.path.join(latent_path, "second_image_noise.pt")
    latent_1 = torch.load(first_latent_path)
    latent_2 = torch.load(second_latent_path)
    noise_list_1 = torch.load(first_noise_path)
    noise_list_2 = torch.load(second_noise_path)
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        invertor.mix_latents(latent_1, latent_2, noise_list_1,
                             noise_list_2, mix_threshold=thresh)


def offline_style_transfer():
    invertor = Invertor(cfg=cfg)
    latent_path = invertor.latents_dir
    first_latent_path = os.path.join(latent_path, "first_image_w.pt")
    second_latent_path = os.path.join(latent_path, "second_image_w.pt")
    first_noise_path = os.path.join(latent_path, "first_image_noise.pt")
    second_noise_path = os.path.join(latent_path, "second_image_noise.pt")
    latent_1 = torch.load(first_latent_path)
    latent_2 = torch.load(second_latent_path)
    noise_list_1 = torch.load(first_noise_path)
    noise_list_2 = torch.load(second_noise_path)
    image_1 = invertor.generate(latent_1, noise_list_1)
    image_2 = invertor.generate(latent_2, noise_list_2)
    save_image(image_1, "image_1.png")
    save_image(image_2, "image_2.png")
    invertor.style_transfer(
        latent_1, latent_2, noise_list_1, noise_list_2, add_random_noise=True)


def get_min_max_noise(noise_list: List[torch.Tensor], scale_factor=1.0):
    min_noise, max_noise = torch.mean(torch.tensor(
        [n.min() for n in noise_list])), torch.mean(torch.tensor([n.max() for n in noise_list]))
    return min_noise * scale_factor, max_noise * scale_factor


def resample_image():
    invertor = Invertor(cfg=cfg)
    latent_path = invertor.latents_dir
    first_latent_path = os.path.join(latent_path, "image_w.pt")
    noise_path = os.path.join(latent_path, "image_noise.pt")
    latent = torch.load(first_latent_path)
    noise = torch.load(noise_path)
    invertor.update_noise(noise)
    num_images_to_generate = 30
    for i in range(num_images_to_generate):
        new_latent = latent.clone() + torch.randn_like(latent) * 0.15
        image = invertor.generate(new_latent)
        save_image(image, f"resampled_image_{i}.png")


if __name__ == '__main__':
    # resample_image()
    # generate_similar_images()
    embed_everything()
    # generate_augmented_images()
    # create_computed_latents_set()
    # test_noise_saving()
