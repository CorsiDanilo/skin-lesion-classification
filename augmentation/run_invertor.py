import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dataloaders.StyleGANDataLoader import StyleGANDataLoader
from .Invertor import Invertor
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from .gan_config import cfg
from shared.constants import DEFAULT_STATISTICS, IMAGENET_STATISTICS


def main():
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

    latent_1, noise_list_1 = invertor.embed_v2(first_image, "first_image")
    latent_2, noise_list_2 = invertor.embed_v2(second_image, "second_image")

    invertor.style_transfer_v2(latent_1, latent_2, noise_list_1, noise_list_2)


def embed_everything():
    batch_size = 32
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
    total_iterations = sum(len(loader) for loader in train_dataloders.values())
    pbar = tqdm(total=total_iterations,
                desc="Embedding images with Images2Stylegan++")
    for dataloader_label in range(8):
        for index, batch in enumerate(train_dataloders[dataloader_label]):
            images, labels, image_paths, augmented_list = batch
            for image, label, image_path, augmented in zip(images, labels, image_paths, augmented_list):
                clean_path = image_path.split("/")[-1].replace(".jpg", "")
                if augmented:
                    clean_path = clean_path + "_augmented"
                assert label == dataloader_label
                image = image.unsqueeze(0)
                print(
                    f'Embedding image with path {clean_path}, label {label}')
                latent, noise_list = invertor.embed_v2(
                    image=image,
                    name=clean_path,
                    save_images=False,
                    w_epochs=1,
                    n_epochs=1)
            pbar.update(1)
            pbar.set_description_str(
                f"Class: {dataloader_label}, batch: {index}")


def offline_mix_latents():
    invertor = Invertor(cfg=cfg)
    latent_path = invertor.latents_dir
    first_latent_path = os.path.join(latent_path, "first_image.pt")
    second_latent_path = os.path.join(latent_path, "second_image.pt")
    latent_1 = torch.load(first_latent_path)
    latent_2 = torch.load(second_latent_path)
    image_1 = invertor.generate(latent_1)
    image_2 = invertor.generate(latent_2)
    save_image(image_1, "image_1.png")
    save_image(image_2, "image_2.png")
    invertor.mix_latents(latent_1, latent_2)


def offline_mix_latents_v2():
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
        invertor.mix_latents_v2(latent_1, latent_2, noise_list_1,
                                noise_list_2, mix_threshold=thresh)


def offline_style_transfer():
    invertor = Invertor(cfg=cfg)
    latent_path = invertor.latents_dir
    first_latent_path = os.path.join(latent_path, "first_image.pt")
    second_latent_path = os.path.join(latent_path, "second_image.pt")
    latent_1 = torch.load(first_latent_path)
    latent_2 = torch.load(second_latent_path)
    image_1 = invertor.generate(latent_1)
    image_2 = invertor.generate(latent_2)
    save_image(image_1, "image_1.png")
    save_image(image_2, "image_2.png")
    invertor.style_transfer(latent_1, latent_2)


def offline_style_transfer_v2():
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
    save_image(image_1, "image_1_v2.png")
    save_image(image_2, "image_2_v2.png")
    invertor.style_transfer_v2(latent_1, latent_2, noise_list_1, noise_list_2)


def generate_resnet_images():
    batch_size = 32
    invertor = Invertor(cfg=cfg)

    fixed_dataloader = ImagesAndSegmentationDataLoader(
        limit=None,
        dynamic_load=True,
        upscale_train=False,
        normalize=True,
        normalization_statistics=IMAGENET_STATISTICS,
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
        first_image = images[0].unsqueeze(0)
        second_image = images[1].unsqueeze(0)
        break

    invertor.generate_from_resnet(first_image, second_image)


def generate_image_with_noise():
    invertor = Invertor(cfg=cfg)
    latent_path = invertor.latents_dir
    first_latent_path = os.path.join(latent_path, "first_image.pt")
    second_latent_path = os.path.join(latent_path, "second_image.pt")
    latent_1 = torch.load(first_latent_path)
    latent_2 = torch.load(second_latent_path)
    invertor.generate_with_noise(
        latent=latent_1, latent_2=latent_2)
    # invertor.generate_with_noise(
    #     latent=latent_2, latent_2=latent_1)


def generate_image_from_labels():
    invertor = Invertor(cfg=cfg)
    latents = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    images = invertor.generate_from_label(latents)
    for i, image in enumerate(images):
        save_image(image, f"image_{i}.png")
    return images


def embed_full_dataset():
    pass


if __name__ == '__main__':
    # main()
    embed_everything()
