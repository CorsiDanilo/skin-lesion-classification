import os
from typing import List, Optional
import torch
from augmentation.CustomLayers import NoiseLayer
from augmentation.GAN import Discriminator, Generator, Resnet50Styles
from torchvision.utils import save_image
from augmentation.Losses import VGG16PerceptualLoss, W_loss, Mkn_loss
from torch.optim import Adam
from utils.utils import select_device
from tqdm import tqdm
from math import log10


class Invertor():
    def __init__(self, cfg, checkpoint: Optional[str] = None):
        self.cfg = cfg
        self.device = select_device()
        self.resolution = cfg.dataset.resolution
        self.gen = Generator(num_channels=3,
                             dlatent_size=512,
                             resolution=self.resolution,
                             structure="fixed",
                             conditional=False,
                             **cfg.model.gen).to(self.device)
        self.dis = Discriminator(num_channels=3,
                                 resolution=self.resolution,
                                 structure="fixed",
                                 conditional=False,
                                 **cfg.model.dis).to(self.device)
        self.gen_checkpoint = checkpoint if checkpoint is not None else "GAN_GEN_7_50.pth"
        self.dis_checkpoint = "GAN_DIS_7_42.pth"

        self.gen.load_checkpoints(os.path.join(
            "checkpoints", self.gen_checkpoint))
        self.dis.load_state_dict(torch.load(os.path.join(
            "checkpoints", self.dis_checkpoint), map_location=self.device)
        )

        self.gen.eval()
        self.dis.eval()
        self.g_synthesis = self.gen.g_synthesis
        self.g_synthesis.eval()

        self.resnet50 = Resnet50Styles().to(self.device)
        self.resnet50.eval()

        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.results_dir = os.path.join(self.current_dir, "invertor_results")
        self.images_dir = os.path.join(self.results_dir, "images")
        self.latents_dir = os.path.join(self.results_dir, "latents")
        self.augmented_images_dir = os.path.join(
            self.results_dir, "augmented_images")
        resample_results_path = os.path.join(
            self.results_dir, "resample_results")

        os.makedirs("invertor_results", exist_ok=True)
        os.makedirs(resample_results_path, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.latents_dir, exist_ok=True)

    def psnr(self, mse, flag=0):
        # flag = 0 if a single image is used and 1 if loss for a batch of images is to be calculated
        if flag == 0:
            psnr = 10 * log10(1 / mse.item())
        return psnr

    def reset_noise(self,
                    min_value: Optional[float] = None,
                    max_value: Optional[float] = None):
        """
        Utility function to reset the noise of the NoiseLayers of the generator.
        """
        def generate_random_tensor(size, min_value, max_value):
            return (min_value + torch.rand(size) * (max_value - min_value)).to(self.device)
        noise_list = []
        num_noise_layers = 8
        for i in range(2, num_noise_layers + 2):
            if min_value is not None and max_value is not None:
                noise_1 = generate_random_tensor(
                    (1, 1, pow(2, i), pow(2, i)), min_value, max_value)
                noise_2 = generate_random_tensor(
                    (1, 1, pow(2, i), pow(2, i)), min_value, max_value)
            else:
                noise_1 = torch.randn(
                    (1, 1, pow(2, i), pow(2, i))).to(self.device)
                noise_2 = torch.randn(
                    (1, 1, pow(2, i), pow(2, i))).to(self.device)

            noise_list.append(noise_1)
            noise_list.append(noise_2)
        self.update_noise(noise_list)

    def update_noise(self,
                     noise_list: torch.Tensor,
                     from_layer: Optional[int] = None,
                     to_layer: Optional[int] = None):
        """
        Utility function to update the noise of the NoiseLayers of the generator.
        """
        noise_layers = [layer for layer in self.g_synthesis.modules(
        ) if isinstance(layer, NoiseLayer)]

        if from_layer is not None and to_layer is not None:
            noise_layers = noise_layers[from_layer:to_layer+1]

        for i, layer in enumerate(noise_layers):
            if from_layer is not None and to_layer is not None:
                if i < from_layer:
                    continue
                if i > to_layer:
                    break
            layer.noise = noise_list[i]

    def embed(self,
              images: torch.Tensor,
              names: List[str],
              save_images: bool = True,
              w_epochs: int = 800,
              n_epochs: int = 500,
              verbose: bool = True,
              show_pbar: bool = True):
        """
        Function taken from https://github.com/Jerry2398/Image2StyleGAN-and-Image2StyleGAN-
        and sligthly modified to fit our needs.

        :param images
        """
        batch_size = images.shape[0]
        saved_latents_paths = [os.path.join(
            self.latents_dir, f"{name}_w.pt") for name in names]
        saved_noise_paths = [os.path.join(
            self.latents_dir, f"{name}_noise.pt") for name in names]

        upsample = torch.nn.Upsample(
            scale_factor=256 / self.resolution, mode='bilinear')

        perceptual = VGG16PerceptualLoss().to(self.device)
        w = torch.zeros((batch_size, 16, 512),
                        requires_grad=True, device=self.device)

        noise_list = []
        num_noise_layers = 8
        for i in range(2, num_noise_layers + 2):
            noise_list.append(torch.randn(
                (batch_size, 1, pow(2, i), pow(2, i)), requires_grad=True, device=self.device))
            noise_list.append(torch.randn(
                (batch_size, 1, pow(2, i), pow(2, i)), requires_grad=True, device=self.device))

        self.update_noise(noise_list)

        # Optimizer to change latent code in each backward step
        w_opt = Adam({w}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        n_opt = Adam(noise_list, lr=0.01,
                     betas=(0.9, 0.999), eps=1e-8)

        if show_pbar:
            pbar = tqdm(range(w_epochs + n_epochs))
        for e in range(w_epochs):
            w_opt.zero_grad()

            syn_imgs = self.g_synthesis(w)
            loss = W_loss(syn_img=syn_imgs,
                          img=images,
                          MSE_loss=perceptual.MSE_loss,
                          upsample=upsample,
                          perceptual=perceptual,
                          lamb_p=1,
                          lamb_mse=1)
            loss.backward()
            w_opt.step()
            pbar.update(1)
            FEEDBACK_INTERVAL = 100
            if (e+1) % FEEDBACK_INTERVAL == 0:
                if verbose:
                    print(f"iter{e}: loss -- {loss}")

                syn_img_paths = [os.path.join(
                    self.images_dir, f"syn_{name}_{e+1}.png") for name in names]

                if (e+1) == FEEDBACK_INTERVAL and save_images:
                    original_img_paths = [os.path.join(
                        self.images_dir, f"original_{name}_{e+1}.png") for name in names]

                    for index, image in enumerate(images):
                        save_image(image.clamp(0, 1),
                                   original_img_paths[index])

                if save_images:
                    for index, syn_img in enumerate(syn_imgs):
                        save_image(syn_img.clamp(0, 1), syn_img_paths[index])

        for e in range(w_epochs, w_epochs + n_epochs):
            n_opt.zero_grad()

            self.update_noise(noise_list)

            syn_imgs = self.g_synthesis(w)
            loss = Mkn_loss(syn_image=syn_imgs,
                            image1=images,
                            image2=images,
                            MSE_loss=perceptual.MSE_loss,
                            lamd_mse1=1,
                            lamb_mse2=0)
            loss.backward()
            n_opt.step()
            pbar.update(1)
            FEEDBACK_INTERVAL = 100
            if (e+1) % FEEDBACK_INTERVAL == 0:
                original_img_paths = [os.path.join(
                    self.images_dir, f"original_{name}_{e+1}.png") for name in names]

                syn_img_paths = [os.path.join(
                    self.images_dir, f"syn_{name}_{e+1}.png") for name in names]

                if verbose:
                    print(f"iter{e}: loss -- {loss}")

                if (e+1) == FEEDBACK_INTERVAL and save_images:

                    for index, image in enumerate(images):
                        save_image(image.clamp(0, 1),
                                   original_img_paths[index])

                if save_images:
                    for index, syn_img in enumerate(syn_imgs):
                        save_image(syn_img.clamp(0, 1), syn_img_paths[index])

        for index, (latents_path, noise_path) in enumerate(zip(saved_latents_paths, saved_noise_paths)):
            torch.save(w[index], latents_path)

            formatted_noise_list = [noise[index] for noise in noise_list]
            torch.save(formatted_noise_list, noise_path)

        for index, (original_img_path, syn_img_path) in enumerate(zip(original_img_paths, syn_img_paths)):
            save_image(images[index].clamp(0, 1), original_img_path)
            save_image(syn_imgs[index].clamp(0, 1), syn_img_path)

        return w, noise_list

    # def style_transfer(self,
    #                    source_latent: torch.Tensor,
    #                    style_latent: torch.Tensor,
    #                    noise_list_1: Optional[List[torch.Tensor]],
    #                    noise_list_2: Optional[List[torch.Tensor]],
    #                    add_random_noise: bool = False):
    #     if noise_list_1 is not None and noise_list_2 is not None:
    #         self.update_noise(noise_list_1[:8], from_layer=0, to_layer=7)
    #         if add_random_noise:
    #             random_layers = 6
    #             assert random_layers < 8
    #             self.update_noise(
    #                 noise_list_2[7:16 - random_layers - 1], from_layer=8, to_layer=16 - random_layers - 1)
    #             print(
    #                 f"Noise list from 7 to 11 shape is {len(noise_list_2[7:16 - random_layers - 1])}")
    #             random_noise_list = [torch.randn_like(
    #                 noise) for noise in noise_list_2[-random_layers:]]
    #             print(f"Random noise list shape is {len(random_noise_list)}")
    #             self.update_noise(
    #                 random_noise_list, from_layer=16 - random_layers, to_layer=15)
    #         else:
    #             self.update_noise(noise_list_2[-8:], from_layer=8, to_layer=15)

    #     image = self.g_synthesis(dlatents_in=source_latent,
    #                              styled_latents=style_latent,
    #                              style_threshold=8)
    #     image = (image+1.0)/2.0
    #     style_transfer_path = os.path.join(
    #         self.results_dir, "style_transfer_results")
    #     os.makedirs(style_transfer_path, exist_ok=True)
    #     image_path = os.path.join(
    #         style_transfer_path, "transferred_image_new.png")
    #     save_image(image.clamp(0, 1), image_path)
    #     return

    # def mix_latents(self,
    #                 source_latent: torch.Tensor,
    #                 style_latent: torch.Tensor,
    #                 noise_list_1: Optional[List[torch.Tensor]],
    #                 noise_list_2: Optional[List[torch.Tensor]],
    #                 mix_threshold: float = 0.5):
    #     mixed_latent = mix_threshold * source_latent + \
    #         (1 - mix_threshold) * style_latent
    #     mixed_noise = [mix_threshold * n1 + (1 - mix_threshold) * n2
    #                    for n1, n2 in zip(noise_list_1, noise_list_2)]

    #     self.update_noise(mixed_noise)
    #     image = self.g_synthesis(mixed_latent)
    #     image = (image+1.0)/2.0
    #     style_transfer_path = os.path.join(
    #         self.results_dir, "style_transfer_results")
    #     os.makedirs(style_transfer_path, exist_ok=True)
    #     image_path = os.path.join(
    #         style_transfer_path, f"transferred_image_{mix_threshold}.png")
    #     save_image(image.clamp(0, 1), image_path)
    #     return

    def generate(self,
                 latent: torch.Tensor,
                 noise_list: Optional[List[torch.Tensor]] = None):
        if noise_list is not None:
            self.update_noise(noise_list)
        image = self.g_synthesis(latent)
        return image
