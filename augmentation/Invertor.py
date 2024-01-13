import os
from typing import Optional
import torch
from augmentation.GAN import GSynthesis, Generator, Resnet50Styles
from torchvision.utils import save_image
from augmentation.Losses import VGG16PerceptualLoss
from torch.optim import Adam
from utils.utils import select_device
from tqdm import tqdm
from math import log10


class Invertor():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = select_device()
        self.gen = Generator(num_channels=3,
                             dlatent_size=512,
                             resolution=cfg.dataset.resolution,
                             structure="fixed",
                             conditional=False,
                             #  n_classes=7,
                             **cfg.model.gen).to(self.device)
        self.gen.load_checkpoints(os.path.join(
            "checkpoints", "512res_512lat_GAN_GEN_7_14.pth"))
        self.gen.eval()
        self.g_synthesis = self.gen.g_synthesis
        self.g_synthesis.eval()

        self.resnet50 = Resnet50Styles().to(self.device)
        self.resnet50.eval()

        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.results_dir = os.path.join(self.current_dir, "invertor_results")
        self.images_dir = os.path.join(self.results_dir, "images")
        self.latents_dir = os.path.join(self.results_dir, "latents")
        os.makedirs("invertor_results", exist_ok=True)

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.latents_dir, exist_ok=True)

    def psnr(self, mse, flag=0):
        # flag = 0 if a single image is used and 1 if loss for a batch of images is to be calculated
        if flag == 0:
            psnr = 10 * log10(1 / mse.item())
        return psnr

    def embed(self,
              image: torch.Tensor,
              embedding_name: str):
        upsample = torch.nn.Upsample(
            scale_factor=256/self.cfg.dataset.resolution, mode='bilinear')
        # assert image.shape == (1, 3, 1024, 1024)
        img_p = image.clone()
        img_p = upsample(img_p)
        # Perceptual loss initialise object
        perceptual = VGG16PerceptualLoss().to(self.device)

        # since the synthesis network expects 18 w vectors of size 1xlatent_size thus we take latent vector of the same size
        latents = torch.zeros(
            (1, 18, 512), requires_grad=True, device=self.device)

        # Optimizer to change latent code in each backward step
        optimizer = Adam(
            {latents}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        # Loop to optimise latent vector to match the ted image to input image
        loss_ = []
        loss_psnr = []
        iterations = 1500
        pbar = tqdm(total=iterations)
        for e in range(iterations):
            optimizer.zero_grad()
            syn_img = self.g_synthesis(latents)
            syn_img = (syn_img+1.0)/2.0
            mse, per_loss = perceptual.loss_function(
                syn_img=syn_img,
                img=image,
                img_p=img_p,
                upsample=upsample
            )
            psnr = self.psnr(mse, flag=0)
            loss = per_loss + mse
            loss.backward()
            optimizer.step()
            loss_np = loss.detach().cpu().numpy()
            loss_p = per_loss.detach().cpu().numpy()
            loss_m = mse.detach().cpu().numpy()
            loss_psnr.append(psnr)
            loss_.append(loss_np)
            pbar.update(1)
            pbar.set_postfix(
                loss=loss.item(), psnr=psnr
            )

            FEEDBACK_INTERVAL = 100
            if (e+1) % FEEDBACK_INTERVAL == 0:
                print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(e +
                                                                                                1, loss_np, loss_m, loss_p, psnr))

                saved_latents_path = os.path.join(
                    self.latents_dir, f"{embedding_name}.pt")

                torch.save(latents, saved_latents_path)

                syn_img_path = os.path.join(
                    self.images_dir, f"syn_{embedding_name}_{e+1}.png")

                if (e+1) == FEEDBACK_INTERVAL:
                    original_img_path = os.path.join(
                        self.images_dir, f"original_{embedding_name}_{e+1}.png")

                save_image(syn_img.clamp(0, 1), syn_img_path)
                save_image(image.clamp(0, 1), original_img_path)
        return latents

    def style_transfer(self,
                       source_latent: torch.Tensor,
                       style_latent: torch.Tensor):
        image = self.g_synthesis(dlatents_in=source_latent,
                                 styled_latents=style_latent,
                                 style_threshold=9)
        style_transfer_path = os.path.join(
            self.results_dir, "style_transfer_results")
        os.makedirs(style_transfer_path, exist_ok=True)
        image_path = os.path.join(style_transfer_path, "transferred_image.png")
        save_image(image.clamp(0, 1), image_path)
        return

    def mix_latents(self,
                    source_latent: torch.Tensor,
                    style_latent: torch.Tensor):
        mixed_latent = (source_latent + style_latent) / 2
        image = self.g_synthesis(mixed_latent)
        style_transfer_path = os.path.join(
            self.results_dir, "style_transfer_results")
        os.makedirs(style_transfer_path, exist_ok=True)
        image_path = os.path.join(style_transfer_path, "transferred_image.png")
        save_image(image.clamp(0, 1), image_path)
        return

    def generate(self, latent: torch.Tensor):
        return self.g_synthesis(latent)

    def generate_from_label(self, labels_in: torch.Tensor):
        noise = torch.randn(7, 256).to(self.device)
        labels_in = labels_in.to(self.device)

        return self.gen(latents_in=noise, labels_in=labels_in, depth=6, alpha=0)

    def generate_from_resnet(self,
                             image1: torch.Tensor,
                             image2: torch.Tensor):
        style1 = self.resnet50(image1)
        style2 = self.resnet50(image2)
        random_styles = torch.randn(
            1, 4, self.resnet50.latent_size).to(self.device)
        styles = torch.cat([style1, style2, random_styles], dim=1)
        image = self.g_synthesis(styles)
        image = (image+1.0)/2.0
        save_image(image.clamp(0, 1), "generated_image.png")
        return image

    def generate_with_noise(self,
                            latent: torch.Tensor,
                            latent_2: Optional[torch.Tensor]):
        noise_layers = None
        transfer_layers = 5
        latent.requires_grad = False
        # print(f"Latent shape is {latent.shape}")
        # if latent_2 is not None:
        #     print(f"Latent 2 shape is {latent_2.shape}")
        if latent_2 is not None and transfer_layers is not None:
            latent[:, :transfer_layers,
                   :] = latent_2[:, :transfer_layers, :]
        if noise_layers is not None:
            latent = latent[:, noise_layers-18:, :]
            noise = torch.randn(1, noise_layers, 512).to(self.device)
            noise = noise * 0.01
            noised_latent = torch.cat([noise, latent], dim=1)
            image = self.g_synthesis(noised_latent)
        else:
            image = self.g_synthesis(latent)
        noise = torch.randn(1, 18, 512).to(self.device)
        # noise = noise * 0.05
        # latent = latent - noise
        # image = self.g_synthesis(latent)
        image = (image+1.0)/2.0
        save_image(image.clamp(0, 1), "generated_image.png")
        return image

    # def train_hierarchical(self, image):
    #     upsample = torch.nn.Upsample(scale_factor=256/1024, mode='bilinear')
    #     img_p = image.clone()
    #     img_p = upsample(img_p)

    #     # Perceptual loss initialise object
    #     perceptual = VGG16PerceptualLoss().to(self.device)
    #     # since the synthesis network expects 18 w vectors of size 1x512 thus we take latent vector of the same size
    #     latent_w = torch.zeros(
    #         (1, 512), requires_grad=True, device=self.device)

    #     # Optimizer to change latent code in each backward step
    #     optimizer = Adam({latent_w}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    #     # Loop to optimise latent vector to match the generated image to input image
    #     loss_ = []
    #     loss_psnr = []
    #     pbar = tqdm(total=1000)
    #     for e in range(1000):
    #         optimizer.zero_grad()
    #         latent_w1 = latent_w.unsqueeze(1).expand(-1, 18, -1)
    #         syn_img = self.g_synthesis(
    #             dlatents_in=latent_w1, depth=self.depth)
    #         syn_img = (syn_img+1.0)/2.0
    #         mse, per_loss = perceptual.loss_function(
    #             syn_img=syn_img,
    #             img=image,
    #             img_p=img_p,
    #             upsample=upsample)
    #         psnr = self.psnr(mse, flag=0)
    #         loss = per_loss + mse
    #         loss.backward()
    #         optimizer.step()
    #         loss_np = loss.detach().cpu().numpy()
    #         loss_p = per_loss.detach().cpu().numpy()
    #         loss_m = mse.detach().cpu().numpy()
    #         loss_psnr.append(psnr)
    #         loss_.append(loss_np)
    #         pbar.update(1)
    #         pbar.set_postfix(
    #             loss=loss.item(),
    #             psnr=psnr)
    #         if (e+1) % 500 == 0:
    #             print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(
    #                 e+1, loss_np, loss_m, loss_p, psnr))
    #             save_image(syn_img.clamp(0, 1),
    #                        "Hier_pass_morphP1-syn-{}.png".format(e+1))
    #             save_image(image.clamp(0, 1),
    #                        "Hier_pass_morphP1-original-{}.png".format(e+1))

    #     latent_w1 = latent_w.unsqueeze(1).expand(-1, 18, -1)
    #     latent_w1 = torch.tensor(latent_w1, requires_grad=True)
    #     optimizer = Adam({latent_w1}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    #     pbar = tqdm(total=1000)
    #     for e in range(1000):
    #         optimizer.zero_grad()
    #         syn_img = self.g_synthesis(
    #             dlatents_in=latent_w1, depth=self.depth)
    #         syn_img = (syn_img+1.0)/2.0
    #         mse, per_loss = perceptual.loss_function(
    #             syn_img, image, img_p, upsample)
    #         psnr = self.psnr(mse, flag=0)
    #         loss = per_loss + mse
    #         loss.backward()
    #         optimizer.step()
    #         loss_np = loss.detach().cpu().numpy()
    #         loss_p = per_loss.detach().cpu().numpy()
    #         loss_m = mse.detach().cpu().numpy()
    #         loss_psnr.append(psnr)
    #         loss_.append(loss_np)
    #         pbar.update(1)
    #         pbar.set_postfix(
    #             loss=loss.item(),
    #             psnr=psnr)
    #         if (e+1) % 500 == 0:
    #             print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(
    #                 e+1, loss_np, loss_m, loss_p, psnr))
    #             save_image(syn_img.clamp(0, 1),
    #                        "Hier_pass_morphP2-syn-{}.png".format(e+1))
    #             save_image(image.clamp(0, 1),
    #                        "Hier_pass_morphP2-original-{}.png".format(e+1))

    #     return latent_w1
