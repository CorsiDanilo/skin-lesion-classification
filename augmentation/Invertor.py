import os
from typing import List, Optional
import torch
from augmentation.CustomLayers import NoiseLayer
from augmentation.GAN import Generator, Resnet50Styles
from torchvision.utils import save_image
from augmentation.Losses import VGG16PerceptualLoss, W_loss, Mkn_loss
from torch.optim import Adam
from utils.utils import select_device
from tqdm import tqdm
from math import log10


class Invertor():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = select_device()
        self.resolution = cfg.dataset.resolution
        self.gen = Generator(num_channels=3,
                             dlatent_size=512,
                             resolution=self.resolution,
                             structure="fixed",
                             conditional=False,
                             #  n_classes=7,
                             **cfg.model.gen).to(self.device)
        self.gen.load_checkpoints(os.path.join(
            "checkpoints", "512res_512lat_GAN_GEN_7_21.pth"))
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
        """
        Function taken from https://github.com/zaidbhat1234/Image2StyleGAN/blob/main/Image2Style_Implementation.ipynb
        and sligthly modified to fit our needs.
        """

        upsample = torch.nn.Upsample(
            scale_factor=256/self.resolution, mode='bilinear')
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
            _lambda = 0.5
            loss = _lambda * per_loss + (1 - _lambda) * mse
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

    # def update_noise(self,
    #                  noise_list: torch.Tensor,
    #                  from_layer: Optional[int] = None,
    #                  to_layer: Optional[int] = None):
    #     # TODO: format this in order to iterate only on the list of NoiseLayers to remove those two ugly indexes
    #     i = 0
    #     layer_count = 0
    #     for param in list(self.g_synthesis.modules()):

    #         if from_layer is not None and to_layer is not None:
    #             if layer_count < from_layer:
    #                 layer_count += 1
    #                 continue
    #             if layer_count > to_layer:
    #                 break

    #         if isinstance(param, NoiseLayer):
    #             param.noise = noise_list[i]
    #             i += 1
    #             layer_count += 1

    def update_noise(self,
                     noise_list: torch.Tensor,
                     from_layer: Optional[int] = None,
                     to_layer: Optional[int] = None):

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
            # print(
            #     f"Updating noise layer {i}. From {from_layer} to {to_layer}")
            layer.noise = noise_list[i]

    def embed_v2(self, image, name):
        """
        Function taken from https://github.com/Jerry2398/Image2StyleGAN-and-Image2StyleGAN-
        and sligthly modified to fit our needs.
        """

        upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')

        perceptual = VGG16PerceptualLoss().to(self.device)
        # since the synthesis network expects 18 w vectors of size 1x512 thus we take latent vector of the same size
        w = torch.zeros((1, 16, 512), requires_grad=True, device=self.device)

        # noise 初始化，noise是直接加在feature map上的
        noise_list = []
        num_noise_layers = 8
        for i in range(2, num_noise_layers + 2):
            noise_list.append(torch.randn(
                (1, 1, pow(2, i), pow(2, i)), requires_grad=True, device=self.device))
            noise_list.append(torch.randn(
                (1, 1, pow(2, i), pow(2, i)), requires_grad=True, device=self.device))

        # Optimizer to change latent code in each backward step
        w_opt = Adam({w}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        n_opt = Adam(noise_list, lr=0.01,
                     betas=(0.9, 0.999), eps=1e-8)

        w_epochs = 800
        n_epochs = 500
        # w_epochs = 1
        # n_epochs = 1
        for e in tqdm(range(w_epochs)):
            w_opt.zero_grad()

            self.update_noise(noise_list)

            syn_img = self.g_synthesis(w)
            syn_img = (syn_img + 1.0) / 2.0
            loss = W_loss(syn_img=syn_img,
                          img=image,
                          MSE_loss=perceptual.MSE_loss,
                          upsample=upsample,
                          perceptual=perceptual,
                          lamb_p=1e-5,
                          lamb_mse=1e-5)
            loss.backward()
            w_opt.step()
            FEEDBACK_INTERVAL = 100
            if (e+1) % FEEDBACK_INTERVAL == 0:
                print(f"iter{e}: loss -- {loss}")

                saved_latents_path = os.path.join(
                    self.latents_dir, f"{name}_w.pt")
                saved_noise_path = os.path.join(
                    self.latents_dir, f"{name}_noise.pt")

                torch.save(w, saved_latents_path)
                torch.save(noise_list, saved_noise_path)

                syn_img_path = os.path.join(
                    self.images_dir, f"syn_{name}_{e+1}.png")

                if (e+1) == FEEDBACK_INTERVAL:
                    original_img_path = os.path.join(
                        self.images_dir, f"original_{name}_{e+1}.png")

                save_image(syn_img.clamp(0, 1), syn_img_path)
                save_image(image.clamp(0, 1), original_img_path)

            # X shape is torch.Size([1, 512, 4, 4]), noise shape is torch.Size([1, 1, 4, 4])
            # X shape is torch.Size([1, 512, 4, 4]), noise shape is torch.Size([1, 1, 4, 4])
            # X shape is torch.Size([1, 512, 8, 8]), noise shape is torch.Size([1, 1, 8, 8])
            # X shape is torch.Size([1, 512, 8, 8]), noise shape is torch.Size([1, 1, 8, 8])
            # X shape is torch.Size([1, 512, 16, 16]), noise shape is torch.Size([1, 1, 16, 16])
            # X shape is torch.Size([1, 512, 16, 16]), noise shape is torch.Size([1, 1, 16, 16])
            # X shape is torch.Size([1, 512, 32, 32]), noise shape is torch.Size([1, 1, 32, 32])
            # X shape is torch.Size([1, 512, 32, 32]), noise shape is torch.Size([1, 1, 32, 32])
            # X shape is torch.Size([1, 256, 64, 64]), noise shape is torch.Size([1, 1, 64, 64])
            # X shape is torch.Size([1, 256, 64, 64]), noise shape is torch.Size([1, 1, 64, 64])
            # X shape is torch.Size([1, 128, 128, 128]), noise shape is torch.Size([1, 1, 128, 128])
            # X shape is torch.Size([1, 128, 128, 128]), noise shape is torch.Size([1, 1, 128, 128])
            # X shape is torch.Size([1, 64, 256, 256]), noise shape is torch.Size([1, 1, 256, 256])
            # X shape is torch.Size([1, 64, 256, 256]), noise shape is torch.Size([1, 1, 256, 256])
            # X shape is torch.Size([1, 32, 512, 512]), noise shape is torch.Size([1, 1, 512, 512])
            # X shape is torch.Size([1, 32, 512, 512]), noise shape is torch.Size([1, 1, 512, 512])

            # if (e + 1) % 500 == 0:
            #     print("iter{}: loss -- {}".format(e + 1, loss.item()))
            #     save_image(syn_img.clamp(
            #         0, 1) "save_images/image2stylegan_v2/image_reconstruct/reconstruct_{}.png".format(e + 1))

        for e in tqdm(range(w_epochs, w_epochs + n_epochs)):
            n_opt.zero_grad()

            # NOTE: Trick to update the noise, don't know if it works
            self.update_noise(noise_list)

            syn_img = self.g_synthesis(w)
            syn_img = (syn_img + 1.0) / 2.0
            loss = Mkn_loss(syn_image=syn_img,
                            image1=image,
                            image2=image,
                            MSE_loss=perceptual.MSE_loss,
                            lamd_mse1=1e-5,
                            lamb_mse2=0)
            loss.backward()
            n_opt.step()

            FEEDBACK_INTERVAL = 100
            if (e+1) % FEEDBACK_INTERVAL == 0:
                print(f"iter{e}: loss -- {loss}")

                saved_latents_path = os.path.join(
                    self.latents_dir, f"{name}.pt")
                saved_noise_path = os.path.join(
                    self.latents_dir, f"{name}_noise.pt")

                torch.save(w, saved_latents_path)
                torch.save(noise_list, saved_noise_path)

                syn_img_path = os.path.join(
                    self.images_dir, f"syn_{name}_{e+1}.png")

                if (e+1) == FEEDBACK_INTERVAL:
                    original_img_path = os.path.join(
                        self.images_dir, f"original_{name}_{e+1}.png")

                save_image(syn_img.clamp(0, 1), syn_img_path)
                save_image(image.clamp(0, 1), original_img_path)

            # if (e + 1) % 500 == 0:
            #     print("iter{}: loss -- {}".format(e + 1, loss.item()))
            #     save_image(syn_img.clamp(
            #         0, 1), "save_images/image2stylegan_v2/image_reconstruct/reconstruct_{}.png".format(e + 1))

        return w, noise_list

    def style_transfer(self,
                       source_latent: torch.Tensor,
                       style_latent: torch.Tensor,
                       noise_list: Optional[List[torch.Tensor]] = None):
        if noise_list is not None:
            self.update_noise(noise_list)
        image = self.g_synthesis(dlatents_in=source_latent,
                                 styled_latents=style_latent,
                                 style_threshold=8)
        image = (image+1.0)/2.0
        style_transfer_path = os.path.join(
            self.results_dir, "style_transfer_results")
        os.makedirs(style_transfer_path, exist_ok=True)
        image_path = os.path.join(style_transfer_path, "transferred_image.png")
        save_image(image.clamp(0, 1), image_path)
        return

    def style_transfer_v2(self,
                          source_latent: torch.Tensor,
                          style_latent: torch.Tensor,
                          noise_list_1: Optional[List[torch.Tensor]],
                          noise_list_2: Optional[List[torch.Tensor]]):
        if noise_list_1 is not None and noise_list_2 is not None:
            self.update_noise(noise_list_1[:8], from_layer=0, to_layer=7)
            self.update_noise(noise_list_2[-8:], from_layer=8, to_layer=15)
        image = self.g_synthesis(dlatents_in=source_latent,
                                 styled_latents=style_latent,
                                 style_threshold=9)
        image = (image+1.0)/2.0
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
        image = (image+1.0)/2.0
        style_transfer_path = os.path.join(
            self.results_dir, "style_transfer_results")
        os.makedirs(style_transfer_path, exist_ok=True)
        image_path = os.path.join(style_transfer_path, "transferred_image.png")
        save_image(image.clamp(0, 1), image_path)
        return

    def mix_latents_v2(self,
                       source_latent: torch.Tensor,
                       style_latent: torch.Tensor,
                       noise_list_1: Optional[List[torch.Tensor]],
                       noise_list_2: Optional[List[torch.Tensor]],
                       mix_threshold: float = 0.5):
        mixed_latent = mix_threshold * source_latent + \
            (1 - mix_threshold) * style_latent
        mixed_noise = [mix_threshold * n1 + (1 - mix_threshold) * n2
                       for n1, n2 in zip(noise_list_1, noise_list_2)]

        self.update_noise(mixed_noise)
        image = self.g_synthesis(mixed_latent)
        image = (image+1.0)/2.0
        style_transfer_path = os.path.join(
            self.results_dir, "style_transfer_results")
        os.makedirs(style_transfer_path, exist_ok=True)
        image_path = os.path.join(
            style_transfer_path, f"transferred_image_{mix_threshold}.png")
        save_image(image.clamp(0, 1), image_path)
        return

    def generate(self,
                 latent: torch.Tensor,
                 noise_list: Optional[List[torch.Tensor]] = None):
        if noise_list is not None:
            self.update_noise(noise_list)
        image = self.g_synthesis(latent)
        image = (image+1.0)/2.0
        return image

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
