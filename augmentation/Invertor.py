import os
import torch
from augmentation.GAN import GSynthesis, Generator, Resnet50Styles
from torchvision.utils import save_image
from augmentation.Losses import VGG16PerceptualLoss
from torch.optim import Adam
from utils.utils import select_device
from tqdm import tqdm
from math import log10


class Invertor():
    def __init__(self, cfg, depth):
        self.cfg = cfg
        self.depth = depth
        self.device = select_device()
        self.gen = Generator(num_channels=3,
                             dlatent_size=512,
                             resolution=cfg.dataset.resolution,
                             structure="linear",
                             conditional=False,
                             **cfg.model.gen).to(self.device)
        self.gen.load_checkpoints(os.path.join(
            "checkpoints", "stylegan_ffhq_1024_gen.pth"))
        self.g_synthesis = self.gen.g_synthesis

    def psnr(self, mse, flag=0):
        # flag = 0 if a single image is used and 1 if loss for a batch of images is to be calculated
        if flag == 0:
            psnr = 10 * log10(1 / mse.item())
        return psnr

    def train(self, image, random_style=True):
        upsample = torch.nn.Upsample(scale_factor=256/1024, mode='bilinear')
        img_p = image.clone()
        img_p = upsample(img_p)
        # Perceptual loss initialise object
        perceptual = VGG16PerceptualLoss().to(self.device)

        # since the synthesis network expects 18 w vectors of size 1xlatent_size thus we take latent vector of the same size
        if random_style:
            latents = torch.zeros(
                (1, 18, 512), requires_grad=True, device=self.device)
        else:
            self.resnet50 = Resnet50Styles().to(self.device)
            styles = self.resnet50(image)
            styles_2 = styles.clone()
            random_styles = torch.randn(
                1, 4, self.resnet50.latent_size, requires_grad=True).to(self.device)
            latents = torch.cat(
                [styles, random_styles, random_styles], dim=1).to(self.device)

            latents = latents.detach().requires_grad_()

        # Optimizer to change latent code in each backward step
        optimizer = Adam(
            {latents}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        # Loop to optimise latent vector to match the ted image to input image
        loss_ = []
        loss_psnr = []
        iterations = 10_000
        pbar = tqdm(total=iterations)
        for e in range(iterations):
            optimizer.zero_grad()
            syn_img = self.g_synthesis(
                dlatents_in=latents, depth=self.depth)
            syn_img = (syn_img+1.0)/2.0
            mse, per_loss = perceptual.loss_function(
                syn_img=syn_img,
                img=image,
                img_p=img_p,
                upsample=upsample)
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
            if (e+1) % 100 == 0:
                print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(e +
                                                                                                1, loss_np, loss_m, loss_p, psnr))
                save_image(syn_img.clamp(0, 1), "syn_img{}.png".format(e+1))
                save_image(image.clamp(0, 1), "original_img{}.png".format(e+1))
        return latents

    def train_hierarchical(self, image):
        upsample = torch.nn.Upsample(scale_factor=256/1024, mode='bilinear')
        img_p = image.clone()
        img_p = upsample(img_p)

        # Perceptual loss initialise object
        perceptual = VGG16PerceptualLoss().to(self.device)
        # since the synthesis network expects 18 w vectors of size 1x512 thus we take latent vector of the same size
        latent_w = torch.zeros(
            (1, 512), requires_grad=True, device=self.device)

        # Optimizer to change latent code in each backward step
        optimizer = Adam({latent_w}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        # Loop to optimise latent vector to match the generated image to input image
        loss_ = []
        loss_psnr = []
        pbar = tqdm(total=1000)
        for e in range(1000):
            optimizer.zero_grad()
            latent_w1 = latent_w.unsqueeze(1).expand(-1, 18, -1)
            syn_img = self.g_synthesis(
                dlatents_in=latent_w1, depth=self.depth)
            syn_img = (syn_img+1.0)/2.0
            mse, per_loss = perceptual.loss_function(
                syn_img=syn_img,
                img=image,
                img_p=img_p,
                upsample=upsample)
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
                loss=loss.item(),
                psnr=psnr)
            if (e+1) % 500 == 0:
                print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(
                    e+1, loss_np, loss_m, loss_p, psnr))
                save_image(syn_img.clamp(0, 1),
                           "Hier_pass_morphP1-syn-{}.png".format(e+1))
                save_image(image.clamp(0, 1),
                           "Hier_pass_morphP1-original-{}.png".format(e+1))

        latent_w1 = latent_w.unsqueeze(1).expand(-1, 18, -1)
        latent_w1 = torch.tensor(latent_w1, requires_grad=True)
        optimizer = Adam({latent_w1}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        pbar = tqdm(total=1000)
        for e in range(1000):
            optimizer.zero_grad()
            syn_img = self.g_synthesis(
                dlatents_in=latent_w1, depth=self.depth)
            syn_img = (syn_img+1.0)/2.0
            mse, per_loss = perceptual.loss_function(
                syn_img, image, img_p, upsample)
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
                loss=loss.item(),
                psnr=psnr)
            if (e+1) % 500 == 0:
                print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(
                    e+1, loss_np, loss_m, loss_p, psnr))
                save_image(syn_img.clamp(0, 1),
                           "Hier_pass_morphP2-syn-{}.png".format(e+1))
                save_image(image.clamp(0, 1),
                           "Hier_pass_morphP2-original-{}.png".format(e+1))

        return latent_w1
