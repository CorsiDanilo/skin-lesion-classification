"""
-------------------------------------------------
   File Name:    Losses.py
   Author:       Zhonghao Huang
   Date:         2019/10/21
   Description:  Module implementing various loss functions
                 Copy from: https://github.com/akanimax/pro_gan_pytorch
-------------------------------------------------
"""

from torchvision.models import vgg16, VGG16_Weights
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

# =============================================================
# Interface for the losses
# =============================================================


class GANLoss:
    """ Base class for all losses

        @args:
        dis: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.criterion = BCEWithLogitsLoss()
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha, labels_in=labels)
        f_preds = self.dis(fake_samps, height, alpha, labels_in=labels)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        preds = self.dis(fake_samps, height, alpha, labels_in=labels)
        return self.criterion(torch.squeeze(preds),
                              torch.ones(fake_samps.shape[0]).to(fake_samps.device))


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, dis):

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, height, alpha):
        preds, _, _ = self.dis(fake_samps, height, alpha)
        return self.criterion(torch.squeeze(preds),
                              torch.ones(fake_samps.shape[0]).to(fake_samps.device))


class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = (torch.mean(nn.ReLU()(1 - r_preds)) +
                torch.mean(nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        return -torch.mean(self.dis(fake_samps, height, alpha))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(nn.ReLU()(1 - r_f_diff))
                + torch.mean(nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(nn.ReLU()(1 + r_f_diff))
                + torch.mean(nn.ReLU()(1 - f_r_diff)))


class LogisticGAN(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    # gradient penalty
    def R1Penalty(self, real_img, height, alpha):

        # TODO: use_loss_scaling, for fp16
        def apply_loss_scaling(x): return x * torch.exp(x *
                                                        torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        def undo_loss_scaling(x): return x * torch.exp(-x *
                                                       torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit = self.dis(real_img, height, alpha)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(
                                             real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha, r1_gamma=10.0):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = torch.mean(nn.Softplus()(f_preds)) + \
            torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(
                real_samps.detach(), height, alpha) * (r1_gamma * 0.5)
            loss += r1_penalty

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        f_preds = self.dis(fake_samps, height, alpha)

        return torch.mean(nn.Softplus()(-f_preds))


# =============================================================
# Losses for Images2StyleGan
# =============================================================


class VGG16PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16PerceptualLoss, self).__init__()
        vgg_pretrained_features = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.MSE_loss = nn.MSELoss(reduction="mean")
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu1_2 = h
        h = self.slice3(h)
        h_relu3_2 = h
        h = self.slice4(h)
        h_relu4_2 = h
        return h_relu1_1, h_relu1_2, h_relu3_2, h_relu4_2

    def loss_function(self, syn_img, img, img_p, upsample):
        # UpSample synthesized image to match the input size of VGG-16 input.
        # Extract mid level features for real and synthesized image and find the MSE loss between them for perceptual loss.
        # Find MSE loss between the real and synthesized images of actual size
        syn_img_p = upsample(syn_img)
        # syn_img_p = syn_img
        syn0, syn1, syn2, syn3 = self.forward(syn_img_p)
        r0, r1, r2, r3 = self.forward(img_p)
        mse = self.MSE_loss(syn_img, img)

        per_loss = 0
        per_loss += self.MSE_loss(syn0, r0)
        per_loss += self.MSE_loss(syn1, r1)
        per_loss += self.MSE_loss(syn2, r2)
        per_loss += self.MSE_loss(syn3, r3)

        return mse, per_loss


def W_loss(syn_img, img, MSE_loss, upsample, perceptual, lamb_p, lamb_mse, M_p=None, M_m=None):
    '''
    For W_l loss
    '''
    # adding mask on image
    if M_m is not None:
        mse = MSE_loss(M_m * syn_img, M_m * img)
    else:
        mse = MSE_loss(syn_img, img)

    if M_p is not None:
        syn_img_p = upsample(M_p * syn_img)
        img_p = upsample(M_p * img)
    else:
        syn_img_p = upsample(syn_img)
        img_p = upsample(img)

    syn0, syn1, syn2, syn3 = perceptual(syn_img_p)
    r0, r1, r2, r3 = perceptual(img_p)

    per_loss = 0
    per_loss += MSE_loss(syn0, r0)
    per_loss += MSE_loss(syn1, r1)
    per_loss += MSE_loss(syn2, r2)
    per_loss += MSE_loss(syn3, r3)

    loss = lamb_p * per_loss + lamb_mse * mse

    return loss


def Mkn_loss(syn_image, image1, image2, MSE_loss, lamd_mse1, lamb_mse2, M=None):
    '''
        For noise optimization loss
    '''
    if M is not None:
        syn_image1 = M * syn_image
        syn_image2 = (1-M) * syn_image
        image1 = M * image1
        image2 = (1-M) * image2
    else:
        syn_image1 = syn_image
        syn_image2 = syn_image
        image1 = image1
        image2 = image2

    mse = lamd_mse1 * MSE_loss(syn_image1, image1) + \
        lamb_mse2 * MSE_loss(syn_image2, image2)
    return mse
