import copy
import datetime
import os
import random
import time
import timeit
from typing import Literal, Optional, Tuple
import warnings
from collections import OrderedDict

from config import NUM_CLASSES
from utils.utils import select_device

from .gan_config import cfg
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn.modules.sparse import Embedding
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
from . import Losses
from .gan_utils import update_average
from .Blocks import DiscriminatorBlock, DiscriminatorTop, GSynthesisBlock, InputBlock
from .CustomLayers import EqualizedConv2d, EqualizedLinear, PixelNormLayer, Truncation
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from shared.constants import IMAGENET_STATISTICS


class GMapping(nn.Module):

    def __init__(self,
                 latent_size: int = 512,
                 dlatent_size: int = 512,
                 dlatent_broadcast: Optional[torch.Tensor] = None,
                 mapping_layers: int = 8,
                 mapping_fmaps: int = 512,
                 mapping_lrmul: float = 0.01,
                 mapping_nonlinearity: str = 'lrelu',
                 use_wscale: bool = True,
                 normalize_latents: bool = True):
        """
        Mapping network used in the StyleGAN paper.

        :param latent_size: Latent vector(Z) dimensionality.
        # :param label_size: Label dimensionality, 0 if no labels.
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param dlatent_broadcast: Output disentangled latent (W) as [minibatch, dlatent_size]
                                  or [minibatch, dlatent_broadcast, dlatent_size].
        :param mapping_layers: Number of mapping layers. (8 in the StyleGAN paper)
        :param mapping_fmaps: Number of activations in the mapping layers.
        :param mapping_lrmul: Learning rate multiplier for the mapping layers.
        :param mapping_nonlinearity: Activation function: 'relu', 'lrelu'.
        :param use_wscale: Enable equalized learning rate?
        :param normalize_latents: Normalize latent vectors (Z) before feeding them to the mapping layers?
        """

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        # Activation function.
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[mapping_nonlinearity]

        # Embed labels and concatenate them with latents.
        # TODO

        layers = []
        # Normalize latents.
        if normalize_latents:
            layers.append(('pixel_norm', PixelNormLayer()))

        layers.append(('dense0', EqualizedLinear(self.latent_size, self.mapping_fmaps,
                                                 gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(
                ('dense{:d}'.format(layer_idx),
                 EqualizedLinear(fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))

        # Output.
        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # First input: Latent vectors (Z) [mini_batch, latent_size].
        x = self.map(x)

        # print(f"x shape after g mapping is {x.shape}")

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(nn.Module):

    def __init__(self,
                 dlatent_size: int,
                 fmap_max: int,
                 num_channels: int = 3,
                 resolution: int = 256,
                 fmap_base: int = 8192,
                 fmap_decay: float = 1.0,
                 use_styles: bool = True,
                 const_input_layer: bool = True,
                 use_noise: bool = True,
                 nonlinearity: str = 'lrelu',
                 use_wscale: bool = True,
                 use_pixel_norm: bool = False,
                 use_instance_norm: bool = True,
                 blur_filter: bool = None,
                 structure: str = 'linear',):
        """
        Synthesis network used in the StyleGAN paper.

        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param num_channels: Number of output color channels.
        :param resolution: Output resolution.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param use_styles: Enable style inputs?
        :param const_input_layer: First layer is a learned constant?
        :param use_noise: Enable noise inputs?
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param use_pixel_norm: Enable pixel_wise feature vector normalization?
        :param use_instance_norm: Enable instance normalization?
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        """

        super().__init__()

        # if blur_filter is None:
        #     blur_filter = [1, 2, 1]

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure

        # 8 if resolution is 256
        # 10 if resolution is 1024
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        # 14 layers if resolution is 256
        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # Early layers.
        self.init_block = InputBlock(nf(1), dlatent_size, const_input_layer, gain, use_wscale,
                                     use_noise, use_pixel_norm, use_instance_norm, use_styles, act)
        # create the ToRGB layers for various outputs
        rgb_converters = [EqualizedConv2d(
            nf(1), num_channels, 1, gain=1, use_wscale=use_wscale)]

        # Building blocks for remaining layers.
        blocks = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale,
                                          use_noise, use_pixel_norm, use_instance_norm, use_styles, act))
            rgb_converters.append(EqualizedConv2d(
                channels, num_channels, 1, gain=1, use_wscale=use_wscale))

        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(rgb_converters)

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, dlatents_in, depth=0, alpha=0., labels_in=None):
        """
            forward pass of the Generator
            :param dlatents_in: Input: Disentangled latents (W) [mini_batch, num_layers, dlatent_size].
            :param labels_in:
            :param depth: current depth from where output is required
            :param alpha: value of alpha for fade-in effect
            :return: y => output
        """
        # assert dlatents_in.shape[1] == 18
        assert depth < self.depth, "Requested output depth cannot be produced"

        # d_latents shape is torch.Size([1, 18, 256])
        if self.structure == 'fixed':
            # In the 'fixed' structure, the generator processes the input latent vectors through all the blocks of the generator network sequentially.
            # It doesn't consider the depth parameter or the alpha parameter for any fade-in effect.
            # This structure is typically used when the generator is fully trained and you're not growing the network any further.
            x = self.init_block(dlatents_in[:, 0:2])
            for i, block in enumerate(self.blocks):
                x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])
            images_out = self.to_rgb[-1](x)
        elif self.structure == 'linear':
            # In the 'linear' structure, the generator uses a progressive growing technique where the depth of the network can be increased over time.
            # This is done by adding new layers to the network as training progresses to generate higher resolution images.
            # The 'alpha' parameter is used for a fade-in effect where the output from the new layer is mixed with the upscaled output from the previous layer.
            # This helps in smooth transition as the network depth increases. This structure is typically used during the training phase when you're growing the
            # network progressively.
            x = self.init_block(dlatents_in[:, 0:2])

            if depth > 0:
                for i, block in enumerate(self.blocks[:depth - 1]):
                    x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])

                residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
                straight = self.to_rgb[depth](
                    self.blocks[depth - 1](x, dlatents_in[:, 2 * depth:2 * (depth + 1)]))

                images_out = (alpha * straight) + ((1 - alpha) * residual)
            else:
                images_out = self.to_rgb[0](x)
        else:
            raise KeyError("Unknown structure: ", self.structure)
        # print(f"Images out shape is {images_out.shape}")
        return images_out


class Generator(nn.Module):

    def __init__(self,
                 resolution: int,
                 latent_size: int,
                 dlatent_size: int,
                 conditional: bool = False,
                 n_classes: int = 0,
                 truncation_psi: float = 0.7,
                 truncation_cutoff: int = 8,
                 dlatent_avg_beta: float = 0.995,
                 style_mixing_prob: float = 0.9):
        """
        # Style-based generator used in the StyleGAN paper.
        # Composed of two sub-networks (G_mapping and G_synthesis).

        :param resolution:
        :param latent_size:
        :param dlatent_size:
        :param truncation_psi: Style strength multiplier for the truncation trick. None = disable.
        :param truncation_cutoff: Number of layers for which to apply the truncation trick. None = disable.
        :param dlatent_avg_beta: Decay for tracking the moving average of W during training. None = disable.
        :param style_mixing_prob: Probability of mixing styles during training. None = disable.
        """

        super(Generator, self).__init__()

        assert latent_size == dlatent_size
        if conditional:
            assert n_classes > 0, "Conditional generation requires n_class > 0"
            self.class_embedding = nn.Embedding(n_classes, latent_size)
            latent_size *= 2

        self.conditional = conditional
        self.style_mixing_prob = style_mixing_prob

        # Setup components.

        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(
            latent_size,
            dlatent_size,
            # dlatent_broadcast=self.num_layers
        )

        self.g_synthesis = GSynthesis(
            resolution=resolution,
            fmap_max=dlatent_size,
            dlatent_size=dlatent_size)

        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=torch.zeros(dlatent_size),
                                         max_layer=truncation_cutoff,
                                         threshold=truncation_psi,
                                         beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self,
                latents_in: torch.Tensor,
                depth: int,
                alpha: float,
                labels_in: Optional[torch.Tensor] = None):
        """
        :param latents_in: First input: Latent vectors (Z) [mini_batch, latent_size].
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :param labels_in: Second input: Conditioning labels [mini_batch, label_size].
        :return:
        """

        if not self.conditional:
            if labels_in is not None:
                warnings.warn(
                    "Generator is unconditional, labels_in will be ignored")
        else:
            assert labels_in is not None, "Conditional discriminatin requires labels"
            embedding = self.class_embedding(labels_in)

            latents_in = latents_in.squeeze(0)

            latents_in = torch.cat([latents_in, embedding], 1)

        dlatents_in = self.g_mapping(latents_in)
        assert dlatents_in.shape == latents_in.shape
        # print(f"Latents in shape is {latents_in.shape}")
        if self.training:
            # Update moving average of W(dlatent).
            # TODO
            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # Perform style mixing regularization.
            # TODO: TODO: TODO: skip for now because of simplicity, make sure to add it later
            # if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
            #     latents2 = torch.randn(latents_in.shape).to(latents_in.device)
            #     dlatents2 = self.g_mapping(latents2)
            #     layer_idx = torch.from_numpy(np.arange(self.num_layers)[np.newaxis, :, np.newaxis]).to(
            #         latents_in.device)
            #     cur_layers = 2 * (depth + 1)
            #     mixing_cutoff = random.randint(1,
            #                                    cur_layers) if random.random() < self.style_mixing_prob else cur_layers
            #     dlatents_in = torch.where(
            #         layer_idx < mixing_cutoff, dlatents_in, dlatents2)

            # Apply truncation trick.
            if self.truncation is not None:
                # print(f"dlatents_in shape is {dlatents_in.shape}")
                if dlatents_in.dim() == 2:
                    dlatents_in = dlatents_in.unsqueeze(0)
                dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, depth, alpha)

        return fake_images


class Discriminator(nn.Module):

    def __init__(self,
                 resolution: int,
                 num_channels: int = 3,
                 conditional: bool = False,
                 n_classes: int = 0,
                 fmap_base: int = 8192,
                 fmap_decay: float = 1.0,
                 fmap_max: int = 512,
                 nonlinearity: Literal["lrelu", "relu"] = 'lrelu',
                 use_wscale: bool = True,
                 mbstd_group_size: int = 4,
                 mbstd_num_features: int = 1,
                 blur_filter: bool = None,
                 structure: Literal["linear", "fixed"] = 'linear'):
        """
        Discriminator used in the StyleGAN paper.

        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param resolution: Input resolution. Overridden based on dataset.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param mbstd_group_size: Group size for the mini_batch standard deviation layer, 0 = disable.
        :param mbstd_num_features: Number of features for the mini_batch standard deviation layer.
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param kwargs: Ignore unrecognized keyword args.
        """
        super(Discriminator, self).__init__()

        if conditional:
            assert n_classes > 0, "Conditional Discriminator requires n_class > 0"
            # self.embedding = nn.Embedding(n_classes, num_channels * resolution ** 2)
            num_channels *= 2
            self.embeddings = []

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.conditional = conditional
        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure
        # if blur_filter is None:
        #     blur_filter = [1, 2, 1]

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # create the remaining layers
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -1):
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 2),
                                             gain=gain, use_wscale=use_wscale, activation_layer=act,
                                             blur_kernel=blur_filter))
            # create the fromRGB layers for various inputs:
            from_rgb.append(EqualizedConv2d(num_channels, nf(res - 1), kernel_size=1,
                                            gain=gain, use_wscale=use_wscale))
            # Create embeddings for various inputs:
            if conditional:
                r = 2 ** (res)
                self.embeddings.append(
                    Embedding(n_classes, (num_channels // 2) * r * r))

        if self.conditional:
            self.embeddings.append(nn.Embedding(
                n_classes, (num_channels // 2) * 4 * 4))
            self.embeddings = nn.ModuleList(self.embeddings)

        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                            in_channels=nf(2), intermediate_channels=nf(2),
                                            gain=gain, use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(EqualizedConv2d(num_channels, nf(2), kernel_size=1,
                                        gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(self, images_in, depth, alpha=1., labels_in=None):
        """
        :param images_in: First input: Images [mini_batch, channel, height, width].
        :param labels_in: Second input: Labels [mini_batch, label_size].
        :param depth: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return:
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.conditional:
            assert labels_in is not None, "Conditional Discriminator requires labels"
        # print(embedding_in.shape, images_in.shape)
        # exit(0)
        # print(self.embeddings)
        # exit(0)
        if self.structure == 'fixed':
            if self.conditional:
                embedding_in = self.embeddings[0](labels_in)
                embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                 images_in.shape[2],
                                                 images_in.shape[3])
                images_in = torch.cat([images_in, embedding_in], dim=1)
            x = self.from_rgb[0](images_in)
            for i, block in enumerate(self.blocks):
                x = block(x)
            scores_out = self.final_block(x)

        elif self.structure == 'linear':
            if depth > 0:
                if self.conditional:
                    embedding_in = self.embeddings[self.depth -
                                                   depth - 1](labels_in)
                    embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                     images_in.shape[2],
                                                     images_in.shape[3])
                    images_in = torch.cat([images_in, embedding_in], dim=1)

                residual = self.from_rgb[self.depth -
                                         depth](self.temporaryDownsampler(images_in))
                straight = self.blocks[self.depth - depth -
                                       1](self.from_rgb[self.depth - depth - 1](images_in))
                x = (alpha * straight) + ((1 - alpha) * residual)

                for block in self.blocks[(self.depth - depth):]:
                    x = block(x)
            else:
                if self.conditional:
                    embedding_in = self.embeddings[-1](labels_in)
                    embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                     images_in.shape[2],
                                                     images_in.shape[3])
                    images_in = torch.cat([images_in, embedding_in], dim=1)
                x = self.from_rgb[-1](images_in)

            scores_out = self.final_block(x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return scores_out


class Resnet50Styles(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        # Global Average Pooling is obtained by AdaptiveAvgPool2d with output size = 1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.latent_size = 512  # styles would be 2 * latent_size
        self.linear64 = nn.Linear(256, self.latent_size)
        self.linear16 = nn.Linear(1024, self.latent_size)
        self.linear8 = nn.Linear(2048, self.latent_size)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x1 = self.resnet50.layer1(x)  # 64x64 feature maps
        x2 = self.resnet50.layer2(x1)  # 32x32 feature maps
        x3 = self.resnet50.layer3(x2)  # 16x16 feature maps
        x4 = self.resnet50.layer4(x3)  # 8x8 feature maps

        # Resnet 64x64 shape: torch.Size([4, 256, 64, 64])
        # Resnet 16x16 shape: torch.Size([4, 1024, 16, 16])                                                                                                                                                                                        | 0/1803 [00:00<?, ?it/s]
        # Resnet 8x8 shape: torch.Size([4, 2048, 8, 8])

        x1 = self.gap(x1)
        x3 = self.gap(x3)
        x4 = self.gap(x4)

        x1 = nn.Flatten()(x1)
        x3 = nn.Flatten()(x3)
        x4 = nn.Flatten()(x4)

        x1 = self.linear64(x1)
        x3 = self.linear16(x3)
        x4 = self.linear8(x4)

        x1 = x1.view(-1, 1, self.latent_size)
        x3 = x3.view(-1, 1, self.latent_size)
        x4 = x4.view(-1, 1, self.latent_size)

        ratio = [2, 2, 3]
        x1 = x1.repeat(1, ratio[0], 1)
        x3 = x3.repeat(1, ratio[1], 1)
        x4 = x4.repeat(1, ratio[2], 1)

        return torch.cat([x1, x3, x4], dim=1)


class StyleGAN:

    def __init__(self,
                 structure,
                 resolution,
                 num_channels,
                 latent_size,
                 conditional=False,
                 n_classes=0,
                 loss="relativistic-hinge",
                 drift=0.001,
                 d_repeats=1,
                 use_ema=False,
                 ema_decay=0.999):
        """
        Wrapper around the Generator and the Discriminator.

        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param resolution: Input resolution. Overridden based on dataset.
        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param latent_size: Latent size of the manifold used by the GAN
        :param g_args: Options for generator network.
        :param d_args: Options for discriminator network.
        :param g_opt_args: Options for generator optimizer.
        :param d_opt_args: Options for discriminator optimizer.
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                          "hinge", "standard-gan" or "relativistic-hinge"]
                     Or an instance of GANLoss
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param d_repeats: How many times the discriminator is trained per G iteration.
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param device: device to run the GAN on (GPU / CPU)
        """

        # state of the object
        assert structure in ['fixed', 'linear']

        # Check conditional validity
        if conditional:
            assert n_classes > 0, "Conditional GANs require n_classes > 0"
        self.structure = structure
        self.depth = int(np.log2(resolution)) - 1
        # print(f"Depth is {self.depth}")
        self.latent_size = latent_size
        self.device = select_device()
        self.d_repeats = d_repeats
        self.conditional = conditional
        self.n_classes = n_classes

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Create the Generator and the Discriminator
        self.gen = Generator(
            latent_size=latent_size,
            dlatent_size=latent_size,
            resolution=resolution,
            conditional=self.conditional,
            n_classes=self.n_classes).to(self.device)

        self.dis = Discriminator(
            num_channels=num_channels,
            resolution=resolution,
            structure=self.structure,
            conditional=self.conditional,
            n_classes=self.n_classes).to(self.device)

        self.resnet50 = Resnet50Styles().to(self.device)

        # if code is to be run on GPU, we can use DataParallel:
        # TODO

        # define the optimizers for the discriminator and generator
        self.__setup_gen_optim()
        self.__setup_dis_optim()

        # define the loss function used for training the GAN
        self.drift = drift
        self.loss = self.__setup_loss(loss)

        # Use of ema
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

        # TODO: review this
        # Total = 101
        # gen_freeze_point = 50
        # for i, (name, param) in enumerate(self.gen.named_parameters()):
        #     # if name.startswith('g_mapping'):
        #     #     continue
        #     param.requires_grad = False
        # # Total = 51
        # dis_freeze_point = 25
        # for i, param in enumerate(self.dis.parameters()):
        #     # if i < dis_freeze_point:
        #     #     param.requires_grad = False
        #     param.requires_grad = False

    def __setup_gen_optim(self):
        learning_rate = cfg.model.g_optim.learning_rate
        beta_1 = cfg.model.g_optim.beta_1
        beta_2 = cfg.model.g_optim.beta_2
        eps = cfg.model.g_optim.eps

        self.gen_optim = torch.optim.Adam(
            [*self.gen.parameters(), *self.resnet50.parameters()], lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_dis_optim(self):
        learning_rate = cfg.model.d_optim.learning_rate
        beta_1 = cfg.model.d_optim.beta_1
        beta_2 = cfg.model.d_optim.beta_2
        eps = cfg.model.d_optim.eps
        self.dis_optim = torch.optim.Adam(
            [*self.dis.parameters()], lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_loss(self, loss):
        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string

            if not self.conditional:
                assert loss in ["logistic", "hinge", "standard-gan",
                                "relativistic-hinge"], "Unknown loss function"
                if loss == "logistic":
                    loss_func = Losses.LogisticGAN(self.dis)
                elif loss == "hinge":
                    loss_func = Losses.HingeGAN(self.dis)
                if loss == "standard-gan":
                    loss_func = Losses.StandardGAN(self.dis)
                elif loss == "relativistic-hinge":
                    loss_func = Losses.RelativisticAverageHingeGAN(self.dis)
            else:
                assert loss in ["conditional-loss"]
                if loss == "conditional-loss":
                    loss_func = Losses.ConditionalGANLoss(self.dis)

        return loss_func

    def __progressive_down_sampling(self, real_batch, depth, alpha):
        """
        private helper for down_sampling the original images in order to facilitate the
        progressive growing of the layers.

        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fade-in alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        if self.structure == 'fixed':
            return real_batch

        # down_sample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(
                AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + \
            ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def optimize_discriminator(self, noise, real_batch, depth, alpha, labels=None):
        """
        performs one step of weight update on discriminator using the batch of data

        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """

        real_samples = self.__progressive_down_sampling(
            real_batch, depth, alpha)

        loss_val = 0
        for _ in range(self.d_repeats):
            # generate a batch of samples
            # print(
            # f"Inside optimize discriminator, noise shape is {noise.shape}")
            fake_samples = self.gen(noise, depth, alpha, labels).detach()

            if not self.conditional:
                loss = self.loss.dis_loss(
                    real_samples, fake_samples, depth, alpha)
            else:
                loss = self.loss.dis_loss(
                    real_samples, fake_samples, labels, depth, alpha)
            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.d_repeats

    def optimize_generator(self, noise, real_batch, depth, alpha, labels=None):
        """
        performs one step of weight update on generator for the given batch_size

        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        real_samples = self.__progressive_down_sampling(
            real_batch, depth, alpha)

        # generate fake samples:
        # print(f"Inside optimize generator, noise shape is {noise.shape}")
        fake_samples = self.gen(noise, depth, alpha, labels)

        # Change this implementation for making it compatible for relativisticGAN
        if not self.conditional:
            loss = self.loss.gen_loss(real_samples, fake_samples, depth, alpha)
        else:
            loss = self.loss.gen_loss(
                real_samples, fake_samples, labels, depth, alpha)

        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=10.)
        self.gen_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss.item()

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples

        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torch.nn.functional import interpolate
        from torchvision.utils import save_image

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
                   normalize=True, scale_each=True, pad_value=128, padding=1)

    def train(self, epochs, batch_sizes, fade_in_percentage, logger, output,
              num_samples=36, start_depth=0, feedback_factor=100, checkpoint_factor=1):
        """
        Utility method for training the GAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.

        :param dataset: object of the dataset used for training.
                        Note that this is not the data loader (we create data loader in this method
                        since the batch_sizes for resolutions can be different)
        :param num_workers: number of workers for reading the data. def=3
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param logger:
        :param output: Output dir for samples,models,and log.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param feedback_factor: number of logs per epoch. def=100
        :param checkpoint_factor:
        :return: None (Writes multiple files to disk)
        """

        assert self.depth <= len(epochs), "epochs not compatible with depth"
        assert self.depth <= len(
            batch_sizes), "batch_sizes not compatible with depth"
        assert self.depth <= len(
            fade_in_percentage), "fade_in_percentage not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_input = torch.randn(
            num_samples, 18, 512).to(self.device)

        fixed_labels = None
        if self.conditional:
            fixed_labels = torch.linspace(
                0, self.n_classes - 1, num_samples).to(torch.int64).to(self.device)
        # config depend on structure
        logger.info("Starting the training process ... \n")
        if self.structure == 'fixed':
            start_depth = self.depth - 1
        step = 1  # counter for number of iterations
        for current_depth in tqdm(range(start_depth, self.depth), desc="Depth loop"):
            current_res = np.power(2, current_depth + 2)
            logger.info("Currently working on depth: %d", current_depth + 1)
            logger.info("Current resolution: %d x %d" %
                        (current_res, current_res))
            logger.info(f"Current batch size is {batch_sizes[current_depth]}")

            ticker = 1

            dataloader = ImagesAndSegmentationDataLoader(
                limit=None,
                dynamic_load=True,
                upscale_train=False,
                normalize=True,
                normalization_statistics=IMAGENET_STATISTICS,
                batch_size=batch_sizes[current_depth],
                resize_dim=(
                    cfg.dataset.resolution,
                    cfg.dataset.resolution,
                )
            )
            data = dataloader.get_train_dataloder()
            for epoch in tqdm(range(1, epochs[current_depth] + 1), desc="Epoch loop at depth %d" % (current_depth + 1)):
                start = timeit.default_timer()  # record time at the start of epoch

                logger.info("Epoch: [%d]" % epoch)
                total_batches = len(iter(data))

                fade_point = int((fade_in_percentage[current_depth] / 100)
                                 * epochs[current_depth] * total_batches)

                for i, batch in enumerate(tqdm(data, desc="training"), 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fade_point if ticker <= fade_point else 1

                    # extract current batch of data for training
                    if self.conditional:
                        images, labels, _ = batch
                        labels = labels.to(self.device)
                    else:
                        images, labels, _ = batch

                    images = images.to(self.device)

                    # extract the desired features from resnet
                    first_image = images[0].unsqueeze(0)
                    second_image = images[1].unsqueeze(0)
                    self.resnet_style_1 = self.resnet50(first_image)
                    # TODO: first_image and second_image need to be from the same class
                    self.resnet_style_2 = self.resnet50(second_image)
                    # TODO: make this trainable
                    self.random_style = torch.randn(
                        1, 4, self.resnet50.latent_size).to(self.device)

                    assert self.resnet_style_1.shape == (
                        1, 7, self.resnet50.latent_size), f"Resnet style 1 is not of shape (7, 256), but {self.resnet_style_1.shape}"
                    assert self.resnet_style_2.shape == (
                        1, 7, self.resnet50.latent_size), f"Resnet style 2 is not of shape (7, 256), but {self.resnet_style_2.shape}"

                    # gan_input shape: torch.Size([1, 18, 256])
                    # og gan_input shape: torch.Size([4, 512])

                    # _og_gan_input = torch.randn(
                    #     images.shape[0], self.latent_size).to(self.device)

                    gan_input = torch.cat(
                        [self.resnet_style_1, self.resnet_style_2, self.random_style], dim=1)

                    gan_input = gan_input.repeat(images.shape[0], 1, 1)
                    # optimize the discriminator:
                    dis_loss = self.optimize_discriminator(
                        gan_input, images, current_depth, alpha, labels)

                    # optimize the generator:
                    gen_loss = self.optimize_generator(
                        gan_input, images, current_depth, alpha, labels)

                    # provide a loss feedback
                    if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(
                            seconds=elapsed)).split('.')[0]
                        logger.info(
                            f"Elapsed: [{elapsed}] Step: {step}  Batch: {i}  D_Loss: {dis_loss}  G_Loss: {gen_loss} avg_loss: {(dis_loss + gen_loss) / 2}")

                        # create a grid of samples and save it
                        os.makedirs(os.path.join(
                            output, 'samples'), exist_ok=True)
                        gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
                                                    + "_" + str(epoch) + "_" + str(i) + ".png")

                        with torch.no_grad():
                            # print(f"Fixed input shape is {fixed_input.shape}")
                            self.create_grid(
                                samples=self.gen(fixed_input, current_depth, alpha, labels_in=fixed_labels).detach() if not self.use_ema
                                else self.gen_shadow(fixed_input, current_depth, alpha, labels_in=fixed_labels).detach(),
                                scale_factor=int(
                                    np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1,
                                img_file=gen_img_file,
                            )

                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1

                elapsed = timeit.default_timer() - start
                elapsed = str(datetime.timedelta(
                    seconds=elapsed)).split('.')[0]
                logger.info("Time taken for epoch: %s\n" % elapsed)

                if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                    save_dir = os.path.join(output, 'models')
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(
                        save_dir, "GAN_GEN_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    dis_save_file = os.path.join(
                        save_dir, "GAN_DIS_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    gen_optim_save_file = os.path.join(
                        save_dir, "GAN_GEN_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    dis_optim_save_file = os.path.join(
                        save_dir, "GAN_DIS_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")

                    torch.save(self.gen.state_dict(), gen_save_file)
                    logger.info("Saving the model to: %s\n" % gen_save_file)
                    torch.save(self.dis.state_dict(), dis_save_file)
                    torch.save(self.gen_optim.state_dict(),
                               gen_optim_save_file)
                    torch.save(self.dis_optim.state_dict(),
                               dis_optim_save_file)

                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(
                            save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + "_" + str(epoch) + ".pth")
                        torch.save(self.gen_shadow.state_dict(),
                                   gen_shadow_save_file)
                        logger.info("Saving the model to: %s\n" %
                                    gen_shadow_save_file)

        logger.info('Training completed.\n')
