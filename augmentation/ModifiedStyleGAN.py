import copy
import datetime
import os
import random
import time
import timeit
from typing import Optional
import warnings
from collections import OrderedDict

from .gan_config import cfg
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn.modules.sparse import Embedding
from tqdm import tqdm

from . import Losses
from .StyleGANPytorch.models import update_average
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

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(nn.Module):

    def __init__(self,
                 dlatent_size: int = 512,
                 num_channels: int = 3,
                 resolution: int = 1024,
                 fmap_base: int = 8192,
                 fmap_decay: float = 1.0,
                 fmap_max: int = 512,
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

        assert depth < self.depth, "Requested output depth cannot be produced"

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

        return images_out


class Generator(nn.Module):

    def __init__(self, resolution, latent_size=512, dlatent_size=512,
                 conditional=False, n_classes=0, truncation_psi=0.7,
                 truncation_cutoff=8, dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9, **kwargs):
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
        :param kwargs: Arguments for sub-networks (G_mapping and G_synthesis).
        """

        super(Generator, self).__init__()

        if conditional:
            assert n_classes > 0, "Conditional generation requires n_class > 0"
            self.class_embedding = nn.Embedding(n_classes, latent_size)
            latent_size *= 2

        self.conditional = conditional
        self.style_mixing_prob = style_mixing_prob

        # Setup components.
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(
            latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)

        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=torch.zeros(dlatent_size),
                                         max_layer=truncation_cutoff,
                                         threshold=truncation_psi,
                                         beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self, latents_in, depth, alpha, labels_in=None):
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
            latents_in = torch.cat([latents_in, embedding], 1)

        dlatents_in = self.g_mapping(latents_in)

        if self.training:
            # Update moving average of W(dlatent).
            # TODO
            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # Perform style mixing regularization.
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                latents2 = torch.randn(latents_in.shape).to(latents_in.device)
                dlatents2 = self.g_mapping(latents2)
                layer_idx = torch.from_numpy(np.arange(self.num_layers)[np.newaxis, :, np.newaxis]).to(
                    latents_in.device)
                cur_layers = 2 * (depth + 1)
                mixing_cutoff = random.randint(1,
                                               cur_layers) if random.random() < self.style_mixing_prob else cur_layers
                dlatents_in = torch.where(
                    layer_idx < mixing_cutoff, dlatents_in, dlatents2)

            # Apply truncation trick.
            if self.truncation is not None:
                dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, depth, alpha)

        return fake_images
