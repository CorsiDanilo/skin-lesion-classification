import torch
from torch import nn
from augmentation.CustomLayers import AdaIN, LayerEpilogue, StyleMod
from augmentation.ModifiedStyleGAN import GMapping


def test_GMapping():
    mapping_network = GMapping()
    latent_codes = torch.randn((32, 512, 512))
    Ws = mapping_network(latent_codes)
    assert Ws.shape == latent_codes.shape


def test_AdaIN():
    adain = AdaIN(latent_size=512,
                  channels=512,
                  use_wscale=True)
    w_vector = torch.randn((32, 512))
    previous_activation_map = torch.randn((32, 512, 16, 16))
    result = adain(previous_activation_map, w_vector)
    assert result.shape == previous_activation_map.shape


def test_LayerEpilogue():
    layer_epilogue = LayerEpilogue(channels=512,
                                   dlatent_size=512,
                                   use_wscale=True,
                                   use_noise=True,
                                   use_pixel_norm=True,
                                   use_instance_norm=True,
                                   use_styles=True,
                                   activation_layer=nn.LeakyReLU(0.2))
    x = torch.randn((32, 512, 16, 16))
    d_latents = torch.randn((32, 512))
    result = layer_epilogue(x, d_latents)
    assert result.shape == x.shape


if __name__ == '__main__':
    test_GMapping()
    test_AdaIN()
    test_LayerEpilogue()
