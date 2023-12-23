import torch

IMAGENET_STATISTICS = tuple((torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([
            0.229, 0.224, 0.225]).view(3, 1, 1)))
DEFAULT_STATISTICS = tuple((torch.tensor([0.7822, 0.5384, 0.5595]).view(3, 1, 1), torch.tensor([
            0.1358, 0.1510, 0.1669]).view(3, 1, 1)))