from argparse import ArgumentParser
from shared.enums import DynamicSegmentationStrategy, SegmentationStrategy
from models.DenseNetPretrained import DenseNetPretrained
from models.InceptionV3Pretrained import InceptionV3Pretrained
from models.ViTEfficient import Conv2d_BN as ViTEfficient
from models.ViTPretrained import ViT_pretrained as ViTPretrained
from models.ViTStandard import ViT_standard as ViTStandard

# TODO: work in progress

hparams_space = {}


def main():
    parser = ArgumentParser()
    parser.add_argument("--segmentation", type=SegmentationStrategy,
                        default=SegmentationStrategy.NO_SEGMENTATION)
    parser.add_argument("--model", type=str)
    parser.add_argument("--limit", type=int, default=None)

    print("Starting training...")
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
