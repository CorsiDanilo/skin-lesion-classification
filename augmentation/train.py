import os
import torch

from utils.utils import select_device
from .GAN import StyleGAN
from .gan_utils import make_logger
from .gan_config import cfg as opt


def main():
    logger = make_logger("project", opt.output_dir, 'log')

    style_gan = StyleGAN(structure=opt.structure,
                         conditional=opt.conditional,
                         n_classes=opt.n_classes,
                         resolution=opt.dataset.resolution,
                         num_channels=opt.dataset.channels,
                         latent_size=opt.model.gen.latent_size,
                         g_args=opt.model.gen,
                         d_args=opt.model.dis,
                         g_opt_args=opt.model.g_optim,
                         d_opt_args=opt.model.d_optim,
                         loss=opt.loss,
                         drift=opt.drift,
                         d_repeats=opt.d_repeats,
                         use_ema=opt.use_ema,
                         ema_decay=opt.ema_decay,
                         device=select_device())

    CHECKPOINT_GEN = os.path.join("checkpoints", "stylegan_ffhq_1024_gen.pth")
    # Load the state dict from the checkpoint
    # style_gan.gen.load_state_dict(torch.load(CHECKPOINT_GEN))
    checkpoint_state_dict = torch.load(CHECKPOINT_GEN)

    # Get the state dict of the current model
    model_state_dict = style_gan.gen.state_dict()

    # Filter out the keys in the checkpoint state dict that are not in the model state dict or have a different size
    compatible_state_dict = {k: v for k, v in checkpoint_state_dict.items(
    ) if k in model_state_dict and v.size() == model_state_dict[k].size()}

    non_compatible_state_dict = {k: v for k, v in checkpoint_state_dict.items(
    ) if k not in model_state_dict or v.size() != model_state_dict[k].size()}

    print("Non compatible state dict keys: ", non_compatible_state_dict.keys())

    # Load the compatible state dict into the model
    style_gan.gen.load_state_dict(compatible_state_dict, strict=False)
    # if args.discriminator_file is not None:
    #     logger.info("Loading discriminator from: %s", args.discriminator_file)
    #     style_gan.dis.load_state_dict(torch.load(args.discriminator_file))

    # if args.gen_shadow_file is not None and opt.use_ema:
    #     logger.info("Loading shadow generator from: %s", args.gen_shadow_file)
    #     # style_gan.gen_shadow.load_state_dict(torch.load(args.gen_shadow_file))
    #     # Load fewer layers of pre-trained models if possible
    #     load(style_gan.gen_shadow, args.gen_shadow_file)

    # if args.gen_optim_file is not None:
    #     logger.info("Loading generator optimizer from: %s",
    #                 args.gen_optim_file)
    #     style_gan.gen_optim.load_state_dict(torch.load(args.gen_optim_file))

    # if args.dis_optim_file is not None:
    #     logger.info("Loading discriminator optimizer from: %s",
    #                 args.dis_optim_file)
    #     style_gan.dis_optim.load_state_dict(torch.load(args.dis_optim_file))

    # train the network
    style_gan.train(epochs=opt.sched.epochs,
                    batch_sizes=opt.sched.batch_sizes,
                    fade_in_percentage=opt.sched.fade_in_percentage,
                    logger=logger,
                    output=opt.output_dir,
                    num_samples=opt.num_samples,
                    start_depth=4,
                    feedback_factor=opt.feedback_factor,
                    checkpoint_factor=opt.checkpoint_factor)


if __name__ == "__main__":
    main()
