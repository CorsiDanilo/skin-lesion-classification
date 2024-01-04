from augmentation.StyleGAN import StyleGAN
from augmentation.StyleGANPytorch.utils.logger import make_logger
from config import BATCH_SIZE
from dataloaders.ImagesAndSegmentationDataLoader import ImagesAndSegmentationDataLoader
from shared.constants import IMAGENET_STATISTICS
from utils.utils import select_device
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
    # # Resume training from checkpoints
    # if args.generator_file is not None:
    #     logger.info("Loading generator from: %s", args.generator_file)
    #     # style_gan.gen.load_state_dict(torch.load(args.generator_file))
    #     # Load fewer layers of pre-trained models if possible
    #     load(style_gan.gen, args.generator_file)
    # else:
    #     logger.info("Training from scratch...")

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
                    start_depth=0,
                    feedback_factor=opt.feedback_factor,
                    checkpoint_factor=opt.checkpoint_factor)


if __name__ == "__main__":
    main()
