from yacs.config import CfgNode as CN

cfg = CN()

cfg.output_dir = 'augmentation'
cfg.device = 'mps'
cfg.device_id = '0'

cfg.structure = 'linear'
cfg.conditional = False
cfg.n_classes = 7
cfg.loss = "logistic"
cfg.drift = 0.001
cfg.d_repeats = 1
cfg.use_ema = True
cfg.ema_decay = 0.999

cfg.num_works = 4
cfg.num_samples = 36
cfg.feedback_factor = 10
cfg.checkpoint_factor = 10

# ---------------------------------------------------------------------------- #
# Options for scheduler
# ---------------------------------------------------------------------------- #
cfg.sched = CN()

# example for {depth:9,resolution:1024}
DEPTH = 7
# cfg.sched.epochs = [2 for _ in range(DEPTH - 1)]
# # batches for oen 1080Ti with 11G memory
# cfg.sched.batch_sizes = [int(256 / 2**i) for i in range(len(cfg.sched.epochs))]

cfg.sched.epochs = [4, 4, 4, 4, 8, 16, 32, 32, 64]
# batches for oen 1080Ti with 11G memory
cfg.sched.batch_sizes = [64, 64, 32, 32, 16, 8, 4, 2, 2]
cfg.sched.fade_in_percentage = [50 for _ in range(len(cfg.sched.epochs))]

# TODO
# cfg.sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
# cfg.sched.D_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.img_dir = ""
cfg.dataset.folder = True
cfg.dataset.resolution = 1024
cfg.dataset.channels = 3

cfg.model = CN()
# ---------------------------------------------------------------------------- #
# Options for Generator
# ---------------------------------------------------------------------------- #
cfg.model.gen = CN()
cfg.model.gen.latent_size = 256
# 8 in original paper
cfg.model.gen.mapping_layers = 8
cfg.model.gen.blur_filter = [1, 2, 1]
cfg.model.gen.truncation_psi = 0.7
cfg.model.gen.truncation_cutoff = 8

# ---------------------------------------------------------------------------- #
# Options for Discriminator
# ---------------------------------------------------------------------------- #
cfg.model.dis = CN()
cfg.model.dis.use_wscale = True
cfg.model.dis.blur_filter = [1, 2, 1]

# ---------------------------------------------------------------------------- #
# Options for Generator Optimizer
# ---------------------------------------------------------------------------- #
cfg.model.g_optim = CN()
cfg.model.g_optim.learning_rate = 0.0015
cfg.model.g_optim.beta_1 = 0
cfg.model.g_optim.beta_2 = 0.99
cfg.model.g_optim.eps = 1e-8

# ---------------------------------------------------------------------------- #
# Options for Discriminator Optimizer
# ---------------------------------------------------------------------------- #
cfg.model.d_optim = CN()
cfg.model.d_optim.learning_rate = 0.0015
cfg.model.d_optim.beta_1 = 0
cfg.model.d_optim.beta_2 = 0.99
cfg.model.d_optim.eps = 1e-8
