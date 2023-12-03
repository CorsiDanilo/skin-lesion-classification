from argparse import ArgumentParser
import config
import wandb


def main():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=config.N_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--use_wandb', type=bool, default=config.USE_WANDB)

    print("Starting training...")
    args = parser.parse_args()
    print(args)


def setup_wandb(args):
    wandb.init(
        project="melanoma",

        # track hyperparameters and run metadata
        config={
            "task": "segmentation",
            "learning_rate": args.lr,
            "architecture": args.architecture,
            "epochs": args.epochs,
            'reg': args.reg,
            "dataset": "HAM10K",
            "optimizer": "AdamW",
            "dataset_limit": args.dataset_limit,
            "normalize": args.normalize,
            "resumed": args.resume,
            "from_epoch": args.from_epoch,
            "balance_undersampling": args.balance_undersampling
        },
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
