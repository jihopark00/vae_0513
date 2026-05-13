"""
DeTok: Reconstruction model training script.

Model / loss / optimization / seed args are loaded from a YAML config file
(see configs/default.yaml); remaining args are parsed from the CLI.
"""

import argparse
import datetime
import logging
import sys
import time

import torch
import torch.distributed
import yaml

import utils.distributed as distributed
from utils.builders import (
    create_loss_module,
    create_optimizer_and_scaler,
    create_reconstruction_model,
    create_train_dataloader,
    create_val_dataloader,
    create_vis_dataloader,
)
from utils.misc import ckpt_resume, save_checkpoint
from utils.train_utils import evaluate_tokenizer, setup, train_one_epoch_tokenizer, visualize_tokenizer

# performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logger = logging.getLogger("DeTok")


# keys loaded from YAML config; everything else stays as a CLI flag
# YAML_SECTIONS = ("model", "loss", "optimization")
# YAML_SCALARS = ("seed",)


def main(args: argparse.Namespace) -> int:
    global logger
    wandb_logger = setup(args)

    # initialize data loaders
    data_loader_train = create_train_dataloader(args)
    data_loader_val = create_val_dataloader(args)
    data_loader_vis = create_vis_dataloader(args)
    vis_iterator = iter(data_loader_vis)

    # initialize models and optimizers
    model, ema_model = create_reconstruction_model(args)
    if args.train_decoder_only and hasattr(model, "freeze_everything_but_decoder"):
        model.freeze_everything_but_decoder()

    optimizer, loss_scaler = create_optimizer_and_scaler(args, model)
    loss_fn = create_loss_module(args)
    discriminator_optimizer, discriminator_loss_scaler = create_optimizer_and_scaler(args, loss_fn)

    # setup distributed training
    if distributed.is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        loss_fn = torch.nn.parallel.DistributedDataParallel(loss_fn, find_unused_parameters=True)

    # get models without DDP wrapper
    model_wo_ddp = model.module if hasattr(model, "module") else model
    loss_module_wo_ddp = loss_fn.module if hasattr(loss_fn, "module") else loss_fn

    # resume from checkpoint if needed
    ckpt_resume(
        args, model_wo_ddp, optimizer, loss_scaler, ema_model,
        loss_module_wo_ddp, discriminator_optimizer, discriminator_loss_scaler
    )

    # initial visualization
    visualize_tokenizer(args, model_wo_ddp, ema_model, next(vis_iterator), args.start_epoch)

    if args.vis_only:
        return 0

    # evaluation-only mode
    if args.evaluate:
        torch.cuda.empty_cache()
        for use_ema in [False, True]:
            evaluate_tokenizer(
                args, model_wo_ddp, ema_model, data_loader_val, args.start_epoch, wandb_logger, use_ema
            )
        return 0

    # training loop
    logger.info(f"Start training from {args.start_epoch} to {args.epochs}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch_tokenizer(
            args, model, data_loader_train, optimizer, loss_scaler, wandb_logger, epoch,
            ema_model, loss_fn, discriminator_optimizer, discriminator_loss_scaler
        )

        # progress logging
        elapsed_t = time.time() - start_time + args.last_elapsed_time
        eta = elapsed_t / (epoch + 1) * (args.epochs - epoch - 1)
        logger.info(
            f"[{epoch}/{args.epochs}] "
            f"Accumulated elapsed time: {str(datetime.timedelta(seconds=int(elapsed_t)))}, "
            f"ETA: {str(datetime.timedelta(seconds=int(eta)))}"
        )

        # checkpointing
        should_save = (
            (epoch + 1) % args.save_freq == 0  # save every n epochs
            or (epoch + 1) == args.epochs  # save at the end of training
        )

        if should_save:
            save_checkpoint(
                args, epoch, model_wo_ddp, optimizer, loss_scaler, ema_model, elapsed_t,
                loss_module_wo_ddp, discriminator_optimizer, discriminator_loss_scaler
            )
            torch.distributed.barrier()

        # periodic visualization
        if (epoch + 1) % args.vis_freq == 0:
            visualize_tokenizer(args, model_wo_ddp, ema_model, next(vis_iterator), epoch)

        # online evaluation
        if (args.online_eval and (epoch + 1) % args.eval_freq == 0 and (epoch + 1) != args.epochs):
            torch.cuda.empty_cache()
            for use_ema in [False, True]:
                evaluate_tokenizer(
                    args, model_wo_ddp, ema_model, data_loader_val, epoch + 1, wandb_logger, use_ema
                )

    # final evaluation
    total_time = int(time.time() - start_time + args.last_elapsed_time)
    logger.info(f"Training time {str(datetime.timedelta(seconds=total_time))}")

    for use_ema in [False, True]:
        evaluate_tokenizer(args, model_wo_ddp, ema_model, data_loader_val, args.epochs, wandb_logger, use_ema)

    return 0


def load_yaml_into_args(args: argparse.Namespace, config_path: str) -> argparse.Namespace:
    """Load model/loss/optimization/seed entries from a YAML config onto args."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
        
    for k, v in cfg.items():
        setattr(args, k, v)

    # for section in YAML_SECTIONS:
    #     for k, v in (cfg.get(section) or {}).items():
    #         setattr(args, k, v)

    # system = cfg.get("system") or {}
    # for k in YAML_SCALARS:
    #     if k in system:
    #         setattr(args, k, system[k])
    #     elif k in cfg:
    #         setattr(args, k, cfg[k])

    return args


def get_args_parser():
    parser = argparse.ArgumentParser("Reconstruction model training", add_help=False)

    # YAML config (provides model / loss / optimization / seed)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # logging parameters
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--vis_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--last_elapsed_time", type=float, default=0.0)

    # checkpoint parameters
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", default=None, help="resume model weights and optimizer state")
    parser.add_argument("--load_from", type=str, default=None, help="load from pretrained model")
    parser.add_argument("--keep_n_ckpts", default=1, type=int, help="keep the last n checkpoints")
    parser.add_argument("--milestone_interval", default=100, type=int, help="keep checkpoints every n epochs")

    # evaluation parameters
    parser.add_argument("--num_images", default=50000, type=int, help="Number of images to evaluate on")
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--fid_stats_path", type=str, default="data/fid_stats/val_fid_statistics_file.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval_bsz", type=int, default=256)

    # dataset parameters
    parser.add_argument("--use_cached_tokens", action="store_true")
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279], type=int, nargs="+")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # wandb parameters
    parser.add_argument("--project", default="lDeTok", type=str)
    parser.add_argument("--entity", default="YOUR_WANDB_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_key", default=None, type=str, help="W&B API key for wandb.login")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args = load_yaml_into_args(args, args.config)
    exit_code = main(args)
    sys.exit(exit_code)
