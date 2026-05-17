"""Joint SSL + Diffusion training.

Trains the DeTok autoencoder (``models/detok_ssldm.py``) and the LightningDiT
flow-matching diffusion model (``models/lightningdit_flow.py``) in a single
optimization loop:

    x --(encoder)--> z
    z, y --(diffusion)--> diffusion loss
    z --(decoder)--> x_pred --> reconstruction / LPIPS / GAN loss

A single posterior sample ``z`` is shared between the decoder and the
diffusion model; the diffusion gradient flows through the encoder.
"""

import argparse
import copy
import datetime
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as T
import yaml
from einops import rearrange
from PIL import Image
from torch import Tensor

import models
import utils.distributed as distributed
import utils.misc as misc
from utils.builders import (
    create_loss_module,
    create_optimizer_and_scaler,
    create_train_dataloader,
    create_val_dataloader,
    create_vis_dataloader,
)
from utils.logger import MetricLogger, SmoothedValue, WandbLogger
from utils.misc import NativeScalerWithGradNormCount, cleanup_checkpoints
from utils.train_utils import (
    evaluate_generator,
    evaluate_tokenizer,
    setup,
    visualize_generator,
    visualize_tokenizer,
)

# performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logger = logging.getLogger("DeTok")


# =============================================================================
# HuggingFace dataset wrappers (mnist / cifar10)
# =============================================================================

# (repo_id, image_key, train_split, val_split, num_classes, default_flip)
_HF_DATASETS = {
    "mnist":   ("ylecun/mnist",     "image", "train", "test", 10, False),
    "cifar10": ("uoft-cs/cifar10",  "img",   "train", "test", 10, True),
}


class HFImageDataset(torch.utils.data.Dataset):
    """Wraps a Hugging Face image dataset to match the (img, label, index) dict
    format used by the rest of the trainer.

    Images are converted to RGB, resized + center-cropped to ``img_size``, and
    normalized to ``[-1, 1]`` (matching ListDataset's preprocessing). Random
    horizontal flip is applied for training when the dataset allows it (CIFAR-10
    yes, MNIST no).
    """

    def __init__(self, hf_split, img_size: int, img_key: str, flip: bool):
        self.split = hf_split
        self.img_key = img_key
        ops = [
            T.Lambda(lambda im: im.convert("RGB") if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im)).convert("RGB")),
            T.Resize(img_size, antialias=True),
            T.CenterCrop(img_size),
        ]
        if flip:
            ops.append(T.RandomHorizontalFlip(p=0.5))
        ops += [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        self.transform = T.Compose(ops)

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int):
        item = self.split[idx]
        img = self.transform(item[self.img_key])
        label = int(item["label"]) if "label" in item else 0
        return {"img": img, "label": label, "index": idx}


def _load_hf_split(name: str, split: str):
    import datasets as hf_datasets
    repo_id = _HF_DATASETS[name][0]
    logger.info(f"[HF dataset] loading {repo_id}:{split}")
    return hf_datasets.load_dataset(repo_id, split=split)


def _build_hf_loader(args, dataset, batch_size: int, shuffle: bool, drop_last: bool):
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=distributed.get_world_size(),
        rank=distributed.get_global_rank(),
        shuffle=shuffle,
    )
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
    )


def create_train_dataloader_hf(args):
    name = args.dataset
    repo_id, img_key, train_split, _, num_classes, default_flip = _HF_DATASETS[name]
    ds = HFImageDataset(_load_hf_split(name, train_split), args.img_size, img_key, flip=default_flip)
    logger.info(f"[HF dataset] train size: {len(ds)} (num_classes={num_classes})")
    return _build_hf_loader(args, ds, args.batch_size, shuffle=True, drop_last=True)


def create_val_dataloader_hf(args):
    name = args.dataset
    repo_id, img_key, _, val_split, _, _ = _HF_DATASETS[name]
    ds = HFImageDataset(_load_hf_split(name, val_split), args.img_size, img_key, flip=False)
    logger.info(f"[HF dataset] val size: {len(ds)}")
    return _build_hf_loader(args, ds, args.eval_bsz, shuffle=False, drop_last=False)


def create_vis_dataloader_hf(args):
    name = args.dataset
    repo_id, img_key, train_split, _, _, _ = _HF_DATASETS[name]
    ds = HFImageDataset(_load_hf_split(name, train_split), args.img_size, img_key, flip=False)
    logger.info(f"[HF dataset] vis size: {len(ds)}")
    return _build_hf_loader(args, ds, batch_size=8, shuffle=True, drop_last=False)


def adjust_args_for_dataset(args: argparse.Namespace) -> argparse.Namespace:
    """Auto-set ``num_classes`` and a sensible ``class_of_interest`` when using
    a Hugging Face dataset, since the YAML defaults assume ImageNet (1000 cls).
    """
    if args.dataset in _HF_DATASETS:
        _, _, _, _, num_classes, _ = _HF_DATASETS[args.dataset]
        if args.num_classes != num_classes:
            logger.info(f"[dataset={args.dataset}] overriding num_classes: {args.num_classes} -> {num_classes}")
            args.num_classes = num_classes
        # If the user is still on the default ImageNet class_of_interest, swap to
        # the first 8 (or fewer) classes of the new dataset.
        imagenet_default = [207, 360, 387, 974, 88, 979, 417, 279]
        if list(getattr(args, "class_of_interest", [])) == imagenet_default:
            args.class_of_interest = list(range(min(8, num_classes)))
            logger.info(f"[dataset={args.dataset}] class_of_interest -> {args.class_of_interest}")
    return args


# =============================================================================
# Joint model
# =============================================================================


class JointSSLDM(nn.Module):
    """Wraps an autoencoder and a diffusion model for joint training.

    forward(x, y) -> (decoded, posteriors, diffusion_loss)
    """

    def __init__(self, autoencoder: nn.Module, diffusion_model: nn.Module, seq_h: int):
        super().__init__()
        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model
        self.seq_h = seq_h

    def forward(self, x: Tensor, y: Tensor):
        z_latents, posteriors, ids_restore = self.autoencoder.encode(
            x, sampling=self.training, mask_ratio=0.0,
        )
        decoded = self.autoencoder.decoder(z_latents, ids_restore=ids_restore)
        z_for_diff = rearrange(z_latents, "b (h w) c -> b c h w", h=self.seq_h)
        diff_loss = self.diffusion_model(z_for_diff, y)
        return decoded, posteriors, diff_loss

    # ---- AE passthroughs (used by visualize_tokenizer / evaluate_tokenizer) --
    def tokenize(self, *a, **kw):
        return self.autoencoder.tokenize(*a, **kw)

    def detokenize(self, *a, **kw):
        return self.autoencoder.detokenize(*a, **kw)

    def reconstruct(self, *a, **kw):
        return self.autoencoder.reconstruct(*a, **kw)

    # ---- DM passthrough (used by visualize_generator / evaluate_generator) ---
    def generate(self, *a, **kw):
        return self.diffusion_model.generate(*a, **kw)


# =============================================================================
# Builders
# =============================================================================


def create_ssldm_model(args: argparse.Namespace):
    logger.info("Creating SSL+DM joint model.")
    if args.ae_model not in models.DeTok_SSLDM_models:
        raise ValueError(f"Unsupported ae_model {args.ae_model}")
    if args.dm_model not in models.LightningDiT_Flow_models:
        raise ValueError(f"Unsupported dm_model {args.dm_model}")

    autoencoder = models.DeTok_SSLDM_models[args.ae_model](
        img_size=args.img_size,
        patch_size=args.patch_size,
        token_channels=args.token_channels,
        mask_ratio=0.0,
        gamma=0.0,
        latent_norm=getattr(args, "latent_norm", None),
        fixed_std=getattr(args, "fixed_std", None),
    )
    diffusion_model = models.LightningDiT_Flow_models[args.dm_model](
        img_size=args.img_size,
        patch_size=args.dm_patch_size,
        tokenizer_patch_size=args.patch_size,
        token_channels=args.token_channels,
        label_drop_prob=args.label_drop_prob,
        num_classes=args.num_classes,
        num_sampling_steps=args.num_sampling_steps,
        grad_checkpointing=args.grad_checkpointing,
        qk_norm=args.qk_norm,
        force_one_d_seq=0,
        prediction=args.prediction,
        loss_target=args.loss_target,
        detach_zt=args.detach_zt,
        detach_target=args.detach_target,
        t_eps=args.t_eps,
        P_mean=args.P_mean,
        P_std=args.P_std,
        noise_scale=args.noise_scale,
        cfg_interval_start=args.cfg_interval_start,
    )

    seq_h = args.img_size // args.patch_size
    model = JointSSLDM(autoencoder, diffusion_model, seq_h=seq_h).cuda()

    logger.info("====Joint model=====")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable params: {n_params / 1e6:.2f}M ({n_params:,})")

    ema = models.SimpleEMAModel(model, decay=args.ema_rate)
    return model, ema


def _override_args(args: argparse.Namespace, overrides: dict | None) -> argparse.Namespace:
    new = copy.copy(args)
    for k, v in (overrides or {}).items():
        setattr(new, k, v)
    return new


def create_joint_optimizers(args: argparse.Namespace, joint: JointSSLDM):
    """Build optimizers for the joint model.

    Returns:
        opt_specs: list of dicts with keys
            {"name", "optimizer", "params", "args"}
            (args is the per-optimizer namespace, used for LR scheduling).
        loss_scaler: a single shared NativeScalerWithGradNormCount.
    """
    loss_scaler = NativeScalerWithGradNormCount()
    if not getattr(args, "separate_optimizers", False):
        # single optimizer over the whole joint model
        opt, _scaler_discarded = create_optimizer_and_scaler(args, joint)
        return [{"name": "joint", "optimizer": opt, "params": list(joint.parameters()), "args": args}], loss_scaler

    ae_args = _override_args(args, getattr(args, "ae_opt", None))
    dm_args = _override_args(args, getattr(args, "dm_opt", None))
    logger.info(f"[opt-ae] lr={ae_args.lr}, blr={ae_args.blr}, wd={ae_args.weight_decay}")
    logger.info(f"[opt-dm] lr={dm_args.lr}, blr={dm_args.blr}, wd={dm_args.weight_decay}")
    opt_ae, _ = create_optimizer_and_scaler(ae_args, joint.autoencoder)
    opt_dm, _ = create_optimizer_and_scaler(dm_args, joint.diffusion_model)
    return [
        {"name": "ae", "optimizer": opt_ae,
         "params": list(joint.autoencoder.parameters()), "args": ae_args},
        {"name": "dm", "optimizer": opt_dm,
         "params": list(joint.diffusion_model.parameters()), "args": dm_args},
    ], loss_scaler


# =============================================================================
# Checkpointing (multi-optimizer aware wrappers)
# =============================================================================


def _save_checkpoint(
    args, epoch, model, opt_specs, loss_scaler, model_ema, elapsed_time,
    loss_module, discriminator_optimizer, discriminator_loss_scaler,
):
    if not distributed.is_main_process():
        return
    ckpt = {
        "model": model.state_dict(),
        "model_ema": model_ema.state_dict() if model_ema is not None else None,
        "loss_scaler": loss_scaler.state_dict(),
        "epoch": epoch,
        "last_elapsed_time": elapsed_time,
    }
    # store optimizers under explicit names so single<->dual mode is unambiguous
    ckpt["optimizers"] = {spec["name"]: spec["optimizer"].state_dict() for spec in opt_specs}
    if loss_module is not None and isinstance(loss_module, torch.nn.Module):
        ckpt["loss_module"] = loss_module.state_dict()
    if discriminator_optimizer is not None:
        ckpt["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    if discriminator_loss_scaler is not None:
        ckpt["discriminator_loss_scaler"] = discriminator_loss_scaler.state_dict()

    path = os.path.join(args.ckpt_dir, f"epoch_{epoch:04d}.pth")
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint: {path}")
    cleanup_checkpoints(args.ckpt_dir, args.keep_n_ckpts, args.milestone_interval)


def _ckpt_resume(
    args, model, opt_specs, loss_scaler, model_ema,
    loss_module, discriminator_optimizer, discriminator_loss_scaler,
):
    from glob import glob
    if args.resume_from or args.auto_resume:
        if args.resume_from is None:
            candidates = [c for c in glob(f"{args.ckpt_dir}/*.pth") if "latest" not in c]
            candidates = sorted(candidates, key=os.path.getmtime)
            if candidates:
                args.resume_from = candidates[-1]

        if args.resume_from and os.path.exists(args.resume_from):
            logger.info(f"[Model-resume] Resuming from: {args.resume_from}")
            ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
            msg = model.load_state_dict(ckpt["model"])
            logger.info(f"[Model-resume] Loaded model: {msg}")

            if "model_ema" in ckpt and ckpt["model_ema"] is not None:
                ema_state = ckpt["model_ema"]
            else:
                ema_state = {k: v for k, v in model.state_dict().items()
                             if k in {n for n, _ in model.named_parameters()}}
            if model_ema is not None:
                model_ema.load_state_dict(ema_state)
                model_ema.to("cuda")
                logger.info("[Model-resume] Loaded EMA")

            if "optimizers" in ckpt:
                for spec in opt_specs:
                    name = spec["name"]
                    if name in ckpt["optimizers"]:
                        spec["optimizer"].load_state_dict(ckpt["optimizers"][name])
                        logger.info(f"[Model-resume] Loaded optimizer:{name}")
            if "loss_scaler" in ckpt:
                loss_scaler.load_state_dict(ckpt["loss_scaler"])
            if "epoch" in ckpt:
                args.start_epoch = ckpt["epoch"] + 1
            if "last_elapsed_time" in ckpt:
                args.last_elapsed_time = float(ckpt["last_elapsed_time"])
                logger.info(
                    f"Loaded elapsed_time: {str(datetime.timedelta(seconds=int(args.last_elapsed_time)))}"
                )
            if "loss_module" in ckpt and loss_module is not None:
                msg = loss_module.load_state_dict(ckpt["loss_module"])
                logger.info(f"[Model-resume] Loaded loss_module: {msg}")
            if "discriminator_optimizer" in ckpt and discriminator_optimizer is not None:
                discriminator_optimizer.load_state_dict(ckpt["discriminator_optimizer"])
                logger.info("[Model-resume] Loaded discriminator_optimizer")
            if "discriminator_loss_scaler" in ckpt and discriminator_loss_scaler is not None:
                discriminator_loss_scaler.load_state_dict(ckpt["discriminator_loss_scaler"])
                logger.info("[Model-resume] Loaded discriminator_loss_scaler")
            del ckpt
        else:
            logger.info(f"[Model-resume] Could not find checkpoint at {args.resume_from}.")

    if args.load_from and not args.resume_from:
        if not os.path.exists(args.load_from):
            raise FileNotFoundError(f"Could not find checkpoint at {args.load_from}")
        logger.info(f"[Model-load] Loading checkpoint from: {args.load_from}")
        ckpt = torch.load(args.load_from, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"[Model-load] Loaded model: {msg}")
        if "model_ema" in ckpt and ckpt["model_ema"] is not None and model_ema is not None:
            model_ema.load_state_dict(ckpt["model_ema"])
            model_ema.to("cuda")
            logger.info("[Model-load] Loaded EMA")
        del ckpt


# =============================================================================
# Training step
# =============================================================================


def _generator_update(combined_loss, loss_scaler, opt_specs, grad_clip):
    """Backward + clip + step for the generator (AE + DM) loss.

    Returns a dict ``{"grad_norm_<name>": tensor}``.
    """
    if len(opt_specs) == 1:
        spec = opt_specs[0]
        gn = loss_scaler(combined_loss, spec["optimizer"], grad_clip, spec["params"])
        return {f"grad_norm_{spec['name']}": gn}

    # shared scaler, manual multi-step
    loss_scaler._scaler.scale(combined_loss).backward()
    grad_norms = {}
    for spec in opt_specs:
        loss_scaler._scaler.unscale_(spec["optimizer"])
        if grad_clip is not None and grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(spec["params"], grad_clip)
        else:
            gn = misc.get_grad_norm_(spec["params"])
        loss_scaler._scaler.step(spec["optimizer"])
        grad_norms[f"grad_norm_{spec['name']}"] = gn
    loss_scaler._scaler.update()
    return grad_norms


def train_one_epoch_ssldm(
    args, model, data_loader, opt_specs, loss_scaler, wandb_logger, epoch,
    ema_model, loss_fn, discriminator_optimizer, discriminator_loss_scaler,
):
    model.train(True)
    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    metric_logger.add_meter("lr", SmoothedValue(1, "{value:.6f}"))
    metric_logger.add_meter("samples/s/gpu", SmoothedValue(args.print_freq, "{avg:.2f}"))
    steps_per_epoch = len(data_loader)
    header = f"Epoch: [{epoch}]"
    logger.info(f"log dir: {args.log_dir}")
    start_time = time.perf_counter()

    for step, data_dict in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        frac_epoch = step / steps_per_epoch + epoch
        calib_global_step = int(frac_epoch * 1000)
        x = data_dict["img"]
        y = data_dict["label"]

        # zero grads + adjust LRs per optimizer
        for spec in opt_specs:
            spec["optimizer"].zero_grad(set_to_none=True)
            misc.adjust_learning_rate(spec["optimizer"], frac_epoch, spec["args"])
        discriminator_optimizer.zero_grad(set_to_none=True)
        misc.adjust_learning_rate(discriminator_optimizer, frac_epoch, args)

        # ---- generator forward ----
        with torch.autocast("cuda", dtype=torch.bfloat16):
            decoded, posteriors, diff_loss = model(x, y)
            targets = x * 0.5 + 0.5
            reconstructions = decoded * 0.5 + 0.5
            ae_loss, loss_dict = loss_fn(targets, reconstructions, posteriors, epoch, "generator")
            combined_loss = ae_loss + args.diffusion_weight * diff_loss

            loss_dict["diffusion_loss"] = diff_loss.detach()
            loss_dict["weighted_diffusion_loss"] = (args.diffusion_weight * diff_loss).detach()

            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight"]:
                    autoencoder_logs[k] = v.cpu().item() if isinstance(v, Tensor) else v
                else:
                    autoencoder_logs[k] = distributed.all_reduce_mean(v)
            loss_dict.update(autoencoder_logs)

        # ---- generator backward / step ----
        grad_norms = _generator_update(combined_loss, loss_scaler, opt_specs, args.grad_clip)

        # ---- EMA ----
        ema_model.step(model)

        # ---- discriminator step ----
        discriminator_logs = {}
        if epoch >= args.discriminator_start_epoch:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                discriminator_loss, ld_disc = loss_fn(
                    targets, reconstructions, posteriors, epoch, mode="discriminator"
                )
            for k, v in ld_disc.items():
                if k in ["logits_real", "logits_fake"]:
                    discriminator_logs[k] = v.cpu().item() if isinstance(v, Tensor) else v
                else:
                    discriminator_logs[k] = distributed.all_reduce_mean(v)
            loss_dict.update(discriminator_logs)

            discriminator_grad_norm = discriminator_loss_scaler(
                discriminator_loss, discriminator_optimizer, args.grad_clip, loss_fn.parameters(),
            )
        else:
            discriminator_grad_norm = 0.0

        torch.cuda.synchronize()
        loss_dict_reduced = {k: distributed.all_reduce_mean(v) for k, v in loss_dict.items()}
        loss_dict_reduced.pop("total_loss", None)
        total_loss_reduced = sum(v for k, v in loss_dict_reduced.items() if "loss" in k)

        samples_per_second_per_gpu = args.batch_size * (step + 1) / (time.perf_counter() - start_time)
        samples_per_second = samples_per_second_per_gpu * args.world_size

        # use the first optimizer's LR as the canonical lr metric
        canonical_lr = opt_specs[0]["optimizer"].param_groups[0]["lr"]

        metric_logger.update(
            loss=total_loss_reduced,
            discriminator_grad_norm=discriminator_grad_norm,
            lr=canonical_lr,
            **{k: v for k, v in grad_norms.items()},
            **loss_dict_reduced,
            **{"samples/s/gpu": samples_per_second_per_gpu, "samples/s": samples_per_second},
        )

        if wandb_logger is not None and step % args.print_freq == 0:
            log_dict = {
                "loss": total_loss_reduced,
                **loss_dict_reduced,
                **{k: (v.item() if isinstance(v, Tensor) else v) for k, v in grad_norms.items()},
                "lr": canonical_lr,
                "discriminator_grad_norm": discriminator_grad_norm,
                "samples_per_sec_per_gpu": samples_per_second_per_gpu,
                "samples_per_sec": samples_per_second,
            }
            # also log per-optimizer LRs when there are multiple
            if len(opt_specs) > 1:
                for spec in opt_specs:
                    log_dict[f"lr_{spec['name']}"] = spec["optimizer"].param_groups[0]["lr"]
            wandb_logger.update(log_dict, step=calib_global_step)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# =============================================================================
# Main
# =============================================================================


def main(args: argparse.Namespace) -> int:
    global logger
    wandb_logger = setup(args)
    adjust_args_for_dataset(args)

    # data loaders — dispatch on args.dataset
    if args.dataset == "imagenet":
        data_loader_train = create_train_dataloader(args)
        data_loader_val = create_val_dataloader(args)
        data_loader_vis = create_vis_dataloader(args)
    elif args.dataset in _HF_DATASETS:
        data_loader_train = create_train_dataloader_hf(args)
        data_loader_val = create_val_dataloader_hf(args)
        data_loader_vis = create_vis_dataloader_hf(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    vis_iterator = iter(data_loader_vis)

    # model + optimizers + loss
    model, ema_model = create_ssldm_model(args)
    opt_specs, loss_scaler = create_joint_optimizers(args, model)
    loss_fn = create_loss_module(args)
    discriminator_optimizer, discriminator_loss_scaler = create_optimizer_and_scaler(args, loss_fn)

    # DDP wrapping
    if distributed.is_enabled():
        # decoder.mask_token is only used when ids_restore is not None (mask_ratio>0).
        # Since SSLDM always calls encode(mask_ratio=0.0), that param is unused —
        # match train_reconstruction.py and let DDP tolerate it.
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True,static_graph=True)
        loss_fn = torch.nn.parallel.DistributedDataParallel(loss_fn, find_unused_parameters=True)

    model_wo_ddp = model.module if hasattr(model, "module") else model
    loss_module_wo_ddp = loss_fn.module if hasattr(loss_fn, "module") else loss_fn

    # NOTE: opt_specs already holds references to the underlying (unwrapped) params,
    # which is what we want for DDP gradient clipping/stepping.

    _ckpt_resume(
        args, model_wo_ddp, opt_specs, loss_scaler, ema_model,
        loss_module_wo_ddp, discriminator_optimizer, discriminator_loss_scaler,
    )

    # initial autoencoder visualization
    try:
        visualize_tokenizer(args, model_wo_ddp, ema_model, next(vis_iterator), args.start_epoch)
    except StopIteration:
        pass

    if args.evaluate:
        torch.cuda.empty_cache()
        for use_ema in [False, True]:
            evaluate_tokenizer(
                args, model_wo_ddp, ema_model, data_loader_val, args.start_epoch, wandb_logger, use_ema,
            )
        if args.online_eval_gen:
            evaluate_generator(
                args, model_wo_ddp, ema_model, tokenizer=None,
                epoch=args.start_epoch, wandb_logger=wandb_logger,
                use_ema=True, cfg=args.cfg, num_images=args.num_images,
            )
        return 0

    logger.info(f"Start training from {args.start_epoch} to {args.epochs}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch_ssldm(
            args, model, data_loader_train, opt_specs, loss_scaler, wandb_logger, epoch,
            ema_model, loss_fn, discriminator_optimizer, discriminator_loss_scaler,
        )

        elapsed_t = time.time() - start_time + args.last_elapsed_time
        eta = elapsed_t / (epoch + 1) * (args.epochs - epoch - 1)
        logger.info(
            f"[{epoch}/{args.epochs}] "
            f"Accumulated elapsed time: {str(datetime.timedelta(seconds=int(elapsed_t)))}, "
            f"ETA: {str(datetime.timedelta(seconds=int(eta)))}"
        )

        should_save = (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs
        if should_save:
            _save_checkpoint(
                args, epoch, model_wo_ddp, opt_specs, loss_scaler, ema_model, elapsed_t,
                loss_module_wo_ddp, discriminator_optimizer, discriminator_loss_scaler,
            )
            torch.distributed.barrier()

        if (epoch + 1) % args.vis_freq == 0:
            try:
                visualize_tokenizer(args, model_wo_ddp, ema_model, next(vis_iterator), epoch)
            except StopIteration:
                vis_iterator = iter(data_loader_vis)
                visualize_tokenizer(args, model_wo_ddp, ema_model, next(vis_iterator), epoch)
            if args.gen_vis_freq > 0 and (epoch + 1) % args.gen_vis_freq == 0:
                visualize_generator(args, model_wo_ddp, ema_model, tokenizer=None, epoch=epoch + 1)

        if args.online_eval and (epoch + 1) % args.eval_freq == 0 and (epoch + 1) != args.epochs:
            torch.cuda.empty_cache()
            for use_ema in [False, True]:
                evaluate_tokenizer(
                    args, model_wo_ddp, ema_model, data_loader_val, epoch + 1, wandb_logger, use_ema,
                )
            if args.online_eval_gen:
                evaluate_generator(
                    args, model_wo_ddp, ema_model, tokenizer=None,
                    epoch=epoch + 1, wandb_logger=wandb_logger,
                    use_ema=True, cfg=args.cfg, num_images=args.num_images_for_eval_and_search,
                )

    total_time = int(time.time() - start_time + args.last_elapsed_time)
    logger.info(f"Training time {str(datetime.timedelta(seconds=total_time))}")

    for use_ema in [False, True]:
        evaluate_tokenizer(args, model_wo_ddp, ema_model, data_loader_val, args.epochs, wandb_logger, use_ema)
    if args.online_eval_gen:
        evaluate_generator(
            args, model_wo_ddp, ema_model, tokenizer=None,
            epoch=args.epochs, wandb_logger=wandb_logger,
            use_ema=True, cfg=args.cfg, num_images=args.num_images,
        )

    return 0


# =============================================================================
# CLI
# =============================================================================


def load_yaml_into_args(args: argparse.Namespace, config_path: str) -> argparse.Namespace:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


def get_args_parser():
    parser = argparse.ArgumentParser("Joint SSL+DM training", add_help=False)

    # YAML config
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # logging parameters
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--vis_freq", type=int, default=5)
    parser.add_argument("--gen_vis_freq", type=int, default=0,
                        help="If >0, run visualize_generator every N epochs")
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
    parser.add_argument("--num_images_for_eval_and_search", default=10000, type=int)
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--online_eval_gen", action="store_true",
                        help="also run generation FID online during training")
    parser.add_argument("--fid_stats_path", type=str, default="data/fid_stats/val_fid_statistics_file.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval_bsz", type=int, default=256)

    # generation parameters used by visualize_generator / evaluate_generator
    parser.add_argument("--cfg", default=4.0, type=float)
    parser.add_argument("--cfg_schedule", default="linear", type=str)
    parser.add_argument("--num_iter", default=64, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--force_class_of_interest", action="store_true")

    # dataset parameters
    parser.add_argument("--dataset", default="imagenet", type=str,
                        choices=["imagenet", "mnist", "cifar10"],
                        help="imagenet uses local data_path; mnist/cifar10 use HuggingFace datasets")
    parser.add_argument("--use_cached_tokens", action="store_true")
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279], type=int, nargs="+")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # wandb parameters
    parser.add_argument("--project", default="ssldm", type=str)
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
