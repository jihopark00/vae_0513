"""DeTok variant for joint SSL+Diffusion training.

Adds an optional normalization layer applied to the **mean** half of the
moments tensor produced by ``Encoder.latent_head`` (the logvar half is left
untouched).

Also supports a ``fixed_std`` mode: when set, the encoder's predicted logvar
is discarded and the posterior std is forced to a fixed scalar value.
"""

import logging
import math
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from .autoencoder import DiagonalGaussianDistribution
from .detok import DeTok

logger = logging.getLogger("DeTok")


class _ChannelBatchNorm1d(nn.Module):
    """BatchNorm1d over the channel dim of a ``(B, seq_len, C)`` tensor."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_channels, affine=False)

    def forward(self, x: Tensor) -> Tensor:
        # (B, L, C) -> (B, C, L) -> bn -> (B, L, C)
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


def _build_latent_norm(kind: str | None, num_channels: int) -> nn.Module:
    if kind is None or kind == "none" or kind is False:
        return nn.Identity()
    if kind == "layernorm":
        return nn.LayerNorm(num_channels, elementwise_affine=False)
    if kind == "rmsnorm":
        return nn.RMSNorm(num_channels, elementwise_affine=False)
    if kind == "batchnorm":
        return _ChannelBatchNorm1d(num_channels)
    raise ValueError(f"unsupported latent_norm: {kind!r}")


class DeTok_SSLDM(DeTok):
    """DeTok + optional norm on the moments tensor + optional fixed posterior std."""

    def __init__(
        self,
        latent_norm: str | None = None,
        fixed_std: float | None = None,
        kl_reduction: str = "sum",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.latent_norm_kind = latent_norm
        self.fixed_std = fixed_std
        # override parent's to_posteriors so the chosen reduction over latent
        # dims is baked into every posterior created via this model.
        if kl_reduction not in ("sum", "mean"):
            raise ValueError(f"kl_reduction must be 'sum' or 'mean', got {kl_reduction!r}")
        self.kl_reduction = kl_reduction
        self.to_posteriors = partial(
            DiagonalGaussianDistribution, channel_dim=-1, reduction=kl_reduction,
        )

        if fixed_std is not None:
            assert fixed_std >= 0, f"fixed_std must be non-negative, got {fixed_std}"
            # rebuild latent_head to predict the mean only (drop the logvar half).
            # parent set encoder.token_channels = token_channels * 2; halve it back.
            mean_dim = self.encoder.token_channels // 2
            width = self.encoder.width
            new_head = nn.Linear(width, mean_dim)
            nn.init.trunc_normal_(new_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(new_head.bias)
            self.encoder.latent_head = new_head
            self.encoder.token_channels = mean_dim
            # clamp to a tiny positive value so log() stays finite. fixed_std == 0
            # then yields std ≈ 1e-6 (near-deterministic, negligible KL).
            safe_std = max(float(fixed_std), 1e-6)
            self._fixed_logvar_value = 2.0 * math.log(safe_std)
            mean_chans = mean_dim
        else:
            # learning logvar: encoder output is (B, L, 2C); mean half is C-dim.
            mean_chans = self.encoder.token_channels // 2

        # norm runs on the **mean** half only — never on logvar.
        self.latent_norm = _build_latent_norm(latent_norm, mean_chans)
        self._mean_chans = mean_chans
        logger.info(f"[DeTok_SSLDM] latent_norm: {latent_norm} (dim={mean_chans}, applied to mean only), "
                    f"fixed_std: {fixed_std}, kl_reduction: {kl_reduction}")

    def encode(self, x: Tensor, sampling: bool = False, mask_ratio: float = -1, noise_level: float = -1.0):
        z, ids_restore = self.encoder(x, mask_ratio=mask_ratio)
        if self.fixed_std is not None:
            # encoder.latent_head outputs the mean only; norm it directly and
            # append a constant logvar to form a moments tensor.
            mean = self.latent_norm(z)
            logvar = torch.full_like(mean, fill_value=self._fixed_logvar_value)
            params = torch.cat([mean, logvar], dim=-1)
        else:
            # split (B, L, 2C) into mean / logvar; norm the mean half only.
            mean, logvar = z[..., :self._mean_chans], z[..., self._mean_chans:]
            mean = self.latent_norm(mean)
            params = torch.cat([mean, logvar], dim=-1)
        posteriors = self.to_posteriors(params)
        z_latents = posteriors.sample() if sampling else posteriors.mean

        if self.training and self.gamma > 0.0:
            device = z_latents.device
            bsz, n_tokens, chans = z_latents.shape
            if noise_level > 0.0:
                noise_level_tensor = torch.full((bsz, 1, 1), noise_level, device=device)
            else:
                noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
            noise_level_tensor = noise_level_tensor.expand(-1, n_tokens, chans)
            noise = torch.randn(bsz, n_tokens, chans, device=device) * self.gamma
            if self.use_additive_noise:
                z_latents = z_latents + noise_level_tensor * noise
            else:
                z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise

        return z_latents, posteriors, ids_restore


def _make(enc: str, dec: str):
    def _ctor(**kwargs) -> DeTok_SSLDM:
        return DeTok_SSLDM(vit_enc_model_size=enc, vit_dec_model_size=dec, **kwargs)

    return _ctor


DeTok_SSLDM_models = {
    "detok_ssldm_SS": _make("small", "small"),
    "detok_ssldm_SB": _make("small", "base"),
    "detok_ssldm_SL": _make("small", "large"),
    "detok_ssldm_BS": _make("base", "small"),
    "detok_ssldm_BB": _make("base", "base"),
    "detok_ssldm_BL": _make("base", "large"),
    "detok_ssldm_LS": _make("large", "small"),
    "detok_ssldm_LB": _make("large", "base"),
    "detok_ssldm_LL": _make("large", "large"),
    "detok_ssldm_XLXL": _make("xl", "xl"),
}
