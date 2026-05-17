"""DeTok variant for joint SSL+Diffusion training.

Adds an optional normalization layer applied to the encoder output (the
moments tensor produced by ``Encoder.latent_head``) before it is split into
the ``(mean, logvar)`` pair consumed by ``DiagonalGaussianDistribution``.

Also supports a ``fixed_std`` mode: when set, the encoder's predicted logvar
is discarded and the posterior std is forced to a fixed scalar value.
"""

import logging
import math

import torch
import torch.nn as nn
from torch import Tensor

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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.latent_norm_kind = latent_norm
        self.fixed_std = fixed_std

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

        # norm runs on the encoder output: full moments (2C) when fixed_std is
        # None, mean-only (C) when fixed_std is set.
        self.latent_norm = _build_latent_norm(latent_norm, self.encoder.token_channels)
        logger.info(f"[DeTok_SSLDM] latent_norm: {latent_norm}, fixed_std: {fixed_std}")

    def encode(self, x: Tensor, sampling: bool = False, mask_ratio: float = -1, noise_level: float = -1.0):
        z, ids_restore = self.encoder(x, mask_ratio=mask_ratio)
        z = self.latent_norm(z)
        if self.fixed_std is not None:
            # encoder.latent_head was rebuilt to output only the mean; build the
            # moments tensor by appending a constant logvar (clamped at init).
            mean = z
            logvar = torch.full_like(mean, fill_value=self._fixed_logvar_value)
            params = torch.cat([mean, logvar], dim=-1)
            posteriors = self.to_posteriors(params)
        else:
            posteriors = self.to_posteriors(z)
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
