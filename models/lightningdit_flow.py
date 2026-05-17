"""
LightningDiT_Flow — same architecture as LightningDiT but with an inline
rectified-flow loss / Euler sampler instead of the external `transport` package.

Convention (t=0 noise, t=1 data) and loss logic follow the JiT reference
denoiser at https://github.com/.../JiT/denoiser.py.
"""

import logging
from functools import partial

import torch
import torch.nn as nn

from .layers import (
    Block,
    LabelEmbedder,
    ModulatedLinear,
    PatchEmbed,
    TimestepEmbedder,
    Transformer,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)
from .model_utils import SIZE_DICT

logger = logging.getLogger("DeTok")


class LightningDiT_Flow(nn.Module):
    """LightningDiT architecture trained with an inline flow-matching loss
    (t=0 noise, t=1 data)."""

    def __init__(
        self,
        img_size=256,
        patch_size=1,
        model_size="base",
        tokenizer_patch_size=16,
        token_channels=16,
        label_drop_prob=0.1,
        num_classes=1000,
        num_sampling_steps=50,
        grad_checkpointing=False,
        force_one_d_seq=0,
        legacy_mode=False,
        qk_norm=False,
        # flow-specific
        prediction="velocity",
        loss_target="velocity",
        detach_zt = False,
        detach_target = False,
        t_eps=0.05,
        P_mean=0.0,
        P_std=1.0,
        noise_scale=1.0,
        cfg_interval_start=0.10,
    ):
        super().__init__()

        assert prediction in ("velocity", "data"), prediction
        assert loss_target in ("velocity", "data"), loss_target

        # --------------------------------------------------------------------------
        # basic configuration
        self.token_channels = self.out_channels = token_channels
        self.input_size = img_size // tokenizer_patch_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.force_one_d_seq = force_one_d_seq
        self.grad_checkpointing = grad_checkpointing
        self.legacy_mode = legacy_mode

        # flow configuration
        self.prediction = prediction
        self.loss_target = loss_target
        self.t_eps = t_eps
        self.P_mean = P_mean
        self.P_std = P_std
        self.noise_scale = noise_scale
        self.num_sampling_steps = int(num_sampling_steps)
        self.cfg_interval_start = cfg_interval_start

        # model architecture configuration
        size_dict = SIZE_DICT[model_size]
        num_layers, num_heads, width = size_dict["layers"], size_dict["heads"], size_dict["width"]

        self.detach_zt = detach_zt    
        self.detach_target = detach_target

        # --------------------------------------------------------------------------
        # embedding layers
        if self.force_one_d_seq > 0:
            self.x_embedder = nn.Linear(token_channels, width)
            self.pos_embed = nn.Parameter(torch.randn(1, self.force_one_d_seq, width) * 0.02)
            self.seq_len = self.force_one_d_seq
        else:
            self.x_embedder = PatchEmbed(self.input_size, patch_size, token_channels, width)
            num_patches = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, width))
            self.rope = VisionRotaryEmbeddingFast(width // num_heads // 2, self.input_size // patch_size)
            self.seq_len = num_patches

        self.t_embedder = TimestepEmbedder(width)
        self.y_embedder = LabelEmbedder(num_classes, width, label_drop_prob)

        # --------------------------------------------------------------------------
        # transformer architecture
        self.transformer = Transformer(
            width,
            num_layers,
            num_heads,
            block_fn=partial(Block, use_modulation=True),
            norm_layer=nn.RMSNorm,
            grad_checkpointing=grad_checkpointing,
            use_swiglu=True,
            qk_norm=qk_norm,
        )
        self.final_layer = ModulatedLinear(width, patch_size**2 * token_channels, use_rmsnorm=True)

        self.initialize_weights()

        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"[LightningDiT_Flow] params: {num_trainable_params:.2f}M size: {model_size}, "
            f"num_layers: {num_layers}, width: {width}, "
            f"prediction: {prediction}, loss_target: {loss_target}, t_eps: {t_eps}"
        )

    def initialize_weights(self):
        """initialize model weights."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if not self.force_one_d_seq:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.transformer.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """convert patch tokens back to image tensor."""
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def net(self, x, t=None, y=None):
        """core network forward pass."""
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, self.training)

        if not self.force_one_d_seq:
            x = self.transformer(x, condition=c, rope=self.rope)
        else:
            x = self.transformer(x, condition=c)

        x = self.final_layer(x, c)
        if not self.force_one_d_seq:
            x = self.unpatchify(x)
        return x

    # ---------------------------------------------------------------- training
    def _sample_t(self, n, device):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, y):
        """flow-matching training loss (t=0 noise, t=1 data)."""
        b = x.size(0)
        t_flat = self._sample_t(b, device=x.device)
        t = t_flat.view(-1, *([1] * (x.ndim - 1)))

        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e

        _z = z.detach() if self.detach_zt else z
        out = self.net(_z, t_flat, y)

        if self.prediction == "data":
            x_hat = out
            v_hat = (x_hat - z) / (1 - t).clamp_min(self.t_eps)
            v_true = (x - z) / (1 - t).clamp_min(self.t_eps)
        else:  # "velocity"
            v_hat = out
            x_hat = z + (1 - t) * v_hat
            v_true = x - e

        if self.loss_target == "velocity":
            v_true = v_true.detach() if self.detach_target else v_true
            loss = (v_hat - v_true).pow(2).mean()
        else:  # "data"
            x = x.detach() if self.detach_target else x
            loss = (x_hat - x).pow(2).mean()

        return loss

    # ---------------------------------------------------------------- sampling
    def _velocity(self, z, t_flat, y, cfg):
        """run net, convert to velocity, apply CFG on velocity (JiT-style)."""
        out = self.net(z, t_flat, y)

        if self.prediction == "data":
            t = t_flat.view(-1, *([1] * (z.ndim - 1)))
            v = (out - z) / (1 - t).clamp_min(self.t_eps)
        else:
            v = out

        if cfg > 1.0:
            v_cond, v_uncond = v.chunk(2, dim=0)
            if t_flat[0].item() >= self.cfg_interval_start:
                v_blend = v_uncond + cfg * (v_cond - v_uncond)
            else:
                v_blend = v_cond
            v = torch.cat([v_blend, v_blend], dim=0)

        return v

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=1.0, args=None):
        """generate samples by Euler-integrating the velocity field from t=0 to t=1."""
        device = labels.device

        if self.force_one_d_seq:
            z = torch.randn(n_samples, self.force_one_d_seq, self.token_channels, device=device)
        else:
            z = torch.randn(n_samples, self.token_channels, self.input_size, self.input_size, device=device)
        z = z * self.noise_scale

        if cfg > 1.0:
            z = torch.cat([z, z], dim=0)
            y_null = torch.full((n_samples,), self.num_classes, device=device, dtype=labels.dtype)
            labels = torch.cat([labels, y_null], dim=0)

        ts = torch.linspace(0.0, 1.0, self.num_sampling_steps + 1, device=device)
        for i in range(self.num_sampling_steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            t_vec = t_cur.expand(z.size(0))
            v_hat = self._velocity(z, t_vec, labels, cfg)
            z = z + (t_next - t_cur) * v_hat

        if cfg > 1.0:
            z, _ = z.chunk(2, dim=0)
        return z


# model size variants
def LightningDiT_Flow_base(**kwargs) -> LightningDiT_Flow:
    return LightningDiT_Flow(model_size="base", **kwargs)


def LightningDiT_Flow_large(**kwargs) -> LightningDiT_Flow:
    return LightningDiT_Flow(model_size="large", **kwargs)


def LightningDiT_Flow_xl(**kwargs) -> LightningDiT_Flow:
    return LightningDiT_Flow(model_size="xl", **kwargs)


def LightningDiT_Flow_huge(**kwargs) -> LightningDiT_Flow:
    return LightningDiT_Flow(model_size="huge", **kwargs)


LightningDiT_Flow_models = {
    "LightningDiT_Flow_base": LightningDiT_Flow_base,
    "LightningDiT_Flow_large": LightningDiT_Flow_large,
    "LightningDiT_Flow_xl": LightningDiT_Flow_xl,
    "LightningDiT_Flow_huge": LightningDiT_Flow_huge,
}
