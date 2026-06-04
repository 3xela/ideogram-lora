"""Flow-matching training math for Ideogram 4.

Conventions reverse-engineered from ``ideogram4.pipeline_ideogram4`` and the
Euler sampler in ``ideogram4.scheduler``:

* Time runs ``t = 0`` (pure noise) -> ``t = 1`` (clean image).
* The probability-flow path is the straight line
  ``x_t = (1 - t) * noise + t * x_clean``.
* The transformer predicts the velocity ``v = dx_t/dt = x_clean - noise``.

The pipeline applies latent normalisation in the *packed* 128-dim token space
(`_decode` does ``z = z * scale + shift`` before un-patchifying), so we patchify
the raw VAE latent first and then normalise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def patchify(x: torch.Tensor, patch: int = 2) -> torch.Tensor:
    """(B, C, H, W) -> (B, (H/p)*(W/p), p*p*C).

    Exact inverse of the un-patchify in ``Ideogram4Pipeline._decode``: the flat
    last dim is ordered (ph, pw, c) with channel innermost.
    """
    b, c, h, w = x.shape
    assert h % patch == 0 and w % patch == 0, (h, w, patch)
    gh, gw = h // patch, w // patch
    x = x.view(b, c, gh, patch, gw, patch)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, gh, gw, ph, pw, C)
    return x.view(b, gh * gw, patch * patch * c)


def unpatchify(z: torch.Tensor, gh: int, gw: int, patch: int = 2) -> torch.Tensor:
    """(B, gh*gw, p*p*C) -> (B, C, gh*p, gw*p). Mirrors ``_decode``."""
    b = z.shape[0]
    c = z.shape[-1] // (patch * patch)
    z = z.view(b, gh, gw, patch, patch, c)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, gh, ph, gw, pw)
    return z.view(b, c, gh * patch, gw * patch)


@torch.no_grad()
def encode_to_latent(
    autoencoder,
    image: torch.Tensor,
    latent_shift: torch.Tensor,
    latent_scale: torch.Tensor,
    *,
    patch: int = 2,
    sample: bool = True,
) -> torch.Tensor:
    """Image in [-1, 1], shape (B, 3, H, W) -> normalised packed latent (B, L, 128).

    The VAE emits KL moments (mean | logvar) over 2*z_channels; we (optionally)
    sample, patchify, then normalise in packed space.
    """
    moments = autoencoder.encoder(image)
    mean, logvar = moments.chunk(2, dim=1)
    if sample:
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        latent = mean + std * torch.randn_like(mean)
    else:
        latent = mean
    packed = patchify(latent.to(torch.float32), patch)
    return (packed - latent_shift) / latent_scale


def sample_timesteps(
    batch: int,
    device: torch.device,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Logit-normal sampling of t in (0, 1) (SD3-style: mass near the middle).

    mean=0 centres on t=0.5; increase to bias toward clean, decrease toward noise.
    """
    n = torch.randn(batch, device=device)
    return torch.sigmoid(mean + std * n)


def make_flow_targets(
    clean_latent: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (x_t, velocity_target) for clean_latent (B, L, 128) and t (B,)."""
    noise = torch.randn_like(clean_latent)
    t_ = t.view(-1, *([1] * (clean_latent.ndim - 1)))
    x_t = (1.0 - t_) * noise + t_ * clean_latent
    velocity = clean_latent - noise
    return x_t, velocity


def flow_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.to(torch.float32), target.to(torch.float32))
