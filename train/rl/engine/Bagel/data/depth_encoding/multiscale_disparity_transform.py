"""
Multi-scale (N-channel) disparity depth encoding, designed to reuse pretrained
RGB VAEs (SD-VAE, Flux-VAE, etc.) without wasting channel capacity.

With three channels:
    y_R = (d - c_1) / (d + c_1)     c_1 = 2   (near)
    y_G = (d - c_2) / (d + c_2)     c_2 = 8   (mid)
    y_B = (d - c_3) / (d + c_3)     c_3 = 32  (far)

Each channel is an independent disparity bijection at a different pivot.
Their resolution curves have "knees" at c_1, c_2, c_3, respectively, and
together give near-flat high-precision coverage across ~2 decades of depth.

Decoding uses each channel's own inverse and blends the three depth estimates
weighted by  w_i = (1 - y_i^2)^2  (saturation-aware confidence).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Default preset for NYU + ETH3D + DIODE + Sintel + KITTI
# ---------------------------------------------------------------------------

DEFAULT_C_RGB: Tuple[float, float, float] = (2.0, 8.0, 32.0)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def multiscale_disparity_encode(
    d: Tensor,                                      # shape (..., 1, H, W) or (..., H, W)
    c_values: Sequence[float] = DEFAULT_C_RGB,
    mask: Optional[Tensor] = None,
    invalid_fill: float = 0.0,
    return_mask: bool = False,
    channel_dim: int = -3,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Encode depth to an N-channel disparity tensor.

    Args:
        d            : depth tensor with a singleton channel dim (or no channel
                       dim); any leading batch dims are fine.
        c_values     : per-channel pivot depths.  len == output channels.
        mask         : optional bool mask, same shape as `d`.
        invalid_fill : value at invalid pixels.
        return_mask  : also return the effective valid mask.
        channel_dim  : where to stack the N channels (default -3, i.e.
                       producing (..., N, H, W) from (..., 1, H, W)).

    Returns:
        y : same as `d` but with N channels on `channel_dim`.
    """
    # Collapse a singleton channel dim on d, if present, so broadcasting works.
    if d.dim() >= -channel_dim and d.shape[channel_dim] == 1:
        d_bare = d.squeeze(channel_dim)
    else:
        d_bare = d

    channels = []
    for c in c_values:
        y = (d_bare - c) / (d_bare + c)
        channels.append(y)
    y = torch.stack(channels, dim=channel_dim)

    d_valid = (d_bare > 0).unsqueeze(channel_dim).expand_as(y)
    if mask is not None:
        if mask.shape == d.shape and d.shape[channel_dim] == 1:
            mask_bare = mask.squeeze(channel_dim)
        else:
            mask_bare = mask
        mask_bc = mask_bare.to(torch.bool).unsqueeze(channel_dim).expand_as(y)
        valid = mask_bc & d_valid
    else:
        valid = d_valid
    y = torch.where(valid, y, torch.full_like(y, invalid_fill))

    if return_mask:
        return y, valid
    return y


def multiscale_disparity_decode(
    y: Tensor,                                      # shape (..., N, H, W)
    c_values: Sequence[float] = DEFAULT_C_RGB,
    inf_cap: Optional[float] = 1e4,
    eps_boundary: float = 1e-4,
    channel_dim: int = -3,
) -> Tensor:
    """Decode an N-channel disparity tensor back to a single-channel depth.

    Each channel is inverted independently;  d_i = c_i * (1 + y_i) / (1 - y_i).
    The final estimate is the confidence-weighted average,

        w_i = (1 - y_i^2)^2,     d_hat = sum_i w_i d_i / sum_i w_i

    which assigns near-zero weight to any saturated channel and full weight to
    whichever channel is most sensitive at the true depth.
    """
    if y.shape[channel_dim] != len(c_values):
        raise ValueError(
            f"y has {y.shape[channel_dim]} channels but "
            f"c_values has {len(c_values)} entries"
        )

    c_t = torch.as_tensor(c_values, dtype=y.dtype, device=y.device)
    view = [1] * y.dim()
    view[channel_dim] = len(c_values)
    c_t = c_t.view(view)

    y_c = y.clamp(min=-1.0 + eps_boundary, max=1.0 - eps_boundary)
    d_per_ch = c_t * (1.0 + y_c) / (1.0 - y_c)                          # (..., N, H, W)

    # Confidence: (1 - y^2)^2  -- peaks at y=0, vanishes at y=+-1.
    w = (1.0 - y_c * y_c).pow(2).clamp(min=1e-12)
    d_hat = (w * d_per_ch).sum(dim=channel_dim) / w.sum(dim=channel_dim)

    if inf_cap is not None:
        d_hat = d_hat.clamp(max=inf_cap)
    return d_hat


# ---------------------------------------------------------------------------
# Container class (parallels DepthDisparityTransform)
# ---------------------------------------------------------------------------

class DepthMultiScaleDisparityTransform:
    """N-channel disparity codec.  Plain container, not an nn.Module.

    Example
    -------
    >>> codec = DepthMultiScaleDisparityTransform()             # c=(2, 8, 32)
    >>> y       = codec.encode(depth, mask=mask)                # (..., 3, H, W)
    >>> d_hat   = codec.decode(y_pred)                          # (..., H, W)
    """

    def __init__(
        self,
        c_values: Sequence[float] = DEFAULT_C_RGB,
        inf_cap: Optional[float] = 1e4,
        invalid_fill: float = 0.0,
        channel_dim: int = -3,
    ):
        self.c_values = tuple(float(c) for c in c_values)
        if any(c <= 0 for c in self.c_values):
            raise ValueError(f"all c must be > 0, got {self.c_values}")
        self.inf_cap = inf_cap
        self.invalid_fill = float(invalid_fill)
        self.channel_dim = channel_dim

    @property
    def num_channels(self) -> int:
        return len(self.c_values)

    def encode(self, d, mask=None, return_mask=False):
        return multiscale_disparity_encode(
            d, c_values=self.c_values, mask=mask,
            invalid_fill=self.invalid_fill, return_mask=return_mask,
            channel_dim=self.channel_dim,
        )

    def decode(self, y):
        return multiscale_disparity_decode(
            y, c_values=self.c_values,
            inf_cap=self.inf_cap, channel_dim=self.channel_dim,
        )

    def __call__(self, d, valid_mask=None, return_mask=False):
        return self.encode(d, mask=valid_mask, return_mask=return_mask)

    def __repr__(self):
        return (f"DepthMultiScaleDisparityTransform(c={self.c_values}, "
                f"inf_cap={self.inf_cap}, invalid_fill={self.invalid_fill})")


# ---------------------------------------------------------------------------
# Self-test: round-trip + VAE-noise accuracy vs single-channel
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    torch.manual_seed(0)

    codec = DepthMultiScaleDisparityTransform()
    print(codec)

    # Round-trip sanity
    d = torch.rand(2, 1, 64, 64, dtype=torch.float64) * 200.0
    y = codec(d)
    d_hat = codec.decode(y)
    print(f"round-trip max err   : {(d.squeeze(1) - d_hat).abs().max():.2e} m")
    print(f"y shape              : {tuple(y.shape)}")

    # Simulate VAE noise + compare to single-channel disparity (c=5)
    def lognormal(mean_m, sigma, n=200_000, cap=None):
        x = torch.randn(n) * sigma + math.log(mean_m)
        x = x.exp()
        return x if cap is None else x.clamp(0.1, cap)

    datasets = {
        "NYU":    lognormal(2.5,  0.5, cap=10.0),
        "ETH3D":  lognormal(5.0,  0.8, cap=70.0),
        "DIODE":  lognormal(8.0,  1.2, cap=350.0),
        "Sintel": lognormal(10.0, 0.6, cap=100.0),
        "KITTI":  lognormal(15.0, 0.7, cap=80.0),
    }

    # Single-channel disparity (c=5), for comparison
    def enc_single(d, c=5.0):
        return (d - c) / (d + c)

    def dec_single(y, c=5.0, eps=1e-4):
        y = y.clamp(-1 + eps, 1 - eps)
        return c * (1 + y) / (1 - y)

    print("\nVAE-noise comparison (sigma_y = 0.01):")
    print(f"{'dataset':<8}  {'single-ch AbsRel':>18}  {'3-ch AbsRel':>14}  {'improvement':>12}")
    print("-" * 60)

    sigma = 0.01
    for ds_name, d in datasets.items():
        d = d.reshape(1, 1, -1, 1)
        # 1 channel
        y_s = enc_single(d.squeeze(1))
        y_s = y_s + torch.randn_like(y_s) * sigma
        d_hat_s = dec_single(y_s)
        abs_rel_s = ((d_hat_s - d.squeeze(1)).abs() / d.squeeze(1)).mean().item()

        # 3 channels (independent noise per channel)
        y_m = codec.encode(d)
        y_m = y_m + torch.randn_like(y_m) * sigma
        d_hat_m = codec.decode(y_m)
        abs_rel_m = ((d_hat_m - d.squeeze(1)).abs() / d.squeeze(1)).mean().item()

        imp = (abs_rel_s - abs_rel_m) / abs_rel_s * 100.0
        print(f"{ds_name:<8}  {abs_rel_s:>18.4f}  {abs_rel_m:>14.4f}  {imp:>11.1f}%")
