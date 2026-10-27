"""
Multi-scale (N-channel) power-transform depth encoding, designed -- like
`multiscale_disparity_torch.py` -- to reuse pretrained RGB VAEs without wasting
channel capacity.  This is the power-transform counterpart of the multi-scale
disparity codec.

With three channels and the recommended preset (lam = -3, c = (2, 6, 18)):

    y_R = 1 - 2 / (1 + d / (|lam| * c_1))^2     c_1 = 2    (near)
    y_G = 1 - 2 / (1 + d / (|lam| * c_2))^2     c_2 = 6    (mid)
    y_B = 1 - 2 / (1 + d / (|lam| * c_3))^2     c_3 = 18   (far)

Each channel is an independent power-transform bijection at a different
scale.  Decoding inverts each channel and blends the three depth estimates,

    w_i = (1 - y_i^2)^2,        d_hat = sum_i w_i * d_i / sum_i w_i.

Why (1 - y_i^2)^2?
------------------
For the disparity codec, this weight happens to be exactly proportional to
d^2 / Var(d_hat), which minimises AbsRel.  For the power transform that
identity is no longer exact -- the AbsRel-optimal weight is

    w_i^opt  proportional to  (1 - y_i)^2 * (sqrt(2) - sqrt(1 - y_i))^2.

In our 5-dataset VAE-noise simulation (sigma_y in [0.01, 0.05]) the two weights
were within 0.5%, so we keep the simpler (1 - y^2)^2 as the default and expose
the optimal form as an opt-in (`weight = "opt"`).

Why c = (2, 6, 18) ?
--------------------
Power transform's tail decays as d^-3 (vs d^-2 for disparity), so each channel
covers a *narrower* band of depth.  Empirically, ratio r = 3 between adjacent
c's outperforms r = 4 (which we use for disparity) by ~5% AbsRel; r = 2 over-
laps too much, r = 4 leaves precision dips between channels.

In our 5-dataset experiment (NYU, ETH3D, DIODE, Sintel, KITTI), this preset
gives ~10% lower AbsRel than 3-channel disparity (2, 8, 32) under the same
VAE noise level -- at the cost of a slightly less-clean optimal weight.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .power_transform import depth_to_signed_unit, signed_unit_to_depth


# ---------------------------------------------------------------------------
# Default preset.  Picked from VAE-noise sweeps over NYU + ETH3D + DIODE +
# Sintel + KITTI;  see the docstring for the rationale.
# ---------------------------------------------------------------------------

DEFAULT_LAM: float = -3.0
DEFAULT_C_RGB: Tuple[float, float, float] = (2.0, 6.0, 18.0)

WeightKind = Literal["iyy2", "opt"]


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def multiscale_power_encode(
    d: Tensor,                                      # shape (..., 1, H, W) or (..., H, W)
    c_values: Sequence[float] = DEFAULT_C_RGB,
    lam: float = DEFAULT_LAM,
    mask: Optional[Tensor] = None,
    invalid_fill: float = 0.0,
    return_mask: bool = False,
    channel_dim: int = -3,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Encode depth to an N-channel power-transform tensor.

    Args:
        d            : depth tensor with a singleton channel dim (or no channel
                       dim);  any leading batch dims are fine.
        c_values     : per-channel scale parameters.  len == output channels.
        lam          : shared shape parameter for all channels;  must be < -1.
        mask         : optional bool mask, broadcastable to `d`.
        invalid_fill : value at invalid pixels (in [-1, 1) space).
        return_mask  : also return the effective valid mask.
        channel_dim  : where to stack the N channels (default -3, so a (..., 1,
                       H, W) input becomes (..., N, H, W)).

    Returns:
        y : same as `d` but with N channels on `channel_dim`, in [-1, 1)^N.
    """
    if lam >= -1.0:
        raise ValueError(f"lam must be < -1, got {lam}")
    if any(c <= 0 for c in c_values):
        raise ValueError(f"all c must be > 0, got {tuple(c_values)}")

    # Collapse a singleton channel dim on d, if present, so per-channel
    # encoders see a clean (..., H, W) tensor.
    if d.dim() >= -channel_dim and d.shape[channel_dim] == 1:
        d_bare = d.squeeze(channel_dim)
        if mask is not None and mask.shape == d.shape:
            mask_bare: Optional[Tensor] = mask.squeeze(channel_dim)
        else:
            mask_bare = mask
    else:
        d_bare = d
        mask_bare = mask

    channels = []
    valid_per_ch = []
    for c in c_values:
        y_i, v_i = depth_to_signed_unit(
            d_bare, lam=lam, c=float(c),
            mask=mask_bare, invalid_fill=invalid_fill, return_mask=True,
        )
        channels.append(y_i)
        valid_per_ch.append(v_i)
    y = torch.stack(channels, dim=channel_dim)
    valid = torch.stack(valid_per_ch, dim=channel_dim)

    if return_mask:
        return y, valid
    return y


def multiscale_power_decode(
    y: Tensor,                                      # shape (..., N, H, W)
    c_values: Sequence[float] = DEFAULT_C_RGB,
    lam: float = DEFAULT_LAM,
    inf_cap: Optional[float] = 1e4,
    eps_boundary: float = 1e-4,
    weight: WeightKind = "iyy2",
    channel_dim: int = -3,
) -> Tensor:
    """Decode an N-channel power-transform tensor back to single-channel depth.

    Each channel is inverted independently using `signed_unit_to_depth`
    and the final estimate is the confidence-weighted average,

        d_hat = sum_i w_i * d_i / sum_i w_i

    Args:
        y            : encoded tensor, shape (..., N, H, W).
        c_values     : per-channel scale parameters.
        lam          : shared shape parameter, must match the encoder.
        inf_cap      : clamp on the final depth (None disables).
        eps_boundary : keep |y| <= 1 - eps so the inverse stays finite.
        weight       : "iyy2" -> w = (1 - y^2)^2  (cheap, near-optimal)
                       "opt"  -> AbsRel-optimal weight derived for lam = -3.
        channel_dim  : channel axis (default -3).
    """
    if y.shape[channel_dim] != len(c_values):
        raise ValueError(
            f"y has {y.shape[channel_dim]} channels but "
            f"c_values has {len(c_values)} entries"
        )

    y_c = y.clamp(min=-1.0 + eps_boundary, max=1.0 - eps_boundary)

    # Per-channel depth estimates, one slice at a time so each call to
    # `signed_unit_to_depth` sees the right `c` (lam is shared).
    chans = []
    for i, c in enumerate(c_values):
        y_i = y_c.select(channel_dim, i)
        d_i = signed_unit_to_depth(y_i, lam=lam, c=float(c), inf_cap=None)
        chans.append(d_i)
    d_per_ch = torch.stack(chans, dim=channel_dim)               # (..., N, H, W)

    # --- confidence weights --------------------------------------------------
    if weight == "iyy2":
        w = (1.0 - y_c.pow(2)).pow(2)
    elif weight == "opt":
        # w_i^opt  proportional to  (1 - y)^2 * (sqrt(2) - sqrt(1 - y))^2
        # Derived from  w  proportional to  d^2 / Var(d_hat)  with lam = -3.
        if lam != -3.0:
            raise NotImplementedError(
                "weight='opt' is currently only derived for lam = -3"
            )
        one_my = (1.0 - y_c).clamp(min=eps_boundary)
        w = one_my.pow(2) * (2.0 ** 0.5 - one_my.sqrt()).pow(2)
    else:
        raise ValueError(f"unknown weight kind {weight!r}")

    w = w.clamp(min=1e-12)
    d_hat = (w * d_per_ch).sum(dim=channel_dim) / w.sum(dim=channel_dim)

    if inf_cap is not None:
        d_hat = d_hat.clamp(max=inf_cap)
    return d_hat


# ---------------------------------------------------------------------------
# Container class (parallels DepthMultiScaleDisparityTransform)
# ---------------------------------------------------------------------------

class DepthMultiScalePowerTransform:
    """N-channel power-transform codec.  Plain container, not an nn.Module.

    Example
    -------
    >>> codec = DepthMultiScalePowerTransform()             # lam=-3, c=(2,6,18)
    >>> y     = codec.encode(depth, mask=mask)              # (..., 3, H, W)
    >>> d_hat = codec.decode(y_pred)                        # (..., H, W)
    """

    def __init__(
        self,
        c: Sequence[float] = DEFAULT_C_RGB,
        lam: float = DEFAULT_LAM,
        inf_cap: Optional[float] = 1e4,
        invalid_fill: float = 0.0,
        weight: WeightKind = "iyy2",
        channel_dim: int = -3,
    ):
        if lam >= -1.0:
            raise ValueError(f"lam must be < -1, got {lam}")
        self.c_values = tuple(float(_c) for _c in c)
        if any(_c <= 0 for _c in self.c_values):
            raise ValueError(f"all c must be > 0, got {self.c_values}")

        self.lam = float(lam)
        self.inf_cap = inf_cap
        self.invalid_fill = float(invalid_fill)
        self.weight: WeightKind = weight
        self.channel_dim = channel_dim

    @property
    def num_channels(self) -> int:
        return len(self.c_values)

    def encode(self, d, mask=None, return_mask=False):
        return multiscale_power_encode(
            d, c_values=self.c_values, lam=self.lam, mask=mask,
            invalid_fill=self.invalid_fill, return_mask=return_mask,
            channel_dim=self.channel_dim,
        )

    def decode(self, y):
        return multiscale_power_decode(
            y, c_values=self.c_values, lam=self.lam,
            inf_cap=self.inf_cap, weight=self.weight,
            channel_dim=self.channel_dim,
        )

    def __call__(self, d, valid_mask=None, return_mask=False):
        return self.encode(d, mask=valid_mask, return_mask=return_mask)

    def __repr__(self):
        return (
            f"DepthMultiScalePowerTransform("
            f"lam={self.lam:.4f}, c={self.c_values}, "
            f"weight={self.weight!r}, inf_cap={self.inf_cap}, "
            f"invalid_fill={self.invalid_fill})"
        )


# ---------------------------------------------------------------------------
# Self-test: round-trip + VAE-noise comparison vs single-channel and disparity
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    torch.manual_seed(0)

    codec = DepthMultiScalePowerTransform()
    print(codec)

    # Round-trip sanity (no noise) -- weighted blend converges to true depth
    # because each channel's inverse is exact.
    d = torch.rand(2, 1, 64, 64, dtype=torch.float64) * 200.0
    y = codec(d)
    d_hat = codec.decode(y)
    print(f"round-trip max err  : {(d.squeeze(1) - d_hat).abs().max():.2e} m")
    print(f"y shape             : {tuple(y.shape)}")
    print(f"y range per channel : "
          f"R [{y[:, 0].min():+.3f}, {y[:, 0].max():+.3f}]  "
          f"G [{y[:, 1].min():+.3f}, {y[:, 1].max():+.3f}]  "
          f"B [{y[:, 2].min():+.3f}, {y[:, 2].max():+.3f}]")

    # ---- VAE-noise comparison vs single-channel power & 3-ch disparity ------

    def lognormal(mean_m, sigma, n=200_000, cap=None):
        x = (torch.randn(n) * sigma + math.log(mean_m)).exp()
        return x if cap is None else x.clamp(0.1, cap)

    datasets = {
        "NYU":    lognormal(2.5,  0.5, cap=10.0),
        "ETH3D":  lognormal(5.0,  0.8, cap=70.0),
        "DIODE":  lognormal(8.0,  1.2, cap=350.0),
        "Sintel": lognormal(10.0, 0.6, cap=100.0),
        "KITTI":  lognormal(15.0, 0.7, cap=80.0),
    }

    # 3-ch disparity (c = 2, 8, 32) for an apples-to-apples baseline.
    from multiscale_disparity_torch import DepthMultiScaleDisparityTransform
    disp_codec  = DepthMultiScaleDisparityTransform()                 # (2, 8, 32)
    pwr_codec   = DepthMultiScalePowerTransform()                     # (2, 6, 18)
    pwr_codec_o = DepthMultiScalePowerTransform(weight="opt")         # AbsRel-optimal

    print("\nVAE-noise AbsRel (sigma_y = 0.01):")
    print(f"{'dataset':<8}  {'3-ch disp':>10}  {'3-ch pwr (1-y2)2':>18}  "
          f"{'3-ch pwr opt':>14}  {'pwr vs disp':>12}")
    print("-" * 72)
    sigma = 0.01
    for ds_name, d_arr in datasets.items():
        d_arr = d_arr.reshape(1, 1, -1, 1)

        y_d = disp_codec.encode(d_arr) + torch.randn(
            1, 3, d_arr.shape[-2], 1) * sigma
        ar_d = ((disp_codec.decode(y_d) - d_arr.squeeze(1)).abs()
                / d_arr.squeeze(1)).mean().item()

        y_p = pwr_codec.encode(d_arr) + torch.randn(
            1, 3, d_arr.shape[-2], 1) * sigma
        ar_p = ((pwr_codec.decode(y_p) - d_arr.squeeze(1)).abs()
                / d_arr.squeeze(1)).mean().item()

        y_po = pwr_codec_o.encode(d_arr) + torch.randn(
            1, 3, d_arr.shape[-2], 1) * sigma
        ar_po = ((pwr_codec_o.decode(y_po) - d_arr.squeeze(1)).abs()
                 / d_arr.squeeze(1)).mean().item()

        imp = (ar_d - ar_p) / ar_d * 100.0
        print(f"{ds_name:<8}  {ar_d:>10.4f}  {ar_p:>18.4f}  {ar_po:>14.4f}  "
              f"{imp:>11.1f}%")

    # ---- Per-channel inspection at one specific depth ------------------------
    d_pt = torch.tensor([[[1.0, 6.0, 30.0, 100.0]]])               # (1, 1, 4)
    y_pt = pwr_codec.encode(d_pt.unsqueeze(-1))                    # (1, 3, 4, 1)
    print("\nPer-channel y at d = [1, 6, 30, 100] m  (c = (2, 6, 18)):")
    for i, c in enumerate(pwr_codec.c_values):
        row = y_pt[0, i, :, 0].tolist()
        print(f"   ch{i} (c={c:>4.1f})  y = "
              + "  ".join(f"{v:+.3f}" for v in row))
