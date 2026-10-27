"""
PyTorch implementation of Barron's power transform for metric depth encoding:

    f(d; lam, c) = 1 - (1 + d / (|lam| * c)) ** (lam + 1)
                 = -expm1( (lam + 1) * log1p( d / (|lam| * c) ) )

The second form is what we actually evaluate: it is numerically stable for
every `lam < -1` and avoids `0 ** 0` / `0 ** neg` pathologies that hit the
naive closed form.

The paper's depth encoding uses `(lam, c) = (-3, 10/3)` so that
`f(d) = 1 - 1 / (1 + d/10)^2`; keep these as defaults.

Valid range for `lam`:  `lam < -1` (strictly).
  * At `lam = -1` the family degenerates (all outputs are 0).
  * For `lam > -1` the image is no longer contained in `[0, 1)`, so the
    depth encoding is ill-defined.

Practical range for `lam`:
  * `lam ~= -3` (the paper's default) gives roughly uniform resolution per
    octave of depth over ~[0.3 m, 100 m].
  * Useful range is roughly `lam in [-10, -1.5]`.  Past `lam ~ -15` the
    transform saturates so aggressively that `y = 2f - 1` cannot resolve
    large depths even in float64 (every `d > c` maps to within one ulp of 1);
    this is a property of the mapping, not of the implementation.

Functions
---------
power_transform(d, lam, c, mask)            : [0, inf)   -> [0, 1)
inverse_power_transform(f, lam, c, ...)     : [0, 1)     -> [0, inf)
depth_to_signed_unit(d, lam, c, mask, ...)  : [0, inf)   -> [-1, 1)
signed_unit_to_depth(y, lam, c, ...)        : [-1, 1)    -> [0, inf)

`lam` and `c` may each be either a Python float, a 0-d tensor, or a tensor
broadcastable against `d` (e.g. a per-sample `lam` of shape `[B, 1, 1, 1]`).

Class
-----
DepthPowerTransform : plain container that bundles `(lam, c, inf_cap,
invalid_fill)` and exposes `encode` / `decode` (also callable via `()`).
It is NOT an `nn.Module` -- this is purely a normalization utility, so it has
no buffers / parameters and doesn't need `.to(device)`.  Device/dtype follow
the input tensor automatically.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

# A scalar "shape parameter" may be passed as either a Python float or a
# Tensor (so it can be a learnable `nn.Parameter`, a per-sample value, ...).
ScalarLike = Union[float, Tensor]


# ---------------------------------------------------------------------------
# Small utility: bring a ScalarLike onto the same dtype / device as a ref
# tensor while preserving its shape (so broadcasting Just Works).
# ---------------------------------------------------------------------------

def _as_like(x: ScalarLike, ref: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


# ---------------------------------------------------------------------------
# Core forward / inverse
# ---------------------------------------------------------------------------

def power_transform(
    d: Tensor,
    lam: ScalarLike = -3.0,
    c: ScalarLike = 10.0 / 3.0,
    mask: Optional[Tensor] = None,
    invalid_fill: float = 0.0,
) -> Tensor:
    """Map metric depth `d` in `[0, inf)` to normalized `f` in `[0, 1)`.

    Computed as
        f = -expm1( (lam + 1) * log1p( d / (|lam| * c) ) )
    which is stable for any `lam < -1` and well-defined for any `d >= 0`.

    Args:
        d            : depth tensor, any shape (e.g. (1, H, W), (B, 1, H, W)).
        lam, c       : power-transform parameters; float, 0-d tensor, or tensor
                       broadcastable to `d` (lets you learn them or set them
                       per-sample).
        mask         : optional bool tensor broadcastable to `d`.  If given,
                       pixels with `mask == False` or `d <= 0` are written as
                       `invalid_fill` in the output.  If `None`, we just clamp
                       `d` to be non-negative.
        invalid_fill : scalar value written at invalid pixels (default 0.0).

    Returns:
        f : same shape as `d`, values in `[0, 1)`.
    """
    lam_t = _as_like(lam, d)
    c_t   = _as_like(c,   d)

    d_safe = d.clamp(min=0.0)
    # |lam| * c is always > 0 in the regime we care about (lam < -1, c > 0),
    # but we still guard against accidental zero.
    denom = lam_t.abs() * c_t
    denom = denom.clamp(min=torch.finfo(d.dtype).tiny)

    f = -torch.expm1((lam_t + 1.0) * torch.log1p(d_safe / denom))

    if mask is not None:
        valid = mask.to(torch.bool) & (d > 0)
        f = torch.where(valid, f, torch.full_like(f, invalid_fill))
    return f


def inverse_power_transform(
    f: Tensor,
    lam: ScalarLike = -3.0,
    c: ScalarLike = 10.0 / 3.0,
    inf_cap: Optional[float] = None,
    _one_minus_f: Optional[Tensor] = None,
) -> Tensor:
    """Map normalized `f` in `[0, 1)` back to metric depth `d` in `[0, inf)`.

    Computed as
        d = |lam| * c * expm1( log(1 - f) / (lam + 1) )

    `_one_minus_f` is an optional precomputed `(1 - f)` supplied by upstream
    callers that already have it in higher precision.  This matters because
    when `f` is close to 1, computing `1 - f` directly in float32 throws away
    much of the precision in `f`; `signed_unit_to_depth` passes `(1 - y)/2`
    through this path.

    At `f = 1` the depth is `+inf`; pass an `inf_cap` (e.g. 1e4) to keep
    gradients finite during VAE training.
    """
    lam_t = _as_like(lam, f)
    c_t   = _as_like(c,   f)

    if _one_minus_f is None:
        one_minus_f = (1.0 - f).clamp(min=0.0, max=1.0)
    else:
        one_minus_f = _one_minus_f.clamp(min=0.0, max=1.0)

    # Avoid exact zero so `log(0) = -inf -> exp(-inf) = 0` is a finite path.
    eps = torch.finfo(one_minus_f.dtype).tiny
    one_minus_f = one_minus_f.clamp(min=eps)

    denom_lam = (lam_t + 1.0)
    # Guard against lam_t -> -1 producing a 0-division at inference time.
    denom_lam = torch.where(
        denom_lam.abs() < eps,
        torch.full_like(denom_lam, -eps),
        denom_lam,
    )

    u = torch.expm1(torch.log(one_minus_f) / denom_lam)
    d = lam_t.abs() * c_t * u

    if inf_cap is not None:
        d = d.clamp(max=inf_cap)
    return d


# ---------------------------------------------------------------------------
# VAE-friendly: [0, inf) <-> [-1, 1) with optional mask
# ---------------------------------------------------------------------------

def depth_to_signed_unit(
    d: Tensor,
    lam: ScalarLike = -3.0,
    c: ScalarLike = 10.0 / 3.0,
    mask: Optional[Tensor] = None,
    invalid_fill: float = 0.0,
    return_mask: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Encode metric depth in `[0, inf)` to `y` in `[-1, 1)` via

        y = 2 * power_transform(d, lam, c) - 1

    Invalid pixels (either `mask == False` or `d <= 0`) are filled with
    `invalid_fill`.  Using `invalid_fill = 0` (the neutral centre of
    `[-1, 1]`) and passing the mask to the loss separately keeps real
    `d = 0` pixels (which encode to `y = -1`) distinct from "unknown"
    pixels.

    Args:
        d            : depth tensor, any shape.
        lam, c       : scalars or tensors broadcastable to `d`.
        mask         : optional bool tensor, same broadcastable shape as `d`.
                       True = valid, False = invalid.  If None, every `d > 0`
                       is considered valid.
        invalid_fill : value written at invalid pixels.
        return_mask  : if True, also return the effective valid-mask.

    Returns:
        y            : same shape as `d`, in `[-1, 1)`.
        valid (opt.) : bool tensor of same shape as `d`.
    """
    f = power_transform(d, lam=lam, c=c)                # raw [0,1)
    y = 2.0 * f - 1.0

    d_valid = d > 0
    valid = d_valid if mask is None else (mask.to(torch.bool) & d_valid)
    y = torch.where(valid, y, torch.full_like(y, invalid_fill))

    if return_mask:
        return y, valid
    return y


def signed_unit_to_depth(
    y: Tensor,
    lam: ScalarLike = -3.0,
    c: ScalarLike = 10.0 / 3.0,
    inf_cap: Optional[float] = 1e4,
) -> Tensor:
    """Decode `y` in `[-1, 1)` back to metric depth `d` in `[0, inf)`.

    Internally uses `(1 - f) = (1 - y) / 2` so that no precision is lost when
    `y` is close to 1 (large depth).
    """
    y_c = y.clamp(min=-1.0, max=1.0)
    one_minus_f = 0.5 * (1.0 - y_c)
    f = 1.0 - one_minus_f                               # for API consistency
    return inverse_power_transform(
        f, lam=lam, c=c, inf_cap=inf_cap, _one_minus_f=one_minus_f
    )


# ---------------------------------------------------------------------------
# Class wrapper: define once, use everywhere
# ---------------------------------------------------------------------------

class DepthPowerTransform:
    """Plain container bundling the power-transform depth codec's settings.

    Pure normalization utility -- no buffers, no parameters, not an `nn.Module`.
    The stored `lam` / `c` are Python floats; the actual forward / inverse
    computations happen entirely inside `depth_to_signed_unit` /
    `signed_unit_to_depth`, and device & dtype are picked up automatically from
    whatever tensor you pass in.

    Example
    -------
    >>> codec = DepthPowerTransform()                     # lam=-3, c=10/3
    >>> y          = codec.encode(depth, mask=mask)       # [0, inf) -> [-1, 1)
    >>> y          = codec(depth, mask=mask)              # __call__ == encode
    >>> y, valid   = codec(depth, mask=mask, return_mask=True)
    >>> d_hat      = codec.decode(y_pred)                 # [-1, 1) -> [0, inf)
    """

    def __init__(
        self,
        lam: float = -3.0,
        c: float = 10.0 / 3.0,
        inf_cap: Optional[float] = 1e4,
        invalid_fill: float = 0.0,
    ):
        if lam >= -1.0:
            raise ValueError(f"lam must be < -1, got {lam}")
        if c <= 0.0:
            raise ValueError(f"c must be > 0, got {c}")

        self.lam = float(lam)
        self.c = float(c)
        self.inf_cap = inf_cap
        self.invalid_fill = float(invalid_fill)

    def encode(
        self,
        d: Tensor,
        mask: Optional[Tensor] = None,
        return_mask: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Depth in `[0, inf)` -> y in `[-1, 1)`.  See `depth_to_signed_unit`."""
        return depth_to_signed_unit(
            d,
            lam=self.lam,
            c=self.c,
            mask=mask,
            invalid_fill=self.invalid_fill,
            return_mask=return_mask,
        )

    def decode(self, y: Tensor) -> Tensor:
        """y in `[-1, 1)` -> depth in `[0, inf)`.  See `signed_unit_to_depth`."""
        return signed_unit_to_depth(
            y,
            lam=self.lam,
            c=self.c,
            inf_cap=self.inf_cap,
        )

    def __call__(
        self,
        d: Tensor,
        valid_mask: Optional[Tensor] = None,
        return_mask: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return self.encode(d, mask=valid_mask, return_mask=return_mask)

    def __repr__(self) -> str:
        return (f"DepthPowerTransform(lam={self.lam:.4f}, c={self.c:.4f}, "
                f"inf_cap={self.inf_cap}, invalid_fill={self.invalid_fill})")


# ---------------------------------------------------------------------------
# Self-test: round-trip, gradient w.r.t. lam, learnable-lam example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) round-trip on a random (1, H, W) depth map with a mask
    d = torch.rand(1, 32, 32) * 100.0
    mask = d > 0.5
    y, valid = depth_to_signed_unit(d, mask=mask, return_mask=True)
    d_hat = signed_unit_to_depth(y)
    print(f"shape              : {tuple(d.shape)}")
    print(f"y range            : [{y.min().item():+.4f}, {y.max().item():+.4f}]")
    print(f"y at invalid pixels: {y[~valid].unique().tolist()}   (expect [0.0])")
    print(f"round-trip max err : {(d - d_hat)[valid].abs().max():.2e} m")

    # 2) match the legacy closed form at (lam, c) = (-3, 10/3) for sanity
    d2 = torch.linspace(0.0, 200.0, 500, dtype=torch.float64)
    f_legacy = 1.0 - 1.0 / (1.0 + d2 / 10.0) ** 2
    f_new    = power_transform(d2, lam=-3.0, c=10.0 / 3.0)
    print(f"\nmax|f_new - f_legacy| = {(f_new - f_legacy).abs().max():.2e}")

    # 3) vary `lam` freely: round-trip for several values (float64 for headroom)
    print("\nround-trip for varying lam (float64, should all be ~0):")
    for lam in [-1.5, -2.0, -3.0, -5.0, -10.0]:
        d3  = torch.rand(1, 64, 64, dtype=torch.float64) * 200.0
        y3  = depth_to_signed_unit(d3, lam=lam, c=10.0 / 3.0)
        d3h = signed_unit_to_depth(y3, lam=lam, c=10.0 / 3.0)
        err = (d3 - d3h).abs().max().item()
        print(f"   lam = {lam:>6.2f}   max|d - d_hat| = {err:.2e} m")

    # 4) per-sample `lam`: different shape parameter for each image in batch
    d4 = torch.rand(4, 1, 32, 32, dtype=torch.float64) * 100.0
    lam4 = torch.tensor([-1.5, -3.0, -5.0, -10.0]).view(4, 1, 1, 1)
    y4 = depth_to_signed_unit(d4, lam=lam4)
    d4h = signed_unit_to_depth(y4, lam=lam4)
    print(f"\nper-sample lam round-trip max err: "
          f"{(d4 - d4h).abs().max():.2e} m")

    # 5) `lam` as a learnable nn.Parameter: gradient flows through it
    lam_p = torch.nn.Parameter(torch.tensor(-3.0))
    c_p   = torch.nn.Parameter(torch.tensor(10.0 / 3.0))
    d5    = (torch.rand(2, 1, 16, 16) * 50.0).requires_grad_(True)
    y5    = depth_to_signed_unit(d5, lam=lam_p, c=c_p)
    loss  = (y5 - 0.1).pow(2).mean()
    loss.backward()
    print(f"\nlearnable lam  |  loss = {loss.item():.4f}"
          f"   d/dlam = {lam_p.grad.item():+.4f}"
          f"   d/dc = {c_p.grad.item():+.4f}"
          f"   max|d/dd| = {d5.grad.abs().max():.4f}")

    # 6) DepthPowerTransform class: define once, use everywhere
    print("\n--- DepthPowerTransform class ---")
    codec = DepthPowerTransform()
    print(codec)
    d6 = torch.rand(2, 1, 48, 48) * 80.0
    m6 = d6 > 0.2
    y6, v6 = codec(d6, mask=m6, return_mask=True)         # __call__ == encode
    d6h = codec.decode(y6)
    print(f"  round-trip err   : {(d6 - d6h)[v6].abs().max():.2e} m")

    # Non-default parameters.
    codec_b = DepthPowerTransform(lam=-4.0, c=5.0, inf_cap=1e3, invalid_fill=-1.0)
    print(codec_b)

    # Device/dtype follow the input tensor; no `.to()` / `.double()` needed.
    d8 = torch.rand(1, 64, 64, dtype=torch.float64) * 200.0
    y8 = codec(d8)
    print(f"  float64 in/out   : y.dtype={y8.dtype}, "
          f"round-trip = {(d8 - codec.decode(y8)).abs().max():.2e} m")
