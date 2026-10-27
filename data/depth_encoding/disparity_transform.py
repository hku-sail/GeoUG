"""
Disparity (Cayley / Möbius) depth encoding:

    y = (d - c) / (d + c)   =   tanh( (1/2) * log(d / c) )

Bijection  [0, inf) -> [-1, 1)  with ONE parameter `c`:
    * c is the "pivot" depth that maps to y = 0.
    * d = 0       ->  y = -1
    * d = c       ->  y =  0
    * d -> inf    ->  y -> +1

Inverse:
    d = c * (1 + y) / (1 - y)

Why this encoding for multi-dataset training
--------------------------------------------
The L1 distance in y-space factors as

    |y_p - y_g|  =  2c * |d_p - d_g| / ((c + d_p)(c + d_g))

which interpolates smoothly between
    *  |d_p - d_g| / c        (near field,  d << c)
    *  2c * |1/d_p - 1/d_g|    (far  field, d >> c)

i.e. the effective metric-depth loss is flat in close-up and transitions to
the `1/depth` loss used by DepthPro / modern disparity-based depth learners
at far range.  That prior ("near depths deserve more precision") is baked
into the encoding itself, with no loss surgery required.

This is a special case of the log-logistic transform (`beta = 1`).  It is
called out as its own module because for multi-dataset (NYU + KITTI + ETH3D
+ DIODE + Sintel) training, `beta = 1` is the principled choice that aligns
the L1-on-y loss with the 1/depth prior.

See also
--------
    loglogistic_transform_torch.py : generalisation with a tunable `beta`.
    power_transform_torch.py       : Barron's power transform family.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def disparity_encode(
    d: Tensor,
    c: float = 5.0,
    mask: Optional[Tensor] = None,
    invalid_fill: float = 0.0,
    return_mask: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Encode metric depth `d` in `[0, inf)` to `y` in `[-1, 1)`.

        y = (d - c) / (d + c)

    Computed in the `(d - c) / (d + c)` algebraic form directly (stable for
    any `d >= 0`, `c > 0`).

    Args:
        d            : depth tensor, any shape (e.g. (1, H, W), (B, 1, H, W)).
        c            : pivot depth (the d that maps to y = 0).
        mask         : optional bool tensor broadcastable to `d`.  Pixels where
                       `mask == False` or `d <= 0` are written as
                       `invalid_fill`.  If `None`, just `d > 0` is required.
        invalid_fill : value at invalid pixels.
        return_mask  : if True, also return the effective valid-mask.
    """
    y = (d - c) / (d + c)

    d_valid = d > 0
    valid = d_valid if mask is None else (mask.to(torch.bool) & d_valid)
    y = torch.where(valid, y, torch.full_like(y, invalid_fill))

    if return_mask:
        return y, valid
    return y


def disparity_decode(
    y: Tensor,
    c: float = 5.0,
    inf_cap: Optional[float] = 1e4,
) -> Tensor:
    """Decode `y` in `[-1, 1)` back to metric depth `d` in `[0, inf)`.

        d = c * (1 + y) / (1 - y)

    `inf_cap` clips the output at a finite maximum (defends against
    `y -> 1` where d would blow up).
    """
    eps = torch.finfo(y.dtype).eps
    y_c = y.clamp(min=-1.0 + eps, max=1.0 - eps)
    d = c * (1.0 + y_c) / (1.0 - y_c)
    if inf_cap is not None:
        d = d.clamp(max=inf_cap)
    return d


# ---------------------------------------------------------------------------
# Parameter fitting
# ---------------------------------------------------------------------------

def fit_c_geometric_mean(
    depth_samples_by_dataset: Mapping[str, Tensor],
    masks_by_dataset: Optional[Mapping[str, Tensor]] = None,
    min_depth: float = 1e-3,
) -> float:
    """Pick `c` so it matches the geometric mean of the per-dataset medians.

    Rationale: Cayley is log-symmetric around `c`, so for cross-dataset
    training we want `c` to sit roughly at the log-center of all dataset
    medians.  The geometric mean of the medians is exactly that.
    """
    log_medians = []
    for name, d in depth_samples_by_dataset.items():
        m = None
        if masks_by_dataset is not None and name in masks_by_dataset:
            m = masks_by_dataset[name]
        if m is not None:
            d = d[m.to(torch.bool) & (d > min_depth)]
        else:
            d = d[d > min_depth]
        if d.numel() == 0:
            continue
        log_medians.append(math.log(float(d.to(torch.float64).median())))
    if not log_medians:
        raise ValueError("No valid depth samples.")
    return math.exp(sum(log_medians) / len(log_medians))


# ---------------------------------------------------------------------------
# Convenience class
# ---------------------------------------------------------------------------

class DepthDisparityTransform:
    """Plain container for the Cayley / disparity depth codec.

    Example
    -------
    >>> codec = DepthDisparityTransform()                     # c=5 (NYU+KITTI+...)
    >>> y = codec.encode(depth, mask=mask)                    # [0, inf) -> [-1, 1)
    >>> d_hat = codec.decode(y_pred)                          # [-1, 1) -> [0, inf)

    Ready-made multi-dataset preset:
    >>> codec = DepthDisparityTransform.for_multidataset()    # c = 5.0
    """

    def __init__(
        self,
        c: float = 5.0,
        inf_cap: Optional[float] = 1e4,
        invalid_fill: float = 0.0,
    ):
        if c <= 0.0:
            raise ValueError(f"c must be > 0, got {c}")
        self.c = float(c)
        self.inf_cap = inf_cap
        self.invalid_fill = float(invalid_fill)

    def encode(
        self,
        d: Tensor,
        mask: Optional[Tensor] = None,
        return_mask: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return disparity_encode(
            d, c=self.c, mask=mask,
            invalid_fill=self.invalid_fill, return_mask=return_mask,
        )

    def decode(self, y: Tensor) -> Tensor:
        return disparity_decode(y, c=self.c, inf_cap=self.inf_cap)

    def __call__(
        self,
        d: Tensor,
        valid_mask: Optional[Tensor] = None,
        return_mask: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return self.encode(d, mask=valid_mask, return_mask=return_mask)

    def __repr__(self) -> str:
        return (f"DepthDisparityTransform(c={self.c:.4f}, "
                f"inf_cap={self.inf_cap}, invalid_fill={self.invalid_fill})")

    # -- presets ------------------------------------------------------------

    @classmethod
    def for_multidataset(cls, **kwargs) -> "DepthDisparityTransform":
        """Preset for training on NYU + KITTI + ETH3D + DIODE + Sintel.

        The value c = 5.0 m is the approximate geometric mean of those
        datasets' medians (NYU ~2-3, ETH3D ~5, DIODE ~8, Sintel ~10,
        KITTI ~15).  With c = 5, every dataset's useful depth range maps to
        a well-spread segment of [-1, 1):

            NYU 0.3-10 m   ->  y in [-0.89, +0.33]
            ETH3D 0.3-70 m ->  y in [-0.89, +0.87]
            DIODE 0.3-350m ->  y in [-0.89, +0.97]
            Sintel 0.5-100m->  y in [-0.82, +0.90]
            KITTI 1-80 m   ->  y in [-0.67, +0.88]
        """
        return cls(c=5.0, **kwargs)

    @classmethod
    def fit(
        cls,
        depth_samples_by_dataset: Mapping[str, Tensor],
        masks_by_dataset: Optional[Mapping[str, Tensor]] = None,
        **kwargs,
    ) -> "DepthDisparityTransform":
        """Fit `c` to the geometric mean of per-dataset medians."""
        c = fit_c_geometric_mean(depth_samples_by_dataset, masks_by_dataset)
        return cls(c=c, **kwargs)


# ---------------------------------------------------------------------------
# Self-test + plot: encoded-y coverage on a simulated 5-dataset pool
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # --- basic round-trip --------------------------------------------------
    codec = DepthDisparityTransform.for_multidataset()
    print(codec)
    d = torch.rand(1, 64, 64, dtype=torch.float64) * 200.0
    y = codec(d)
    d_hat = codec.decode(y)
    print(f"  round-trip max err         : {(d - d_hat).abs().max():.2e} m")

    # --- simulate 5 datasets -----------------------------------------------
    def lognormal(mean_m, sigma, n=50_000):
        return torch.distributions.LogNormal(math.log(mean_m), sigma).sample([n])

    datasets = {
        "NYU":    lognormal(2.5,  0.5),   # indoor
        "ETH3D":  lognormal(5.0,  0.8),
        "DIODE":  lognormal(8.0,  1.2),
        "Sintel": lognormal(10.0, 0.6),
        "KITTI":  lognormal(15.0, 0.7),
    }
    # Clamp to reasonable per-dataset ranges
    caps = {"NYU": 10.0, "ETH3D": 70.0, "DIODE": 350.0,
            "Sintel": 100.0, "KITTI": 80.0}
    for name in datasets:
        datasets[name] = datasets[name].clamp(0.1, caps[name])

    # --- sanity: coverage of each dataset under the preset -----------------
    print("\ncoverage of c=5 codec:")
    for name, d_ds in datasets.items():
        y_ds = codec(d_ds)
        p05, p50, p95 = torch.quantile(y_ds, torch.tensor([0.05, 0.5, 0.95]))
        print(f"  {name:<7}  y5%={p05:+.3f}  y50%={p50:+.3f}  y95%={p95:+.3f}")

    # --- plot: y-histogram comparison across encodings ---------------------
    try:
        import os, tempfile
        os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Encodings to compare
        def enc_power(d, lam=-3.0, c=10.0 / 3.0):
            return 2.0 * (1.0 - (1.0 + d / (abs(lam) * c)) ** (lam + 1.0)) - 1.0

        def enc_loglog(d, mu=5.0, beta=2.0):
            return torch.tanh(0.5 * beta * torch.log(d.clamp(min=1e-6) / mu))

        def enc_disp(d, c=5.0):
            return (d - c) / (d + c)

        encodings = {
            "power (c=10/3, indoor)": lambda d: enc_power(d, lam=-3.0, c=10.0 / 3.0),
            "power (c=20, driving)":  lambda d: enc_power(d, lam=-3.0, c=20.0),
            "loglogistic (mu=5, beta=2)": lambda d: enc_loglog(d, mu=5.0, beta=2.0),
            "disparity (c=5)  <-- proposal": lambda d: enc_disp(d, c=5.0),
        }

        fig, axes = plt.subplots(
            len(encodings), 1, figsize=(9, 2.2 * len(encodings)), sharex=True)
        bins = torch.linspace(-1, 1, 41)
        colors = {
            "NYU":    "#1f77b4",
            "ETH3D":  "#2ca02c",
            "DIODE":  "#ff7f0e",
            "Sintel": "#d62728",
            "KITTI":  "#9467bd",
        }

        for ax, (name, enc) in zip(axes, encodings.items()):
            for ds_name, d_ds in datasets.items():
                y_ds = enc(d_ds)
                ax.hist(y_ds.numpy(), bins=bins.numpy(), density=True,
                        histtype="step", linewidth=2.0,
                        label=ds_name, color=colors[ds_name])
            ax.set_ylabel("density")
            ax.set_title(name, fontsize=10)
            ax.set_xlim(-1.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=5, loc="upper center")
        axes[-1].set_xlabel("encoded  y  in  [-1, 1]")
        fig.suptitle("y-distribution coverage across 5 datasets under different encodings",
                     y=1.02, fontsize=12)
        plt.tight_layout()
        out = "multidataset_encoding_coverage.png"
        plt.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nSaved coverage plot to  {out}")
    except Exception as e:
        print(f"\n(Skipped plot: {e})")
