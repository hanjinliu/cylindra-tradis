from __future__ import annotations

import math

import impy as ip
import numpy as np
import polars as pl
from cylindra.components import CylTomogram
from cylindra.components._ftprops import PeakDetector
from cylindra.utils import ceilint

from cylindra_tardis.spline_filter.results import PeriodicityResult


def axial_periodicity(
    tomo: CylTomogram,
    i: int,
    *,
    binsize: int = 1,
    depth: float = 50.0,
    sum_every: float = 200.0,
) -> PeriodicityResult:
    spl = tomo.splines[i]
    rmin, rmax = spl.radius_range()
    rc = (rmin + rmax) / 2
    num_points = ceilint(spl.length() / sum_every)
    points = np.arange(1, num_points + 1) / (num_points + 1)
    npf = (
        spl.config.npf_range.center
    )  # this is just for limiting the range of rise angle
    ps_v = 0.0
    ps_h = 0.0
    for polar in tomo.iter_local_image(i, depth, points, binsize):
        # `polar` is already normalized to mean=0
        det = PeakDetector(polar, nsamples=4)
        ps_v += det.get_local_power_spectrum(**_params_v(polar, npf=npf))
        ps_h += det.get_local_power_spectrum(**_params_h(polar, radius=rc))
    snr_v = _safe_div(ps_v.max(), np.median(ps_v))
    snr_h = _safe_div(ps_h.max(), np.median(ps_h))

    return PeriodicityResult(snr_v=snr_v, snr_h=snr_h)


def measure_axial_periodicity(
    tomo: CylTomogram,
    idx: list[int],
    *,
    bin_size: int = 1,
    depth: float = 50.0,
    sum_every: float = 200.0,
) -> pl.DataFrame:
    if not idx:
        return pl.DataFrame()

    rows: list[dict[str, object]] = []
    for i in idx:
        res = axial_periodicity(
            tomo, i, binsize=bin_size, depth=depth, sum_every=sum_every
        )
        rows.append(
            {
                "spline": i,
                "length": float(tomo.splines[i].length()),
                "snr_v": res.snr_v,
                "snr_h": res.snr_h,
            }
        )
    return pl.DataFrame(rows)


def _safe_div(a, b):
    """Safely divide two numbers, returning `inf` if the denominator is zero."""
    return float(a / b) if b > 0 else float("inf")


def _params_h(img: ip.ImgArray, radius: float):
    spacing_arr = np.array([3.0, 5.5])[np.newaxis]
    y_factor = np.abs(radius / spacing_arr / img.shape.a * img.shape.y)
    tan_twist_min, tan_twist_max = -20, 20
    npf_min_max = (9, 18)
    return {
        "range_y": (
            np.min(tan_twist_min * y_factor * npf_min_max),
            np.max(tan_twist_max * y_factor * npf_min_max),
        ),
        "range_a": npf_min_max,
        "up_y": 5,
        "up_a": 5,
    }


def _params_v(img: ip.ImgArray, npf: float):
    tan_rise_min, tan_rise_max = 3, 45
    ya_scale_ratio = img.scale.y / img.scale.a
    _y_min = img.shape.y * img.scale.y / 5.5
    _y_max = img.shape.y * img.scale.y / 3.0
    _a_min = _y_max * tan_rise_min * img.shape.a / img.shape.y / ya_scale_ratio
    _a_max = _y_min * tan_rise_max * img.shape.a / img.shape.y / ya_scale_ratio
    _a_min, _a_max = sorted([-_a_min, -_a_max])
    return {
        "range_y": (_y_min, _y_max),
        "range_a": (int(math.floor(-npf / 2)), int(math.ceil(npf / 2))),
        "up_y": 5,
        "up_a": 5,
    }
