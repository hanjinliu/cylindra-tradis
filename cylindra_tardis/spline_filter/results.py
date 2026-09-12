from typing import NamedTuple

import numpy as np


class FilterResult(NamedTuple):
    kept: list[int]
    dropped: list[int]


class JoinResult(NamedTuple):
    consumed: list[int]
    merged: list[np.ndarray]


class PeriodicityResult(NamedTuple):
    snr_v: float
    snr_h: float
