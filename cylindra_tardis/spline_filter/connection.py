from __future__ import annotations

from typing import NamedTuple

import numpy as np
from cylindra.components import CylSpline
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


class Samples(NamedTuple):
    """A spline sampled at regular intervals, extended beyond both ends."""

    points: NDArray[np.float64]
    """(N, 3) zyx coordinates [nm], including the two linear extensions."""

    arclength: NDArray[np.float64]
    """(N,) arclength [nm] from the original start; negative inside the leading
    extension and greater than ``length`` inside the trailing one."""

    length: float
    """Arclength [nm] of the original (unextended) spline."""

    @classmethod
    def from_spline(cls, spl: CylSpline, interval: float, extension: float) -> Samples:
        """Sample a spline, linearly extended by ``extension`` [nm] on both ends.

        Extending by half the allowed gap turns an end-to-end gap between two traces
        into an overlap, so the minimum of their distance matrix measures the lateral
        separation rather than the longitudinal gap.
        """
        length = float(spl.length())
        spl = spl.with_extrapolation("linear")
        num = max(int((length + 2 * extension) / interval) + 1, 4)
        u = np.linspace(-extension / length, 1.0 + extension / length, num)
        pts = np.asarray(spl.map(u), dtype=np.float64)

        # Exact arclength along the sampled polyline, zeroed at u = 0.
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(steps)])
        s0 = float(np.interp(0.0, u, s))
        s1 = float(np.interp(1.0, u, s))
        return cls(pts, s - s0, s1 - s0)


class Join(NamedTuple):
    """A candidate connection between two splines."""

    spline_a: int
    end_a: int
    spline_b: int
    end_b: int
    lateral: float
    """Minimum distance [nm] between the two extended curves."""
    gap: float
    """Distance [nm] between the two junction endpoints of the original curves.
    Roughly the gap length for separated traces and the overlap length for
    overlapping ones."""
    overlap: float
    """Arclength [nm] over which the two original curves run side by side.
    Reported for inspection only; it is not used as a rejection criterion."""
    overlap_fraction: float
    angle: float
    """Deviation [deg] from perfectly opposed inward directions: 0 means the two
    bodies are exactly collinear and run away from the junction in opposite
    directions."""
    cost: float


def evaluate_pair(
    ia: int,
    ib: int,
    sa: Samples,
    sb: Samples,
    *,
    max_distance: float,
    max_lateral: float,
    max_angle: float,
    direction_baseline: float,
    junction_tolerance: float,
) -> Join | None:
    """Score a candidate join from the full distance matrix of two extended curves.

    Every criterion is invariant under reversing either spline, so parallel and
    antiparallel pairs are treated identically -- there is no direction
    information in the segmentation to distinguish them with.

    The minimum of the extended-vs-extended matrix is the lateral separation, but
    it is degenerate for collinear curves, so it is never used to locate the
    junction. The junction is the closest pair of original endpoints, and
    ``max_distance`` bounds their separation -- a gap for traces that stop short
    of each other, an overlap length for traces that run past each other.
    """
    dist = cdist(sa.points, sb.points)
    lateral = float(dist.min())
    if lateral > max_lateral:
        return None

    # The two bodies must run away from the junction in opposite directions.
    # This rejects duplicates and containments, whose bodies run the same way,
    # and hairpins, whose ends meet but whose bodies run back alongside.
    chosen: tuple[int, int, float, float] | None = None
    for end_a, end_b, gap in _junction_candidates(sa, sb, junction_tolerance):
        if gap > max_distance:
            continue
        in_a = _inward(sa, end_a, direction_baseline)
        in_b = _inward(sb, end_b, direction_baseline)
        cos = min(max(float(np.dot(in_a, in_b)), -1.0), 1.0)
        angle = 180.0 - float(np.degrees(np.arccos(cos)))
        if angle <= max_angle:
            chosen = (end_a, end_b, gap, angle)
            break
    if chosen is None:
        return None
    end_a, end_b, gap, angle = chosen

    # Reported only: how much of the two ORIGINAL traces run side by side.
    body_a = (sa.arclength >= 0.0) & (sa.arclength <= sa.length)
    body_b = (sb.arclength >= 0.0) & (sb.arclength <= sb.length)
    sub = dist[np.ix_(body_a, body_b)]
    overlap = 0.0
    for arc, dmin in (
        (sa.arclength[body_a], sub.min(axis=1)),
        (sb.arclength[body_b], sub.min(axis=0)),
    ):
        hit = dmin <= max_lateral
        if hit.any():
            span = arc[hit]
            overlap = max(overlap, float(span.max() - span.min()))
    shortest = min(sa.length, sb.length)
    overlap_fraction = overlap / shortest if shortest > 0 else 0.0

    return Join(
        spline_a=ia,
        end_a=end_a,
        spline_b=ib,
        end_b=end_b,
        lateral=lateral,
        gap=gap,
        overlap=overlap,
        overlap_fraction=overlap_fraction,
        angle=angle,
        cost=lateral + 0.1 * gap,
    )


def _junction_candidates(
    sa: Samples, sb: Samples, tolerance: float
) -> list[tuple[int, int, float]]:
    """Closest pairs of original endpoints as ``(end of a, end of b, distance)``.

    Choosing the pair jointly, rather than each end independently, is what keeps
    duplicates and hairpins out: for those the closest endpoints are the ones
    sitting on top of each other, whose bodies then run the same way and fail the
    direction test. Selection is by physical position only, so it is unaffected
    by which way either spline happens to be parameterized.

    All pairs within ``tolerance`` of the closest are returned, nearest first,
    because the four endpoint distances can tie: when two traces overlap by about
    half their length, the junction pair is exactly as far apart as a pair
    involving the far end. The caller keeps the first candidate that also passes
    the direction test, so a true tie resolves correctly while a clear winner is
    never overruled.
    """
    ends_a = np.stack([_point_at(sa, 0.0), _point_at(sa, sa.length)])
    ends_b = np.stack([_point_at(sb, 0.0), _point_at(sb, sb.length)])
    dist = cdist(ends_a, ends_b)
    dmin = float(dist.min())
    out = [
        (i, j, float(dist[i, j]))
        for i in range(2)
        for j in range(2)
        if dist[i, j] <= dmin + tolerance
    ]
    out.sort(key=lambda t: t[2])
    return out


def _point_at(smp: Samples, arclength: float) -> NDArray[np.float64]:
    """Sampled point closest to the given arclength."""
    return smp.points[int(np.argmin(np.abs(smp.arclength - arclength)))]


def _inward(smp: Samples, end: int, baseline: float) -> NDArray[np.float64]:
    """Unit vector from one endpoint of the original spline into its own body.

    Defined from geometry alone, so it does not depend on which way the spline
    happens to be parameterized: flipping a spline swaps its two ends and leaves
    each end's inward direction unchanged.
    """
    reach = min(baseline, smp.length)
    if end == 0:
        return _unit(_point_at(smp, reach) - _point_at(smp, 0.0))
    return _unit(_point_at(smp, smp.length - reach) - _point_at(smp, smp.length))


def _unit(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec
