from __future__ import annotations

from typing import Callable, Iterator, Sequence

import numpy as np
import polars as pl
from cylindra.components import CylTomogram
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from cylindra_tardis.spline_filter.connection import Join, Samples, evaluate_pair
from cylindra_tardis.spline_filter.periodicity import measure_axial_periodicity
from cylindra_tardis.spline_filter.results import FilterResult, JoinResult

ReportHook = Callable[[str, pl.DataFrame], None]

TARDIS_ID_COLUMN = "IDs"
TARDIS_XYZ_COLUMNS = ("X [A]", "Y [A]", "Z [A]")


def _report(name: str, df: pl.DataFrame, hook: ReportHook | None) -> pl.DataFrame:
    """Log a report and hand it to the caller's hook, if any."""
    if hook is not None:
        hook(name, df)
    return df


def _norm_indices(tomo: CylTomogram, indices: Sequence[int] | None) -> list[int]:
    """Normalize a spline selection into a list of non-negative indices."""
    n = len(tomo.splines)
    if indices is None:
        return list(range(n))
    return [int(i) % n for i in indices]


def _drop_splines(tomo: CylTomogram, indices: Sequence[int]) -> None:
    """Delete splines by index, highest first so the rest stay valid."""
    for i in sorted({int(x) for x in indices}, reverse=True):
        del tomo.splines[i]


# --------------------------------------------------------------------------- #
#  Step 1: TARDIS output -> splines
# --------------------------------------------------------------------------- #


def tardis_filaments(
    df: pl.DataFrame,
    *,
    id_column: str = TARDIS_ID_COLUMN,
    xyz_columns: tuple[str, str, str] = TARDIS_XYZ_COLUMNS,
    unit_scale: float = 0.1,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """Split a TARDIS instance table into per-filament coordinate arrays.

    Yields ``(filament id, (N, 3) array)`` where the array is in cylindra's zyx
    order and in nanometres. Points keep the order they have in the table, which
    is already sorted along each filament by TARDIS.

    Parameters
    ----------
    df : pl.DataFrame
        Contents of a ``*_instances.csv`` written by ``tardis_mt``.
    id_column : str, default "IDs"
        Column holding the filament id.
    xyz_columns : tuple of str, default ("X [A]", "Y [A]", "Z [A]")
        Columns holding the coordinates, in x, y, z order as TARDIS writes them.
    unit_scale : float, default 0.1
        Factor converting the table's unit to nanometres. TARDIS writes
        Angstrom, hence 0.1.
    origin : tuple of float, default (0, 0, 0)
        Offset [nm] in zyx added after scaling, for the case where the tomogram
        cylindra loaded is not exactly the volume TARDIS ran on.
    """
    col_x, col_y, col_z = xyz_columns
    missing = [c for c in (id_column, *xyz_columns) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in the TARDIS table: {missing}")

    off = np.asarray(origin, dtype=np.float64)
    for key, sub in df.group_by(id_column, maintain_order=True):
        fid = key[0] if isinstance(key, tuple) else key
        # cylindra uses zyx; TARDIS writes xyz.
        zyx = sub.select(col_z, col_y, col_x).to_numpy().astype(np.float64)
        yield int(fid), zyx * unit_scale + off


def filter_by_mean_curvature(
    tomo: CylTomogram,
    max_curvature: float = 0.01,
) -> FilterResult:
    """Filter splines by their mean curvature.

    Parameters
    ----------
    tomo : CylTomogram
        The tomogram containing the splines.
    max_curvature : float
        Maximum mean curvature [1/nm] for a spline to be kept.
    """
    kept: list[int] = []
    dropped: list[int] = []
    for i, spl in enumerate(tomo.splines):
        u = spl.prep_anchor_positions(interval=2.0)
        mean_curv = float(np.mean(spl.curvature(u)))
        if mean_curv <= max_curvature:
            kept.append(i)
        else:
            dropped.append(i)
    return FilterResult(kept=kept, dropped=dropped)


def filter_duplicate_splines(
    tomo: CylTomogram,
    indices: Sequence[int] | None = None,
    *,
    max_lateral: float = 5.0,
    min_overlap_fraction: float = 0.8,
    sample_interval: float = 5.0,
    report_hook: ReportHook | None = None,
) -> FilterResult:
    """Drop splines that are near-duplicates of another, longer spline.

    TARDIS sometimes emits two (or more) separate traces for the same
    microtubule, offset laterally by only a few nanometers. Those bodies run
    the same way rather than away from a shared junction, so
    `connect_collinear_splines` correctly refuses to merge them -- and without
    a separate check they both survive as nearly coincident splines. This
    keeps the longest spline in each overlapping group and drops the rest.

    Parameters
    ----------
    max_lateral : float, default 5.0
        Maximum separation [nm] for a sampled point on one spline to count as
        covered by the other. Keep this well below the typical spacing
        between distinct microtubules so that genuinely adjacent filaments
        are not mistaken for duplicates.
    min_overlap_fraction : float, default 0.8
        Minimum fraction of the shorter spline's arclength that must be
        covered by a longer spline for the shorter one to be dropped.
    """
    idx = _norm_indices(tomo, indices)
    if len(idx) < 2:
        return FilterResult(kept=idx, dropped=[])

    samples = [Samples.from_spline(tomo.splines[i], sample_interval, 0.0) for i in idx]
    pairs = _candidate_pairs(samples, max_lateral)

    # Longest pair first, so a chain of duplicates collapses onto the longest
    # spline instead of onto whichever pair happens to be evaluated first.
    order = sorted(
        pairs, key=lambda p: -max(samples[p[0]].length, samples[p[1]].length)
    )

    rows: list[dict[str, object]] = []
    dropped: set[int] = set()
    for ia, ib in order:
        dist = cdist(samples[ia].points, samples[ib].points)
        frac_a = float(np.mean(dist.min(axis=1) <= max_lateral))
        frac_b = float(np.mean(dist.min(axis=0) <= max_lateral))
        if samples[ia].length <= samples[ib].length:
            shorter, longer, frac = ia, ib, frac_a
        else:
            shorter, longer, frac = ib, ia, frac_b
        is_dup = frac >= min_overlap_fraction
        rows.append(
            {
                "spline_a": idx[ia],
                "spline_b": idx[ib],
                "shorter": idx[shorter],
                "longer": idx[longer],
                "overlap_fraction": frac,
                "duplicate": is_dup,
            }
        )
        if is_dup:
            dropped.add(idx[shorter])

    report = pl.DataFrame(rows)
    _report("filter_duplicate_splines", report, report_hook)

    kept = [i for i in idx if i not in dropped]
    return FilterResult(kept=kept, dropped=sorted(dropped))


def filter_by_cft(
    tomo: CylTomogram,
    indices: Sequence[int] | None = None,
    *,
    min_snr: float = 3.0,
    bin_size: int = 1,
    report_hook: ReportHook | None = None,
) -> FilterResult:
    """Delete splines without a convincing CFT peak.

    Parameters
    ----------
    min_snr : float, default 3.0
        Minimum ratio of the peak power to the background around peak.
    """
    periodicity_result = measure_axial_periodicity(
        tomo,
        _norm_indices(tomo, indices),
        bin_size=bin_size,
    )
    if periodicity_result.height == 0:
        return _report("filter_by_axial_periodicity", periodicity_result, report_hook)

    report = periodicity_result.with_columns(
        ((pl.col("snr_v") >= min_snr) & (pl.col("snr_h") >= min_snr)).alias("keep")
    )
    kept = report.filter(pl.col("keep"))["spline"].to_list()
    dropped = report.filter(~pl.col("keep"))["spline"].to_list()
    _report("filter_by_axial_periodicity", report, report_hook)

    return FilterResult(kept=kept, dropped=dropped)


def _candidate_pairs(
    samples: list[Samples], max_lateral: float
) -> set[tuple[int, int]]:
    """Spline pairs with at least one pair of sampled points within ``max_lateral``.

    A KD-tree pre-filter keeps the full distance matrices to the few pairs that
    can possibly connect, instead of all N^2 of them.
    """
    if not samples:
        return set()
    pts = np.concatenate([s.points for s in samples], axis=0)
    owner = np.concatenate(
        [np.full(len(s.points), i, dtype=np.int64) for i, s in enumerate(samples)]
    )
    tree = cKDTree(pts)
    close = tree.query_pairs(r=max_lateral, output_type="ndarray")
    if close.size == 0:
        return set()
    oa, ob = owner[close[:, 0]], owner[close[:, 1]]
    keep = oa != ob
    return {
        (int(min(a, b)), int(max(a, b)))
        for a, b in zip(oa[keep], ob[keep], strict=True)
    }


def _stitch_pieces(
    pieces: list[NDArray[np.float64]], max_trim: float
) -> NDArray[np.float64]:
    """Concatenate oriented pieces of a chain into one monotonic polyline.

    Pieces that overlap would otherwise make the merged point set double back on
    itself, and a spline fitted through that wobbles at every junction. Each new
    piece therefore truncates the accumulated path at the point closest to the
    new piece's first point, which removes exactly the shared stretch and is a
    no-op when the two pieces are separated by a gap.

    ``max_trim`` caps how much may be removed, so a pathological match can never
    eat an entire piece.
    """
    merged = pieces[0]
    for piece in pieces[1:]:
        dist = np.linalg.norm(merged - piece[0], axis=1)
        k = int(np.argmin(dist))
        tail = merged[k:]
        trimmed = float(np.linalg.norm(np.diff(tail, axis=0), axis=1).sum())
        if trimmed <= max_trim:
            merged = merged[: k + 1]
        merged = np.concatenate([merged, piece], axis=0)
    return merged


def _iter_chains(
    ids: list[int], matches: dict[tuple[int, int], tuple[int, int]]
) -> Iterator[list[tuple[int, bool]]]:
    """Walk the matching into chains of ``(local id, traversed backwards)``."""
    visited: set[int] = set()
    heads = [i for i in ids if (i, 0) not in matches or (i, 1) not in matches]
    for start in heads:
        if start in visited:
            continue
        entry = 0 if (start, 0) not in matches else 1
        chain: list[tuple[int, bool]] = []
        cur = start
        while True:
            chain.append((cur, entry == 1))
            visited.add(cur)
            nxt = matches.get((cur, 1 - entry))
            if nxt is None or nxt[0] in visited:
                break
            cur, entry = nxt
        yield chain


def connect_collinear_splines(
    tomo: CylTomogram,
    indices: Sequence[int] | None = None,
    *,
    max_distance: float = 250.0,
    max_lateral: float = 15.0,
    max_angle: float = 20.0,
    direction_baseline: float = 50.0,
    junction_tolerance: float = 10.0,
    sample_interval: float = 5.0,
    interval: float = 5.0,
    report_hook: ReportHook | None = None,
) -> JoinResult:
    """Join splines of `tomo` that lie on a common axis.

    Consumed splines are deleted and each chain is re-registered as one spline,
    so `tomo.splines` is rebuilt. The merged splines carry no properties; run
    `CylTomogram.fit` and the analyses again on them afterwards.

    Purely geometric and direction-free: parallel and antiparallel pairs are not
    distinguished, since segmentation output carries no polarity and the
    parameter direction of a spline is arbitrary. Endpoints are not compared
    directly either, because two TARDIS traces of the same microtubule
    frequently overlap by ~100 nm. Instead every spline is linearly extended by
    `max_distance / 2` on both ends; the minimum of the distance matrix
    between two extended curves is the lateral separation, the closest pair of
    original endpoints is the junction, and the two bodies must run away from
    that junction in opposite directions.

    Parameters
    ----------
    max_distance : float, default 250.0
        Largest separation [nm] between the two junction endpoints: a gap when
        the traces stop short of each other, an overlap length when they run
        past each other. Each spline is extended by half of it.
    max_lateral : float, default 15.0
        Maximum separation [nm] of the two extended curves at closest approach.
        Keep this below the typical microtubule spacing in the sample.
    max_angle : float, default 20.0
        Maximum deviation [deg] from perfectly opposed inward directions, i.e.
        how much of a kink is tolerated at the junction.
    direction_baseline : float, default 50.0
        Baseline [nm] over which the inward direction of each spline is measured.
        A longer baseline is less sensitive to noise near the endpoint but blurs
        a genuinely curved trace.
    junction_tolerance : float, default 10.0
        Endpoint pairs within this distance [nm] of the closest one are all tried
        as the junction, nearest first. Only matters when the four endpoint
        distances nearly tie, which happens when two traces overlap by about half
        their length.
    sample_interval : float, default 5.0
        Sampling interval [nm] used to build the distance matrices.
    interval : float, default 5.0
        Sampling interval [nm] used when resampling a chain into one point set.
    err_max : float, default 1.0
        Fitting error tolerance [nm] passed to ``CylTomogram.add_spline``.
    """
    idx = _norm_indices(tomo, indices)
    if len(idx) < 2:
        _report("connect_collinear_splines", pl.DataFrame(), report_hook)
        return JoinResult(consumed=[], merged=[])

    extension = max_distance / 2
    samples = [
        Samples.from_spline(tomo.splines[i], sample_interval, extension) for i in idx
    ]
    pairs = _candidate_pairs(samples, max_lateral)

    joins: list[Join] = []
    for ia, ib in pairs:
        join = evaluate_pair(
            ia,
            ib,
            samples[ia],
            samples[ib],
            max_distance=max_distance,
            max_lateral=max_lateral,
            max_angle=max_angle,
            direction_baseline=direction_baseline,
            junction_tolerance=junction_tolerance,
        )
        if join is not None:
            joins.append(join)
    joins.sort(key=lambda j: j.cost)

    # Greedy matching; union-find keeps the chains acyclic.
    parent: dict[int, int] = {i: i for i in range(len(idx))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    matches: dict[tuple[int, int], tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()
    accepted: list[dict[str, object]] = []
    for join in joins:
        ka = (join.spline_a, join.end_a)
        kb = (join.spline_b, join.end_b)
        if ka in used or kb in used:
            continue
        ra, rb = find(join.spline_a), find(join.spline_b)
        if ra == rb:
            continue
        parent[ra] = rb
        matches[ka], matches[kb] = kb, ka
        used.add(ka)
        used.add(kb)
        # Report global spline indices, not the local ones used internally.
        row = dict(join._asdict())
        row["spline_a"] = idx[join.spline_a]
        row["spline_b"] = idx[join.spline_b]
        accepted.append(row)

    report = pl.DataFrame(accepted)
    _report("connect_collinear_splines", report, report_hook)

    chains = [c for c in _iter_chains(list(range(len(idx))), matches) if len(c) > 1]

    # Resample every chain before touching the spline list, since deleting
    # splines invalidates the indices.
    merged: list[NDArray[np.float64]] = []
    for chain in chains:
        pieces: list[NDArray[np.float64]] = []
        for local, reverse in chain:
            spl = tomo.splines[idx[local]]
            num = max(int(spl.length() / interval) + 1, 2)
            pts = np.asarray(spl.map(np.linspace(0, 1, num)), dtype=np.float64)
            pieces.append(pts[::-1] if reverse else pts)
        merged.append(_stitch_pieces(pieces, max_trim=max_distance))

    consumed = [idx[local] for chain in chains for local, _ in chain]

    return JoinResult(consumed=consumed, merged=merged)
