"""GUI-independent implementation of the TARDIS trace curation workflow.

Each step operates on a `CylTomogram` in-place, so that the same logic can be used
both by the GUI step functions and by the batch processing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from cylindra_tardis import spline_filter
from cylindra_tardis.io import iter_tardis_coords
from cylindra_tardis.spline_filter.results import JoinResult
from cylindra_tardis.spline_filter.spline_filter import ReportHook

if TYPE_CHECKING:
    from cylindra.components import CylTomogram
    from cylindra.components.spline import SplineConfig
    from numpy.typing import NDArray


@dataclass(frozen=True)
class CurationParams:
    """Parameters of the full curation workflow."""

    bin_size: int = 1
    min_length: float = 10.0
    trim_edge_by: float = 24.5
    filter_duplicates: bool = True
    dedup_max_lateral: float = 5.0
    dedup_min_overlap: float = 0.8
    fit_max_interval: float = 30.0
    fit_err_max: float = 1.0
    fit_edge_sigma: float = 2.0
    filter_by_curvature: bool = True
    max_curvature: float = 0.05
    radius: float | None = 11.0
    filter_by_cft: bool = True
    min_snr: float = 6.0
    max_gap: float = 40.0
    max_lateral: float = 12.5
    max_angle: float = 15.0

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    def curate_tomogram(
        self,
        tomo: CylTomogram,
        path_coords: Path,
        *,
        config: SplineConfig | dict[str, Any] = {},
        report_hook: ReportHook | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> None:
        """Curate the tomogram using the TARDIS traces."""

        def _step(desc: str):
            if on_step is not None:
                on_step(desc)

        _step("Registering TARDIS traces as splines")
        tomo.splines.clear()
        coords = read_tardis_traces(
            tomo,
            path_coords,
            min_length=self.min_length,
            trim_edge_by=self.trim_edge_by,
        )
        register_traces(tomo, coords, err_max=self.fit_err_max, config=config)
        if self.filter_duplicates:
            _step("Filtering duplicate traces")
            filter_duplicates(
                tomo,
                max_lateral=self.dedup_max_lateral,
                min_overlap=self.dedup_min_overlap,
                report_hook=report_hook,
            )
        _step("Fitting splines")
        fit_all(
            tomo,
            bin_size=self.bin_size,
            max_interval=self.fit_max_interval,
            err_max=self.fit_err_max,
            edge_sigma=self.fit_edge_sigma,
        )
        if self.filter_by_curvature:
            _step("Filtering by curvature")
            filter_curvature(tomo, max_curvature=self.max_curvature)
        if self.filter_by_cft:
            _step("Setting radius")
            set_radius(tomo, self.radius, bin_size=self.bin_size)
            _step("Filtering by CFT peaks")
            filter_cft(
                tomo,
                min_snr=self.min_snr,
                bin_size=self.bin_size,
                report_hook=report_hook,
            )
        _step("Connecting collinear splines")
        connect_collinear(
            tomo,
            max_gap=self.max_gap,
            max_lateral=self.max_lateral,
            max_angle=self.max_angle,
            err_max=self.fit_err_max,
            config=config,
            report_hook=report_hook,
        )


def search_prediction_dir(tomogram_source: Path) -> Path:
    """Find the directory where TARDIS outputs of a tomogram are stored."""
    candidates = ["TARDIS", "Predictions"]
    for candidate in candidates:
        prediction_dir = Path(tomogram_source).parent / candidate
        if prediction_dir.exists():
            return prediction_dir
    raise ValueError(f"No TARDIS prediction directory found for {tomogram_source}.")


def default_instances_csv(tomogram_source: Path) -> Path:
    """Default path of the TARDIS instance CSV file of a tomogram."""
    tomogram_source = Path(tomogram_source)
    return (
        search_prediction_dir(tomogram_source) / f"{tomogram_source.stem}_instances.csv"
    )


def read_tardis_traces(
    tomo: CylTomogram,
    path_coords: Path,
    *,
    min_length: float = 10.0,
    trim_edge_by: float = 24.5,
) -> list[NDArray[np.float64]]:
    """Read the TARDIS traces that fit in the tomogram."""
    path_coords = Path(path_coords)
    if not path_coords.exists():
        raise FileNotFoundError(f"TARDIS coordinates not found: {path_coords}")
    return list(
        iter_tardis_coords(
            path_coords,
            tomo_size=np.array(tomo.image.shape) * tomo.scale,
            trim_edge_by=trim_edge_by,
            min_length=min_length,
        )
    )


def register_traces(
    tomo: CylTomogram,
    coords_list: Sequence[NDArray[np.float64]],
    *,
    err_max: float = 1.0,
    config: SplineConfig | dict[str, Any] = {},
) -> None:
    """Add coordinates to the tomogram as new splines."""
    for coords in coords_list:
        tomo.add_spline(coords, config=config, err_max=err_max)


def drop_splines(tomo: CylTomogram, indices: Sequence[int]) -> None:
    """Delete splines by index, highest first so the rest stay valid."""
    for i in sorted({int(x) for x in indices}, reverse=True):
        tomo.splines.pop(i)


def filter_duplicates(
    tomo: CylTomogram,
    *,
    max_lateral: float = 5.0,
    min_overlap: float = 0.8,
    report_hook: ReportHook | None = None,
) -> list[int]:
    """Drop near-duplicate splines and return the dropped indices."""
    result = spline_filter.filter_duplicate_splines(
        tomo,
        max_lateral=max_lateral,
        min_overlap_fraction=min_overlap,
        report_hook=report_hook,
    )
    drop_splines(tomo, result.dropped)
    return result.dropped


def fit_all(
    tomo: CylTomogram,
    *,
    bin_size: int = 1,
    max_interval: float = 30.0,
    err_max: float = 1.0,
    edge_sigma: float | None = 2.0,
    degree_precision: float = 0.5,
    max_shift: float = 5.0,
) -> None:
    """Fit all the splines to the tomogram."""
    for i in range(len(tomo.splines)):
        tomo.fit(
            i,
            max_interval=max_interval,
            binsize=bin_size,
            err_max=err_max,
            degree_precision=degree_precision,
            edge_sigma=edge_sigma,
            max_shift=max_shift,
        )


def filter_curvature(tomo: CylTomogram, *, max_curvature: float = 0.05) -> list[int]:
    """Drop splines with a large mean curvature and return the dropped indices."""
    result = spline_filter.filter_by_mean_curvature(tomo, max_curvature=max_curvature)
    drop_splines(tomo, result.dropped)
    return result.dropped


def set_radius(
    tomo: CylTomogram,
    radius: float | None,
    *,
    bin_size: int = 1,
    min_radius: float = 7.0,
    max_radius: float = 18.0,
) -> None:
    """Set the radius of all splines, or measure them if `radius` is None."""
    for i, spl in enumerate(tomo.splines):
        if radius is None:
            tomo.measure_radius(
                i, binsize=bin_size, min_radius=min_radius, max_radius=max_radius
            )
        else:
            spl.radius = radius


def filter_cft(
    tomo: CylTomogram,
    *,
    min_snr: float = 6.0,
    bin_size: int = 1,
    report_hook: ReportHook | None = None,
) -> list[int]:
    """Drop splines without a clear CFT peak and return the dropped indices."""
    result = spline_filter.filter_by_cft(
        tomo, min_snr=min_snr, bin_size=bin_size, report_hook=report_hook
    )
    drop_splines(tomo, result.dropped)
    return result.dropped


def connect_collinear(
    tomo: CylTomogram,
    *,
    max_gap: float = 40.0,
    max_lateral: float = 12.5,
    max_angle: float = 15.0,
    err_max: float = 1.0,
    config: SplineConfig | dict[str, Any] = {},
    report_hook: ReportHook | None = None,
) -> list[NDArray[np.float64]]:
    """Merge collinear splines into single splines and return the merged coords."""
    result = connect_collinear_coords(
        tomo,
        max_gap=max_gap,
        max_lateral=max_lateral,
        max_angle=max_angle,
        report_hook=report_hook,
    )
    drop_splines(tomo, result.consumed)
    register_traces(tomo, result.merged, err_max=err_max, config=config)
    return result.merged


def connect_collinear_coords(
    tomo: CylTomogram,
    *,
    max_gap: float = 40.0,
    max_lateral: float = 12.5,
    max_angle: float = 15.0,
    report_hook: ReportHook | None = None,
) -> JoinResult:
    """Find collinear splines without modifying the tomogram."""
    return spline_filter.connect_collinear_splines(
        tomo,
        max_distance=max_gap,
        max_lateral=max_lateral,
        max_angle=max_angle,
        report_hook=report_hook,
    )
