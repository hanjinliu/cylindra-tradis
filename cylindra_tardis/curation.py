"""Plugin functions for curating TARDIS traces step by step or in batch."""

from pathlib import Path as _Path
from typing import Annotated, Any

import polars as pl
from cylindra.components import CylTomogram
from cylindra.components.spline import SplineConfig
from cylindra.plugin import register_function
from cylindra.project import CylindraProject
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.batch._sequence import PathInfo
from magicclass.types import Optional, Path
from magicclass.utils import thread_worker

from cylindra_tardis import workflow
from cylindra_tardis.core import _get_loader_paths


@register_function(name="Register TARDIS traces")
@thread_worker.with_progress("Registering TARDIS traces")
def register_tardis_traces(
    ui: CylindraMainWidget,
    path_coords: Annotated[
        Optional[Path.Read["*.csv"]],
        {"label": "TARDIS coordinates", "text": "Search by tomogram path"},
    ] = None,
    min_length: Annotated[float, {"label": "min length (nm)"}] = 10.0,
    trim_edge_by: Annotated[float, {"label": "trim edge by (nm)"}] = 24.5,
    err_max: Annotated[float, {"label": "max fit error (nm)", "step": 0.1}] = 1.0,
):
    """Replace all the existing splines with the TARDIS traces.

    Parameters
    ----------
    path_coords : Path, optional
        Path to the TARDIS instance CSV file. If not given, it will be searched in the
        "TARDIS" or "Predictions" directory next to the tomogram.
    min_length : float, default 10.0
        Traces shorter than this length (nm) are ignored.
    trim_edge_by : float, default 24.5
        Trace points closer to the XY edges of the tomogram than this (nm) are trimmed.
    err_max : float, default 1.0
        Maximum fitting error (nm) used to register the traces as splines.
    """
    tomo = _get_tomogram(ui)
    if path_coords is None:
        path_coords = workflow.default_instances_csv(tomo.source)
    ui.logger.print(f"Reading TARDIS coordinates: {path_coords}")
    coords = workflow.read_tardis_traces(
        tomo, path_coords, min_length=min_length, trim_edge_by=trim_edge_by
    )
    tomo.splines.clear()
    workflow.register_traces(tomo, coords, err_max=err_max, config=ui.default_config)
    ui.logger.print(f"{len(coords)} traces registered.")
    yield _on_spline_updated.with_args(ui)


@register_function(name="Filter duplicate splines")
@thread_worker.with_progress("Filtering duplicate splines")
def filter_duplicate_splines(
    ui: CylindraMainWidget,
    max_lateral: Annotated[float, {"label": "max lateral (nm)"}] = 5.0,
    min_overlap: Annotated[
        float, {"label": "min overlap fraction", "min": 0.0, "max": 1.0}
    ] = 0.8,
):
    """Delete splines that are near-duplicates of another, longer spline.

    Parameters
    ----------
    max_lateral : float, default 5.0
        Maximum separation (nm) for a point on one spline to count as covered by the
        other.
    min_overlap : float, default 0.8
        Minimum fraction of the shorter spline covered by the longer one for the
        shorter one to be deleted.
    """
    dropped = workflow.filter_duplicates(
        _get_tomogram(ui),
        max_lateral=max_lateral,
        min_overlap=min_overlap,
        report_hook=_report_hook_for(ui),
    )
    _log_dropped(ui, dropped)
    yield _on_spline_updated.with_args(ui)


@register_function(name="Filter splines by curvature")
@thread_worker.with_progress("Filtering splines by curvature")
def filter_splines_by_curvature(
    ui: CylindraMainWidget,
    max_curvature: Annotated[
        float, {"step": 0.00001, "min": 0.00001, "label": "max curvature (nm^-1)"}
    ] = 0.05,
):
    """Delete splines whose mean curvature is too large.

    Parameters
    ----------
    max_curvature : float, default 0.05
        Maximum mean curvature (1/nm) for a spline to be kept.
    """
    dropped = workflow.filter_curvature(_get_tomogram(ui), max_curvature=max_curvature)
    _log_dropped(ui, dropped)
    yield _on_spline_updated.with_args(ui)


@register_function(name="Filter splines by CFT")
@thread_worker.with_progress("Filtering splines by CFT peaks")
def filter_splines_by_cft(
    ui: CylindraMainWidget,
    bin_size: Annotated[int, {"min": 1}] = 1,
    min_snr: Annotated[float, {"step": 0.1, "min": 0.1, "label": "min S/N"}] = 6.0,
):
    """Delete splines without clear axial and lateral peaks in the CFT.

    Spline radius must be set before running this filter.

    Parameters
    ----------
    bin_size : int, default 1
        Bin size of the tomogram used for the CFT.
    min_snr : float, default 6.0
        Minimum ratio of the peak power to the background for a spline to be kept.
    """
    dropped = workflow.filter_cft(
        _get_tomogram(ui),
        min_snr=min_snr,
        bin_size=bin_size,
        report_hook=_report_hook_for(ui),
    )
    _log_dropped(ui, dropped)
    yield _on_spline_updated.with_args(ui)


@register_function(name="Connect collinear splines")
@thread_worker.with_progress("Connecting collinear splines")
def connect_collinear_splines(
    ui: CylindraMainWidget,
    max_gap: Annotated[float, {"label": "max gap (nm)"}] = 40.0,
    max_lateral: Annotated[float, {"label": "max lateral (nm)"}] = 12.5,
    max_angle: Annotated[float, {"label": "max angle (deg)"}] = 15.0,
    err_max: Annotated[float, {"label": "max fit error (nm)", "step": 0.1}] = 1.0,
):
    """Merge splines that lie on a common axis into single splines.

    The merged splines are appended to the end of the spline list and carry no
    properties.

    Parameters
    ----------
    max_gap : float, default 40.0
        Largest separation (nm) between the two junction endpoints.
    max_lateral : float, default 12.5
        Maximum lateral separation (nm) of the two splines.
    max_angle : float, default 15.0
        Maximum kink angle (deg) tolerated at the junction.
    err_max : float, default 1.0
        Maximum fitting error (nm) used to register the merged splines.
    """
    merged = workflow.connect_collinear(
        _get_tomogram(ui),
        max_gap=max_gap,
        max_lateral=max_lateral,
        max_angle=max_angle,
        err_max=err_max,
        config=ui.default_config,
    )
    ui.logger.print(f"{len(merged)} merged splines created.")
    yield _on_spline_updated.with_args(ui)


@register_function(name="Curate TARDIS traces", record=False)
def open_curation_widget(ui: CylindraMainWidget):
    """Open a widget to run the TARDIS trace curation on the current tomogram."""
    from cylindra_tardis.widget import TardisCurationWidget, open_widget

    return open_widget(TardisCurationWidget, ui)


@register_function(name="Curate TARDIS traces (batch)", record=False)
def open_batch_curation_widget(ui: CylindraMainWidget):
    """Open a widget to run the TARDIS trace curation on the batch projects."""
    from cylindra_tardis.widget import TardisBatchCurationWidget, open_widget

    return open_widget(TardisBatchCurationWidget, ui)


@register_function
@thread_worker.with_progress("Curating TARDIS traces (batch)", total="len(path_sets)")
def curate_tardis_traces_batch(
    ui: CylindraMainWidget,
    path_sets: Annotated[Any, {"bind": _get_loader_paths}],
    bin_size: Annotated[int, {"min": 1}] = 1,
    min_length: Annotated[float, {"label": "min length (nm)"}] = 10.0,
    trim_edge_by: Annotated[float, {"label": "trim edge by (nm)"}] = 24.5,
    filter_duplicates: bool = True,
    dedup_max_lateral: Annotated[float, {"label": "duplicate max lateral (nm)"}] = 5.0,
    dedup_min_overlap: Annotated[
        float, {"label": "duplicate min overlap fraction", "min": 0.0, "max": 1.0}
    ] = 0.8,
    fit_max_interval: Annotated[float, {"label": "fit max interval (nm)"}] = 30.0,
    fit_err_max: Annotated[float, {"label": "max fit error (nm)", "step": 0.1}] = 1.0,
    fit_edge_sigma: Annotated[float, {"label": "fit edge σ (nm)"}] = 2.0,
    filter_by_curvature: bool = True,
    max_curvature: Annotated[
        float, {"step": 0.00001, "min": 0.00001, "label": "max curvature (nm^-1)"}
    ] = 0.05,
    filter_by_cft: bool = True,
    radius: Annotated[
        Optional[float], {"label": "radius (nm)", "text": "Measure one by one"}
    ] = 11.0,
    min_snr: Annotated[float, {"step": 0.1, "min": 0.1, "label": "min S/N"}] = 6.0,
    max_gap: Annotated[float, {"label": "max gap (nm)"}] = 40.0,
    max_lateral: Annotated[float, {"label": "max lateral (nm)"}] = 12.5,
    max_angle: Annotated[float, {"label": "max angle (deg)"}] = 15.0,
):
    """Run the TARDIS trace curation on all the projects checked in batch analyzer.

    For each project, TARDIS traces are read from the "TARDIS" or "Predictions"
    directory next to the tomogram, curated, and saved in the project as
    "spline-*.json", replacing all the existing splines.
    """
    params = workflow.CurationParams(
        bin_size=bin_size,
        min_length=min_length,
        trim_edge_by=trim_edge_by,
        filter_duplicates=filter_duplicates,
        dedup_max_lateral=dedup_max_lateral,
        dedup_min_overlap=dedup_min_overlap,
        fit_max_interval=fit_max_interval,
        fit_err_max=fit_err_max,
        fit_edge_sigma=fit_edge_sigma,
        filter_by_curvature=filter_by_curvature,
        max_curvature=max_curvature,
        filter_by_cft=filter_by_cft,
        radius=radius,
        min_snr=min_snr,
        max_gap=max_gap,
        max_lateral=max_lateral,
        max_angle=max_angle,
    )
    default_config = ui.default_config
    ui.logger.print(f"Curating TARDIS traces in {len(path_sets)} projects")
    for path_set in path_sets:
        path_set = PathInfo(*path_set)
        prj = path_set.project_instance(missing_ok=False)
        ui.logger.print(f"Processing {path_set.project}")
        path_coords = workflow.default_instances_csv(path_set.image)
        tomo = _load_tomogram_for_batch(path_set, prj, bin_size)
        with prj.open_project() as dir:
            cfg_path = prj._default_spline_config_path(dir)
            if cfg_path.exists():
                config = SplineConfig.from_file(cfg_path)
            else:
                config = default_config
            params.curate_tomogram(tomo, path_coords, config=config)
            _save_splines(prj, dir, tomo)
            prj.rewrite(dir)
        ui.logger.print(f"{len(tomo.splines)} splines saved.")
        if prj.molecules_info:
            ui.logger.print(
                "Warning: this project has molecules whose source splines are now "
                "replaced by the curated splines."
            )
        yield

    @thread_worker.callback
    def _update_splines_in_widget():
        for proj in ui.batch.constructor.projects:
            if ui._project_dir and Path(proj.path).samefile(ui._project_dir):
                proj._update_from_project()

    return _update_splines_in_widget


def _load_tomogram_for_batch(
    path_set: PathInfo, prj: CylindraProject, bin_size: int
) -> CylTomogram:
    tomo = CylTomogram.imread(
        path_set.image,
        scale=prj.scale,
        tilt=prj.missing_wedge.as_param(),
        binsize=[bin_size] if bin_size > 1 else [],
    )
    if prj.invert:
        tomo.invert()
    return tomo


def _save_splines(prj: CylindraProject, dir: _Path, tomo: CylTomogram) -> None:
    """Replace the splines saved in the project directory."""
    for path in dir.glob("spline-*.json"):
        path.unlink()
    for path, df in [
        (prj._localprops_path(dir), tomo.splines.collect_localprops(allow_none=True)),
        (prj._globalprops_path(dir), tomo.splines.collect_globalprops(allow_none=True)),
    ]:
        if path.exists():
            path.unlink()
        if df is not None:
            df.write_csv(path)
    for i, spl in enumerate(tomo.splines):
        spl.to_json(dir / f"spline-{i}.json")


def _get_tomogram(ui: CylindraMainWidget) -> CylTomogram:
    if ui.tomogram is None:
        raise ValueError("No tomogram is currently loaded.")
    return ui.tomogram


def _report_hook_for(ui: CylindraMainWidget):
    def _hook(title: str, report: pl.DataFrame) -> None:
        ui.logger.print(f"{title} report:")
        ui.logger.print_table(report, index=False, precision=4)

    return _hook


def _log_dropped(ui: CylindraMainWidget, dropped: list[int]) -> None:
    ui.logger.print(f"{len(dropped)} splines deleted: {dropped}")


@thread_worker.callback
def _on_spline_updated(ui: CylindraMainWidget):
    ui._update_splines_in_images()
    ui.reset_choices()
    if (n := len(ui.tomogram.splines)) > 0:
        ui.SplineControl.num = n - 1
