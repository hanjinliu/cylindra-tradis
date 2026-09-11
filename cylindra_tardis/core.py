from typing import Annotated, Literal, Any
from cylindra.plugin import register_function
from cylindra.widgets import CylindraMainWidget
from cylindra.types import get_available_binsize
from cylindra.widgets.batch._sequence import PathInfo

import numpy as np
import polars as pl
from magicclass.types import Path

from cylindra_tardis.segmentation import TardisSegmentationModel

@register_function(name="Open TARDIS output CSV")
def open_tardis_output_csv(
    ui: CylindraMainWidget,
    path: Path.Read["*.csv"],
    trim_edge_by: Annotated[float, {"label": "Trim edge by (nm)"}] = 24.5,
    min_length: Annotated[float, {"label": "Minimum length (nm)"}] = 50,
):
    ui.logger.print(f"Reading TARDIS coordinates: {path}")
    df = pl.read_csv(path)
    tomogram_size_nm = np.array(ui.tomogram.image.shape) * ui.tomogram.scale
    for _, sub in df.group_by("IDs", maintain_order=True):
        coords = sub.select("Z [A]", "Y [A]", "X [A]").to_numpy() / 10
        mask = np.all((coords > trim_edge_by) & (coords < (tomogram_size_nm - trim_edge_by)), axis=1)
        # Keep separate paths when trimming removes points from the middle
        # of a trace, rather than joining the remaining pieces together.
        breaks = np.flatnonzero(~mask)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(mask)]))
        for start, end in zip(starts, ends):
            if start < end and mask[start]:
                coords = coords[start:end]
                length = np.linalg.norm(np.diff(coords, axis=0))
                if length >= min_length:
                    ui.register_path(coords)

@register_function(name="Open TARDIS segmentation")
def open_tardis_segmentation(
    ui: CylindraMainWidget,
):
    """Open the segmentation result for the current tomogram"""
    tomo = ui.tomogram
    if tomo is None:
        raise ValueError("No tomogram is currently loaded.")
    candidates = ["TARDIS", "Predictions"]
    for candidate in candidates:
        prediction_dir = tomo.source.parent / candidate
        if prediction_dir.exists():
            break
    else:
        raise ValueError("No TARDIS prediction directory found.")
    basename = tomo.source.stem
    # TODO: other image output types
    path_semantic = prediction_dir / f"{basename}_semantic.mrc"
    path_coords = prediction_dir / f"{basename}_instances_filter.csv"
    open_tardis_output_csv(ui, path_coords)
    ui.open_label_image(path_semantic)

@register_function(name="Segment microtubules")
def segment_microtubules(
    ui: CylindraMainWidget,
    bin_size: Annotated[int, {"choices": get_available_binsize}],
    prefilter_cutoff_nm: Annotated[float, {"label": "Prefilter cutoff (nm)"}] = 8.0,
    patch_size: Literal[32, 64, 96, 128, 256, 512] = 128,
    min_length: Annotated[int, {"label": "Minimum length (nm)"}] = 50,
    cnn_threshold: Annotated[float, {"label": "CNN threshold", "min": 0.0, "max": 1.0}] = 0.25,
    dist_threshold: Annotated[float, {"label": "Distance threshold", "min": 0.0, "max": 1.0}] = 0.5,
    connect_splines: Annotated[int, {"label": "Connect splines (nm)"}] = 250,
    connect_cylinder: Annotated[int, {"label": "Connect cylinder (nm)"}] = 25,
    device: str = "0",
):
    """Segment microtubules in the current tomogram"""
    tomo = ui.tomogram
    if tomo is None:
        raise ValueError("No tomogram is currently loaded.")
    img = tomo._get_multiscale_or_original(bin_size)
    
    if tomo.is_inverted:
        img = -img  # revert to the light-background image
    
    model = TardisSegmentationModel(
        prefilter_cutoff_nm=prefilter_cutoff_nm,
        patch_size=patch_size,
        min_length=min_length,
        cnn_threshold=cnn_threshold,
        dist_threshold=dist_threshold,
        connect_splines=connect_splines,
        connect_cylinder=connect_cylinder,
        device=device
    )
    model.run(img, tomo.source)
    tardis_dir = model.tardis_dir(tomo.source)
    basename = tomo.source.stem
    open_tardis_output_csv(ui, tardis_dir / f"{basename}_instances_filter.csv", min_length=min_length)
    ui.open_label_image(tardis_dir / f"{basename}_semantic.mrc")


def _get_loader_paths(*_):
    from cylindra import instance

    ui = instance()
    return ui.batch._get_loader_paths(*_)

@register_function(name="Segment microtubules (batch)")
def segment_microtubules_batch(
    ui: CylindraMainWidget,
    path_sets: Annotated[Any, {"bind": _get_loader_paths}],
    bin_size: int,
    prefilter_cutoff_nm: Annotated[float, {"label": "Prefilter cutoff (nm)"}] = 8.0,
    patch_size: Literal[32, 64, 96, 128, 256, 512] = 128,
    min_length: Annotated[int, {"label": "Minimum length (nm)"}] = 50,
    cnn_threshold: Annotated[float, {"label": "CNN threshold", "min": 0.0, "max": 1.0}] = 0.25,
    dist_threshold: Annotated[float, {"label": "Distance threshold", "min": 0.0, "max": 1.0}] = 0.5,
    connect_splines: Annotated[int, {"label": "Connect splines (nm)"}] = 250,
    connect_cylinder: Annotated[int, {"label": "Connect cylinder (nm)"}] = 25,
    device: str = "0",
):
    model = TardisSegmentationModel(
        prefilter_cutoff_nm=prefilter_cutoff_nm,
        patch_size=patch_size,
        min_length=min_length,
        cnn_threshold=cnn_threshold,
        dist_threshold=dist_threshold,
        connect_splines=connect_splines,
        connect_cylinder=connect_cylinder,
        device=device
    )
    ui.logger.print(f"Running TARDIS segmentation on {len(path_sets)} tomograms")
    for path_set in path_sets:
        path_set = PathInfo(*path_set)
        img = path_set.lazy_imread()
        if not path_set.need_invert:
            img = -img  # revert to the light-background image
        img_binned = img.binning(bin_size, check_edges=False)
        model.run(img_binned, path_set.image)
