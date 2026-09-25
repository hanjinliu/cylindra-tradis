from typing import Annotated, Any, Literal

import impy as ip
import numpy as np
from cylindra.plugin import register_function
from cylindra.types import get_available_binsize
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.batch._sequence import PathInfo
from magicclass.types import Optional, Path
from magicclass.utils import thread_worker

from cylindra_tardis.io import iter_tardis_coords
from cylindra_tardis.segmentation import TardisSegmentationModel
from cylindra_tardis.workflow import search_prediction_dir


@register_function(name="Open TARDIS output CSV")
def open_tardis_output_csv(
    ui: CylindraMainWidget,
    path: Optional[Path.Read["*.csv"]],
    trim_edge_by: Annotated[float, {"label": "Trim edge by (nm)"}] = 24.5,
    min_length: Annotated[float, {"label": "Minimum length (nm)"}] = 10,
):
    if path is None:
        root = search_prediction_dir(ui.tomogram.source)
        paths = list(root.glob("*_instances.csv"))
        if not paths:
            raise ValueError(f"No TARDIS output CSV files found in {root}.")
        path = paths[0]
    ui.logger.print(f"Reading TARDIS coordinates: {path}")
    for coords in iter_tardis_coords(
        path,
        tomo_size=np.array(ui.tomogram.image.shape) * ui.tomogram.scale,
        trim_edge_by=trim_edge_by,
        min_length=min_length,
    ):
        ui.register_path(coords)


@register_function(name="Open TARDIS segmentation")
def open_tardis_segmentation(
    ui: CylindraMainWidget,
):
    """Open the segmentation result for the current tomogram"""
    tomo = ui.tomogram
    if tomo is None:
        raise ValueError("No tomogram is currently loaded.")
    prediction_dir = search_prediction_dir(tomo.source)
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
    points_in_patch: Annotated[int, {"min": 2}] = 1000,
    filter_by_length: Annotated[int, {"label": "Minimum length (nm)"}] = 10,
    cnn_threshold: Annotated[
        float, {"label": "CNN threshold", "min": 0.0, "max": 1.0}
    ] = 0.25,
    dist_threshold: Annotated[
        float, {"label": "Distance threshold", "min": 0.0, "max": 1.0}
    ] = 0.5,
    device: str = "0",
    debug: bool = False,
):
    """Segment microtubules in the current tomogram.

    Parameters
    ----------
    bin_size : int
        The bin size to use for the input tomogram image.
    prefilter_cutoff_nm : float, optional
        Binned tomogram will be prefiltered with this cutoff (in nm).
    patch_size : int, optional
        Argument of `tardis_mt`. Window size used for prediction. This will break
        tomograms into smaller patches with 25% overlap. Smaller values consume less
        GPU, but also may lead to worse segmentation results.
    points_in_patch : int, optional
        Argument of `tardis_mt`. Maximum number of points that could be found in each
        point cloud cropped view. A higher number will always lead to faster
        inference, and may slightly improve segmentation.
    filter_by_length : int, optional
        Argument of `tardis_mt`. Minimum microtubule length.
    cnn_threshold : float, optional
        Argument of `tardis_mt`. Threshold used for semantic prediction. Higher value
        will lead to a reduction in noise and microtubule prediction recall. A lower
        value will increase microtubules prediction recall but may lead to increased
        noise.
    dist_threshold : float, optional
        Argument of `tardis_mt`. Threshold used for instance prediction. Higher value
        will lower number of the predicted instances.
    device : str, optional
        Argument of `tardis_mt`. "cpu" for running on the CPU, "gpu" for the GPU. int
        for specifying a particular GPU by its index.
    debug : bool, optional
        Argument of `tardis_mt`. Enable debugging mode.
    """
    tomo = ui.tomogram
    if tomo is None:
        raise ValueError("No tomogram is currently loaded.")
    img = tomo._get_multiscale_or_original(bin_size)

    if tomo.is_inverted:
        img = -img  # revert to the light-background image
    if isinstance(img, ip.LazyImgArray):
        img = img.compute()

    model = TardisSegmentationModel(
        prefilter_cutoff_nm=prefilter_cutoff_nm,
        patch_size=patch_size,
        filter_by_length=filter_by_length,
        cnn_threshold=cnn_threshold,
        dist_threshold=dist_threshold,
        points_in_patch=points_in_patch,
        device=device,
    )
    model.run(img, tomo.source, debug=debug)
    tardis_dir = model.tardis_dir(tomo.source)
    basename = tomo.source.stem

    ui.open_label_image(tardis_dir / f"{basename}_semantic.mrc")


def _get_loader_paths(*_):
    from cylindra import instance

    ui = instance()
    return [
        prj._get_loader_paths() for prj in ui.batch.constructor.projects if prj.check
    ]


@register_function(name="Segment microtubules (batch)")
@thread_worker.with_progress("Segmenting microtubules (batch)", total="len(path_sets)")
def segment_microtubules_batch(
    ui: CylindraMainWidget,
    path_sets: Annotated[Any, {"bind": _get_loader_paths}],
    bin_size: Annotated[int, {"min": 1}],
    prefilter_cutoff_nm: Annotated[float, {"label": "Prefilter cutoff (nm)"}] = 8.0,
    patch_size: Literal[32, 64, 96, 128, 256, 512] = 128,
    points_in_patch: Annotated[int, {"label": "Points in patch"}] = 1000,
    filter_by_length: Annotated[int, {"label": "Minimum length (nm)", "min": 1}] = 10,
    cnn_threshold: Annotated[
        float, {"label": "CNN threshold", "min": 0.0, "max": 1.0}
    ] = 0.25,
    dist_threshold: Annotated[
        float, {"label": "Distance threshold", "min": 0.0, "max": 1.0}
    ] = 0.5,
    device: str = "0",
    debug: bool = False,
):
    model = TardisSegmentationModel(
        prefilter_cutoff_nm=prefilter_cutoff_nm,
        patch_size=patch_size,
        points_in_patch=points_in_patch,
        filter_by_length=filter_by_length,
        cnn_threshold=cnn_threshold,
        dist_threshold=dist_threshold,
        device=device,
    )
    ui.logger.print(f"Running TARDIS segmentation on {len(path_sets)} tomograms")
    for path_set in path_sets:
        path_set = PathInfo(*path_set)
        print(f"Preparing input image using {path_set.image}")
        img = path_set.lazy_imread()
        img.set_scale(zyx=path_set.project_instance().scale)
        if not path_set.need_invert:
            img = -img  # revert to the light-background image
        img_binned = img.binning(bin_size, check_edges=False).compute()
        model.run(img_binned, path_set.image, debug=debug)
        yield
