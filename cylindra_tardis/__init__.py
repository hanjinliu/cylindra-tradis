from magicgui.types import Separator

from .core import (
    open_tardis_output_csv,
    open_tardis_segmentation,
    segment_microtubules,
    segment_microtubules_batch,
)
from .curation import (
    connect_collinear_splines,
    filter_duplicate_splines,
    filter_splines_by_cft,
    filter_splines_by_curvature,
    open_batch_curation_widget,
    open_curation_widget,
    register_tardis_traces,
)
from .curation import (
    curate_tardis_traces_batch as curate_tardis_traces_batch,
)

__version__ = "0.1.0"

__cylindra_methods__ = [
    segment_microtubules,
    segment_microtubules_batch,
    Separator,
    open_tardis_output_csv,
    open_tardis_segmentation,
    Separator,
    open_curation_widget,
    open_batch_curation_widget,
    Separator,
    register_tardis_traces,
    filter_duplicate_splines,
    filter_splines_by_curvature,
    filter_splines_by_cft,
    connect_collinear_splines,
]
