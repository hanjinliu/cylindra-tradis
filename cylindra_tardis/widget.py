"""Widgets to run the TARDIS trace curation workflow."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, TypeVar

from cylindra.types import get_available_binsize
from magicclass import (
    MagicTemplate,
    abstractapi,
    field,
    magicclass,
    set_design,
    vfield,
)
from magicclass.types import Optional, Path
from magicclass.utils import thread_worker

from cylindra_tardis import curation
from cylindra_tardis.workflow import CurationParams

if TYPE_CHECKING:
    from cylindra.widgets import CylindraMainWidget

_W = TypeVar("_W", bound="_CurationWidgetBase")
_WIDGETS: weakref.WeakKeyDictionary[CylindraMainWidget, dict[type, MagicTemplate]] = (
    weakref.WeakKeyDictionary()
)


def open_widget(cls: type[_W], ui: CylindraMainWidget) -> _W:
    """Show the widget of given class for the main widget, create one if needed."""
    widgets = _WIDGETS.setdefault(ui, {})
    if (widget := widgets.get(cls)) is None:
        widget = cls()
        widget._main_ref = weakref.ref(ui)
        widget.native.setParent(ui.native, widget.native.windowFlags())
        widgets[cls] = widget
    widget.reset_choices()
    widget.show()
    return widget


class _Switchable(MagicTemplate):
    """Group box whose parameters are enabled only if `apply` is checked."""

    def _on_apply_changed(self, value: bool):
        for widget in self:
            if widget.name != "apply":
                widget.enabled = value


@magicclass(widget_type="groupbox", layout="horizontal")
class Registration(MagicTemplate):
    """Registration of TARDIS traces as splines.

    Attributes
    ----------
    min_length : float
        Traces shorter than this length (nm) are ignored.
    trim_edge_by : float
        Trace points closer to the XY edges of the tomogram than this (nm) are
        trimmed.
    """

    min_length = vfield(10.0, label="min length (nm)")
    trim_edge_by = vfield(24.5, label="trim edge by (nm)")


@magicclass(widget_type="groupbox")
class Duplicates(_Switchable):
    """Filtering of near-duplicate traces.

    Attributes
    ----------
    apply : bool
        If unchecked, this step is skipped.
    max_lateral : float
        Maximum separation (nm) for a point to count as covered by another spline.
    min_overlap : float
        Minimum covered fraction of the shorter spline for it to be deleted.
    """

    apply = vfield(True, label="filter duplicates")

    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
    class params(MagicTemplate):
        max_lateral = abstractapi()
        min_overlap = abstractapi()

    max_lateral = vfield(5.0, label="max lateral (nm)", location=params)
    min_overlap = vfield(0.8, label="min overlap", location=params).with_options(
        min=0.0, max=1.0, step=0.05
    )
    apply.connect(_Switchable._on_apply_changed)


@magicclass(widget_type="groupbox", layout="horizontal")
class Fitting(MagicTemplate):
    """Spline fitting.

    Attributes
    ----------
    max_interval : float
        Maximum interval (nm) between spline anchors.
    err_max : float
        Maximum fitting error (nm).
    edge_sigma : float
        Sigma (nm) of the edge mask used in fitting.
    """

    max_interval = vfield(30.0, label="max interval (nm)")
    err_max = vfield(1.0, label="max error (nm)").with_options(step=0.1)
    edge_sigma = vfield(2.0, label="edge σ (nm)")


@magicclass(widget_type="groupbox")
class Curvature(_Switchable):
    """Filtering by the mean curvature.

    Attributes
    ----------
    apply : bool
        If unchecked, this step is skipped.
    max_curvature : float
        Maximum mean curvature (1/nm) for a spline to be kept.
    """

    apply = vfield(True, label="filter by curvature")
    max_curvature = vfield(0.05, label="max curvature (nm^-1)").with_options(
        min=0.00001, step=0.00001
    )
    apply.connect(_Switchable._on_apply_changed)


@magicclass(widget_type="groupbox")
class CFT(_Switchable):
    """Filtering by the peaks of the cylindric Fourier transformation.

    Attributes
    ----------
    apply : bool
        If unchecked, this step is skipped.
    measure_radius : bool
        If checked, the radius is measured for each spline.
    radius : float
        Radius (nm) of all the splines.
    min_snr : float
        Minimum S/N of the CFT peaks for a spline to be kept.
    """

    apply = vfield(True, label="filter by CFT peak")

    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
    class params(MagicTemplate):
        measure_radius = abstractapi()
        radius = abstractapi()
        min_snr = abstractapi()

    measure_radius = vfield(False, label="measure radius one by one", location=params)
    radius = vfield(11.0, label="radius (nm)", location=params).with_options(
        min=0.1, step=0.1
    )
    min_snr = vfield(6.0, label="min S/N", location=params).with_options(
        min=0.1, step=0.1
    )

    @apply.connect
    @measure_radius.connect
    def _update_enabled(self):
        self._on_apply_changed(self.apply)
        self.params["radius"].enabled = not self.measure_radius


@magicclass(widget_type="groupbox", layout="horizontal")
class Connection(MagicTemplate):
    """Connection of collinear splines.

    Attributes
    ----------
    max_gap : float
        Largest separation (nm) between the two junction endpoints.
    max_lateral : float
        Maximum lateral separation (nm) of the two splines.
    max_angle : float
        Maximum kink angle (deg) tolerated at the junction.
    """

    max_gap = vfield(40.0, label="max gap (nm)")
    max_lateral = vfield(12.5, label="max lateral (nm)")
    max_angle = vfield(15.0, label="max angle (deg)")


class _CurationWidgetBase(MagicTemplate):
    _main_ref = None

    def _get_main(self) -> CylindraMainWidget:
        if self._main_ref is None or (ui := self._main_ref()) is None:
            from cylindra import instance

            if (ui := instance()) is None:
                raise RuntimeError("No cylindra main widget found.")
        return ui

    def _get_params(self, bin_size: int) -> CurationParams:
        """Get the curation parameters from the widget."""
        return CurationParams(
            bin_size=bin_size,
            min_length=self.Registration.min_length,
            trim_edge_by=self.Registration.trim_edge_by,
            filter_duplicates=self.Duplicates.apply,
            dedup_max_lateral=self.Duplicates.max_lateral,
            dedup_min_overlap=self.Duplicates.min_overlap,
            fit_max_interval=self.Fitting.max_interval,
            fit_err_max=self.Fitting.err_max,
            fit_edge_sigma=self.Fitting.edge_sigma,
            filter_by_curvature=self.Curvature.apply,
            max_curvature=self.Curvature.max_curvature,
            radius=None if self.cft.measure_radius else self.cft.radius,
            filter_by_cft=self.cft.apply,
            min_snr=self.cft.min_snr,
            max_gap=self.Connection.max_gap,
            max_lateral=self.Connection.max_lateral,
            max_angle=self.Connection.max_angle,
        )


@magicclass(name="Curate TARDIS traces", record=False)
class TardisCurationWidget(_CurationWidgetBase):
    """Curate TARDIS traces of the current tomogram into splines.

    Attributes
    ----------
    path_coords : Path
        Path to the TARDIS instance CSV file. If not given, it will be searched in
        the "TARDIS" or "Predictions" directory next to the tomogram.
    bin_size : int
        Bin size of the tomogram used for fitting, radius measurement and CFT.
    """

    path_coords = vfield(
        Optional[Path.Read["*.csv"]], label="TARDIS coordinates"
    ).with_options(text="Search by tomogram path")
    bin_size = vfield(int, label="bin size").with_choices(get_available_binsize)
    Registration = field(Registration, name="Registration")
    Duplicates = field(Duplicates, name="Duplicates")
    Fitting = field(Fitting, name="Fitting")
    Curvature = field(Curvature, name="Curvature")
    cft = field(CFT, name="CFT")
    Connection = field(Connection, name="Connection")

    @set_design(text="Run")
    @thread_worker
    def run(self):
        """Run the curation workflow on the current tomogram.

        All the existing splines will be replaced by the curated splines.
        """
        ui = self._get_main()
        p = self._get_params(self.bin_size)
        path_coords = self.path_coords
        yield from curation.register_tardis_traces.arun(
            ui,
            path_coords=path_coords,
            min_length=p.min_length,
            trim_edge_by=p.trim_edge_by,
            err_max=p.fit_err_max,
        )
        if p.filter_duplicates:
            yield from curation.filter_duplicate_splines.arun(
                ui, max_lateral=p.dedup_max_lateral, min_overlap=p.dedup_min_overlap
            )
        yield from ui.fit_splines.arun(
            max_interval=p.fit_max_interval,
            bin_size=p.bin_size,
            err_max=p.fit_err_max,
            degree_precision=0.5,
            edge_sigma=p.fit_edge_sigma,
            max_shift=5.0,
        )
        if p.filter_by_curvature:
            yield from curation.filter_splines_by_curvature.arun(
                ui, max_curvature=p.max_curvature
            )
        if p.filter_by_cft:
            if p.radius is None:
                yield from ui.measure_radius.arun(
                    bin_size=p.bin_size, min_radius=7.0, max_radius=18.0
                )
            else:
                ui.set_radius(radius=p.radius)
            yield from curation.filter_splines_by_cft.arun(
                ui, bin_size=p.bin_size, min_snr=p.min_snr
            )
        yield from curation.connect_collinear_splines.arun(
            ui,
            max_gap=p.max_gap,
            max_lateral=p.max_lateral,
            max_angle=p.max_angle,
            err_max=p.fit_err_max,
        )

    @run.started.connect
    def _on_run_started(self):
        self.close()


@magicclass(name="Curate TARDIS traces (batch)", record=False)
class TardisBatchCurationWidget(_CurationWidgetBase):
    """Curate TARDIS traces of all the projects checked in the batch analyzer.

    TARDIS traces are searched in the "TARDIS" or "Predictions" directory next to
    each tomogram. Curated splines are saved in each project as "spline-*.json".

    Attributes
    ----------
    bin_size : int
        Bin size of the tomograms used for fitting, radius measurement and CFT.
    """

    bin_size = vfield(1, label="bin size").with_options(min=1, max=32)
    Registration = field(Registration, name="Registration")
    Duplicates = field(Duplicates, name="Duplicates")
    Fitting = field(Fitting, name="Fitting")
    Curvature = field(Curvature, name="Curvature")
    cft = field(CFT, name="CFT")
    Connection = field(Connection, name="Connection")

    @set_design(text="Run")
    @thread_worker
    def run(self):
        """Run the curation workflow on the projects checked in the batch analyzer.

        All the existing splines in each project will be replaced by the curated
        splines.
        """
        ui = self._get_main()
        path_sets = curation._get_loader_paths()
        if len(path_sets) == 0:
            raise ValueError("No project is checked in the batch analyzer.")
        params = self._get_params(self.bin_size)
        yield from curation.curate_tardis_traces_batch.arun(
            ui, path_sets, **params.asdict()
        )

    @run.started.connect
    def _on_run_started(self):
        self.close()
