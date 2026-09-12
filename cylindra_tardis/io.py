from pathlib import Path

import numpy as np
import polars as pl


def iter_tardis_coords(
    path,
    tomo_size: tuple[float, float, float],
    trim_edge_by: float = 24.5,
    min_length: float = 25,
):
    path = Path(path)
    if not path.exists():
        return
    edge = np.array([10, trim_edge_by, trim_edge_by])
    tomogram_size_nm = np.array(tomo_size)
    for coords in iter_all_tardis_coords(path):
        mask = np.all((coords > edge) & (coords < (tomogram_size_nm - edge)), axis=1)
        # Keep separate paths when trimming removes points from the middle
        # of a trace, rather than joining the remaining pieces together.
        breaks = np.flatnonzero(~mask)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(mask)]))
        for start, end in zip(starts, ends, strict=True):
            if start < end and mask[start]:
                coords = coords[start:end]
                length = np.linalg.norm(np.diff(coords, axis=0))
                if length >= min_length:
                    yield coords


def iter_all_tardis_coords(path):
    path = Path(path)
    df = pl.read_csv(path)
    for _, sub in df.group_by("IDs", maintain_order=True):
        coords = sub.select("Z [A]", "Y [A]", "X [A]").to_numpy() / 10
        yield coords


def save_segments(segments: np.ndarray, path):
    path = Path(path)
    df = pl.DataFrame(segments, schema=["ID", "Z [A]", "Y [A]", "X [A]"])
    df.write_csv(path)
