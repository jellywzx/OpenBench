"""Diff Plot comparison scenario dispatcher."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import xarray as xr

from openbench.config.adapter import build_fig_nml
from openbench.core._comparison_helpers import _comparison_sim_groups
from openbench.core._comparison_diff_grid import process_grid_diff_plot
from openbench.core._comparison_diff_station import process_station_diff_plot
from openbench.util.filenames import diff_grid_difference_filename, join_filename_components
from openbench.visualization._validation import finite_values

DIFF_KDE_COMPARISON = "Diff_Kernel_Density_Estimate"


def _diff_kde_options(option, evaluation_item, item_type):
    plot_option = option.copy()
    plot_option["title"] = f"Kernel Density Estimate of {evaluation_item.replace('_', ' ')} Difference"
    plot_option["xticklabel"] = f"{item_type.replace('_', ' ')} Difference"
    return plot_option


@dataclass(frozen=True)
class DiffMetadata:
    evaluation_item: str
    ref_source: str
    sim1: str
    sim2: str
    item_type: str

    @property
    def variable(self) -> str:
        return f"{self.item_type}_diff"

    @property
    def curve_label(self) -> str:
        return f"{self.sim1} − {self.sim2}"


def parse_diff_filename(path: Path) -> DiffMetadata:
    """Parse an OpenBench gridded pairwise-difference filename."""
    if path.suffix.lower() != ".nc":
        raise ValueError(f"Expected a .nc Diff_Plot file: {path}")

    parts = path.stem.split("__")
    if len(parts) != 8 or parts[1] != "ref" or parts[4] != "vs" or parts[7] != "diff":
        raise ValueError(
            f"Not an OpenBench pairwise diff filename: {path.name}; expected "
            "<item>__ref__<reference>__<sim1>__vs__<sim2>__<metric>__diff.nc"
        )

    item, _ref, ref_source, sim1, _vs, sim2, item_type, _diff = (unquote(part) for part in parts)
    if not all((item, ref_source, sim1, sim2, item_type)):
        raise ValueError(f"Diff_Plot filename contains an empty component: {path.name}")
    return DiffMetadata(item, ref_source, sim1, sim2, item_type)


def _validate_metadata(ds: xr.Dataset, metadata: DiffMetadata, path: Path) -> None:
    data_array = ds.get(metadata.variable)
    if data_array is None:
        available = ", ".join(ds.data_vars) or "none"
        raise ValueError(f"{path}: missing variable {metadata.variable!r}; available variables: {available}")

    expected = {"item": metadata.evaluation_item, "ref_source": metadata.ref_source}
    for key, value in expected.items():
        for attributes in (ds.attrs, data_array.attrs):
            actual = attributes.get(key)
            if actual not in (None, "") and str(actual) != value:
                raise ValueError(f"{path}: attribute {key}={actual!r} conflicts with filename value {value!r}")

    declared_type = data_array.attrs.get("score") or data_array.attrs.get("metric")
    if declared_type not in (None, "") and str(declared_type) != metadata.item_type:
        raise ValueError(
            f"{path}: variable metadata declares {declared_type!r}, but filename declares {metadata.item_type!r}"
        )


def read_diff_values(path: Path, metadata: DiffMetadata) -> np.ndarray:
    """Load and validate the finite, equally weighted samples for one diff file."""
    if not path.is_file():
        raise FileNotFoundError(f"Diff_Plot NetCDF does not exist: {path}")
    with xr.open_dataset(path) as ds:
        _validate_metadata(ds, metadata, path)
        values = np.asarray(ds[metadata.variable].values).reshape(-1)

    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise ValueError(f"{path}: KDE requires at least two finite values; found {finite.size}")
    if np.unique(finite).size < 2:
        raise ValueError(f"{path}: KDE requires at least two distinct finite values")
    return finite


def default_output_dir(path: Path) -> Path:
    if path.parent.name != "Diff_Plot":
        raise ValueError(f"Cannot infer output directory because {path} is not inside a Diff_Plot directory")
    return path.parent.parent / DIFF_KDE_COMPARISON


def expected_output_path(output_dir: Path, metadata: DiffMetadata, option: dict) -> Path:
    sim_token = hashlib.sha1(metadata.curve_label.encode("utf-8")).hexdigest()[:8]
    filename = join_filename_components(
        "Kernel_Density_Estimate",
        metadata.evaluation_item,
        metadata.ref_source,
        metadata.variable,
        sim_token,
    )
    return output_dir / f"{filename}.{option['saving_format']}"


def plot_diff_file(path: Path, output_dir: Path | None = None) -> Path:
    """Create one KDE figure from a gridded OpenBench pairwise diff file."""
    path = path.expanduser().resolve()
    metadata = parse_diff_filename(path)
    values = read_diff_values(path, metadata)
    destination = output_dir.expanduser().resolve() if output_dir is not None else default_output_dir(path)

    option = build_fig_nml()["Comparison"][DIFF_KDE_COMPARISON]
    option = _diff_kde_options(option, metadata.evaluation_item, metadata.item_type)

    _comparison_callable("make_scenarios_comparison_Kernel_Density_Estimate")(
        str(destination),
        metadata.evaluation_item,
        metadata.ref_source,
        [metadata.curve_label],
        metadata.variable,
        [values],
        option,
    )
    return expected_output_path(destination, metadata, option)


def _comparison_callable(name: str):
    """Resolve monkeypatch-friendly callables from openbench.core.comparison."""
    comparison = sys.modules.get("openbench.core.comparison")
    if comparison is not None and hasattr(comparison, name):
        return getattr(comparison, name)
    raise AttributeError(f"openbench.core.comparison.{name} is not available")


def render_grid_diff_kdes(basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
    """Render KDE curves from the pairwise gridded Diff_Plot NetCDF outputs."""
    diff_dir = os.path.join(basedir, "comparisons", "Diff_Plot")
    output_dir = os.path.join(basedir, "comparisons", DIFF_KDE_COMPARISON)
    item_types = [*metrics, *(score for score in scores if score != "nSpatialScore")]

    for evaluation_item in evaluation_items:
        sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
        ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
        if isinstance(sim_sources, str):
            sim_sources = [sim_sources]
        if isinstance(ref_sources, str):
            ref_sources = [ref_sources]

        for ref_source in ref_sources:
            groups = _comparison_sim_groups(evaluation_item, sim_sources, ref_source, sim_nml, ref_nml)
            grid_sources = groups.get("grid", [])
            if groups.get("stn"):
                logging.info(
                    "Skipping Diff Kernel Density Estimate for station group %s/%s; only gridded diff NetCDF is supported",
                    evaluation_item,
                    ref_source,
                )
            for index, sim1 in enumerate(grid_sources):
                for sim2 in grid_sources[index + 1 :]:
                    curve_label = f"{sim1} − {sim2}"
                    for item_type in item_types:
                        variable = f"{item_type}_diff"
                        path = os.path.join(
                            diff_dir,
                            diff_grid_difference_filename(evaluation_item, ref_source, sim1, sim2, item_type),
                        )
                        with xr.open_dataset(path) as dataset:
                            if variable not in dataset:
                                raise KeyError(f"{variable!r} not found in Diff_Plot file: {path}")
                            values = finite_values(
                                dataset[variable].values,
                                label=f"{evaluation_item}/{ref_source}/{curve_label}/{variable} KDE",
                            )

                        plot_option = _diff_kde_options(option, evaluation_item, item_type)
                        _comparison_callable("make_scenarios_comparison_Kernel_Density_Estimate")(
                            output_dir,
                            evaluation_item,
                            ref_source,
                            [curve_label],
                            variable,
                            [values],
                            plot_option,
                        )


class DiffPlotScenarioMixin:
    def scenarios_Diff_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        """Compare metrics and scores between simulations, then render Diff Plot outputs."""
        dir_path = os.path.join(f"{basedir}", "comparisons", "Diff_Plot")
        os.makedirs(dir_path, exist_ok=True)

        for evaluation_item in evaluation_items:
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for ref_source in ref_sources:
                groups = _comparison_sim_groups(evaluation_item, sim_sources, ref_source, sim_nml, ref_nml)
                for data_type, sources in groups.items():
                    if not sources:
                        continue
                    kwargs = dict(
                        basedir=basedir,
                        dir_path=dir_path,
                        evaluation_item=evaluation_item,
                        ref_source=ref_source,
                        sim_sources=sources,
                        metrics=metrics,
                        scores=scores,
                    )
                    if data_type == "stn":
                        process_station_diff_plot(sim_nml=sim_nml, **kwargs)
                    else:
                        process_grid_diff_plot(**kwargs)
                    _comparison_callable("make_scenarios_comparison_Diff_Plot")(
                        dir_path,
                        metrics,
                        scores,
                        evaluation_item,
                        ref_source,
                        sources,
                        self.general_config,
                        sim_nml,
                        data_type,
                        option,
                    )
                if len(groups) > 1:
                    logging.warning(
                        "%s/%s: station and grid comparisons were produced separately; "
                        "cross-representation differences require a common spatial support",
                        evaluation_item,
                        ref_source,
                    )

    def scenarios_Diff_Kernel_Density_Estimate_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        """Render KDE figures from pairwise gridded Diff_Plot outputs."""
        render_grid_diff_kdes(basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option)
