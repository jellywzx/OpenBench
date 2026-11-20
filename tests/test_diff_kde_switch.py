from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from openbench.cli.init_cmd import COMPARISON_OPTIONS
from openbench.config.adapter import build_fig_nml
from openbench.core import _comparison_diff_plot as diff_kde
from openbench.core import comparison
from openbench.core._comparison_diff_plot import render_grid_diff_kdes
from openbench.runner.postprocessing import _comparison_execution_order, run_comparison
from openbench.util.filenames import diff_grid_difference_filename
from openbench.visualization import only_drawing
from openbench.visualization.only_drawing import ComparisonProcessing_only_drawing

SWITCH = "Diff_Kernel_Density_Estimate"


def test_diff_kde_switch_is_registered():
    assert SWITCH in COMPARISON_OPTIONS
    assert SWITCH in build_fig_nml()["Comparison"]
    assert hasattr(comparison.ComparisonProcessing, f"scenarios_{SWITCH}_comparison")
    assert hasattr(ComparisonProcessing_only_drawing, f"scenarios_{SWITCH}_comparison")


def test_diff_kde_switch_runs_diff_plot_first_without_duplicates():
    assert _comparison_execution_order([SWITCH]) == ["Diff_Plot", SWITCH]
    assert _comparison_execution_order([SWITCH, "Diff_Plot", "HeatMap"]) == ["Diff_Plot", SWITCH, "HeatMap"]
    assert _comparison_execution_order(["Diff_Plot", SWITCH]) == ["Diff_Plot", SWITCH]


@pytest.mark.parametrize("entry", ["helper", "comparison", "only_drawing"])
def test_render_grid_diff_kdes_uses_expected_pairwise_file(tmp_path: Path, monkeypatch, entry):
    diff_dir = tmp_path / "comparisons" / "Diff_Plot"
    diff_dir.mkdir(parents=True)
    filename = diff_grid_difference_filename("Runoff", "RefA", "SimA", "SimB", "Overall_Score")
    xr.Dataset({"Overall_Score_diff": (("lat", "lon"), [[-0.2, np.nan], [0.1, 0.2]])}).to_netcdf(diff_dir / filename)

    calls = []
    monkeypatch.setattr(
        comparison,
        "make_scenarios_comparison_Kernel_Density_Estimate",
        lambda *args: calls.append(args),
    )
    sim_nml = {
        "general": {"Runoff_sim_source": ["SimA", "SimB"]},
        "Runoff": {"SimA_data_type": "grid", "SimB_data_type": "grid"},
    }
    ref_nml = {
        "general": {"Runoff_ref_source": "RefA"},
        "Runoff": {"RefA_data_type": "grid"},
    }

    original_data = (diff_dir / filename).read_bytes()
    arguments = (
        str(tmp_path),
        sim_nml,
        ref_nml,
        ["Runoff"],
        ["Overall_Score", "nSpatialScore"],
        [],
        {"title": "", "xticklabel": ""},
    )

    if entry == "helper":
        render_grid_diff_kdes(*arguments)
    elif entry == "comparison":
        comparison.ComparisonProcessing.scenarios_Diff_Kernel_Density_Estimate_comparison(None, *arguments)
    else:
        ComparisonProcessing_only_drawing.scenarios_Diff_Kernel_Density_Estimate_comparison(None, *arguments)

    assert (diff_dir / filename).read_bytes() == original_data
    assert diff_kde.default_output_dir(diff_dir / filename) == tmp_path / "comparisons" / SWITCH
    output_dir, item, ref, labels, variable, datasets, option = calls[0]
    assert Path(output_dir) == tmp_path / "comparisons" / SWITCH
    assert (item, ref, labels, variable) == ("Runoff", "RefA", ["SimA − SimB"], "Overall_Score_diff")
    np.testing.assert_array_equal(datasets[0], [-0.2, 0.1, 0.2])
    assert option["title"] == "Kernel Density Estimate of Runoff Difference"
    assert option["xticklabel"] == "Overall Score Difference"
    assert len(calls) == 1


@pytest.mark.parametrize("drawing_only", [False, True])
def test_ordinary_kde_keeps_its_output_directory(tmp_path, monkeypatch, drawing_only):
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    for sim in ("SimA", "SimB"):
        xr.Dataset({"Overall_Score": (("lat", "lon"), [[0.1, 0.2], [0.3, 0.4]])}).to_netcdf(
            score_dir / f"Runoff_ref_RefA_sim_{sim}_Overall_Score.nc"
        )
    sim_nml = {
        "general": {"Runoff_sim_source": ["SimA", "SimB"]},
        "Runoff": {"SimA_data_type": "grid", "SimB_data_type": "grid"},
    }
    ref_nml = {"general": {"Runoff_ref_source": "RefA"}, "Runoff": {"RefA_data_type": "grid"}}
    calls = []
    module = only_drawing if drawing_only else comparison
    processor = ComparisonProcessing_only_drawing if drawing_only else comparison.ComparisonProcessing
    monkeypatch.setattr(module, "make_scenarios_comparison_Kernel_Density_Estimate", lambda *args: calls.append(args))

    processor.scenarios_Kernel_Density_Estimate_comparison(
        None, str(tmp_path), sim_nml, ref_nml, ["Runoff"], ["Overall_Score", "nSpatialScore"], [], {}
    )

    assert len(calls) == 1
    assert Path(calls[0][0]) == tmp_path / "comparisons" / "Kernel_Density_Estimate"
    assert calls[0][3:5] == (["SimA", "SimB"], "Overall_Score")


@pytest.mark.parametrize("drawing_only", [False, True])
def test_runner_executes_diff_plot_before_diff_kde(tmp_path, monkeypatch, drawing_only):
    from openbench.visualization import Mod_Only_Drawing

    calls = []

    class Processor:
        def __init__(self, *args):
            pass

        def scenarios_Diff_Plot_comparison(self, *args):
            calls.append(("Diff_Plot", args[-1]))

        def scenarios_Diff_Kernel_Density_Estimate_comparison(self, *args):
            calls.append((SWITCH, args[-1]))

    module = Mod_Only_Drawing if drawing_only else comparison
    name = "ComparisonProcessing_only_drawing" if drawing_only else "ComparisonProcessing"
    monkeypatch.setattr(module, name, Processor)
    context = SimpleNamespace(
        namelists=SimpleNamespace(main={}, simulation={}, reference={}),
        score_vars=[],
        metric_vars=[],
        evaluation_items=["Runoff"],
        comparison_fig={"Diff_Plot": {"title": "diff"}, SWITCH: {"title": "kde"}},
    )
    bindings = SimpleNamespace(build_comparison_context=lambda: context)

    errors = run_comparison(
        bindings,
        [SWITCH, "Diff_Plot", SWITCH],
        tmp_path,
        make_phase_error_fn=lambda *args, **kwargs: {"error": args},
        bindings_only_drawing_fn=lambda _: drawing_only,
        post_phase_preflight_errors_fn=lambda *args, **kwargs: [],
        filter_evaluation_items_with_outputs_fn=lambda _, items: items,
    )

    assert errors == []
    assert calls == [("Diff_Plot", {"title": "diff"}), (SWITCH, {"title": "kde"})]
