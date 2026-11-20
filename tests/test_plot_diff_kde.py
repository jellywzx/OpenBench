from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scripts import plot_diff_kde

from openbench.core import _comparison_diff_plot as diff_kde
from openbench.core import comparison

FILENAME = "Runoff__ref__RefA__Sim_A__vs__Sim_B__Overall_Score__diff.nc"


def _write_diff(path: Path, values=None, *, attrs=None, variable_attrs=None, variable="Overall_Score_diff") -> Path:
    data = np.asarray(values if values is not None else [[-0.2, 0.0], [0.1, np.nan]], dtype=float)
    array = xr.DataArray(
        data,
        dims=("lat", "lon"),
        attrs={"score": "Overall_Score", **(variable_attrs or {})},
    )
    xr.Dataset(
        {variable: array},
        attrs={"item": "Runoff", "ref_source": "RefA", **(attrs or {})},
    ).to_netcdf(path)
    return path


def test_parse_standard_diff_filename():
    metadata = diff_kde.parse_diff_filename(Path(FILENAME))

    assert metadata.evaluation_item == "Runoff"
    assert metadata.ref_source == "RefA"
    assert metadata.sim1 == "Sim_A"
    assert metadata.sim2 == "Sim_B"
    assert metadata.item_type == "Overall_Score"
    assert metadata.variable == "Overall_Score_diff"


def test_plot_diff_file_filters_values_and_calls_openbench_renderer(tmp_path, monkeypatch):
    diff_dir = tmp_path / "comparisons" / "Diff_Plot"
    diff_dir.mkdir(parents=True)
    path = _write_diff(diff_dir / FILENAME)
    calls = []
    monkeypatch.setattr(
        comparison,
        "make_scenarios_comparison_Kernel_Density_Estimate",
        lambda *args: calls.append(args),
    )

    output = diff_kde.plot_diff_file(path)

    basedir, item, ref, simulations, variable, datasets, option = calls[0]
    assert Path(basedir) == tmp_path / "comparisons" / "Diff_Kernel_Density_Estimate"
    assert (item, ref, simulations, variable) == (
        "Runoff",
        "RefA",
        ["Sim_A − Sim_B"],
        "Overall_Score_diff",
    )
    np.testing.assert_array_equal(datasets[0], [-0.2, 0.0, 0.1])
    assert option["title"] == "Kernel Density Estimate of Runoff Difference"
    assert option["xticklabel"] == "Overall Score Difference"
    assert output.parent == Path(basedir)
    assert "Overall_Score_diff" in output.name


def test_main_processes_multiple_inputs_and_reports_partial_failure(tmp_path, monkeypatch, capsys):
    paths = [tmp_path / "one.nc", tmp_path / "two.nc"]
    calls = []

    def fake_plot(path, output_dir):
        calls.append((path, output_dir))
        if path == paths[1]:
            raise ValueError("broken")
        return tmp_path / "one.jpg"

    monkeypatch.setattr(plot_diff_kde, "plot_diff_file", fake_plot)

    result = plot_diff_kde.main([str(paths[0]), str(paths[1]), "--output-dir", str(tmp_path / "plots")])

    assert result == 1
    assert [call[0] for call in calls] == paths
    assert all(call[1] == tmp_path / "plots" for call in calls)
    assert "Created" in capsys.readouterr().out


@pytest.mark.parametrize(
    "filename",
    ["not-a-diff.nc", "Runoff__ref__RefA__sim__SimA__bias__diff.nc", "Runoff__ref__RefA__A__vs__B.nc"],
)
def test_rejects_nonstandard_diff_filenames(filename):
    with pytest.raises(ValueError, match="pairwise diff filename"):
        diff_kde.parse_diff_filename(Path(filename))


def test_rejects_missing_variable(tmp_path):
    path = _write_diff(tmp_path / FILENAME, variable="Other_diff")
    metadata = diff_kde.parse_diff_filename(path)

    with pytest.raises(ValueError, match="missing variable 'Overall_Score_diff'"):
        diff_kde.read_diff_values(path, metadata)


def test_rejects_conflicting_metadata(tmp_path):
    path = _write_diff(tmp_path / FILENAME, variable_attrs={"item": "NotRunoff"})
    metadata = diff_kde.parse_diff_filename(path)

    with pytest.raises(ValueError, match="conflicts with filename"):
        diff_kde.read_diff_values(path, metadata)


@pytest.mark.parametrize("values", [[[np.nan, np.nan]], [[1.0, 1.0]]])
def test_rejects_data_that_cannot_form_kde(tmp_path, values):
    path = _write_diff(tmp_path / FILENAME, values=values)
    metadata = diff_kde.parse_diff_filename(path)

    with pytest.raises(ValueError, match="KDE requires"):
        diff_kde.read_diff_values(path, metadata)


@pytest.mark.parametrize("custom_output", [False, True])
def test_real_renderer_creates_jpg(tmp_path, custom_output):
    rng = np.random.default_rng(42)
    diff_dir = tmp_path / "comparisons" / "Diff_Plot"
    diff_dir.mkdir(parents=True)
    path = _write_diff(diff_dir / FILENAME, values=rng.normal(size=(10, 10)))

    original_data = path.read_bytes()
    old_dir = tmp_path / "comparisons" / "Kernel_Density_Estimate"
    old_dir.mkdir()
    old_figure = old_dir / "existing.jpg"
    old_figure.write_bytes(b"existing experiment figure")
    destination = tmp_path / "custom" if custom_output else diff_dir.parent / "Diff_Kernel_Density_Estimate"
    if custom_output:
        # A custom destination also permits inputs outside a Diff_Plot directory.
        moved_path = tmp_path / FILENAME
        path.rename(moved_path)
        path = moved_path
    args = [str(path)]
    if custom_output:
        args.extend(["--output-dir", str(destination)])

    assert plot_diff_kde.main(args) == 0
    output = diff_kde.expected_output_path(
        destination,
        diff_kde.parse_diff_filename(path),
        diff_kde.build_fig_nml()["Comparison"]["Diff_Kernel_Density_Estimate"],
    )

    assert path.read_bytes() == original_data
    assert list(old_dir.iterdir()) == [old_figure]
    assert old_figure.read_bytes() == b"existing experiment figure"
    assert output.is_file()
    assert output.suffix == ".jpg"
