"""Native reservoir histories use the real station preprocessing/evaluation path."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.config.adapter import build_legacy_namelists
from openbench.config.runtime_info import GeneralInfoReader
from openbench.config.schema import EvaluationConfig, OpenBenchConfig, ProjectConfig, ReferenceConfig, SimulationEntry
from openbench.data.processing import DatasetProcessing
from openbench.data.registry import manager
from openbench.data.reservoir import read_reservoir_series, valid_reservoir_ids

BINDINGS = {"Dam_Inflow": ("qresv_in", "inflow"), "Dam_Outflow": ("qresv_out", "outflow"),
            "Dam_Water_Storage": ("volresv", "storage")}


def fixture(tmp_path, monkeypatch, item="Dam_Inflow", monthly=True, sims=1):
    registry = manager.RegistryManager(user_dir=tmp_path / "registry")
    monkeypatch.setattr(manager, "get_registry", lambda: registry)
    valid_reservoir_ids.cache_clear()
    ref = tmp_path / "ref"
    ref.mkdir()
    months = range(1, 13) if monthly else [1]
    time = pd.DatetimeIndex([pd.Timestamp(2001, month, day) for month in months for day in (1, 2)])
    native, observed = BINDINGS[item]
    def values(station):
        return np.asarray([station + month + day / 10 for month in months for day in (1, 2)])
    rows = []
    for station in (101, 202, 303, 404):
        scale = 1e6 if item == "Dam_Water_Storage" else 1
        unit = "million cubic meters" if scale > 1 else "m3 s-1"
        xr.Dataset({observed: (("time", "lat", "lon"), (values(station)/scale)[:, None, None], {"units": unit})},
                   coords={"time": time, "lat": [40.], "lon": [-100.]}).to_netcdf(ref / f"{station}.nc")
        rows.append(dict(ID=f"ResOpsUS_{station}", SYEAR=2001, EYEAR=2001, LAT=40., LON=-100.,
                         DIR=f"{station}.nc", Flag=1))
    csv = ref / "stations.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    simulations = {}
    for index in range(sims):
        label = f"Sim{index}"
        directory = tmp_path / label
        directory.mkdir()
        for month in months:
            ids = [101, 202, 303] if month % 2 else [303, 202, 101]
            data = np.asarray([[station + month + day / 10 for station in ids] for day in (1, 2)])
            data[:, ids.index(303)] = -1e36
            ds = xr.Dataset({native: (("time", "reservoir"), data,
                                     {"units": "m^3" if item == "Dam_Water_Storage" else "m^3/s"}),
                             "resv_GRAND_ID": ("reservoir", ids)},
                            coords={"time": pd.date_range(f"2001-{month:02d}-01T12:00", periods=2)})
            ds.to_netcdf(directory / (f"case_hist_unitcat_2001-{month:02d}.nc" if monthly else "case_hist_unitcat_.nc"))
        simulations[label] = SimulationEntry(model="CoLM2024", root_dir=str(directory), data_type="grid",
                                             tim_res="Day", data_groupby="Month" if monthly else "single",
                                             prefix="case_hist_unitcat_")
    cfg = OpenBenchConfig(project=ProjectConfig(name="reservoir", output_dir=str(tmp_path / "output"),
                          years=[2001, 2001], tim_res="Day", grid_res=0.2, min_year_threshold=0,
                          num_cores=1, generate_report=False, station_plots=False),
                          evaluation=EvaluationConfig(variables=[item]),
                          reference=ReferenceConfig(sources={item: "ResOpsUS"},
                              overrides={"ResOpsUS": {"root_dir": str(ref), "fulllist": str(csv)}}),
                          simulation=simulations, metrics=["RMSE", "KGE"], scores=[])
    return cfg


def runtime(cfg, main, refs, sims, label="Sim0"):
    return GeneralInfoReader(main, sims, refs, cfg.metrics, [], [], [], cfg.evaluation.variables[0], label, "ResOpsUS")


@pytest.mark.parametrize("item", BINDINGS)
def test_monthly_histories_reach_station_metrics(tmp_path, monkeypatch, item):
    from openbench.core.evaluation import Evaluation_stn
    import openbench.core.evaluation as evaluation

    cfg = fixture(tmp_path, monkeypatch, item, sims=2)
    main, refs, sims = build_legacy_namelists(cfg)
    assert cfg.simulation["Sim0"].data_type == "grid"  # No user config mutation.
    assert sims[item]["Sim0_data_type"] == "stn"
    assert refs[item]["ResOpsUS_syear"] == 2001  # Not the old catalogue years.
    info = runtime(cfg, main, refs, sims)
    assert info.stn_list.ID.tolist() == ["101", "202"]
    assert {"sim_dir", "ref_dir"} <= set(info.stn_list)
    proc = DatasetProcessing(vars(info))
    proc.prepare_source("ref")
    proc.prepare_source("sim")
    monkeypatch.setattr(evaluation, "make_plot_index_stn", lambda *args, **kwargs: None)
    evaluator = Evaluation_stn(proc.to_dict(), {})
    evaluator.make_evaluation_P()
    result = pd.read_csv(Path(info.casedir) / "metrics" / f"{item}_stn_ResOpsUS_Sim0_evaluations.csv")
    assert result.ID.tolist() == [101, 202]
    np.testing.assert_allclose(result.RMSE, 0, atol=1e-10)
    np.testing.assert_allclose(result.KGE, 1, atol=1e-10)
    audit = json.loads(next((Path(info.casedir)/"reservoir_matching").glob("*.json")).read_text())
    assert audit["all_missing_ids"] == ["303"]
    assert audit["unmatched_observations"] == ["404"]
    # Rebuilding runtime info must not restore excluded stations.
    assert runtime(cfg, main, refs, sims, "Sim1").stn_list.ID.tolist() == ["101", "202"]


def change(path, transform):
    with xr.open_dataset(path) as ds:
        updated = transform(ds.load())
    updated.to_netcdf(path, mode="w")


@pytest.mark.parametrize("case,match", [("duplicate", "Duplicate"), ("missing_id", "Missing resv"),
    ("bad_id", "Invalid GRanD"), ("units", "units"), ("time", "increasing and unique"),
    ("missing_month", "missing or duplicate monthly"), ("missing_variable", "missing reservoir variable")])
def test_invalid_histories_fail_without_spatial_fallback(tmp_path, monkeypatch, case, match):
    cfg = fixture(tmp_path, monkeypatch)
    path = tmp_path / "Sim0/case_hist_unitcat_2001-02.nc"
    if case == "missing_month":
        path.unlink()
    else:
        def transform(ds):
            if case == "duplicate":
                return ds.assign(resv_GRAND_ID=("reservoir", [101, 101, 202]))
            if case == "missing_id":
                return ds.drop_vars("resv_GRAND_ID")
            if case == "bad_id":
                return ds.assign(resv_GRAND_ID=("reservoir", [101, -1, 202]))
            if case == "time":
                return ds.assign_coords(time=[ds.time.values[0]] * 2)
            if case == "missing_variable":
                return ds.rename(qresv_in="other")
            ds.qresv_in.attrs["units"] = "mm/day"
            return ds
        change(path, transform)
    with pytest.raises(ValueError, match=match):
        build_legacy_namelists(cfg)


def test_common_ids_across_simulations_and_months(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, sims=2)
    path = tmp_path / "Sim1/case_hist_unitcat_2001-02.nc"
    change(path, lambda ds: ds.isel(reservoir=[0, 1]))  # Missing 101 this month.
    main, refs, sims = build_legacy_namelists(cfg)
    for label in cfg.simulation:
        assert runtime(cfg, main, refs, sims, label).stn_list.ID.tolist() == ["202"]


def test_ctl_cannot_be_silently_dropped(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch)
    cfg.simulation["CTL"] = SimulationEntry(model="CoLM2024", root_dir=str(tmp_path/"absent"), data_groupby="Month")
    with pytest.raises(ValueError, match="no reservoir output.*CTL"):
        build_legacy_namelists(cfg)


def test_single_file_and_changed_input(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    _, _, sims = build_legacy_namelists(cfg)
    layout = json.loads(sims["Dam_Inflow"]["Sim0_reservoir_manifest"])["sources"]["Sim0"]
    data = read_reservoir_series(layout, ["202", "101"])
    np.testing.assert_allclose(data.sel(station="101"), [102.1, 102.2])
    Path(layout["files"][0]["path"]).touch()
    with pytest.raises(ValueError, match="changed since"):
        read_reservoir_series(layout, ["101"])


@pytest.mark.parametrize("case,match", [("duplicate", "Duplicate ResOpsUS"), ("missing_path", "observation not found"),
                                      ("region", "No ResOpsUS stations"), ("no_ids", "No common GRanD")])
def test_reference_errors_are_fatal(tmp_path, monkeypatch, case, match):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    path = tmp_path / "ref/stations.csv"
    rows = pd.read_csv(path)
    if case == "duplicate":
        rows = pd.concat([rows, rows.iloc[:1]])
    elif case == "missing_path":
        rows.loc[0, "DIR"] = "absent.nc"
    elif case == "no_ids":
        rows.ID = ["900", "901", "902", "903"]
    else:
        cfg.project.lat_range = [-22, 11]
    rows.to_csv(path, index=False)
    main, refs, sims = build_legacy_namelists(cfg)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        runtime(cfg, main, refs, sims)


def test_runner_hash_and_drawing_restore(tmp_path, monkeypatch):
    """Exercise task planning, preprocessing, outputs and saved runtime metadata."""
    import shutil
    import openbench.core.evaluation as evaluation
    from openbench.runner.local import run_evaluation

    cfg = fixture(tmp_path, monkeypatch, monthly=False, sims=2)
    monkeypatch.setattr(evaluation, "make_plot_index_stn", lambda *args, **kwargs: None)
    result = run_evaluation(cfg)
    assert result["status"] == "success", result
    case = Path(cfg.project.output_dir) / cfg.project.name
    manifest = json.loads((case / "run_manifest.json").read_text())
    assert manifest["resolved_runtime"]["namelists"]["simulation"]["Dam_Inflow"]["Sim0_data_type"] == "stn"
    # A peer's inputs affect the common roster and therefore both task hashes.
    from openbench.config.adapter import build_runner_bindings
    from openbench.runner.hashing import task_hash_payload
    bindings = build_runner_bindings(cfg)
    kwargs = dict(cfg=cfg, bindings=bindings, var_name="Dam_Inflow", sim_source="Sim0", ref_source="ResOpsUS",
                  metric_vars=cfg.metrics, score_vars=[], comparison_vars=[], statistic_vars=[])
    before = task_hash_payload(**kwargs)
    path = tmp_path / "Sim1/case_hist_unitcat_.nc"
    change(path, lambda ds: ds.isel(reservoir=[1, 2]))
    kwargs["bindings"] = build_runner_bindings(cfg)
    assert before != task_hash_payload(**kwargs)
    # Drawing-only restores effective types and station lists without raw files.
    shutil.rmtree(tmp_path / "Sim0")
    shutil.rmtree(tmp_path / "Sim1")
    shutil.rmtree(tmp_path / "ref")
    cfg.project.only_drawing = True
    main, refs, sims = build_legacy_namelists(cfg)
    info = runtime(cfg, main, refs, sims)
    assert info.sim_data_type == "stn"
    assert info.stn_list.ID.astype(str).tolist() == ["101", "202"]


def test_time_offset_uses_actual_file_once(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    original = tmp_path / "Sim0/case_hist_unitcat_.nc"
    path = tmp_path / "Sim0/case_hist_cama_.nc"
    original.rename(path)
    change(path, lambda ds: ds.assign_coords(time=ds.time + np.timedelta64(1, "D")))
    cfg.simulation["Sim0"].prefix = "case_hist_cama_"
    main, refs, sims = build_legacy_namelists(cfg)
    info = runtime(cfg, main, refs, sims)
    proc = DatasetProcessing(vars(info))
    layout = json.loads(sims["Dam_Inflow"]["Sim0_reservoir_manifest"])["sources"]["Sim0"]
    data = read_reservoir_series(layout, ["101"], proc)
    assert pd.Timestamp(data.time.values[0]) == pd.Timestamp("2001-01-01T12:00")
    proc.prepare_source("sim")
    path = Path(info.casedir)/"data/stn_ResOpsUS_Sim0/Dam_Inflow_sim_101_2001_2001.nc"
    with xr.open_dataset(path) as ds:
        assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("2001-01-01T12:00")
        np.testing.assert_allclose(ds.qresv_in.values[:2].ravel(), [102.1, 102.2])


def test_grid_water_variable_keeps_grid_path(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    path = tmp_path / "Sim0/case_hist_unitcat_.nc"
    with xr.open_dataset(path) as ds:
        times = ds.time.values
    xr.Dataset({"daminf": (("time", "lat", "lon"), np.ones((2, 1, 1)), {"units": "m3 s-1"})},
               coords={"time": times, "lat": [40.], "lon": [-100.]}).to_netcdf(path, mode="w")
    _, _, sims = build_legacy_namelists(cfg)
    assert sims["Dam_Inflow"]["Sim0_data_type"] == "grid"
    assert "Sim0_reservoir_files" not in sims["Dam_Inflow"]


def test_explicit_reference_years_are_not_overridden(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    cfg.reference.overrides["ResOpsUS"]["years"] = [1908, 1957]
    main, refs, sims = build_legacy_namelists(cfg)
    assert refs["Dam_Inflow"]["ResOpsUS_syear"] == 1908
    with pytest.raises(ValueError, match="No ResOpsUS stations"):
        runtime(cfg, main, refs, sims)


def test_preconverted_station_simulation_preserves_paths(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    root = tmp_path / "converted"
    root.mkdir()
    rows = []
    for station in (101, 202):
        path = root / f"{station}.nc"
        xr.Dataset({"qresv_in": ("time", [102., 103.])},
                   coords={"time": pd.date_range("2001-01-01", periods=2)}).to_netcdf(path)
        rows.append(dict(ID=str(station), SYEAR=2001, EYEAR=2001, LON=-100., LAT=40., DIR=str(path)))
    csv = root / "sim.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    entry = cfg.simulation["Sim0"]
    entry.root_dir, entry.data_type, entry.fulllist = str(root), "stn", str(csv)
    entry.variables = {"Dam_Inflow": {"varname": "qresv_in", "varunit": "m3 s-1"}}
    main, refs, sims = build_legacy_namelists(cfg)
    info = runtime(cfg, main, refs, sims)
    assert info.stn_list.ID.tolist() == ["101", "202"]
    assert info.stn_list.sim_dir.tolist() == [row["DIR"] for row in rows]


def test_cli_check_uses_reservoir_metadata_without_reading_series(tmp_path, monkeypatch):
    from dataclasses import asdict
    from click.testing import CliRunner
    import yaml
    from openbench.cli.main import cli
    import openbench.data.reservoir as reservoir

    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    cfg.simulation["Sim0"].variables = {"Dam_Inflow": {
        "station_dim": "reservoir", "station_id_var": "resv_GRAND_ID",
    }}
    raw = asdict(cfg)
    raw["reference"] = {"Dam_Inflow": "ResOpsUS", "overrides": cfg.reference.overrides}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    def forbidden(*args, **kwargs):
        raise AssertionError("check must not read reservoir time series")
    monkeypatch.setattr(reservoir, "read_reservoir_series", forbidden)
    result = CliRunner().invoke(cli, ["check", str(path)])
    assert result.exit_code == 0, result.output
    assert "reservoir → station" in result.output


def test_profile_prefix_fallback_and_all_missing_failure(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    cfg.simulation["Sim0"].prefix = "case_hist_"
    main, refs, sims = build_legacy_namelists(cfg)
    assert sims["Dam_Inflow"]["Sim0_reservoir_files"][0]["path"].endswith("case_hist_unitcat_.nc")
    path = tmp_path / "Sim0/case_hist_unitcat_.nc"
    change(path, lambda ds: ds.assign(qresv_in=ds.qresv_in.where(False)))
    main, refs, sims = build_legacy_namelists(cfg)
    with pytest.raises(ValueError, match="No reservoir stations with finite"):
        runtime(cfg, main, refs, sims)


def test_missing_reference_variable_cannot_use_another_quantity(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    path = tmp_path / "ref/101.nc"
    change(path, lambda ds: ds.rename(inflow="outflow"))
    main, refs, sims = build_legacy_namelists(cfg)
    info = runtime(cfg, main, refs, sims)
    proc = DatasetProcessing(vars(info))
    with pytest.raises(ValueError, match="observation variable.*inflow.*not found"):
        proc.prepare_source("ref")


def test_custom_waterbody_fields_and_legacy_list_resolution(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from openbench.data.custom.ResOpsUS_filter import _resolve_station_list

    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    path = tmp_path / "Sim0/case_hist_unitcat_.nc"
    change(path, lambda ds: ds.rename(reservoir="waterbody", resv_GRAND_ID="grand"))
    cfg.simulation["Sim0"].variables = {"Dam_Inflow": {"station_dim": "waterbody", "station_id_var": "grand"}}
    main, refs, sims = build_legacy_namelists(cfg)
    info = runtime(cfg, main, refs, sims)
    proc = DatasetProcessing(vars(info))
    proc.prepare_source("sim")
    with xr.open_dataset(Path(info.casedir)/"data/stn_ResOpsUS_Sim0/Dam_Inflow_sim_101_2001_2001.nc") as ds:
        np.testing.assert_allclose(ds.qresv_in.values[:2].ravel(), [102.1, 102.2])
    for resolution, suffix in [(0.1, "06min"), (0.25, "15min")]:
        expected = tmp_path / "list" / f"ResOpsUS_reservoir_matched_{suffix}.csv"
        expected.parent.mkdir(exist_ok=True)
        expected.write_text("ID,SYEAR,EYEAR,LON,LAT,DIR\n")
        assert _resolve_station_list(SimpleNamespace(ref_dir=str(tmp_path), sim_grid_res=resolution)) == expected


def test_year_boundary_finite_screening_uses_adjusted_time(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    original = tmp_path / "Sim0/case_hist_unitcat_.nc"
    path = original.with_name("case_hist_cama_.nc")
    original.rename(path)
    change(path, lambda ds: ds.assign_coords(time=pd.date_range("2002-01-01", periods=2)))
    cfg.simulation["Sim0"].prefix = "case_hist_cama_"
    main, refs, sims = build_legacy_namelists(cfg)
    # The first sample labels 31 Dec 2001 after the existing CaMa -1 day correction.
    assert runtime(cfg, main, refs, sims).stn_list.ID.tolist() == ["101", "202"]


def test_reference_units_are_checked_from_the_file(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    path = tmp_path / "ref/101.nc"
    def corrupt(ds):
        ds.inflow.attrs["units"] = "mm/day"
        return ds
    change(path, corrupt)
    main, refs, sims = build_legacy_namelists(cfg)
    proc = DatasetProcessing(vars(runtime(cfg, main, refs, sims)))
    with pytest.raises(ValueError, match="incompatible or unknown.*units"):
        proc.prepare_source("ref")


def test_yaml_null_filename_suffix_is_an_empty_suffix(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, monthly=False)
    cfg.simulation["Sim0"].variables = {"Dam_Inflow": {"prefix": "case_hist_unitcat_", "suffix": None}}
    main, refs, sims = build_legacy_namelists(cfg)
    assert runtime(cfg, main, refs, sims).stn_list.ID.tolist() == ["101", "202"]
