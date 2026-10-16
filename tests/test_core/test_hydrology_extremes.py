"""Regressions for quantile-flow bias, flood timing, and inundation CSI."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openbench.core.metrics import metrics
from openbench.core.registry import METRICS_ITEMS, METRIC_LABELS
from openbench.core._hydrology_extreme_metrics import inundation_spatial_csi


def daily(values):
    return xr.DataArray(values, dims=("time",), coords={
        "time": pd.date_range("2001-01-01", periods=len(values), freq="D")
    })


def test_q95_and_q99_bias_are_percentile_values_not_high_flow_sample_pbias():
    m = metrics()
    obs = daily([1., 2., 3., 4., 100.])
    sim = obs * 1.2
    assert np.isclose(float(m.Q95_Bias(sim, obs)), 20.0)
    assert np.isclose(float(m.Q99_Bias(sim, obs)), 20.0)
    sim = sim.where(sim.time != sim.time[4])
    obs = obs.where(obs.time != obs.time[1])
    assert np.isclose(float(m.Q95_Bias(sim, obs)), 20.0)
    assert np.isnan(float(m.Q99_Bias(daily([2., 2., 2.]), daily([0., 0., 0.]))))


def test_peak_timing_matches_observed_events_and_reports_fraction():
    obs = np.zeros(90)
    sim = np.zeros(90)
    for time, height in ((12, 12), (45, 15), (78, 18)):
        obs[time-1:time+2] = [height/4, height, height/4]
    for time, height in ((14, 13), (43, 16)):
        sim[time-1:time+2] = [height/4, height, height/4]
    m = metrics()
    assert np.isclose(float(m.PeakTimingError(daily(sim), daily(obs))), 2.0)
    assert np.isclose(float(m.PeakTimingMatchRate(daily(sim), daily(obs))), 2/3)


def test_peak_timing_missing_match_and_monthly_are_not_false_perfect_scores():
    obs = np.zeros(60)
    sim = np.zeros(60)
    obs[15], sim[36] = 10, 10
    m = metrics()
    assert np.isnan(float(m.PeakTimingError(daily(sim), daily(obs))))
    assert float(m.PeakTimingMatchRate(daily(sim), daily(obs))) == 0.0
    months = pd.date_range("2001-01-01", periods=36, freq="MS")
    values = xr.DataArray(np.arange(36), dims=("time",), coords={"time": months})
    assert np.isnan(float(m.PeakTimingError(values, values)))


def test_inundation_temporal_grid_csi_masks_never_wet_and_invalid_units():
    times = pd.date_range("2001-01-01", periods=3, freq="D")
    obs = xr.DataArray([[[.2, 0, 0]], [[0, 0, 0]], [[.5, 0, 0]]],
                       dims=("time", "lat", "lon"),
                       coords={"time": times, "lat": [0], "lon": [0, 1, 2]})
    sim = xr.DataArray([[[.1, .2, 0]], [[.4, 0, 0]], [[0, 0, 0]]],
                       dims=obs.dims, coords=obs.coords)
    got = metrics().Inundation_CSI(sim, obs)
    assert np.isclose(float(got.sel(lat=0, lon=0)), 1/3)
    assert np.isclose(float(got.sel(lat=0, lon=1)), 0.0)
    assert np.isnan(float(got.sel(lat=0, lon=2)))
    invalid = sim.copy()
    invalid.loc[dict(lon=1, time=times[0])] = 20.0
    assert np.isnan(float(metrics().Inundation_CSI(invalid, obs).sel(lat=0, lon=1)))


def test_spatial_csi_is_separate_and_area_weighted():
    times = pd.date_range("2001-01-01", periods=2, freq="D")
    obs = xr.DataArray([[[.2, 0]], [[0, .2]]], dims=("time", "lat", "lon"),
                       coords={"time": times, "lat": [0.], "lon": [0., 1.]})
    sim = xr.DataArray([[[.3, .3]], [[np.nan, 0]]],
                       dims=obs.dims, coords=obs.coords)
    result = inundation_spatial_csi(sim, obs)
    np.testing.assert_allclose(result.CSI_spatial.values, [0.5, 0.0])
    assert result.valid_weight.isel(time=1).item() == 1
    elevated_obs = xr.DataArray(
        np.ones((1, 2, 1)), dims=("time", "lat", "lon"),
        coords={"time": times[:1], "lat": [0., 60.], "lon": [0.]}
    )
    elevated_sim = elevated_obs.copy()
    elevated_sim.loc[dict(lat=60.)] = 0.
    assert np.isclose(inundation_spatial_csi(elevated_sim, elevated_obs).CSI_spatial.item(),
                      1/1.5)


def test_grid_evaluation_writes_spatial_csi_csv(tmp_path, monkeypatch):
    from openbench.core.evaluation import Evaluation_grid
    times = pd.date_range("2001-01-01", periods=2, freq="MS")
    obs = xr.DataArray([[[.2, 0]], [[0, .3]]],
                       dims=("time", "lat", "lon"),
                       coords={"time": times, "lat": [0.], "lon": [0., 1.]})
    sim = xr.DataArray([[[.2, .3]], [[0, .3]]],
                       dims=obs.dims, coords=obs.coords)
    evaluator = object.__new__(Evaluation_grid)
    evaluator.item, evaluator.ref_source, evaluator.sim_source = "Inundation_Fraction", "OBS", "D"
    evaluator.casedir = str(tmp_path)
    monkeypatch.setattr(Evaluation_grid, "_save_metric_array", lambda *args, **kwargs: None)
    evaluator.process_metric("Inundation_CSI", sim, obs)
    base = "Inundation_Fraction_ref_OBS_sim_D_Inundation_CSI"
    data = pd.read_csv(tmp_path / "metrics" / (base + "_spatial_timeseries.csv"))
    summary = pd.read_csv(tmp_path / "metrics" / (base + "_spatial_summary.csv"))
    np.testing.assert_allclose(data.CSI_spatial.to_numpy(), [0.5, 1.0])
    assert np.isclose(summary.CSI_spatial_pooled.iloc[0], 2/3)


def test_metrics_registered_for_gui():
    shown = {name for group in METRICS_ITEMS.values() for name in group}
    expected = {"Q95_Bias", "Q99_Bias", "PeakTimingError",
                "PeakTimingMatchRate", "Inundation_CSI"}
    assert expected <= shown
    assert expected <= METRIC_LABELS.keys()


def test_csi_requires_inundation_fraction_in_evaluation():
    evaluator = metrics()
    evaluator.item = "Inundation_Area"
    with pytest.raises(ValueError, match="Inundation_Fraction"):
        evaluator.Inundation_CSI(daily([0.1, 0.2]), daily([0.1, 0.2]))

