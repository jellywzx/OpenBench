from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.export_sediment_station_info import (
    find_evaluated_station_file,
    merge_evaluated_with_metadata,
    plot_station_map,
    read_reference_station_metadata,
)


def test_metadata_reader_loads_only_station_dimension(tmp_path: Path, monkeypatch):
    path = tmp_path / "sediment.nc"
    xr.Dataset(
        {
            "station_name": ("station", ["Alpha", "Beta"]),
            "river": ("station", ["R1", "R2"]),
            "upstream_area": ("station", [1000.0, 2000.0]),
            "cama_lon_15min": ("station", [10.25, 20.25]),
            "cama_lat_15min": ("station", [1.25, 2.25]),
            "ssc": (("station", "time"), np.ones((2, 3))),
        },
        coords={
            "station": ["0007", "B"],
            "lon": ("station", [10.0, 20.0]),
            "lat": ("station", [1.0, 2.0]),
            "time": [1, 2, 3],
        },
    ).to_netcdf(path)

    original = xr.DataArray.values.fget

    def guarded_values(array):
        assert "time" not in array.dims
        return original(array)

    monkeypatch.setattr(xr.DataArray, "values", property(guarded_values))
    result = read_reference_station_metadata(
        path, variable="Suspended_Sediment_Concentration", grid_resolution=0.25
    )

    assert result["station_id"].tolist() == ["0007", "B"]
    assert result["station_name"].tolist() == ["Alpha", "Beta"]
    assert result["matched_lon"].tolist() == [10.25, 20.25]


def test_merge_preserves_ids_and_uses_coordinate_fallback():
    evaluated = pd.DataFrame(
        {
            "ID": ["0007", "SRC_B"],
            "ref_lon": [10.25, 20.25],
            "ref_lat": [1.25, 2.25],
            "use_syear": [2001, 2002],
            "use_eyear": [2003, 2004],
        }
    )
    metadata = pd.DataFrame(
        {
            "station_id": ["7", "B"],
            "station_name": ["Alpha", "Beta"],
            "river": ["R1", "R2"],
            "country": ["X", "Y"],
            "country_code": ["XX", "YY"],
            "data_source": ["A", "B"],
            "source_station_id": ["a", "b"],
            "original_lon": [10.0, 20.0],
            "original_lat": [1.0, 2.0],
            "matched_lon": [10.25, 20.25],
            "matched_lat": [1.25, 2.25],
            "upstream_area_km2": [1000.0, 2000.0],
            "matched_upstream_area_km2": [1100.0, 2100.0],
            "allocation_error": [0.1, 0.2],
        }
    )

    result = merge_evaluated_with_metadata(evaluated, metadata)

    assert result["station_id"].tolist() == ["0007", "SRC_B"]
    assert result["station_name"].tolist() == ["Alpha", "Beta"]
    assert result["metadata_match_status"].tolist() == ["id", "qualified_id_and_coordinates"]


def test_result_lookup_prefers_metrics_and_is_pair_specific(tmp_path: Path):
    case = tmp_path / "case"
    metrics = case / "metrics"
    scores = case / "scores"
    metrics.mkdir(parents=True)
    scores.mkdir()
    name = "Suspended_Sediment_Load_stn_RefA_SimA_evaluations.csv"
    (scores / name).write_text("ID\nscore\n", encoding="utf-8")
    (metrics / name).write_text("ID\nmetric\n", encoding="utf-8")

    found = find_evaluated_station_file(case, "Suspended_Sediment_Load", "RefA", "SimA")

    assert found == metrics / name


def test_station_map_labels_every_station(tmp_path: Path):
    stations = pd.DataFrame(
        {
            "station_id": ["0007", "B"],
            "matched_lon": [10.0, 20.0],
            "matched_lat": [1.0, 2.0],
            "original_lon": [10.0, 20.0],
            "original_lat": [1.0, 2.0],
        }
    )
    output = tmp_path / "stations.png"

    labels = plot_station_map(
        stations,
        output,
        title="Station test",
        extent=(0.0, 30.0, -5.0, 5.0),
        dpi=72,
        label_fontsize=5,
    )

    assert labels == len(stations)
    assert output.stat().st_size > 0
