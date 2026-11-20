"""CoLM reservoir vectors adapted to the existing station evaluation contract.

Layout discovery reads headers/IDs only. Values are selected by identity in each
file before concatenation; array positions are never identities across files.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from openbench.data.file_lookup import select_data_files
from openbench.util.netcdf import write_file_atomic

ITEMS = {"Dam_Inflow", "Dam_Outflow", "Dam_Water_Storage"}


def grand_id(value) -> str:
    """Normalize ResOpsUS's filename IDs without changing general station IDs."""
    text = str(value).strip()
    text = re.sub(r"^ResOpsUS_", "", text, flags=re.IGNORECASE)
    if not re.fullmatch(r"\d+(?:\.0+)?", text) or int(text.split(".")[0]) <= 0:
        raise ValueError(f"Invalid GRanD ID: {value!r}")
    return str(int(text.split(".")[0]))


def reservoir_index(ds, dim="reservoir", id_var="resv_GRAND_ID"):
    if id_var not in ds or ds[id_var].dims != (dim,):
        raise ValueError(f"Missing {id_var}({dim}); reservoir identity must be explicit")
    ids = [grand_id(value) for value in ds[id_var].values]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate {id_var}; cannot choose a reservoir column")
    return {station: index for index, station in enumerate(ids)}


def reservoir_unit(unit, item):
    compact = str(unit).lower().replace(" ", "").replace("^", "")
    if item == "Dam_Water_Storage":
        if compact == "m3":
            return "m3"
        if compact in {"mcm", "millioncubicmeters", "millioncubicmetres"}:
            return "mcm"
    elif compact in {"m3/s", "m3s-1"}:
        return "m3 s-1"
    raise ValueError(f"{item}: incompatible or unknown reservoir units {unit!r}")


def _discover(section, label, years, item):
    def get(key, default=None):
        return section.get(f"{label}_{key}", default)

    if get("data_type") == "stn" and get("fulllist"):
        return None  # Existing per-station/merged station files remain unchanged.
    grouping = str(get("data_groupby", "")).lower()
    if grouping not in {"month", "single"}:
        return None
    dim, id_var = get("station_dim", "reservoir"), get("station_id_var", "resv_GRAND_ID")
    candidates = list(dict.fromkeys([get("varname"), *(fb["varname"] for fb in get("fallbacks", []))]))
    paths = []
    missing_years = []
    for year in [None] if grouping == "single" else range(years[0], years[1] + 1):
        found, _ = select_data_files(
            get("dir"),
            get("prefix") or "",
            get("suffix") or "",
            year,
            prefix_fallback=get("prefix_fallback"),
            candidate_varnames=candidates,
        )
        if not found:
            missing_years.append(year)
        paths.extend(found)
    paths = sorted(set(paths))
    files, common, kinds, months = [], None, set(), []
    previous_end = None
    for path in paths:
        with xr.open_dataset(path) as ds:
            names = {str(name).casefold(): name for name in ds.data_vars}
            name = next(
                (names[str(candidate).casefold()] for candidate in candidates if str(candidate).casefold() in names),
                None,
            )
            if name is None:
                kinds.add("missing")
                if dim in ds.dims:
                    kinds.add("reservoir")
                continue
            if dim not in ds[name].dims:
                kinds.add("grid")
                continue
            kinds.add("reservoir")
            if set(ds[name].dims) != {"time", dim} or ds[name].ndim != 2:
                raise ValueError(f"{path}: expected {name}(time, {dim})")
            mapping = reservoir_index(ds, dim, id_var)
            unit = reservoir_unit(ds[name].attrs.get("units", ""), item)
            if not ds.sizes.get("time"):
                raise ValueError(f"{path}: empty reservoir time axis")
            index = ds.indexes["time"]
            if not index.is_monotonic_increasing or index.has_duplicates:
                raise ValueError(f"{path}: reservoir time must be increasing and unique")
            if previous_end is not None and index[0] <= previous_end:
                raise ValueError(f"{path}: reservoir time must be increasing and unique across files")
            previous_end = index[-1]
            common = set(mapping) if common is None else common & set(mapping)
            stat = Path(path).stat()
            files.append(
                dict(
                    path=str(Path(path).resolve()),
                    variable=name,
                    unit=unit,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
            match = re.search(r"(?<!\d)(\d{4})[-_](\d{2})(?!\d)", Path(path).name)
            months.append(
                (int(match[1]), int(match[2])) if match else (int(ds.time.dt.year[0]), int(ds.time.dt.month[0]))
            )
    if "reservoir" not in kinds:
        return None
    if kinds != {"reservoir"}:
        raise ValueError(f"{label}: mixed grid/reservoir layouts or missing reservoir variable")
    if missing_years:
        raise ValueError(f"{label}: missing reservoir files for years {missing_years}")
    if grouping == "month":
        expected = {(year, month) for year in range(years[0], years[1] + 1) for month in range(1, 13)}
        if set(months) != expected or len(months) != len(expected):
            raise ValueError(f"{label}: missing or duplicate monthly reservoir files: {sorted(expected - set(months))}")
    units = {entry["unit"] for entry in files}
    if len(units) != 1:
        raise ValueError(f"{label}: reservoir units change between files: {units}")
    from openbench.data.registry.manager import get_registry

    profile = get_registry().get_model(get("model"))
    return dict(
        files=files,
        ids=sorted(common, key=int),
        station_dim=dim,
        station_id_var=id_var,
        variable=files[0]["variable"],
        unit=files[0]["unit"],
        item=item,
        model=get("model"),
        tim_res=get("tim_res"),
        years=list(years),
        time_offset=profile.time_offset if profile else {},
    )


def resolve_reservoir_namelists(cfg, ref_nml, sim_nml):
    """Resolve effective station types before task planning, without writing data."""
    for item in cfg.evaluation.variables:
        if item not in ITEMS:
            continue
        section = sim_nml[item]
        sources = sim_nml["general"][f"{item}_sim_source"]
        layouts = {label: _discover(section, label, cfg.project.years, item) for label in sources}
        if not any(layouts.values()):
            continue
        references = ref_nml["general"].get(f"{item}_ref_source", [])
        references = [references] if isinstance(references, str) else references
        if references != ["ResOpsUS"]:
            raise ValueError(f"{item}: native reservoir evaluation currently requires the ResOpsUS station reference")
        absent = [label for label, layout in layouts.items() if layout is None]
        if absent:
            raise ValueError(
                f"{item}: selected simulations have no reservoir output: {absent}; select simulations explicitly"
            )
        common = set.intersection(*(set(layout["ids"]) for layout in layouts.values()))
        if not common:
            raise ValueError(f"{item}: no common reservoir IDs across selected simulations/months")
        manifest = json.dumps(
            dict(version=1, item=item, common_ids=sorted(common, key=int), sources=layouts),
            sort_keys=True,
        )
        for label, layout in layouts.items():
            section.update(
                {
                    f"{label}_data_type": "stn",
                    f"{label}_varname": layout["variable"],
                    f"{label}_varunit": layout["unit"],
                    f"{label}_station_dim": layout["station_dim"],
                    f"{label}_station_id_var": layout["station_id_var"],
                    f"{label}_reservoir_files": layout["files"],
                    f"{label}_reservoir_manifest": manifest,
                }
            )


def _unchanged(entry):
    stat = Path(entry["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (entry["size"], entry["mtime_ns"]):
        raise ValueError(f"Reservoir input changed since layout discovery: {entry['path']}; rebuild configuration")


def read_reservoir_series(layout, ids, processor=None):
    """Select every file by ID before concatenating, keeping source time rules."""
    from openbench.data._processing_time_adjustments import TimeAdjustmentMixin

    time_context = processor or SimpleNamespace(
        sim_source="reservoir",
        reservoir_model=layout["model"],
        item=layout["item"],
        sim_dir="",
        sim_prefix="",
    )
    pieces = []
    for entry in layout["files"]:
        _unchanged(entry)
        with xr.open_dataset(entry["path"]) as ds:
            mapping = reservoir_index(ds, layout["station_dim"], layout["station_id_var"])
            if reservoir_unit(ds[entry["variable"]].attrs.get("units", ""), layout["item"]) != entry["unit"]:
                raise ValueError(f"{entry['path']}: reservoir units changed")
            if processor is not None:
                # Reuse the common station selector, with the explicit waterbody dimension/ID.
                selected = [
                    processor._select_merged_station_data(ds, pd.Series({"ID": station}), "sim")[entry["variable"]]
                    for station in ids
                ]
                data = xr.concat(selected, dim=pd.Index(ids, name="station")).transpose("time", "station").load()
            else:
                data = ds[entry["variable"]].isel({layout["station_dim"]: [mapping[station] for station in ids]})
                data = data.transpose("time", layout["station_dim"]).load()
            values = data.values
            clean = np.where(np.isfinite(values) & ~np.isclose(values, -1e36, rtol=1e-6, atol=0), values, np.nan)
            data = xr.DataArray(
                clean,
                dims=("time", "station"),
                coords={"time": ds.time.values, "station": ids},
                name=layout["variable"],
                attrs={"units": layout["unit"]},
            )
            # Finite-data screening and extraction use the same physical time window.
            data = TimeAdjustmentMixin.apply_model_specific_time_adjustment(
                time_context,
                data,
                "sim",
                layout["years"][0],
                layout["years"][1],
                layout["tim_res"],
                source_files=[entry["path"]],
            )
            pieces.append(data)
        _unchanged(entry)
    combined = xr.concat(pieces, dim="time")
    index = combined.indexes["time"]
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Reservoir time must be increasing and unique across source files")
    return combined


@lru_cache(maxsize=16)
def valid_reservoir_ids(manifest_json, windows_json):
    """Share finite-data screening across sibling simulations and runtime rebuilds."""
    manifest = json.loads(manifest_json)
    windows = json.loads(windows_json)
    valid = set(windows)
    for layout in manifest["sources"].values():
        ids = sorted(valid, key=int)
        if not ids:
            break
        data = read_reservoir_series(layout, ids)
        for station in ids:
            start, end = windows[station]
            series = data.sel(station=station, time=slice(f"{start}-01-01", f"{end}-12-31T23:59:59"))
            if not bool(series.notnull().any()):
                valid.remove(station)
    return tuple(sorted(valid, key=int))


def match_reservoir_stations(info, rows):
    manifest_json = getattr(info, f"{info.sim_source}_reservoir_manifest")
    manifest = json.loads(manifest_json)
    candidates = rows[rows.ID.isin(manifest["common_ids"])].copy()
    if candidates.empty:
        raise ValueError("No common GRanD IDs between reservoir simulations and ResOpsUS observations")
    windows = {row.ID: [int(row.use_syear), int(row.use_eyear)] for row in candidates.itertuples()}
    valid = valid_reservoir_ids(manifest_json, json.dumps(windows, sort_keys=True))
    selected = candidates[candidates.ID.isin(valid)].copy()
    if selected.empty:
        raise ValueError("No reservoir stations with finite simulation data across all selected simulations")
    layout = manifest["sources"][info.sim_source]
    selected["sim_dir"] = layout["files"][0]["path"]
    selected["sim_syear"], selected["sim_eyear"] = info.sim_syear, info.sim_eyear
    selected["sim_lon"], selected["sim_lat"] = selected.ref_lon, selected.ref_lat
    audit = dict(
        manifest,
        selected_ids=selected.ID.tolist(),
        unmatched_observations=rows.loc[~rows.ID.isin(manifest["common_ids"]), "ID"].tolist(),
        all_missing_ids=sorted(set(candidates.ID) - set(valid), key=int),
        matching="exact GRanD ID; no spatial fallback",
    )
    target = Path(info.casedir) / "reservoir_matching" / f"{info.item}_{info.ref_source}_{info.sim_source}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_file_atomic(target, lambda path: Path(path).write_text(json.dumps(audit, indent=2)), suffix=".tmp.json")
    logging.info("Reservoir matching: %d common stations for %s", len(selected), info.sim_source)
    return selected


def process_reservoir_source(processor):
    manifest = json.loads(getattr(processor, f"{processor.sim_source}_reservoir_manifest"))
    layout = manifest["sources"][processor.sim_source]
    ids = [str(value) for value in processor.station_list.ID]
    data = read_reservoir_series(layout, ids, processor)
    for _, station in processor.station_list.iterrows():
        series = data.sel(station=str(station.ID), drop=True).to_dataset()
        processed = processor.process_single_station_data(series, int(station.use_syear), int(station.use_eyear), "sim")
        processor.save_station_data(processed, station, "sim")
