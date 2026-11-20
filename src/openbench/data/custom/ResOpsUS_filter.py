"""Custom station filter for the ResOpsUS dataset."""

import logging
from pathlib import Path

import pandas as pd

from openbench.util.station_ids import read_station_csv


def _resolve_station_list(info) -> Path:
    """Return the station list path that matches the routing grid resolution."""
    explicit = getattr(info, "ref_fulllist", "")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit ResOpsUS station list not found: {path}")
        return path
    try:
        sim_grid_res = float(info.sim_grid_res)
    except Exception as exc:  # pragma: no cover - defensive
        logging.error(f"Invalid sim_grid_res for ResOpsUS: {info.sim_grid_res} ({exc})")
        raise

    file_map = {
        0.25: "ResOpsUS_reservoir_matched_15min.csv",
        0.1: "ResOpsUS_reservoir_matched_06min.csv",
    }
    if sim_grid_res not in file_map:
        logging.error("ResOpsUS only supports sim_grid_res values of 0.25 or 0.1 degrees.")
        raise ValueError(f"Unsupported sim_grid_res: {sim_grid_res}")

    filename = file_map[sim_grid_res]
    candidates = [
        Path(info.ref_dir) / "list" / filename,
        Path(info.ref_dir).parent / "list" / filename,
    ]

    for station_list in candidates:
        if station_list.exists():  # pragma: no cover - IO guard
            return station_list

    logging.error(f"ResOpsUS station list not found. Tried: {', '.join(str(p) for p in candidates)}")
    raise FileNotFoundError(candidates[0])


def filter_ResOpsUS(info, ds=None):
    """Load and filter ResOpsUS station metadata based on simulation settings.

    Args:
        info: Configuration/info object with processing parameters
        ds: Optional xarray Dataset to filter (for data filtering mode)

    Returns:
        For data filtering mode: Tuple of (info, filtered_data)
        For initialization mode: None (modifies info in place)
    """
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        from openbench.util.names import get_xarray_key_case_insensitive

        requested = getattr(info, "ref_varname", None)
        requested = requested[0] if isinstance(requested, list) and requested else requested
        actual = get_xarray_key_case_insensitive(ds, requested) if requested else None
        if actual is None:
            raise ValueError(f"ResOpsUS observation variable {requested!r} not found; available: {list(ds.data_vars)}")
        return info, ds[actual]

    station_list_path = _resolve_station_list(info)
    info.ref_fulllist = str(station_list_path)

    from openbench.config.runtime_info import GeneralInfoReader
    from openbench.data.reservoir import grand_id, match_reservoir_stations
    from openbench.util.station_ids import merge_on_station_id

    df = read_station_csv(station_list_path)
    if "ID" not in df:
        raise ValueError("ResOpsUS station list requires ID")
    df["ID"] = df["ID"].map(grand_id)
    if df.ID.duplicated().any():
        raise ValueError(f"Duplicate ResOpsUS GRanD IDs: {df.loc[df.ID.duplicated(False), 'ID'].tolist()}")
    rename = {"SYEAR": "ref_syear", "EYEAR": "ref_eyear", "LON": "ref_lon", "LAT": "ref_lat", "DIR": "ref_dir"}
    df.rename(columns={key: val for key, val in rename.items() if val not in df}, inplace=True)
    required = set(rename.values())
    if not required <= set(df):
        raise ValueError(f"ResOpsUS station list missing columns: {sorted(required - set(df))}")
    for col in ("ref_syear", "ref_eyear", "ref_lon", "ref_lat"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if (df.ref_syear > df.ref_eyear).any():
        raise ValueError("ResOpsUS station list has reversed years")
    flags = df["Flag"] if "Flag" in df else pd.Series(True, index=df.index)
    text = flags.astype(str).str.strip().str.lower()
    if not text.isin(["true", "false", "1", "0"]).all():
        raise ValueError("ResOpsUS Flag must be true/false or 1/0")
    df["Flag"] = text.isin(["true", "1"])
    GeneralInfoReader._resolve_relative_paths(df, "ref_dir", str(station_list_path), info.ref_dir)
    df["use_syear"] = df.ref_syear.clip(lower=max(int(info.sim_syear), int(info.syear), int(info.ref_syear)))
    df["use_eyear"] = df.ref_eyear.clip(upper=min(int(info.sim_eyear), int(info.eyear), int(info.ref_eyear)))
    selected = (
        df.Flag
        & ((df.use_eyear - df.use_syear) >= info.min_year)
        & df.ref_lat.between(info.min_lat, info.max_lat)
        & df.ref_lon.between(info.min_lon, info.max_lon)
    )
    df = df[selected].copy()
    if df.empty:
        raise ValueError("No ResOpsUS stations selected in the requested region/years")
    for path in df.ref_dir:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"ResOpsUS observation not found: {path}; check DIR relative to {station_list_path} or {info.ref_dir}"
            )
    if getattr(info, f"{info.sim_source}_reservoir_manifest", None):
        df = match_reservoir_stations(info, df)
    elif getattr(info, "sim_data_type", "grid") == "stn":
        sim = getattr(info, "sim_stn_list", None)
        if sim is None or sim.empty:
            raise ValueError("ResOpsUS station simulation requires a simulation fulllist")
        sim = sim.copy()
        sim["ID"] = sim.ID.map(grand_id)
        if sim.ID.duplicated().any():
            raise ValueError("Duplicate simulation GRanD IDs")
        # Keep reference filtering and both paths; never restore unselected rows.
        df = merge_on_station_id(sim, df, suffixes=("_simdup", ""))
        df["use_syear"] = df[["sim_syear", "use_syear"]].max(axis=1).astype(int)
        df["use_eyear"] = df[["sim_eyear", "use_eyear"]].min(axis=1).astype(int)
        df = df[(df.use_eyear - df.use_syear) >= info.min_year].copy()
        if df.empty:
            raise ValueError("No shared ResOpsUS station IDs/years")
    info.stn_list = df
    info.use_syear = int(df.use_syear.min())
    info.use_eyear = int(df.use_eyear.max())
