# pylint: disable-msg=E1101,W0613,W0603


from collections import OrderedDict
import itertools
import re
import sys

import numpy as np
import pandas as pd

from pandas_datareader.io.util import _read_content


def read_jsdmx(path_or_buf):
    """
    Convert a SDMX-JSON 1.0 string to pandas object.

    Parameters
    ----------
    path_or_buf : a valid SDMX-JSON string or file-like
        https://github.com/sdmx-twg/sdmx-json

    Returns
    -------
    results : Series, DataFrame, or dictionary of Series or DataFrame.
    """

    jdata = _read_content(path_or_buf)

    try:
        import simplejson as json
    except ImportError as exc:
        if sys.version_info[:2] < (2, 7):
            raise ImportError("simplejson is required in python 2.6") from exc
        import json

    if isinstance(jdata, dict):
        data = jdata
    else:
        data = json.loads(jdata, object_pairs_hook=OrderedDict)

    structure = data["structure"]
    index = _parse_dimensions(structure["dimensions"]["observation"])
    columns = _parse_dimensions(structure["dimensions"]["series"])

    dataset = data["dataSets"]
    if len(dataset) != 1:
        raise ValueError("length of 'dataSets' must be 1")
    dataset = dataset[0]
    values = _parse_values(dataset, index=index, columns=columns)

    df = pd.DataFrame(values, columns=columns, index=index)
    return df


def read_jsdmx2(data):
    """
    Convert a SDMX-JSON 2.0 dict (as returned by the new OECD API) to a
    pandas DataFrame.

    The new OECD REST API (https://sdmx.oecd.org/public/rest/) returns
    SDMX-JSON 2.0 where all dimensions are in ``structure.dimensions.observation``
    and observations are keyed by colon-separated dimension indices.

    Parameters
    ----------
    data : dict
        Parsed SDMX-JSON 2.0 response (the ``data`` sub-key or the full
        response dict).

    Returns
    -------
    DataFrame
    """
    # The SDMX-JSON 2.0 payload may be wrapped in a top-level "data" key
    if "data" in data and isinstance(data["data"], dict):
        data = data["data"]

    structure = data["structure"]
    obs_dims = structure["dimensions"]["observation"]

    # Build the dimension value lists
    dim_values = []
    dim_names = []
    time_dim_idx = None
    for i, dim in enumerate(obs_dims):
        values = [v.get("name", v.get("id", str(j))) for j, v in enumerate(dim["values"])]
        role = dim.get("role") or ""
        if role.upper() in ("TIME", "TIME_PERIOD") or dim.get("id", "").upper() in (
            "TIME_PERIOD",
            "TIME",
        ):
            time_dim_idx = i
            values = [_fix_quarter_values(v) for v in values]
        dim_values.append(values)
        dim_names.append(dim.get("name", dim.get("id", f"dim{i}")))

    dataset = data["dataSets"]
    if len(dataset) != 1:
        raise ValueError("length of 'dataSets' must be 1")
    observations = dataset[0]["observations"]

    # Collect all unique index tuples for each dimension
    n_dims = len(obs_dims)
    records = []
    for key_str, obs_val in observations.items():
        indices = [int(x) for x in key_str.split(":")]
        if len(indices) != n_dims:
            continue
        record = {}
        for dim_i, idx in enumerate(indices):
            record[dim_names[dim_i]] = dim_values[dim_i][idx]
        record["_value"] = obs_val[0] if obs_val else np.nan
        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # If there's a single time dimension, use it as the index; otherwise create
    # a MultiIndex from all non-value columns
    value_col = "_value"
    dim_cols = [c for c in df.columns if c != value_col]

    if time_dim_idx is not None and len(dim_cols) == 1:
        # Simple case: one time dimension → flat Series/DataFrame
        time_col = dim_names[time_dim_idx]
        df = df.set_index(time_col)
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = time_col
        df = df[[value_col]].rename(columns={value_col: "value"})
    elif time_dim_idx is not None:
        # Pivot: time as index, remaining dims as columns
        time_col = dim_names[time_dim_idx]
        other_cols = [c for c in dim_cols if c != time_col]
        if other_cols:
            # Combine non-time dims into a column label
            df["_col"] = df[other_cols].apply(
                lambda row: " | ".join(str(row[c]) for c in other_cols), axis=1
            )
            df = df.pivot(index=time_col, columns="_col", values=value_col)
            df.index = pd.DatetimeIndex(df.index)
            df.index.name = time_col
        else:
            df = df.set_index(time_col)[[value_col]]
            df.index = pd.DatetimeIndex(df.index)
    else:
        df = df.set_index(dim_cols) if len(dim_cols) > 1 else df.set_index(dim_cols[0])

    return df


def _get_indexer(index):
    if index.nlevels == 1:
        return [str(i) for i in range(len(index))]
    else:
        it = itertools.product(*[range(len(level)) for level in index.levels])
        return [":".join(map(str, i)) for i in it]


def _fix_quarter_values(value):
    """Make raw quarter values Pandas-friendly (e.g. 'Q4-2018' -> '2018Q4')."""
    m = re.match(r"Q([1-4])-(\d\d\d\d)", value)
    if not m:
        return value
    quarter, year = m.groups()
    value = f"{quarter}Q{year}"
    return value


def _parse_values(dataset, index, columns):
    size = len(index)
    series = dataset["series"]

    values = []
    # for s_key, s_value in iteritems(series):
    for s_key in _get_indexer(columns):
        try:
            observations = series[s_key]["observations"]
            observed = []
            for o_key in _get_indexer(index):
                try:
                    observed.append(observations[o_key][0])
                except KeyError:
                    observed.append(np.nan)
        except KeyError:
            observed = [np.nan] * size

        values.append(observed)

    return np.transpose(np.array(values))


def _parse_dimensions(dimensions):
    arrays = []
    names = []
    for key in dimensions:
        values = [v["name"] for v in key["values"]]

        role = key.get("role", None)
        if role in ("time", "TIME_PERIOD"):
            values = [_fix_quarter_values(v) for v in values]
            values = pd.DatetimeIndex(values)

        arrays.append(values)
        names.append(key["name"])
    midx = pd.MultiIndex.from_product(arrays, names=names)
    if len(arrays) == 1 and isinstance(midx, pd.MultiIndex):
        # Fix for pandas >= 0.21
        midx = midx.levels[0]

    return midx
