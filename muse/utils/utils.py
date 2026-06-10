"""
Functions whose scope is not limited to one part of the muse package.
"""

import inspect
import datetime
from collections.abc import Callable

import numpy as np
import xarray as xr

import astropy.units as u

import muse

__all__ = [
    "add_history",
]


def add_history(ds: xr.Dataset, local_vars: dict, func: Callable) -> None:
    """
    Add a history entry to a dataset.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset to update.
    local_vars : `dict`
        Local variables from the calling function.
    func : `Callable`
        Function being recorded in the history.
    """
    string_vals = []
    for arg, value in local_vars.items():
        if arg in inspect.signature(func).parameters:
            # Handle astropy Quantity objects
            if isinstance(value, u.Quantity):
                string_vals.append(f"{arg}={value.value}")
            elif isinstance(value, xr.Dataset | xr.DataArray | np.ndarray):
                if isinstance(value, np.ndarray) and value.shape in [(), (1,)]:
                    string_vals.append(f"{arg}={value.tolist()}")
                elif isinstance(value, xr.DataArray) and value.size == 1:
                    string_vals.append(f"{arg}={value.values.tolist()}")
                else:
                    string_vals.append(f"{arg}={arg}")
            else:
                string_vals.append(f"{arg}={value}")

    history_entry = f"{func.__name__}({', '.join(string_vals)})"
    if "HISTORY" in ds.attrs:
        if isinstance(ds.attrs["HISTORY"], list):
            ds.attrs["HISTORY"].append(history_entry)
        else:
            ds.attrs["HISTORY"] = [ds.attrs["HISTORY"], history_entry]
    else:
        ds.attrs["HISTORY"] = [history_entry]

    today = datetime.datetime.now(tz=datetime.UTC)
    if "date created" in ds.attrs:
        ds.attrs["date modified"] = today.strftime("%d-%b-%Y")
    else:
        ds.attrs["date created"] = today.strftime("%d-%b-%Y")
    ds.attrs["version"] = muse.__version__
