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
from muse.log import logger

__all__ = ["add_history", "numpy_to_torch", "torch_to_numpy", "update_attrs"]


def torch_to_numpy(torch_tensor, cuda_device: int | None = None):
    """
    Convert a `torch.Tensor` to a `numpy.ndarray`.

    Parameters
    ----------
    torch_tensor : `torch.Tensor`
        Torch tensor to convert.
    cuda_device : `int`, optional
        If provided, transfer the tensor from the GPU to the CPU before conversion.


    Returns
    -------
    `numpy.ndarray`
        The converted NumPy array.
    """
    import torch  # NOQA: PLC0415 - Avoid a heavy import unless we need it

    if cuda_device is not None:
        with torch.cuda.device(f"cuda:{cuda_device}"):
            return torch_tensor.cpu().numpy()
    try:
        return torch_tensor.numpy()
    except RuntimeError:
        return np.array(torch_tensor.tolist())


def numpy_to_torch(numpy_array: np.ndarray, cuda_device: int | None = None):
    """
    Convert a `numpy.ndarray` to a `torch.Tensor`.

    Parameters
    ----------
    numpy_array : `numpy.ndarray`
        The array to convert.
    cuda_device : `int` or `None`, optional
        If provided, transfer the tensor to the specified CUDA device.

    Returns
    -------
    `torch.Tensor`
        The converted Torch tensor.
    """
    import torch  # NOQA: PLC0415 - Avoid a heavy import unless we need it

    if cuda_device is not None:
        with torch.cuda.device(f"cuda:{cuda_device}"):
            return torch.tensor(numpy_array).cuda()
    return torch.tensor(numpy_array)


def _history_entries(history) -> list:
    if history is None:
        return []
    if isinstance(history, list):
        return history.copy()
    return [history]


def _touch_attrs(ds: xr.Dataset | xr.DataArray) -> None:
    today = datetime.datetime.now(tz=datetime.UTC).strftime("%d-%b-%Y")
    ds.attrs["date modified" if "date created" in ds.attrs else "date created"] = today
    ds.attrs["version"] = muse.__version__


def _attr_safe(value):
    """
    Coerce ``value`` to a form serializable by both netCDF4 and Zarr v3, or `None` to skip it.

    The safe intersection of the two backends is ``str``, ``int``, ``float`` and
    (small) lists of those; ``Quantity`` becomes a ``"<value> <unit>"`` string,
    ``bool`` is cast to ``int``, and anything else (arrays larger than 8 elements,
    datasets, ``None``, custom objects) is dropped.
    """
    if isinstance(value, u.Quantity):
        return f"{value.value} {value.unit}" if value.isscalar else None
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 8 else None
    if isinstance(value, bool):  # before int - netCDF4 attrs have no bool type, and bool is an int subclass
        return int(value)
    if isinstance(value, str | int | float):
        return value
    if isinstance(value, list | tuple):
        return [safe for safe in (_attr_safe(item) for item in value) if safe is not None]
    return None


def add_history(
    ds: xr.Dataset | xr.DataArray,
    func_or_local_vars: Callable | str | dict,
    func: Callable | str | None = None,
) -> None:
    """
    Record a call in the dataset history and store its keyword inputs as attributes.

    When ``locals()`` and ``func`` are passed, every keyword input (a parameter
    of ``func`` that has a default) is stored on ``ds.attrs`` after being coerced
    to a netCDF/zarr-serializable form (see `_attr_safe`); required positional
    parameters such as the data itself are skipped.

    Parameters
    ----------
    ds : `xarray.Dataset` or `xarray.DataArray`
        Dataset to update.
    func_or_local_vars : `Callable`, `str`, or `dict`
        Function being recorded in the history.
        For the full-call record, pass ``locals()`` here and ``func`` as
        the third argument.
    func : `Callable` or `str`, optional
        Function being recorded with its input values.
    """
    if func is None:
        if isinstance(func_or_local_vars, dict):
            msg = "func must be provided when local variables are passed"
            raise TypeError(msg)
        # A bare string label or a function passed alone both record just the name, no attrs.
        history_entry = func_or_local_vars if isinstance(func_or_local_vars, str) else func_or_local_vars.__name__
    else:
        local_vars = func_or_local_vars
        params = inspect.signature(func).parameters
        string_vals = []
        for arg, value in local_vars.items():
            if arg not in params:
                continue
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
            if params[arg].default is not inspect.Parameter.empty:  # keyword input
                safe = _attr_safe(value)
                if safe is not None:
                    ds.attrs[arg] = safe
                elif value is not None:
                    logger.warning(
                        f"Not storing keyword input {arg!r} as an attribute: a "
                        f"{type(value).__name__} is not netCDF/zarr serializable.",
                    )

        name = func if isinstance(func, str) else func.__name__
        history_entry = f"{name}({', '.join(string_vals)})"

    ds.attrs["HISTORY"] = [*_history_entries(ds.attrs.get("HISTORY")), history_entry]
    _touch_attrs(ds)


def update_attrs(
    ds: xr.Dataset | xr.DataArray,
    source: xr.Dataset | xr.DataArray | None = None,
    **attrs,
) -> None:
    """
    Copy source attributes and apply explicit updates.

    Parameters
    ----------
    ds : `xarray.Dataset` or `xarray.DataArray`
        Data whose attributes are updated in place.
    source : `xarray.Dataset` or `xarray.DataArray`, optional
        Data to copy attributes from.
    **attrs
        Attributes to add or update after source attributes are copied.
    """
    if source is not None:
        history = _history_entries(ds.attrs.get("HISTORY"))
        source_history = _history_entries(source.attrs.get("HISTORY"))
        ds.attrs.update({key: value for key, value in source.attrs.items() if key != "HISTORY"})
        if source_history:
            if history[: len(source_history)] == source_history:
                ds.attrs["HISTORY"] = history
            else:
                ds.attrs["HISTORY"] = [*history, *source_history]
        elif history:
            ds.attrs["HISTORY"] = history
    ds.attrs.update(attrs)
