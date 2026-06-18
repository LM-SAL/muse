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
    import torch  # NOQA: PLC0415 - keep torch out of add_history-only imports

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
    import torch  # NOQA: PLC0415 - keep torch out of add_history-only imports

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


def add_history(
    ds: xr.Dataset | xr.DataArray,
    func_or_local_vars: Callable | str | dict,
    func: Callable | str | None = None,
) -> None:
    """
    Add a history entry to a dataset.

    Parameters
    ----------
    ds : `xarray.Dataset` or `xarray.DataArray`
        Dataset to update.
    func_or_local_vars : `Callable`, `str`, or `dict`
        Function being recorded in the history.
        For the legacy full-call record, pass ``locals()`` here and ``func`` as
        the third argument.
    func : `Callable` or `str`, optional
        Function being recorded with its input values.
    """
    if func is None:
        if isinstance(func_or_local_vars, dict):
            msg = "func must be provided when local variables are passed"
            raise TypeError(msg)
        name = func_or_local_vars if isinstance(func_or_local_vars, str) else func_or_local_vars.__name__
        history_entry = name
    else:
        local_vars = func_or_local_vars
        string_vals = []
        for arg, value in local_vars.items():
            if arg in inspect.signature(func).parameters:
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
