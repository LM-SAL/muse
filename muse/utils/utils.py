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

__all__ = ["add_history", "numpy_to_torch", "torch_to_numpy"]


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
