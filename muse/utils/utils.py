"""
Functions whose scope is not limited to one part of the muse package.
"""

import inspect
import datetime
import importlib.util
from collections.abc import Callable

import numpy as np
import xarray as xr

import astropy.units as u

import muse
from muse.log import logger

__all__ = [
    "add_history",
    "jax_to_numpy",
    "numpy_to_jax",
    "numpy_to_torch",
    "require_unit",
    "torch_to_numpy",
    "update_attrs",
]


def require_unit(ds: xr.Dataset, name: str, label: str, *, coord_only: bool = False, convertible_to=None):
    """
    Validate that ``ds[name]`` exists and carries a usable ``astropy`` unit.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset to inspect.
    name : `str`
        Variable or coordinate name to look up.
    label : `str`
        Human-readable name used in error messages (e.g. ``"raster.vdem"`` or
        ``"x coordinate"``).
    coord_only : `bool`, optional
        When `True`, require ``name`` to be a coordinate rather than any
        variable, by default `False`.
    convertible_to : `astropy.units.Unit`, optional
        When given, also require the unit to be convertible to this unit.

    Returns
    -------
    `astropy.units.Unit`
        The parsed unit of ``ds[name]``.
    """
    if name not in (ds.coords if coord_only else ds):
        msg = f"{label} is missing"
        raise ValueError(msg)
    array = ds[name]
    if "units" not in array.attrs:
        msg = f"{label} must define units"
        raise ValueError(msg)
    try:
        unit = u.Unit(array.attrs["units"])
    except (TypeError, ValueError) as exc:
        msg = f"{label} units must be a valid astropy unit"
        raise ValueError(msg) from exc
    if convertible_to is not None:
        try:
            unit.to(convertible_to)
        except u.UnitConversionError as exc:
            msg = f"{label} units must be convertible to {convertible_to}"
            raise ValueError(msg) from exc
    return unit


def jax_to_numpy(jax_array):
    """
    Convert a JAX array to a `numpy.ndarray`.

    The array is copied back to host memory first so arrays on accelerator
    devices convert cleanly.

    Parameters
    ----------
    jax_array : `jax.Array`
        JAX array to convert.

    Returns
    -------
    `numpy.ndarray`
        The converted NumPy array.
    """
    return np.asarray(jax_array)


def _jax_device(cuda_device: int | None):
    import jax  # NOQA: PLC0415

    if cuda_device is None:
        return jax.devices("cpu")[0]
    try:
        return jax.devices("gpu")[cuda_device]
    except (RuntimeError, IndexError) as exc:
        msg = f"CUDA device {cuda_device} is not available to JAX"
        raise ValueError(msg) from exc


def _resolve_backend(cuda_device: int | None = None, backend: str = "numpy") -> str:
    """
    Validate and resolve the array backend, returning ``"numpy"``, ``"jax"``, or ``"torch"``.

    JAX and Torch are opt-in: ``backend`` defaults to ``"numpy"`` (also for `None`), so
    the array library never changes with what happens to be installed in the environment
    (the accelerator paths are float32, the NumPy path keeps the input dtype). This is the
    single source of truth for which backends are valid.

    Parameters
    ----------
    cuda_device : `int` or `None`, optional
        CUDA device index for GPU use (requires ``backend="jax"`` or ``"torch"``), or
        `None` for CPU.
    backend : `str`, optional
        ``"numpy"`` (default), ``"jax"``, or ``"torch"``.

    Raises
    ------
    ValueError
        For an unknown ``backend``, an accelerator backend that is not installed,
        NumPy asked for a CUDA device, or a negative CUDA device. A device index the
        backend cannot serve is reported later, when the array is placed.
    """
    if backend not in ("numpy", "jax", "torch"):
        msg = f"Unknown backend {backend!r}; choose 'numpy', 'jax', or 'torch'"
        raise ValueError(msg)
    if backend == "numpy":
        if cuda_device is not None:
            msg = "The numpy backend does not support cuda_device; use backend='jax' or 'torch'"
            raise ValueError(msg)
        return "numpy"
    if cuda_device is not None and cuda_device < 0:
        msg = f"CUDA device {cuda_device} is not valid"
        raise ValueError(msg)
    if importlib.util.find_spec(backend) is None:
        name = "JAX" if backend == "jax" else "Torch"
        msg = f"backend={backend!r} requested but {name} is not installed"
        raise ValueError(msg)
    return backend


def numpy_to_jax(numpy_array: np.ndarray, cuda_device: int | None = None):
    """
    Convert a `numpy.ndarray` to a JAX array.

    Floating-point precision is capped at float32: ``float64`` input is downcast
    to ``float32``, while narrower dtypes (``float16``, integers) are left as-is.

    Parameters
    ----------
    numpy_array : `numpy.ndarray`
        The array to convert.
    cuda_device : `int` or `None`, optional
        If provided, transfer the array to the specified CUDA device. If omitted,
        keep the old CPU default used by the previous tensor bridge.

    Returns
    -------
    `jax.Array`
        The converted JAX array.
    """
    import jax  # NOQA: PLC0415

    return jax.device_put(np.asarray(numpy_array), _jax_device(cuda_device))


def torch_to_numpy(torch_tensor):
    """
    Convert a `torch.Tensor` to a `numpy.ndarray`.

    The tensor is detached from the autograd graph and moved to the CPU first
    so tensors on CUDA devices convert cleanly.

    Parameters
    ----------
    torch_tensor : `torch.Tensor`
        Torch tensor to convert.

    Returns
    -------
    `numpy.ndarray`
        The converted NumPy array.
    """
    tensor = torch_tensor.detach().cpu()
    try:
        return tensor.numpy()
    except TypeError:  # dtypes numpy lacks, e.g. bfloat16 / float8
        return np.array(tensor.tolist())


def numpy_to_torch(numpy_array: np.ndarray, cuda_device: int | None = None):
    """
    Convert a `numpy.ndarray` to a `torch.Tensor`.

    Floating-point precision is capped at float32: ``float64`` input is downcast
    to ``float32``, while narrower dtypes (``float16``, integers) are left as-is.

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
    import torch  # NOQA: PLC0415

    tensor = torch.tensor(numpy_array)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    if cuda_device is not None:
        with torch.cuda.device(f"cuda:{cuda_device}"):
            return tensor.cuda()
    return tensor


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
