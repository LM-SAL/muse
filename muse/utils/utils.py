"""
Functions whose scope is not limited to one part of the muse package.
"""

import inspect
import datetime
from collections.abc import Callable, Sequence

import dask.array as da
import numpy as np
import xarray as xr

import astropy.units as u

from muse.log import logger
from muse.version import version as _version

__all__ = [
    "add_history",
    "coord_as_unit",
    "require_unit",
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


def coord_as_unit(ds: xr.Dataset, name: str, target_unit, label: str) -> xr.DataArray:
    """
    Return coordinate ``name`` converted to ``target_unit``.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset to inspect.
    name : `str`
        Coordinate name.
    target_unit : `str`
        Unit to convert the coordinate values to.
        Must be a valid astropy unit string.
    label : `str`
        Human-readable name used in error messages.

    Returns
    -------
    `xarray.DataArray`
        Coordinate values in ``target_unit`` with updated ``units`` attrs.
    """
    target_unit = u.Unit(target_unit)
    unit = require_unit(ds, name, label, coord_only=True, convertible_to=target_unit)
    converted = ds.coords[name] * unit.to(target_unit)
    return xr.DataArray(
        converted.data,
        dims=converted.dims,
        attrs={**ds.coords[name].attrs, "units": str(target_unit)},
        name=name,
    )


# Provenance is owned by add_history alone; update_attrs never copies these keys.
_PROVENANCE_ATTRS = ("HISTORY", "date created", "date modified", "version")


def _history_entries(history) -> list:
    if history is None:
        return []
    if isinstance(history, list):
        return history.copy()
    return [history]


def _inherit_history(ds: xr.Dataset | xr.DataArray, source: xr.Dataset | xr.DataArray) -> None:
    """
    Append ``source`` history to ``ds``, skipping any shared-ancestry prefix so that
    merging sources with nested histories never duplicates entries, whatever their
    order.
    """
    source_history = _history_entries(source.attrs.get("HISTORY"))
    if not source_history:
        return
    history = _history_entries(ds.attrs.get("HISTORY"))
    shared = 0
    for ours, theirs in zip(history, source_history, strict=False):
        if ours != theirs:
            break
        shared += 1
    ds.attrs["HISTORY"] = [*history, *source_history[shared:]]


def _touch_attrs(ds: xr.Dataset | xr.DataArray) -> None:
    today = datetime.datetime.now(tz=datetime.UTC).strftime("%d-%b-%Y")
    ds.attrs["date modified" if "date created" in ds.attrs else "date created"] = today
    ds.attrs["version"] = _version


def _attr_safe(value):
    """
    Coerce ``value`` to a form serializable by both netCDF4 and Zarr v3, or `None` to
    skip it.

    The safe intersection of the two backends is ``str``, ``int``, ``float`` and (small)
    lists of those; ``Quantity`` becomes a ``"<value> <unit>"`` string, ``bool`` is cast
    to ``int``, and anything else (arrays larger than 8 elements, datasets, ``None``,
    custom objects) is dropped.
    """
    if isinstance(value, u.Quantity):
        return f"{value.value} {value.unit}" if value.isscalar else None
    if isinstance(value, np.ndarray | da.Array):
        return value.tolist() if isinstance(value, np.ndarray) and value.size <= 8 else None
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
    *,
    sources: Sequence[xr.Dataset | xr.DataArray] = (),
) -> None:
    """
    Record a call in the dataset history and store its keyword inputs as attributes.

    This is a finalizer: it mutates ``ds`` in place, so ``ds`` must be a newly
    constructed output that the calling function owns, never a caller-owned
    input. It is the sole owner of the provenance attributes ``HISTORY``,
    ``date created``, ``date modified``, and ``version``.

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
    sources : sequence of `xarray.Dataset` or `xarray.DataArray`, optional
        Inputs whose ``HISTORY`` the result inherits, in order, before the new
        entry is appended. Omit it for a result that starts a new lineage.
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
            elif isinstance(value, xr.Dataset | xr.DataArray | np.ndarray | da.Array):
                if isinstance(value, np.ndarray) and value.shape in [(), (1,)]:
                    string_vals.append(f"{arg}={value.tolist()}")
                elif isinstance(value, xr.DataArray) and value.size == 1 and isinstance(value.data, np.ndarray):
                    string_vals.append(f"{arg}={value.values.tolist()}")
                else:
                    string_vals.append(f"{arg}={arg}")
            else:
                string_vals.append(f"{arg}={value}")
            if params[arg].default is not inspect.Parameter.empty:  # keyword input
                safe = _attr_safe(value)
                if safe is not None:
                    ds.attrs[arg] = safe
                elif value is not None and not isinstance(value, np.ndarray | xr.DataArray | xr.Dataset | da.Array):
                    # Drop Arrays/datasets silently; only warn for other types.
                    logger.warning(
                        f"Not storing keyword input {arg!r} as an attribute: a "
                        f"{type(value).__name__} is not netCDF/zarr serializable.",
                    )

        name = func if isinstance(func, str) else func.__name__
        history_entry = f"{name}({', '.join(string_vals)})"

    for source in sources:
        _inherit_history(ds, source)
    ds.attrs["HISTORY"] = [*_history_entries(ds.attrs.get("HISTORY")), history_entry]
    _touch_attrs(ds)


def update_attrs(
    ds: xr.Dataset | xr.DataArray,
    source: xr.Dataset | xr.DataArray | None = None,
    **attrs,
) -> None:
    """
    Copy non-provenance source attributes and apply explicit updates.

    This is a finalizer: it mutates ``ds`` in place, so ``ds`` must be a newly
    constructed output that the calling function owns, never a caller-owned
    input. The provenance attributes (``HISTORY``, ``date created``,
    ``date modified``, ``version``) are owned by `add_history` and never copied
    here; passing one as an explicit update raises `ValueError`. Pass the
    inputs to ``add_history(..., sources=...)`` to inherit their lineage.

    Parameters
    ----------
    ds : `xarray.Dataset` or `xarray.DataArray`
        Data whose attributes are updated in place.
    source : `xarray.Dataset` or `xarray.DataArray`, optional
        Data to copy attributes from.
    **attrs
        Attributes to add or update after source attributes are copied.
    """
    if forbidden := [key for key in attrs if key in _PROVENANCE_ATTRS]:
        msg = f"update_attrs cannot set provenance attributes {forbidden}; they are owned by add_history"
        raise ValueError(msg)
    if source is not None:
        ds.attrs.update({key: value for key, value in source.attrs.items() if key not in _PROVENANCE_ATTRS})
    ds.attrs.update(attrs)
