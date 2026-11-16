"""
Functions whose scope is not limited to one part of the muse package.
"""

import re
import inspect
import zipfile
import datetime as dt
import tempfile
from pathlib import Path
from collections.abc import Callable, Sequence

import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr

import astropy.units as u

from muse.log import logger
from muse.variables import DEFAULTS_MUSE
from muse.variables_schema import InstrumentDefaults
from muse.version import version as _version

__all__ = [
    "add_history",
    "coord_as_unit",
    "require_unit",
    "update_attrs",
]


def require_unit(
    ds: xr.Dataset,
    name: str,
    label: str,
    *,
    coord_only: bool = False,
    convertible_to: u.UnitBase | str | None = None,
) -> u.UnitBase:
    """
    Validate that ``ds[name]`` exists and carries a usable ``astropy`` unit.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset to inspect.
    name : `str`
        Variable or coordinate name to look up.
    label : `str`
        Human-readable name used in error messages (e.g. ``"vdem.vdem"`` or
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


def coord_as_unit(ds: xr.Dataset, name: str, target_unit: u.UnitBase | str, label: str) -> xr.DataArray:
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
        Coordinate values in ``target_unit`` with updated ``units`` attrs. Coordinates
        attached to the source coordinate are not retained.
    """
    target_unit = u.Unit(target_unit)
    unit = require_unit(ds, name, label, coord_only=True, convertible_to=target_unit)
    converted = unit.to(target_unit, ds.coords[name].data)
    # Rebuilt without coords on purpose: multiplying an index coordinate scales the
    # data but not the index, so the converted values must not stay self-indexed.
    return xr.DataArray(
        converted,
        dims=ds.coords[name].dims,
        attrs={**ds.coords[name].attrs, "units": str(target_unit)},
        name=name,
    )


def _require_increasing_axis(values: npt.ArrayLike, label: str, *, positive: bool = False) -> None:
    """
    Validate that ``values`` form a usable one-dimensional axis.

    Raises `ValueError` unless the values are one-dimensional, non-empty, finite,
    strictly increasing and, when ``positive``, greater than zero. The conditions are
    checked in one short-circuiting expression so ``numpy.diff`` never sees a zero-
    dimensional input.
    """
    values = np.asarray(values)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or (positive and np.any(values <= 0))
        or np.any(np.diff(values) <= 0)
    ):
        qualifier = "positive, " if positive else ""
        msg = f"{label} must be one-dimensional, finite, {qualifier}and strictly increasing"
        raise ValueError(msg)


# Provenance is owned by add_history alone; update_attrs never copies these keys.
_PROVENANCE_ATTRS = ("history", "date_created", "date_modified", "muse_version")
# Keyword inputs that say how the code ran, not what the result is: the history entry records
# them, the attributes do not, so that they neither crowd out the description of the data nor
# overwrite one another from call to call.
_EXECUTION_OPTIONS = frozenset(
    {
        "workers",
        "threads",
        "chunked",
        "chunks",
        "create_if_missing",
        "skip_existing",
        "logT_method",
        "doppler_velocity_method",
        "device",
        "progress",
    }
)


# The time and version that end a history entry.
_STAMP = re.compile(r" \[[^\[\]]* muse [^\[\]]*\]$")


def _history_entries(history: str | list[str] | None) -> list[str]:
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
    source_history = _history_entries(_current_attrs(source.attrs).get("history"))
    if not source_history:
        return
    history = _history_entries(ds.attrs.get("history"))
    shared = 0
    # Without the stamps: the two outputs of one call, such as align_response_and_vdem, may be
    # stamped a second apart and are still one step.
    for ours, theirs in zip(history, source_history, strict=False):
        if _STAMP.sub("", ours) != _STAMP.sub("", theirs):
            break
        shared += 1
    ds.attrs["history"] = [*history, *source_history[shared:]]


def _instrument_defaults_label(instrument_defaults: InstrumentDefaults) -> str:
    """
    How the history and the attributes name an
    `~muse.variables_schema.InstrumentDefaults`:

    the whole object does not fit an attribute, so only whether it is the MUSE one.
    """
    # shortcut: two different custom objects get the same label; the values they set are in the
    # other attributes of a product. Store a digest of the values if runs with several must not mix.
    return "DEFAULTS_MUSE" if instrument_defaults == DEFAULTS_MUSE else "custom"


def _touch_attrs(ds: xr.Dataset | xr.DataArray, now: str) -> None:
    """
    Date ``ds``: its creation and the muse that made it the first time, a modification
    after that.
    """
    if "date_created" in ds.attrs:
        ds.attrs["date_modified"] = now
    else:
        ds.attrs["date_created"] = now
        ds.attrs["muse_version"] = _version


def _entry_value(value: object) -> str:
    """
    How a history entry writes one input: with its unit, and an array as its size and,
    when it is one-dimensional and in memory, its range.

    Nothing lazy is computed.
    """
    # .data reads a file-backed array, which is never a Quantity
    if isinstance(value, xr.DataArray) and _loaded(value) and isinstance(value.data, u.Quantity):
        value = value.data  # a Quantity in a DataArray, as the line-list builders take a temperature
    if isinstance(value, xr.Dataset):
        return f"<Dataset of {', '.join(map(str, value.data_vars))}>" if value.data_vars else "<Dataset>"
    if isinstance(value, u.Quantity):
        return str(value) if value.isscalar else _array_summary(value.value, f" {value.unit}")
    if isinstance(value, xr.DataArray):
        unit = f" {value.attrs['units']}" if "units" in value.attrs else ""
        if value.ndim == 0 and value.chunks is None:
            return f"{value.values.item()}{unit}"
        return _array_summary(value, unit)
    if isinstance(value, np.ndarray | da.Array):
        return str(value.item()) if isinstance(value, np.ndarray) and value.ndim == 0 else _array_summary(value, "")
    return str(value)


def _loaded(array: object) -> bool:
    """
    Whether the values of ``array`` are at hand: neither computed with dask nor read
    from a file, which an entry must never do.
    """
    in_memory = getattr(getattr(array, "variable", None), "_in_memory", True)  # xarray's own test
    return not isinstance(array, da.Array) and getattr(array, "chunks", None) is None and in_memory


def _array_summary(array: np.ndarray | da.Array | xr.DataArray, unit: str) -> str:
    if array.ndim == 1 and _loaded(array) and np.issubdtype(array.dtype, np.number) and array.size:
        values = np.asarray(array)
        if array.size == 1:
            return f"[{values[0]}{unit}]"
        return f"<{array.size} values, {values.min()} to {values.max()}{unit}>"
    return f"<array of shape {tuple(array.shape)}{unit}>"


def _attr_safe(value: object) -> str | int | float | list | None:
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
    local_vars: dict,
    func: Callable | str,
    *,
    sources: Sequence[xr.Dataset | xr.DataArray] = (),
    reading: bool = False,
) -> None:
    """
    Record a call in the dataset history and store its keyword inputs as attributes.

    This is a finalizer: it mutates ``ds`` in place, so ``ds`` must be a newly
    constructed output that the calling function owns, never a caller-owned
    input. It is the sole owner of the provenance attributes ``history``,
    ``date_created``, ``date_modified`` and ``muse_version``.

    The entry reads ``name(inputs) [<UTC time> muse <version>]``, with units kept
    and arrays summarised. Every keyword input (a parameter of ``func`` that has a
    default) is also stored on ``ds.attrs`` after being coerced to a
    netCDF/zarr-serializable form (see `_attr_safe`), except the options that only
    say how the code ran, such as ``workers``; required positional parameters such
    as the data itself are skipped.

    Parameters
    ----------
    ds : `xarray.Dataset` or `xarray.DataArray`
        Dataset to update.
    local_vars : `dict`
        Inputs of the call being recorded, normally ``locals()``. Pass an empty
        mapping to record the name alone.
    func : `Callable` or `str`
        Function being recorded. A plain string is used as the label as-is and
        records no inputs.
    sources : sequence of `xarray.Dataset` or `xarray.DataArray`, optional
        Inputs whose ``history`` the result inherits, in order, before the new
        entry is appended. Omit it for a result that starts a new lineage.
    reading : `bool`, optional
        `True` for a call that reads a file: its entry is added, and the dates and
        version stay as the file has them. By default `False`.
    """
    # A dataset that took the attributes of a file of muse 0.3 by xarray keeps their lineage.
    ds.attrs = _current_attrs(ds.attrs)
    # A string label has no signature to inspect, so it records no inputs.
    params = inspect.signature(func).parameters if callable(func) else {}
    formatted_args = []
    for arg, value in local_vars.items():
        if arg not in params:
            continue
        if isinstance(value, InstrumentDefaults):
            value = _instrument_defaults_label(value)  # NOQA: PLW2901
        formatted_args.append(f"{arg}={_entry_value(value)}")
        if params[arg].default is not inspect.Parameter.empty and arg not in _EXECUTION_OPTIONS:
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
    now = dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    history_entry = f"{name}({', '.join(formatted_args)}) [{now} muse {_version}]"

    for source in sources:
        _inherit_history(ds, source)
    ds.attrs["history"] = [*_history_entries(ds.attrs.get("history")), history_entry]
    # A reader returns the provenance of its file: dating the read would claim the product.
    if not reading:
        _touch_attrs(ds, now)


def update_attrs(
    ds: xr.Dataset | xr.DataArray,
    source: xr.Dataset | xr.DataArray | None = None,
    **attrs,
) -> None:
    """
    Copy non-provenance source attributes and apply explicit updates.

    This is a finalizer: it mutates ``ds`` in place, so ``ds`` must be a newly
    constructed output that the calling function owns, never a caller-owned
    input. The provenance attributes (``history``, ``date_created``,
    ``date_modified``, ``muse_version``) are owned by `add_history` and never copied
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
        # Under their current names, also for a source that was read without the library.
        ds.attrs.update(
            {key: value for key, value in _current_attrs(source.attrs).items() if key not in _PROVENANCE_ATTRS}
        )
    ds.attrs.update(attrs)


# shortcut: the names that muse 0.3 and earlier wrote into line lists, continua, effective areas
# and responses; remove once the caches are migrated and the tutorial data is remade.
_OLD_ATTRS = {
    "Chianti": "chianti_version",
    "Chiantipy": "chiantipy_version",
    "fiasco": "fiasco_version",
    "aiapy": "aiapy_version",
    "continuum_Chianti": "continuum_chianti_version",
    "continuum_Chiantipy": "continuum_chiantipy_version",
    "continuum_fiasco": "continuum_fiasco_version",
    "HISTORY": "history",
    "date created": "date_created",
    "date modified": "date_modified",
    "version": "muse_version",
}


def _current_attrs(attrs: dict) -> dict:
    """
    ``attrs`` under their current names; where both an old and a current name are there,
    the current one wins.

    An old date takes the ISO form, and the options that only said how the code ran are
    left out.
    """
    # muse 0.3 wrote "HISTORY" into every file: a "version", "workers" or "device" without it is
    # another tool's own.
    of_muse_0_3 = "HISTORY" in attrs
    old_names = {old: new for old, new in _OLD_ATTRS.items() if old != "version" or of_muse_0_3}
    renamed = {
        old_names[name]: _iso_date(value) if name in ("date created", "date modified") else value
        for name, value in attrs.items()
        if name in old_names
    }
    kept = {
        name: value
        for name, value in attrs.items()
        if name not in old_names and not (of_muse_0_3 and name in _EXECUTION_OPTIONS)
    }
    return renamed | kept


def _iso_date(value: object) -> object:
    """
    A date as muse 0.3 wrote it, such as ``01-Mar-2026``, in ISO 8601; anything else as
    it is.
    """
    try:
        return dt.datetime.strptime(value, "%d-%b-%Y").date().isoformat()  # NOQA: DTZ007
    except (TypeError, ValueError):
        return value


def _with_current_names(dataset: xr.Dataset) -> xr.Dataset:
    """
    ``dataset`` as read from a file, with the version and provenance attributes of
    earlier versions under their current names.
    """
    dataset.attrs = _current_attrs(dataset.attrs)
    return dataset


def _to_zarr(dataset: xr.Dataset, destination: Path, **kwargs) -> None:
    """
    Write ``dataset`` as a zarr store of format 3: a directory or, for a destination
    that ends in ``.zip``, one file.

    ``kwargs`` go to `xarray.Dataset.to_zarr`.
    """
    options = {"mode": "w", "zarr_format": 3, "consolidated": False, **kwargs}
    if destination.suffix.lower() != ".zip":
        dataset.to_zarr(destination, **options)
        return
    # zarr writes its metadata more than once, and a zip keeps every version as an entry of the
    # same name. So the store is written as a directory first and packed whole.
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}.") as staging:
        store = Path(staging) / "store.zarr"
        dataset.to_zarr(store, **options)
        try:
            # Stored, not deflated: the chunks are compressed already.
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
                for path in sorted(store.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(store).as_posix())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
