"""
Functions shared across the muse package.
"""

import re
import time
import uuid
import shutil
import hashlib
import inspect
import zipfile
import datetime as dt
import tempfile
import contextlib
from pathlib import Path
from collections.abc import Mapping, Callable, Sequence

import attrs
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
        Variable or coordinate name.
    label : `str`
        Name used in error messages, e.g. ``"x coordinate"``.
    coord_only : `bool`, optional
        Require a coordinate rather than any variable, by default `False`.
    convertible_to : `astropy.units.Unit`, optional
        Also require the unit to convert to this one.

    Returns
    -------
    `astropy.units.Unit`
        The unit of ``ds[name]``.
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
        Astropy unit to convert to.
    label : `str`
        Name used in error messages.

    Returns
    -------
    `xarray.DataArray`
        The coordinate in ``target_unit``, ``units`` attribute included, without the
        coordinates attached to it.
    """
    target_unit = u.Unit(target_unit)
    unit = require_unit(ds, name, label, coord_only=True, convertible_to=target_unit)
    converted = unit.to(target_unit, ds.coords[name].data)
    # No coords: scaling an index coordinate scales its data but not its index.
    return xr.DataArray(
        converted,
        dims=ds.coords[name].dims,
        attrs={**ds.coords[name].attrs, "units": str(target_unit)},
        name=name,
    )


def _require_increasing_axis(values: npt.ArrayLike, label: str, *, positive: bool = False) -> None:
    """
    Raise `ValueError` unless ``values`` are one-dimensional, non-empty, finite,
    strictly increasing and, when ``positive``, above zero.

    ``ndim`` is checked first because a scalar cannot be sliced.
    """
    values = np.asarray(values)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or (positive and np.any(values <= 0))
        # Neighbours are compared, not subtracted, as differences of unsigned integers wrap around.
        or np.any(values[1:] <= values[:-1])
    ):
        qualifier = "positive, " if positive else ""
        msg = f"{label} must be one-dimensional, finite, {qualifier}and strictly increasing"
        raise ValueError(msg)


# Provenance is owned by add_history alone; update_attrs never copies these keys.
_PROVENANCE_ATTRS = ("history", "date_created", "date_modified", "muse_version")
# How the code ran, not what the result is: the history records them, the attributes do not.
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
    Append the history of ``source`` to ``ds``, less the entries they start with in
    common, so nested histories never repeat an entry.
    """
    source_history = _history_entries(source.attrs.get("history"))
    if not source_history:
        return
    history = _history_entries(ds.attrs.get("history"))
    shared = 0
    # Without the stamps: the two outputs of one call may be stamped a second apart.
    for ours, theirs in zip(history, source_history, strict=False):
        if _STAMP.sub("", ours) != _STAMP.sub("", theirs):
            break
        shared += 1
    ds.attrs["history"] = [*history, *source_history[shared:]]


def _instrument_defaults_label(instrument_defaults: InstrumentDefaults) -> str:
    """
    Name an `InstrumentDefaults` in the history and the attributes.

    ``DEFAULTS_MUSE`` or ``custom``, and a digest of every value, which tells apart
    products made with other values, those of another muse version included.
    """
    digest = hashlib.sha256()
    for name in attrs.fields_dict(type(instrument_defaults)):
        digest.update(f"{name}={_written_in_full(getattr(instrument_defaults, name))};".encode())
    kind = "DEFAULTS_MUSE" if instrument_defaults == DEFAULTS_MUSE else "custom"
    return f"{kind} {digest.hexdigest()[:12]}"


def _written_in_full(value: object) -> str:
    """
    ``value`` as text with every array element, unlike the numpy repr of a large array.

    Equal values give equal text: numbers as floats, coordinates sorted by name.
    """
    if isinstance(value, xr.DataArray):
        # dims and unit too: the same numbers on another axis or in another unit differ
        coords = {name: _written_in_full(value.coords[name].values) for name in sorted(value.coords)}
        return f"{_written_in_full(value.data)} {value.dims} {value.attrs.get('units')} {coords}"
    if isinstance(value, u.Quantity):
        return f"{_written_in_full(value.value)} {value.unit}"
    if isinstance(value, np.ndarray | np.number | int | float):
        array = np.asarray(value)
        return str((array.astype(float) if np.issubdtype(array.dtype, np.number) else array).tolist())
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return "{" + ", ".join(f"{_written_in_full(key)}: {_written_in_full(item)}" for key, item in items) + "}"
    if isinstance(value, list | tuple | set | frozenset):
        # sorted: the order of a set changes with the hash seed
        items = sorted(value, key=_written_in_full) if isinstance(value, set | frozenset) else value
        return "[" + ", ".join(map(_written_in_full, items)) + "]"
    # the label of an InstrumentDefaults digests every value
    return repr(_instrument_defaults_label(value) if isinstance(value, InstrumentDefaults) else value)


def _touch_attrs(ds: xr.Dataset | xr.DataArray, now: str) -> None:
    """
    Stamp the creation date and muse version the first time, the modification date
    after.
    """
    if "date_created" in ds.attrs:
        ds.attrs["date_modified"] = now
    else:
        ds.attrs["date_created"] = now
        ds.attrs["muse_version"] = _version


def _entry_value(value: object) -> str:
    """
    One input as a history entry writes it, unit included.

    An array is its size and, when one-dimensional and in memory, its range; nothing
    lazy is computed.
    """
    # .data reads a file-backed array, which is never a Quantity
    if isinstance(value, xr.DataArray) and _loaded(value) and isinstance(value.data, u.Quantity):
        value = value.data  # a Quantity in a DataArray, such as a field of InstrumentDefaults
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
    Whether ``array`` is in memory: an entry never computes dask or reads a file.
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
    ``value`` in a form both netCDF4 and Zarr v3 store, or `None` to skip it.

    ``str``, ``int``, ``float`` and lists of them pass; a scalar ``Quantity`` becomes
    ``"<value> <unit>"``, a ``bool`` an ``int``, an array of up to 8 elements a list and
    a mapping one text of ``key=value`` pairs, written as in the history entry.
    """
    if isinstance(value, Mapping):  # netCDF and Zarr store no mapping, so it becomes the text returned below
        texts = (
            " ".join(map(_entry_value, item)) if isinstance(item, tuple) else _entry_value(item)
            for item in value.values()
        )
        value = ", ".join(f"{key}={text}" for key, text in zip(value, texts, strict=True))
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
    Record a call in the history of ``ds`` and store its keyword inputs as attributes.

    A finalizer: it mutates ``ds`` in place, so ``ds`` must be a new output the calling
    function owns, never a caller's input. It alone sets ``history``,
    ``date_created``, ``date_modified`` and ``muse_version``.

    The entry reads ``name(inputs) [<UTC time> muse <version>]``, with units kept and
    arrays summarised. Each input with a default is also stored in ``ds.attrs`` as a
    ``str``, number or short list, which netCDF and zarr can write; options of how the
    code ran, such as ``workers``, and values with no such form are not.

    Parameters
    ----------
    ds : `xarray.Dataset` or `xarray.DataArray`
        Dataset to update.
    local_vars : `dict`
        Inputs of the call, normally ``locals()``; empty to record the name alone.
    func : `Callable` or `str`
        Function recorded, or a label as-is, which records no inputs.
    sources : sequence of `xarray.Dataset` or `xarray.DataArray`, optional
        Inputs whose ``history`` comes first, in order. Omit for a new lineage.
    reading : `bool`, optional
        `True` for a call that reads a file: the dates and version stay as the file
        has them. By default `False`.
    """
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
    Copy the attributes of ``source`` to ``ds``, then set ``attrs``.

    A finalizer like `add_history`, which alone sets the provenance attributes: they are
    never copied, and passing one in ``attrs`` raises `ValueError`. Pass the inputs to
    ``add_history(..., sources=...)`` to inherit their lineage.

    Parameters
    ----------
    ds : `xarray.Dataset` or `xarray.DataArray`
        Data to update in place.
    source : `xarray.Dataset` or `xarray.DataArray`, optional
        Data to copy attributes from.
    **attrs
        Attributes to set after the copy.
    """
    if forbidden := [key for key in attrs if key in _PROVENANCE_ATTRS]:
        msg = f"update_attrs cannot set provenance attributes {forbidden}; they are owned by add_history"
        raise ValueError(msg)
    if source is not None:
        ds.attrs.update({key: value for key, value in source.attrs.items() if key not in _PROVENANCE_ATTRS})
    ds.attrs.update(attrs)


def _replace_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """
    Call ``write`` on a staging path beside ``path`` and rename the result into place: a
    reader never finds a half-written file, and a failed write leaves nothing behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Not named after path, whose name may already be near the 255-character limit; the suffix picks the writer.
    staging = path.with_name(f".{uuid.uuid4().hex}{path.suffix}")
    try:
        write(staging)
        staging.replace(path)
    finally:
        # Still there only if the write or the rename failed.
        _remove(staging)


def _remove(path: Path) -> None:
    """
    Remove the file or directory ``path`` left by a failed write, if there, without
    raising over the error of the write.
    """
    # A failed dask write raises while its threads still finish their chunks, which recreate what was removed:
    # remove until the path stays gone. shortcut: gives up after 1 s; wait on dask's threads if a path is left.
    for _ in range(20):
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            # Windows refuses to delete a file still open.
            with contextlib.suppress(OSError):
                path.unlink()
        time.sleep(0.05)
    if path.exists():
        logger.warning(f"Could not remove {path}, left by a failed write; delete it")


def _to_zarr(dataset: xr.Dataset, destination: Path, **kwargs) -> None:
    """
    Write ``dataset`` as a zarr v3 store: a directory, or one file for a ``.zip``
    destination.

    ``kwargs`` go to `xarray.Dataset.to_zarr`.
    """
    options = {"mode": "w", "zarr_format": 3, "consolidated": False, **kwargs}
    if destination.suffix.lower() != ".zip":
        dataset.to_zarr(destination, **options)
        return
    # zarr rewrites its metadata and a zip would keep every version: write a directory, then pack it.
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}."))
    try:
        store = staging / "store.zarr"
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
        # Written in full, so no thread is still writing: no need for the wait of _remove.
        shutil.rmtree(staging)
    finally:
        _remove(staging)
