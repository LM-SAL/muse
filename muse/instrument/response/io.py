import numbers
from pathlib import Path
from collections.abc import Mapping, Sequence

import dask
import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec
from zarr.storage import ZipStore

import astropy.units as u

from muse.log import logger
from muse.utils.utils import (
    _replace_atomically,
    _to_zarr,
    add_history,
    coord_as_unit,
    require_unit,
    update_attrs,
)
from muse.variables_schema import FrozenDict

__all__ = [
    "align_response_and_vdem",
    "read_response",
    "read_responses",
    "read_spectral_response",
    "save_response",
    "save_spectral_response",
]

_DEFAULT_RESPONSE_CHUNKS = {"line": 1, "doppler_velocity": 20, "logT": 1, "pressure": 1, "abundance": 1}
_NETCDF_SUFFIXES = {".nc", ".ncdf", ".netcdf"}
# A zarr store as a directory, or packed into one zip file.
_ZARR_SUFFIXES = {".zarr", ".zip"}


def save_response(
    response: xr.Dataset,
    response_file: str | Path,
    *,
    chunks: Mapping[str, int] | None = None,
) -> None:
    """
    Save a detector response as Zarr or NetCDF.

    Parameters
    ----------
    response : `xarray.Dataset`
        Dataset containing ``detector_response``.
    response_file : `str` or `pathlib.Path`
        A zarr store, as a ``.zarr`` directory or one ``.zip`` file, or a netCDF file ending
        in ``.nc``, ``.ncdf`` or ``.netcdf``. An existing one is not overwritten.
    chunks : mapping of `str` to `int`, optional
        Per-dimension overrides of the defaults ``line=1``, ``doppler_velocity=20``, ``logT=1``,
        ``pressure=1``, ``abundance=1`` and whole ``slit``/``detector_x_pixel`` planes. A value
        larger than its dimension takes the whole dimension.

    Notes
    -----
    Zarr uses Blosc/Zstd level 3 with bit-shuffle, netCDF zlib level 1 with shuffle, each
    chosen by benchmark on the full 171 response.
    """
    if not isinstance(response, xr.Dataset):
        msg = "response must be an xarray.Dataset"
        raise TypeError(msg)
    _validate_response(response, "response")
    response_file = _check_destination(response_file)
    chunked = response.drop_encoding().chunk(_response_chunks(response, chunks))
    _replace_atomically(response_file, lambda staging: _write_chunked(chunked, staging, "detector_response"))


def _check_destination(destination: str | Path, *, what: str = "response", argument: str = "response_file") -> Path:
    """
    The destination as a path, refusing an existing one or an unknown suffix.
    """
    destination = Path(destination)
    if destination.exists():
        msg = f"Refusing to overwrite existing {what}: {destination}"
        raise ValueError(msg)
    suffix = destination.suffix.lower()
    if suffix not in _ZARR_SUFFIXES and suffix not in _NETCDF_SUFFIXES:
        msg = f"{argument} must end in .zarr, .zip, .nc, .ncdf, or .netcdf"
        raise ValueError(msg)
    return destination


def _write_chunked(chunked: xr.Dataset, response_file: Path, variable: str) -> None:
    """
    Write a chunked dataset with the compression every response file gets.
    """
    if response_file.suffix.lower() in _ZARR_SUFFIXES:
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
        _to_zarr(chunked, response_file, encoding={variable: {"compressors": (compressor,)}})
        return
    chunksizes = tuple(axis[0] for axis in chunked[variable].chunks)
    chunked.to_netcdf(
        response_file, encoding={variable: {"chunksizes": chunksizes, "zlib": True, "complevel": 1, "shuffle": True}}
    )


def _response_chunks(response: xr.Dataset, overrides: Mapping[str, int] | None) -> dict[str, int]:
    chunks = {
        name: min(size, response.sizes[name])
        for name, size in _DEFAULT_RESPONSE_CHUNKS.items()
        if name in response.dims
    }
    for name in ("slit", "detector_x_pixel"):
        if name in response.dims:
            chunks[name] = response.sizes[name]
    if overrides is None:
        return chunks
    if not isinstance(overrides, Mapping):
        msg = "chunks must be a mapping of dimension names to positive integers"
        raise TypeError(msg)
    for name, size in overrides.items():
        if name not in response.dims:
            msg = f"chunks contains unknown dimension: {name}"
            raise ValueError(msg)
        if isinstance(size, bool) or not isinstance(size, int | np.integer) or size <= 0:
            msg = f"chunks[{name!r}] must be a positive integer"
            raise ValueError(msg)
        chunks[name] = min(int(size), response.sizes[name])
    return chunks


def _validate_response(response: xr.Dataset, label: str) -> None:
    """
    Reject a response not in the layout `save_response` writes.

    Parameters
    ----------
    response : `xarray.Dataset`
        Must hold ``detector_response``, ``logT`` and ``doppler_velocity`` coordinates in
        temperature and velocity units, a ``line`` or ``channel`` dimension, and
        ``line_wavelength`` and any ``detector_wavelength`` in a wavelength unit.
    label : `str`
        Name of the dataset in error messages.

    Raises
    ------
    ValueError
        At the first failing requirement, naming a legacy MUSE file as such.
    """
    if "detector_response" not in response.data_vars:
        legacy = " (it has the legacy MUSE names; convert it to the layout save_response writes)"
        hint = legacy if "SG_resp" in response else ""
        msg = f"{label} must contain detector_response{hint}"
        raise ValueError(msg)
    require_unit(response, "logT", f"{label}.logT", coord_only=True, convertible_to=u.dex(u.K))
    require_unit(response, "doppler_velocity", f"{label}.doppler_velocity", coord_only=True, convertible_to=u.km / u.s)
    if "line" not in response.dims and "channel" not in response.dims:
        msg = f"{label} must have a line or channel dimension"
        raise ValueError(msg)
    require_unit(response, "line_wavelength", f"{label}.line_wavelength", coord_only=True, convertible_to=u.AA)
    if "detector_wavelength" in response.coords:
        require_unit(
            response, "detector_wavelength", f"{label}.detector_wavelength", coord_only=True, convertible_to=u.AA
        )


def _open_dataset_file(response_file: Path, *, chunked: bool = False, decode_coords: bool | str = True) -> xr.Dataset:
    # chunks=None: lazily indexed arrays without dask, as open_dataset gives for netCDF.
    kwargs = {"chunks": {} if chunked else None, "decode_coords": decode_coords}
    if response_file.suffix == ".zip":
        # xarray closes only the stores it opens itself; zarr reopens a closed archive on demand.
        store = ZipStore(response_file, mode="r")
        response = xr.open_zarr(store, consolidated=False, **kwargs)
        response.set_close(store.close)
    elif response_file.is_dir():
        # consolidated=False reads zarr v2 and v3 alike, without the missing-metadata warning.
        response = xr.open_zarr(response_file, consolidated=False, **kwargs)
    else:
        response = xr.open_dataset(response_file, **kwargs)
    return response


def read_response(
    response_file: str | Path,
    *,
    logT: u.Quantity | None = None,
    doppler_velocity: u.Quantity | None = None,
    number_of_slits: int | None = None,
    logT_method: str = "nearest",
    doppler_velocity_method: str = "linear",
    chunked: bool = False,
) -> xr.Dataset:
    """
    Read a response file, resampled onto ``logT`` and ``doppler_velocity`` if given.

    Resampling sets the values of ``detector_response`` that are not finite or are negative to zero, and logs how many when the response is in memory (``chunked=False``).

    Parameters
    ----------
    response_file : `str` | `pathlib.Path`
        A zarr store (a directory or a ``.zip``) or a netCDF file, in the layout
        `save_response` writes: ``detector_response`` and ``line_wavelength``, a ``line`` or
        ``channel`` dimension, and units on ``logT``, ``doppler_velocity`` and the wavelengths.
    logT : `astropy.units.Quantity`, optional
        Temperature axis to resample onto, in a unit convertible to the response's, e.g.
        ``np.arange(5.0, 7.0, 0.1) * u.dex(u.K)``.
    doppler_velocity : `astropy.units.Quantity`, optional
        Velocity axis to resample onto, as for ``logT``.
    number_of_slits : `int`, optional
        Keep the first ``number_of_slits`` slits, relabelled from 0, e.g. to match a raster
        from `muse.transforms.reshape_x_to_slit_step`. By default all.
    logT_method : `str`, optional
        Interpolation method for ``logT``, by default "nearest".
    doppler_velocity_method : `str`, optional
        Interpolation method for ``doppler_velocity``, by default "linear".
    chunked : `bool`, optional
        If `True`, dask-backed with the on-disk chunks, so the response stays lazy and memory
        bounded through resampling and synthesis. By default `False`: numpy arrays read on
        first use (resampling reads them at once).

    Returns
    -------
    `xarray.Dataset`
        The response, with ``gain`` and ``pair_energy`` coordinates when the file states them.

    Raises
    ------
    TypeError
        If ``logT`` or ``doppler_velocity`` is not a Quantity.
    ValueError
        If a requested axis is empty, not finite, not convertible to the response's units or
        wholly outside its range, or the file is not in that layout.
    """
    response_file = Path(response_file)

    logT = _requested_axis(logT, "logT")
    doppler_velocity = _requested_axis(doppler_velocity, "doppler_velocity")
    for name, axis in (("logT", logT), ("doppler_velocity", doppler_velocity)):
        if axis is None:
            continue
        if len(axis.data) == 0:
            msg = f"{name} array must not be empty"
            raise ValueError(msg)
        if not np.all(np.isfinite(axis.data)):
            msg = f"{name} must contain only finite values"
            raise ValueError(msg)

    response = _open_dataset_file(response_file, chunked=chunked)

    _validate_response(response, "response")

    response = _resample_axis(response, "logT", logT, logT_method)
    response = _resample_axis(response, "doppler_velocity", doppler_velocity, doppler_velocity_method)

    if number_of_slits is not None:
        if isinstance(number_of_slits, bool) or not isinstance(number_of_slits, numbers.Integral):
            msg = f"number_of_slits must be an integer, got {number_of_slits!r}"
            raise TypeError(msg)
        if "slit" not in response.dims:
            msg = "number_of_slits needs a response with a slit dimension"
            raise ValueError(msg)
        if not 1 <= number_of_slits <= response.sizes["slit"]:
            msg = f"number_of_slits must be between 1 and the {response.sizes['slit']} slits of the response"
            raise ValueError(msg)
        number_of_slits = int(number_of_slits)  # a numpy integer is not stored as an attribute
        # Relabelled from 0, so that downstream exact joins see byte-identical axes.
        response = response.isel(slit=slice(number_of_slits)).assign_coords(slit=np.arange(number_of_slits))

    # Per line, so transform_flux_units converts with the detector this response was made for.
    dim = "channel" if "channel" in response.dims else "line"
    for name, unit in (("gain", u.electron / u.DN), ("pair_energy", u.eV / u.electron)):
        if name in response.coords:
            response = response.assign_coords({name: coord_as_unit(response, name, unit, f"{name} coordinate")})
        elif name in response.attrs:
            values = np.broadcast_to(u.Quantity(response.attrs[name]).to_value(unit), response.sizes[dim])
            response = response.assign_coords({name: (dim, values, {"units": str(unit)})})

    add_history(response, locals(), read_response, reading=True)
    return response


def _requested_axis(axis: u.Quantity | None, name: str) -> xr.DataArray | None:
    """
    ``axis`` as an `xarray.DataArray` along ``name`` with a ``units`` attribute.
    """
    if axis is None:
        return None
    if not isinstance(axis, u.Quantity):
        msg = f"{name} must be an astropy Quantity"
        raise TypeError(msg)
    return xr.DataArray(np.atleast_1d(axis.value), dims=name, attrs={"units": str(axis.unit)})


def _resample_axis(response: xr.Dataset, name: str, axis: xr.DataArray | None, method: str) -> xr.Dataset:
    """
    Resample onto ``axis`` along ``name``, dropping points beyond the response range.

    ``"nearest"`` selects samples, any other method interpolates; either way the result
    is clamped to finite, non-negative values, and a response in memory logs how many
    values that changed.
    """
    if axis is None:
        return response
    response_unit = require_unit(response, name, f"response.{name}", coord_only=True)
    axis_attrs = dict(response[name].attrs)
    try:
        converted = u.Unit(axis.attrs["units"]).to(response_unit, np.asarray(axis.values, dtype=float))
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        msg = f"{name} axis units {axis.attrs['units']!r} are not convertible to the response's {response_unit}"
        raise ValueError(msg) from exc
    # An end a few rounding steps from the one asked for still counts: a grid made with arange ends at
    # 7.8999999999999995 for 7.9, and float32 stores 7.1 as 7.0999999, so the steps are those of the coarser dtype.
    eps = max(
        np.finfo(d).eps if np.issubdtype(d, np.floating) else np.finfo(float).eps
        for d in (axis.dtype, response[name].dtype)
    )
    axis = axis.copy(data=converted)
    low, high = float(response[name].min()), float(response[name].max())
    tolerance = 16 * eps * max(abs(low), abs(high), 1.0)
    in_range = (axis >= low - tolerance) & (axis <= high + tolerance)
    if not bool(in_range.all()):
        logger.info(f"Requested {name} extends beyond the response range; trimming to the response grid")
        axis = axis.where(in_range, drop=True)
        if axis.size == 0:
            msg = f"Requested {name} axis has no overlap with the response range"
            raise ValueError(msg)
    if method == "nearest":
        response = response.sel({name: axis}, drop=True, method="nearest")
    else:
        nearest = response[name].sel({name: axis}, method="nearest").values
        if np.allclose(nearest, np.asarray(axis.values, dtype=float), rtol=1e-12, atol=0.0):
            # On the grid, selection equals interpolation and reads only the needed dask slabs.
            response = response.sel({name: axis}, drop=True, method="nearest")
        else:
            # Clipped to the grid, as interpolating a rounding step past its end gives NaN, which becomes zero below.
            response = response.interp({name: axis.clip(low, high)}, method=method)
    values = response.detector_response
    usable = np.isfinite(values) & (values >= 0)
    if values.chunks is None:
        # Counted where the response is in memory, at the cost of one pass; a dask-backed one stays lazy.
        clamped = int((~usable).sum())
        if clamped:
            logger.warning(
                f"{clamped} of the {values.size} values of detector_response are not finite or are negative after "
                f"resampling {name}: they are set to zero"
            )
    # Not fillna and clip: they leave positive infinity as it is.
    response["detector_response"] = values.where(usable, 0).assign_attrs(values.attrs)
    if response.detector_response.chunks is not None:
        # Point selection leaves one dask chunk per point, which shatters downstream graphs.
        response = response.chunk({name: -1})
    return response.assign_coords({name: axis.assign_attrs(axis_attrs)})


def align_response_and_vdem(
    response: xr.Dataset,
    vdem: xr.Dataset,
    *,
    coordinate_methods: Mapping[str, tuple] = FrozenDict(
        {"logT": ("nearest", u.dex(u.K)), "doppler_velocity": ("linear", u.km / u.s)}
    ),
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Resample the response onto the overlapping VDEM temperature and velocity grids.

    Parameters
    ----------
    response : `xarray.Dataset`
        Dataset containing ``detector_response``.
    vdem : `xarray.Dataset`
        Dataset containing ``vdem``.
    coordinate_methods : `~collections.abc.Mapping`, optional
        A ``(resampling method, unit)`` for each coordinate to align, by default ``logT``
        nearest in dex(K) and ``doppler_velocity`` linear in km / s. Others are left alone.

    Returns
    -------
    response, vdem : `tuple` of `xarray.Dataset`
        New datasets with identical aligned coordinates.

    Raises
    ------
    TypeError
        If either input is not an `xarray.Dataset`.
    ValueError
        If a required variable or coordinate is missing, malformed, or has no overlap.
    """
    for name, dataset in (("response", response), ("vdem", vdem)):
        if not isinstance(dataset, xr.Dataset):
            msg = f"{name} must be an xarray.Dataset"
            raise TypeError(msg)
    if "detector_response" not in response:
        msg = "response must contain detector_response"
        raise ValueError(msg)
    if "vdem" not in vdem:
        msg = "vdem must contain vdem"
        raise ValueError(msg)

    matched_response = response
    matched_vdem = vdem

    for name, (method, unit) in coordinate_methods.items():
        normalized_axes = []
        for dataset_name, dataset in (("response", matched_response), ("vdem", matched_vdem)):
            axis = coord_as_unit(dataset, name, unit, f"{dataset_name}.{name}")
            if axis.size == 0 or not bool(np.isfinite(axis).all()):
                msg = f"{dataset_name} {name} coordinate must contain finite values"
                raise ValueError(msg)
            normalized_axes.append(axis)

        matched_response = matched_response.assign_coords({name: normalized_axes[0]})
        matched_vdem = matched_vdem.assign_coords({name: normalized_axes[1]})

        matched_response = _resample_axis(matched_response, name, matched_vdem[name], method)
        axis = matched_response[name]
        if axis.size < matched_vdem.sizes[name]:
            matched_vdem = matched_vdem.sel({name: axis})
        # Share vdem's coordinate values so downstream exact joins see byte-identical axes.
        matched_response = matched_response.assign_coords({name: matched_vdem[name]})

    matched_response = matched_response.drop_attrs(deep=False)
    matched_vdem = matched_vdem.drop_attrs(deep=False)
    update_attrs(matched_response, response)
    update_attrs(matched_vdem, vdem)
    add_history(matched_response, locals(), align_response_and_vdem, sources=(response, vdem))
    add_history(matched_vdem, locals(), align_response_and_vdem, sources=(response, vdem))
    return matched_response, matched_vdem


def read_responses(
    response_files: Sequence[str | Path],
    *,
    logT: u.Quantity | None = None,
    doppler_velocity: u.Quantity | None = None,
    number_of_slits: int | None = None,
    logT_method: str = "nearest",
    doppler_velocity_method: str = "linear",
    chunked: bool = False,
) -> xr.Dataset:
    """
    Read several response files and concatenate them along ``line``.

    Parameters
    ----------
    response_files : `Sequence` of `str` or `pathlib.Path`
        Response files, in order, each with a ``channel`` coordinate.
    logT, doppler_velocity, number_of_slits, logT_method, doppler_velocity_method
        As for `muse.instrument.read_response`.
    chunked : `bool`, optional
        If `True`, the result stays dask-backed. By default `False`: in memory.

    Returns
    -------
    `xarray.Dataset`
        The concatenated response, with the files' ``channel`` along ``line``.

    Raises
    ------
    ValueError
        If no file is given, a file has no ``channel`` coordinate, or the files
        are of different instrument defaults or units.
    TypeError
        If ``logT`` or ``doppler_velocity`` is not a Quantity.
    """
    if not response_files:
        msg = "pass at least one response file"
        raise ValueError(msg)

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        datasets = []
        for response_file in response_files:
            dataset = read_response(
                Path(response_file),
                logT=logT,
                doppler_velocity=doppler_velocity,
                number_of_slits=number_of_slits,
                logT_method=logT_method,
                doppler_velocity_method=doppler_velocity_method,
                chunked=chunked,
            ).drop_vars("effective_area", errors="ignore")
            if "channel" not in dataset.coords:
                msg = f"{response_file} has no channel coordinate, which read_responses takes the channel from"
                raise ValueError(msg)
            # A file may state one channel for all lines.
            dataset = dataset.assign_coords(channel=dataset.channel.broadcast_like(dataset.line))
            unused_dims = [dim for dim in dataset.dims if dim not in dataset.detector_response.dims]
            datasets.append(dataset.drop_dims(unused_dims))
        for what, value_of in (
            ("instrument defaults", lambda dataset: dataset.attrs.get("instrument_defaults")),
            ("units", lambda dataset: dataset.detector_response.attrs.get("units")),
        ):
            if len(values := {str(value_of(dataset)) for dataset in datasets}) > 1:
                msg = (
                    f"the files are of different {what} ({', '.join(sorted(values))}); read those of one kind together"
                )
                raise ValueError(msg)
        response = xr.concat(
            datasets,
            dim="line",
            data_vars="all",
            coords="different",
            compat="equals",
            join="exact",
            combine_attrs="drop_conflicts",
        )
    # combine_attrs="drop_conflicts" kept what every file agrees on, their dates included.
    add_history(response, locals(), read_responses, sources=datasets, reading=True)
    return response


def save_spectral_response(response: xr.Dataset, response_file: str | Path) -> None:
    """
    Save a wavelength-space response, e.g. from `create_spectral_response` or
    `create_continuum_response`.

    Parameters
    ----------
    response : `xarray.Dataset`
        Dataset containing ``spectral_response``, ``wavelength_grid`` and ``line_wavelength``.
    response_file : `str` or `pathlib.Path`
        As for `save_response`.

    Raises
    ------
    TypeError
        If ``response`` is not an `xarray.Dataset`.
    ValueError
        If one of those variables is missing, or the destination exists or has an unknown
        suffix.
    """
    if not isinstance(response, xr.Dataset):
        msg = "response must be an xarray.Dataset"
        raise TypeError(msg)
    missing = [
        name for name in ("spectral_response", "wavelength_grid", "line_wavelength") if name not in response.variables
    ]
    if missing:
        msg = f"response is missing required variables: {', '.join(missing)}"
        raise ValueError(msg)
    response_file = _check_destination(response_file)
    chunks = {name: 1 for name in ("line", "logT", "pressure", "logD", "abundance") if name in response.dims}
    chunked = response.drop_encoding().chunk(chunks)
    _replace_atomically(response_file, lambda staging: _write_chunked(chunked, staging, "spectral_response"))


def read_spectral_response(response_file: str | Path) -> xr.Dataset:
    """
    Read a wavelength-space response saved by `save_spectral_response`.

    Parameters
    ----------
    response_file : `str` or `pathlib.Path`
        A zarr store (a directory or a ``.zip``) or a netCDF file.

    Returns
    -------
    `xarray.Dataset`
        The response, with ``wavelength_bounds`` as a coordinate. Its arrays are read on
        demand, so the file stays open: ``.load()`` them, or ``.close()`` it when done.

    Raises
    ------
    ValueError
        If the file does not contain ``spectral_response``.
    """
    response_file = Path(response_file)
    response = _open_dataset_file(response_file, decode_coords="all")
    if "spectral_response" not in response:
        msg = f"{response_file} does not contain spectral_response"
        raise ValueError(msg)
    return response
