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
from muse.utils.utils import _to_zarr, add_history, coord_as_unit, require_unit, update_attrs
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
        Destination ending in ``.zarr``, ``.zarr.zip``, ``.nc``, ``.ncdf``, or ``.netcdf``.
        A ``.zarr.zip`` destination is one file: the zarr store packed into a zip.
        Existing destinations are not overwritten.
    chunks : mapping of `str` to `int`, optional
        Per-dimension overrides for the benchmark-backed defaults:
        ``line=1``, ``doppler_velocity=20``, ``logT=1``, ``pressure=1``,
        ``abundance=1``, and complete ``slit``/``detector_x_pixel`` planes.
        Unspecified dimensions retain the defaults. Values larger than a
        dimension use the full dimension.

    Notes
    -----
    Zarr responses use Blosc/Zstd level 3 with bit-shuffle. NetCDF responses
    use zlib level 1 with shuffle. These defaults were selected independently
    using the full 171 response.
    """
    if not isinstance(response, xr.Dataset):
        msg = "response must be an xarray.Dataset"
        raise TypeError(msg)
    _validate_response(response, "response")
    response_file = _check_destination(response_file)
    chunked = response.drop_encoding().chunk(_response_chunks(response, chunks))
    _write_chunked(chunked, response_file, "detector_response")


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
        msg = f"{argument} must end in .zarr, .zarr.zip, .nc, .ncdf, or .netcdf"
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
    Reject a response that is not in the layout `save_response` writes.

    Parameters
    ----------
    response : `xarray.Dataset`
        The response to check. It must hold ``detector_response``, the ``logT`` and
        ``doppler_velocity`` coordinates in temperature and velocity units, a ``line`` or
        ``channel`` dimension, ``line_wavelength`` in a wavelength unit and, if present,
        ``detector_wavelength`` in one too.
    label : `str`
        Name of the dataset in error messages.

    Raises
    ------
    ValueError
        With the first failing requirement; a file in the legacy MUSE layout is named as such.
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
    # open_zarr chunks with dask by default; chunks=None gives the file's lazily indexed arrays
    # without dask, as open_dataset does for netCDF.
    kwargs = {"chunks": {} if chunked else None, "decode_coords": decode_coords}
    if response_file.suffix == ".zip":
        # Reads lazily like a directory. xarray closes only the stores it opens itself, and
        # zarr reopens the archive on demand after a close.
        store = ZipStore(response_file, mode="r")
        response = xr.open_zarr(store, consolidated=False, **kwargs)
        response.set_close(store.close)
    # One call covers both zarr layouts, v3 (zarr.json) and v2 (.zgroup); consolidated=False
    # also stops the missing-consolidated-metadata fallback warning from firing.
    elif response_file.is_dir():
        response = xr.open_zarr(response_file, consolidated=False, **kwargs)
    else:
        response = xr.open_dataset(response_file, **kwargs)
    return response


@u.quantity_input(gain=u.electron / u.DN)
def read_response(
    response_file: str | Path,
    *,
    logT: u.Quantity | xr.DataArray | None = None,
    doppler_velocity: u.Quantity | xr.DataArray | None = None,
    number_of_slits: int | None = None,
    logT_method: str = "nearest",
    doppler_velocity_method: str = "nearest",
    gain: u.Quantity | None = None,
    chunked: bool = False,
) -> xr.Dataset:
    """
    Reads a response function into an `xarray.Dataset` interpolating if needed in
    doppler_velocity, and logT.

    Parameters
    ----------
    response_file : `str` | `pathlib.Path`
        Response function in Xarray readable format (netCDF file or Zarr store).
    logT : `astropy.units.Quantity` or `xarray.DataArray`, optional
        Temperature axis to (re)sample onto, in a unit convertible to the response's,
        e.g. ``np.arange(5.0, 7.0, 0.1) * u.dex(u.K)`` or the ``logT`` coordinate of a
        VDEM. A `xarray.DataArray` must carry a ``units`` attribute.
    doppler_velocity : `astropy.units.Quantity` or `xarray.DataArray`, optional
        Velocity axis to (re)sample onto, in the same way as ``logT``.
    number_of_slits : `int`, optional
        Keep the first ``number_of_slits`` slits of the response, relabelled from 0,
        e.g. to match a raster from `muse.transforms.reshape_x_to_slit_step`. By
        default every slit is kept.
    logT_method : `str`, optional
        Interpolation method for logT, by default "nearest".
    doppler_velocity_method : `str`, optional
        Interpolation method for doppler_velocity, by default "nearest".
    gain : `astropy.units.Quantity`, optional
        Camera gain, convertible to electron/DN. By default the ``gain`` attribute of
        the file, which the generators write.
    chunked : `bool`, optional
        When `True`, open the file dask-backed using its on-disk chunking, so
        the response stays lazy through resampling and downstream synthesis and
        peak memory stays bounded by the chunks. By default `False` (eager).
        A zipped zarr store (``.zip``) behaves like a zarr directory.

    Returns
    -------
    `xarray.Dataset`
        The response dataset, with a ``gain`` coordinate. Only the layout `save_response`
        writes is read: ``detector_response`` and ``line_wavelength``, a ``line`` or ``channel``
        dimension, and units on ``logT``, ``doppler_velocity`` and the wavelengths.

    Raises
    ------
    ValueError
        If the requested ``logT`` or ``doppler_velocity`` axis has no units, is empty, not
        finite, in units that do not convert to the response's or wholly outside its range;
        if the file is not
        in the layout `save_response` writes; or if ``gain`` is not given and the file states
        none, in a ``gain`` coordinate or attribute.
    """
    response_file = Path(response_file)

    # Rebound under their own names, so that the history names the axes instead of listing them.
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

    if "channel" not in response.dims and "line" not in response.dims:
        response = response.expand_dims("line")

    gain_unit = u.electron / u.DN
    if gain is None and "gain" in response.coords:
        # A file that holds a gain per line, such as a saved set of responses, keeps it.
        response = response.assign_coords(gain=coord_as_unit(response, "gain", gain_unit, "gain coordinate"))
    else:
        if gain is None:
            if "gain" not in response.attrs:
                msg = f"{response_file} states no gain, in a gain coordinate or attribute; pass gain"
                raise ValueError(msg)
            gain = u.Quantity(response.attrs["gain"])
        else:
            # The given gain replaces that of the file; add_history records a scalar one.
            response.attrs.pop("gain", None)
        gain = gain.to(gain_unit)  # a statement so add_history records the converted value
        gain_dim = "channel" if "channel" in response.dims else "line"
        gain_values = np.broadcast_to(np.atleast_1d(gain.value), response.sizes[gain_dim])
        response = response.assign_coords(gain=(gain_dim, gain_values, {"units": str(gain_unit)}))

    add_history(response, locals(), read_response, reading=True)
    return response


def _requested_axis(axis: u.Quantity | xr.DataArray | None, name: str) -> xr.DataArray | None:
    """
    ``axis`` as an `xarray.DataArray` along ``name`` with a ``units`` attribute.

    Raises
    ------
    ValueError
        If ``axis`` is neither a Quantity nor a DataArray that carries units.
    """
    if axis is None:
        return None
    if isinstance(axis, u.Quantity):
        return xr.DataArray(np.atleast_1d(axis.value), dims=name, attrs={"units": str(axis.unit)})
    if not isinstance(axis, xr.DataArray) or "units" not in axis.attrs:
        msg = f"{name} must be an astropy Quantity or an xarray.DataArray with a units attribute"
        raise ValueError(msg)
    return axis


def _resample_axis(response: xr.Dataset, name: str, axis: xr.DataArray | None, method: str) -> xr.Dataset:
    """
    Select or interpolate the response onto ``axis`` along ``name`` (``logT`` or
    ``doppler_velocity``).

    Out-of-range requested points are trimmed to the response grid first. The
    ``"nearest"`` method selects existing samples; any other method interpolates and
    clamps the result to be finite and non-negative.
    """
    if axis is None:
        return response
    # The requested axis is converted to the response's units, so the range check, the
    # resampling and the result share one unit.
    response_unit = require_unit(response, name, f"response.{name}", coord_only=True)
    axis_attrs = dict(response[name].attrs)
    try:
        converted = u.Unit(axis.attrs["units"]).to(response_unit, np.asarray(axis.values, dtype=float))
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        msg = f"{name} axis units {axis.attrs['units']!r} are not convertible to the response's {response_unit}"
        raise ValueError(msg) from exc
    axis = axis.copy(data=converted)
    in_range = (axis >= response[name].min()) & (axis <= response[name].max())
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
            # Every requested point already sits on the response grid, where interpolation
            # reduces to selection; selecting also reads only the needed slabs of a
            # dask-backed response instead of the full axis.
            response = response.sel({name: axis}, drop=True, method="nearest")
        else:
            response = response.interp({name: axis}, method=method)
    # Clamp on every path so nearest and interpolated responses behave consistently.
    response["detector_response"] = (
        response.detector_response.fillna(0).clip(min=0).assign_attrs(response.detector_response.attrs)
    )
    if response.detector_response.chunks is not None:
        # Response files store tiny chunks along logT; point selection above keeps one
        # dask chunk per selected point, which shatters downstream synthesis graphs.
        # Consolidate the resampled axis into a single chunk.
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
    Resample response onto the overlapping VDEM temperature and velocity grids.

    Parameters
    ----------
    response : `xarray.Dataset`
        Response dataset containing ``detector_response``.
    vdem : `xarray.Dataset`
        VDEM dataset containing ``vdem``.
    coordinate_methods : `~collections.abc.Mapping`, optional
        Maps each coordinate name to align to a ``(resampling method, unit)``
        tuple, by default ``{"logT": ("nearest", u.dex(u.K)),
        "doppler_velocity": ("linear", u.km / u.s)}``. Coordinates not listed
        are left unaligned.

    Returns
    -------
    response, vdem : `tuple` of `xarray.Dataset`
        New datasets with identical ``logT`` and ``doppler_velocity`` coordinates.

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
    logT: u.Quantity | xr.DataArray | None = None,
    doppler_velocity: u.Quantity | xr.DataArray | None = None,
    number_of_slits: int | None = None,
    logT_method: str = "nearest",
    doppler_velocity_method: str = "linear",
    chunked: bool = False,
) -> xr.Dataset:
    """
    Load several response functions and concatenate them along ``line``.

    Each file gives its channel in its ``channel`` coordinate and its camera gain in
    its ``gain`` coordinate or attribute, as the products of the generators do.

    Parameters
    ----------
    response_files : `Sequence` of `str` or `pathlib.Path`
        Response files, in order.
    logT : `astropy.units.Quantity` or `xarray.DataArray`, optional
        Temperature axis to (re)sample onto, with units. Passed to
        `muse.instrument.read_response`.
    doppler_velocity : `astropy.units.Quantity` or `xarray.DataArray`, optional
        Velocity axis to (re)sample onto, with units. Passed to
        `muse.instrument.read_response`.
    number_of_slits : `int`, optional
        How many slits to keep. Passed to `muse.instrument.read_response`.
    logT_method : `str`, optional
        Interpolation method for logT, by default "nearest".
        Passed to `muse.instrument.read_response`.
    doppler_velocity_method : `str`, optional
        Interpolation method for doppler_velocity, by default "linear".
        Passed to `muse.instrument.read_response`.
    chunked : `bool`, optional
        When `True`, load each response dask-backed so the concatenated
        response stays lazy and synthesis peak memory stays bounded by the
        chunks. Passed to `muse.instrument.read_response`. By default `False`.

    Returns
    -------
    `xarray.Dataset`
        Concatenated response dataset with the ``channel`` coordinate of the files
        along ``line``.

    Raises
    ------
    ValueError
        If no file is given, a file has no ``channel`` coordinate or no gain, the files
        are of different instrument defaults or units, or ``logT`` or
        ``doppler_velocity`` has no units.
    """
    if not response_files:
        msg = "pass at least one response file"
        raise ValueError(msg)
    # Rebound under their own names, so that the history names the axes instead of listing them.
    logT = _requested_axis(logT, "logT")
    doppler_velocity = _requested_axis(doppler_velocity, "doppler_velocity")

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
            # one channel per line, also from a file that states one for all
            dataset = dataset.assign_coords(channel=dataset.channel.broadcast_like(dataset.line))
            unused_dims = [dim for dim in dataset.dims if dim not in dataset.detector_response.dims]
            datasets.append(dataset.drop_dims(unused_dims))
    for what, value_of in (
        ("instrument defaults", lambda dataset: dataset.attrs.get("instrument_defaults")),
        ("units", lambda dataset: dataset.detector_response.attrs.get("units")),
    ):
        if len(values := {str(value_of(dataset)) for dataset in datasets}) > 1:
            msg = f"the files are of different {what} ({', '.join(sorted(values))}); read those of one kind together"
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
        Destination ending in ``.zarr``, ``.zarr.zip``, ``.nc``, ``.ncdf``, or ``.netcdf``. Existing
        destinations are not overwritten.

    Raises
    ------
    TypeError
        If ``response`` is not an `xarray.Dataset`.
    ValueError
        If one of those variables is missing, or the destination exists or does not end in
        one of those suffixes.
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
    _write_chunked(response.drop_encoding().chunk(chunks), response_file, "spectral_response")


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
        The response, with ``wavelength_bounds`` as a coordinate. The arrays are read from
        the file on demand, so the file stays open: call ``.load()`` to bring them into
        memory, or ``.close()`` when done.

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
