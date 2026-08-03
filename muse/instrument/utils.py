from pathlib import Path
from collections.abc import Mapping, Sequence

import dask
import numpy as np
import xarray as xr
from zarr.codecs import BloscCname, BloscCodec, BloscShuffle

import astropy.units as u

from muse.log import logger
from muse.utils.utils import add_history, coord_as_unit, update_attrs
from muse.variables import DEFAULTS_MUSE
from muse.variables_schema import FrozenDict

__all__ = ["align_response_and_vdem", "load_and_concat_responses", "read_response", "save_response"]

_DEFAULT_RESPONSE_CHUNKS = {"line": 1, "doppler_velocity": 20, "logT": 1, "pressure": 1, "abundance": 1}
_LEGACY_RESPONSE_NAMES = {
    "SG_xpixel": "detector_x_pixel",
    "SG_wvl": "detector_wavelength",
    "SG_resp": "detector_response",
    "line_wvl": "line_wavelength",
    "vdop": "doppler_velocity",
}
_NETCDF_SUFFIXES = {".nc", ".ncdf", ".netcdf"}


def _channel_as_angstrom(channel: u.Quantity, allowed_channels: xr.DataArray | None, detector: str) -> int | float:
    if not channel.isscalar:
        msg = "channel must be a scalar wavelength"
        raise ValueError(msg)
    channel_value = channel.to_value(u.AA)
    if not np.isfinite(channel_value) or channel_value <= 0:
        msg = "channel must be a finite, positive wavelength"
        raise ValueError(msg)
    if allowed_channels is None:
        return channel_value
    allowed = np.asarray(allowed_channels)
    matches = np.isclose(allowed, channel_value, rtol=0, atol=1e-12)
    if not matches.any():
        msg = f"unsupported MUSE {detector.upper()} channel {channel}"
        raise ValueError(msg)
    return allowed[matches][0].item()


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
        Destination ending in ``.zarr``, ``.nc``, ``.ncdf``, or ``.netcdf``.
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
    if "detector_response" not in response.data_vars:
        msg = "response must contain detector_response"
        raise ValueError(msg)
    response_file = Path(response_file)
    if response_file.exists():
        msg = f"Refusing to overwrite existing response: {response_file}"
        raise ValueError(msg)
    suffix = response_file.suffix.lower()
    if suffix != ".zarr" and suffix not in _NETCDF_SUFFIXES:
        msg = "response_file must end in .zarr, .nc, .ncdf, or .netcdf"
        raise ValueError(msg)

    chunked = response.drop_encoding().chunk(_response_chunks(response, chunks))
    if suffix == ".zarr":
        compressor = BloscCodec(cname=BloscCname.zstd, clevel=3, shuffle=BloscShuffle.bitshuffle)
        chunked.to_zarr(
            response_file,
            mode="w",
            zarr_format=3,
            consolidated=False,
            encoding={"detector_response": {"compressors": (compressor,)}},
        )
        return
    chunksizes = tuple(axis[0] for axis in chunked.detector_response.chunks)
    chunked.to_netcdf(
        response_file,
        encoding={
            "detector_response": {
                "chunksizes": chunksizes,
                "zlib": True,
                "complevel": 1,
                "shuffle": True,
            }
        },
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


def _open_response_file(response_file: Path, *, chunked: bool = False) -> xr.Dataset:
    kwargs = {"chunks": {}} if chunked else {}
    # One call covers both zarr layouts, v3 (zarr.json) and v2 (.zgroup); consolidated=False
    # also stops the missing-consolidated-metadata fallback warning from firing.
    if response_file.is_dir():
        return xr.open_zarr(response_file, consolidated=False, **kwargs)
    return xr.open_dataset(response_file, **kwargs)


def _canonicalize_response_names(response: xr.Dataset) -> xr.Dataset:
    for old_name, new_name in _LEGACY_RESPONSE_NAMES.items():
        if old_name not in response.variables and old_name not in response.dims:
            continue
        response = response.drop_vars(new_name, errors="ignore").rename({old_name: new_name})
    return response


@u.quantity_input(gain=u.electron / u.DN)
def read_response(
    response_file: str | Path,
    *,
    logT: xr.DataArray | None = None,
    doppler_velocity: xr.DataArray | None = None,
    slit: xr.DataArray | None = None,
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
    logT : `xarray.DataArray`, optional
        Temperature axis to (re)sample onto.
    doppler_velocity : `xarray.DataArray`, optional
        Velocity axis to (re)sample onto.
    slit : `xarray.DataArray`, optional
        Number of slits array of integers.
    logT_method : `str`, optional
        Interpolation method for logT, by default "nearest".
    doppler_velocity_method : `str`, optional
        Interpolation method for doppler_velocity, by default "nearest".
    gain : `astropy.units.Quantity`, optional
        Camera gain, convertible to electron/DN. If `None`, use the per-channel
        values from `~muse.variables_schema.InstrumentDefaults.ccd_gain_sg`
        selected by the response's ``channel`` coordinate.
    chunked : `bool`, optional
        When `True`, open the file dask-backed using its on-disk chunking, so
        the response stays lazy through resampling and downstream synthesis and
        peak memory stays bounded by the chunks. By default `False` (eager).

    Returns
    -------
    `xarray.Dataset`
        The response dataset using the canonical ``detector_response``,
        ``detector_wavelength``, ``detector_x_pixel``, and ``line_wavelength``
        names. Existing files that use the legacy MUSE names are normalized on
        load.

    Raises
    ------
    ValueError
        If the ``logT``/``doppler_velocity`` axes are malformed, or the loaded dataset is
        missing the ``detector_response`` variable or the ``logT``/``doppler_velocity`` coordinates.
    """
    response_file = Path(response_file)

    for name, axis in (("logT", logT), ("doppler_velocity", doppler_velocity)):
        if axis is None:
            continue
        if len(axis.data) == 0:
            msg = f"{name} array must not be empty"
            raise ValueError(msg)
        if not np.all(np.isfinite(axis.data)):
            msg = f"{name} must contain only finite values"
            raise ValueError(msg)

    r = _canonicalize_response_names(_open_response_file(response_file, chunked=chunked))

    if "detector_response" not in r.data_vars:
        msg = "Response dataset must contain 'detector_response' variable"
        raise ValueError(msg)
    for name in ("logT", "doppler_velocity"):
        if name not in r.coords and name not in r.dims:
            msg = f"Response must have {name} coordinate"
            raise ValueError(msg)

    r = _resample_axis(r, "logT", logT, logT_method)
    r = _resample_axis(r, "doppler_velocity", doppler_velocity, doppler_velocity_method)

    if slit is not None:
        r = r.sel(slit=np.arange(slit.max() + 1), drop=True, method="nearest")

    if "channel" not in r.dims and "line" not in r.dims:
        r = r.expand_dims("line")

    if "line_wavelength" not in r:
        fallback = r.attrs.get("LINE_WVL", r.attrs.get("MAIN_LINE_WVL"))
        if fallback is None:
            hint = ""
            if "channel" in r.coords:
                hint = f"; channel is {r.channel.values.tolist()}, a band label, not a line wavelength"
            msg = f"Response must define line_wavelength or LINE_WVL/MAIN_LINE_WVL metadata{hint}"
            raise ValueError(msg)
        r = r.assign_coords(line_wavelength=fallback)

    gain_unit = u.electron / u.DN
    if gain is None:
        if "channel" not in r.coords:
            msg = "response has no channel coordinate to select the per-channel default gain; pass gain explicitly"
            raise ValueError(msg)
        try:
            gain = u.Quantity(DEFAULTS_MUSE.ccd_gain_sg.sel(channel=r.channel).data)
        except KeyError:
            msg = f"unsupported MUSE SG channel(s) {np.unique(r.channel.values).tolist()}; pass gain explicitly"
            raise ValueError(msg) from None
    gain = gain.to(gain_unit)  # a statement so add_history records the converted value
    gain_dim = "channel" if "channel" in r.dims else "line"
    gain_values = np.broadcast_to(np.atleast_1d(gain.value), r.sizes[gain_dim])
    r = r.assign_coords(gain=(gain_dim, gain_values, {"units": str(gain_unit)}))

    # Current response files do not carry units on every coordinate; warn and
    # assume the established MUSE units until the files are migrated.
    assumed_units = {
        "logT": u.dex(u.K),
        "doppler_velocity": u.km / u.s,
        "detector_wavelength": u.AA,
        "line_wavelength": u.AA,
    }
    for name, unit in assumed_units.items():
        if name in r and "units" not in r[name].attrs:
            logger.warning(
                f"Response {name} is missing the 'units' attribute; assuming {unit}. "
                f"This will raise an error in a future release once response files carry units."
            )
            r[name].attrs.update({"units": str(unit)})

    add_history(r, locals(), read_response)
    return r


def _resample_axis(r: xr.Dataset, name: str, axis: xr.DataArray | None, method: str) -> xr.Dataset:
    """
    Select or interpolate the response onto ``axis`` along ``name`` (``logT`` or
    ``doppler_velocity``).

    Out-of-range requested points are trimmed to the response grid first. The
    ``"nearest"`` method selects existing samples; any other method interpolates and
    clamps the result to be finite and non-negative.
    """
    if axis is None:
        return r
    in_range = (axis >= r[name].min()) & (axis <= r[name].max())
    if not bool(in_range.all()):
        logger.info(f"Requested {name} extends beyond the response range; trimming to the response grid")
        axis = axis.where(in_range, drop=True)
        if axis.size == 0:
            msg = f"Requested {name} axis has no overlap with the response range"
            raise ValueError(msg)
    if method == "nearest":
        r = r.sel({name: axis}, drop=True, method="nearest")
    else:
        r = r.interp({name: axis}, method=method)
    # Clamp on every path so nearest and interpolated responses behave consistently.
    r["detector_response"] = r.detector_response.fillna(0).clip(min=0).assign_attrs(r.detector_response.attrs)
    return r.assign_coords({name: axis})


def align_response_and_vdem(
    response: xr.Dataset,
    vdem: xr.Dataset,
    *,
    coord_methods: Mapping[str, tuple] = FrozenDict(
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
    coord_methods : `~collections.abc.Mapping`, optional
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

    for name, (method, unit) in coord_methods.items():
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


def load_and_concat_responses(
    response_directory: str | Path,
    response_files: Sequence[str],
    *,
    channels: Sequence[int],
    logT: xr.DataArray | None = None,
    doppler_velocity: xr.DataArray | None = None,
    slit: xr.DataArray | None = None,
    logT_method: str = "nearest",
    doppler_velocity_method: str = "linear",
    chunked: bool = False,
) -> xr.Dataset:
    """
    Load multiple response functions and concatenate them along ``line``.

    Parameters
    ----------
    response_directory : `str` or `pathlib.Path`
        Directory containing the response files.
    response_files : `Sequence` of `str`
        Filenames of response functions to load, in order.
    channels : `Sequence` of `int`
        One MUSE SG channel per response file. Selects the per-channel default
        gain and is repeated for every line when a file contains multiple lines.
    logT : `xarray.DataArray`, optional
        Temperature axis to (re)sample onto. Passed to `muse.instrument.read_response`.
    doppler_velocity : `xarray.DataArray`, optional
        Velocity axis to (re)sample onto. Passed to `muse.instrument.read_response`.
    slit : `xarray.DataArray`, optional
        Number of slits array of integers. Passed to `muse.instrument.read_response`.
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
        Concatenated response dataset with assigned channel coordinates.

    Raises
    ------
    ValueError
        If the length of ``channels`` does not match ``response_files`` or a
        channel is unsupported.
    """
    if len(channels) != len(response_files):
        msg = f"channels ({len(channels)}) must match the number of response_files ({len(response_files)})"
        raise ValueError(msg)

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        datasets = []
        for filename, channel in zip(response_files, channels, strict=True):
            try:
                gain = u.Quantity(DEFAULTS_MUSE.ccd_gain_sg.sel(channel=channel).data)
            except KeyError:
                msg = f"unsupported MUSE SG channel {channel}"
                raise ValueError(msg) from None
            dataset = read_response(
                Path(response_directory) / filename,
                logT=logT,
                doppler_velocity=doppler_velocity,
                slit=slit,
                logT_method=logT_method,
                doppler_velocity_method=doppler_velocity_method,
                gain=gain,
                chunked=chunked,
            ).drop_vars("effective_area", errors="ignore")
            unused_dims = [dim for dim in dataset.dims if dim not in dataset.detector_response.dims]
            datasets.append(dataset.drop_dims(unused_dims))
        response = xr.concat(
            datasets,
            dim="line",
            data_vars="all",
            coords="different",
            compat="equals",
            join="exact",
            combine_attrs="drop_conflicts",
        )
    line_channels = [
        channel for dataset, channel in zip(datasets, channels, strict=True) for _ in range(dataset.sizes["line"])
    ]
    result = response.assign_coords(channel=("line", line_channels)).drop_attrs(deep=False)
    update_attrs(result, response)
    add_history(result, locals(), load_and_concat_responses, sources=datasets)
    return result
