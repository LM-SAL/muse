from pathlib import Path
from collections.abc import Mapping, Sequence

import dask
import numpy as np
import xarray as xr
from zarr.codecs import BloscCname, BloscCodec, BloscShuffle

import astropy.units as u

from muse.log import logger
from muse.utils.utils import add_history
from muse.variables import DEFAULTS_MUSE

__all__ = ["load_and_concat_responses", "read_response", "save_response"]

_DEFAULT_RESPONSE_CHUNKS = {"line": 1, "doppler_velocity": 20, "logT": 1, "pressure": 1, "abundance": 1}
_LEGACY_RESPONSE_NAMES = {
    "SG_xpixel": "detector_x_pixel",
    "SG_wvl": "detector_wavelength",
    "SG_resp": "detector_response",
    "line_wvl": "line_wavelength",
    "vdop": "doppler_velocity",
}
_NETCDF_SUFFIXES = {".nc", ".ncdf", ".netcdf"}


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
        values from `~muse.variables_schema.InstrumentDefaults.ccd_gain`
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
            gain = u.Quantity(DEFAULTS_MUSE.ccd_gain.sel(channel=r.channel).data)
        except KeyError:
            msg = f"unsupported MUSE SG channel(s) {np.unique(r.channel.values).tolist()}; pass gain explicitly"
            raise ValueError(msg) from None
    gain = gain.to(gain_unit)  # a statement so add_history records the converted value
    gain_dim = "channel" if "channel" in r.dims else "line"
    gain_values = np.broadcast_to(np.atleast_1d(gain.value), r.sizes[gain_dim])
    r = r.assign_coords(gain=(gain_dim, gain_values, {"units": str(gain_unit)}))

    # The current response files carry no wavelength units; warn and assume Angstrom for now.
    # This is intended to become a hard error once every response file carries units.
    for name in ("detector_wavelength", "line_wavelength"):
        if name in r and "units" not in r[name].attrs:
            logger.warning(
                f"Response {name} is missing the 'units' attribute; assuming Angstrom. "
                f"This will raise an error in a future release once response files carry units."
            )
            r[name].attrs.update({"units": str(u.AA)})

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
        logger.info(
            f"Requested {name} extends beyond the response range; trimming to the response grid. "
            f"Run vdem.sel(logT=response.logT, doppler_velocity=response.doppler_velocity, drop=True, method='nearest') to match."
        )
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
                gain = u.Quantity(DEFAULTS_MUSE.ccd_gain.sel(channel=channel).data)
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
        response = xr.concat(datasets, dim="line", data_vars="all", coords="different", compat="equals", join="exact")
    line_channels = [
        channel for dataset, channel in zip(datasets, channels, strict=True) for _ in range(dataset.sizes["line"])
    ]
    return response.assign_coords(channel=("line", line_channels))


def match_responses_and_vdems(
    responses: xr.Dataset,
    vdems: xr.Dataset,
    *,
    logT_method: str = "nearest",
    doppler_velocity_method: str = "linear",
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Resample the response and VDEM datasets onto a common logT and doppler_velocity grid.

    Parameters
    ----------
    responses : `xarray.Dataset`
        Response dataset from `muse.instrument.read_response` or
        `muse.instrument.load_and_concat_responses`.
    vdems : `xarray.Dataset`
        VDEM dataset from `muse.vdem.read_vdem`.
    logT_method : `str`, optional
        Interpolation method for logT, by default "nearest".
        Passed to `muse.instrument.read_response`.
    doppler_velocity_method : `str`, optional
        Interpolation method for doppler_velocity, by default "linear".
        Passed to `muse.instrument.read_response`.

    Returns
    -------
    tuple of `xarray.Dataset`
        The resampled response and VDEM datasets.
    """
    larger_logT_bin = False
    if responses.logT.size > 1 and vdems.logT.size > 1:
        response_logt_gradient = np.abs(np.gradient(responses.logT.values)).max()
        vdem_logt_gradient = np.abs(np.gradient(vdems.logT.values)).min()
        if response_logt_gradient > vdem_logt_gradient:
            logger.warning(
                "Response logT grid is coarser than the VDEM logT grid "
                f"(response max dlogT={response_logt_gradient:.3g}, "
                f"VDEM min dlogT={vdem_logt_gradient:.3g})."
            )
            larger_logT_bin = True
    larger_doppler_velocity_bin = False
    if responses.doppler_velocity.size > 1 and vdems.doppler_velocity.size > 1:
        response_doppler_velocity_gradient = np.abs(np.gradient(responses.doppler_velocity.values)).max()
        vdem_doppler_velocity_gradient = np.abs(np.gradient(vdems.doppler_velocity.values)).min()
        if response_doppler_velocity_gradient > vdem_doppler_velocity_gradient:
            logger.warning(
                "Response doppler_velocity grid is coarser than the VDEM doppler_velocity grid "
                f"(response max ddoppler_velocity={response_doppler_velocity_gradient:.3g}, "
                f"VDEM min ddoppler_velocity={vdem_doppler_velocity_gradient:.3g})."
            )
            larger_doppler_velocity_bin = True

    if logT_method == "nearest":
        if larger_logT_bin:
            vr = np.arange(np.min(vdems.logT), np.max(vdems.logT), np.min(np.diff(responses.logT)))
            vdem_red = vdems.isel(logT=np.arange(np.size(vr)))
            for iii, ii in enumerate(vr):
                data = vdems.vdem.where(vdems.logT<=ii)
                if iii>0:
                    if ii == np.max(vr):
                        data = vdems.vdem.where(vdems.logT>vr[iii-1]).sum(dim="logT")
                    else:
                        data = data.where(data.logT>vr[iii-1]).sum(dim="logT")
                else: 
                    data = data.sum(dim='logT')
                vdem_red.vdem.loc[{"logT": vdem_red.logT.isel(logT=iii).data}] = data
            vdem_red.coords["logT"] = vr
        else:
            shared_logT = vdems.logT.where(
                (vdems.logT >= responses.logT.min())
                & (vdems.logT <= responses.logT.max()),
                drop=True,
            )
            if shared_logT.size < vdems.logT.size:
                logger.warning(
                    "High logT values are present in responses.logT; "
                    "these will be dropped to match the VDEM logT axis."
                )

            vdems = vdems.sel(logT=shared_logT, method="nearest", drop=True)

            # Prefer exact common logT values; if none match exactly, fall back to overlap + nearest.
            responses_logt = np.asarray(responses.logT.values)
            vdem_logt = np.asarray(vdems.logT.values)
            exact_mask = np.any(
                np.isclose(vdem_logt[:, np.newaxis], responses_logt[np.newaxis, :], rtol=0, atol=1e-12),
                axis=1,
            )
            shared_logT = vdems.logT.where(exact_mask, drop=True)
            if shared_logT.size != vdems.logT.size:
                msg = "logT axes have no overlap between response and VDEM"
                raise ValueError(msg)

            responses = responses.sel(logT=shared_logT, method="nearest", drop=True)
            vdems = vdems.sel(logT=shared_logT, method="nearest", drop=True)

    else:
        shared_logT = vdems.logT.where((vdems.logT >= responses.logT.min()) & (vdems.logT <= responses.logT.max()), drop=True)
        if shared_logT.size == 0:
            msg = "logT axes have no overlap between response and VDEM"
            raise ValueError(msg)

        responses = responses.interp(logT=shared_logT, method=logT_method)

    if doppler_velocity_method == "nearest":      
        if larger_doppler_velocity_bin:
            vr = np.arange(np.min(vdems.doppler_velocity), np.max(vdems.doppler_velocity), np.min(np.diff(responses.doppler_velocity)))
            vdem_red = vdems.isel(doppler_velocity=np.arange(np.size(vr)))
            for iii, ii in enumerate(vr):
                data = vdems.vdem.where(vdems.doppler_velocity<=ii)
                if iii>0:
                    if ii == np.max(vr):
                        data = vdems.vdem.where(vdems.doppler_velocity>vr[iii-1]).sum(dim="doppler_velocity")
                    else:
                        data = data.where(data.doppler_velocity>vr[iii-1]).sum(dim="doppler_velocity")
                else: 
                    data = data.sum(dim='doppler_velocity')
                vdem_red.vdem.loc[{"doppler_velocity": vdem_red.doppler_velocity.isel(doppler_velocity=iii).data}] = data
            vdem_red.coords["doppler_velocity"] = vr
            vdems = vdem_red.copy(deep=True)
        else: 
            shared_doppler_velocity = vdems.doppler_velocity.where(
                (vdems.doppler_velocity >= responses.doppler_velocity.min())
                & (vdems.doppler_velocity <= responses.doppler_velocity.max()),
                drop=True,
            )
            if shared_doppler_velocity.size == 0:
                msg = "doppler_velocity axes have no overlap between response and VDEM"
                raise ValueError(msg)
            if shared_doppler_velocity.size < vdems.doppler_velocity.size:
                logger.warning(
                    "High speed values are present in responses.doppler_velocity; "
                    "these will be dropped to match the VDEM doppler_velocity axis."
                )
            vdems = vdems.sel(doppler_velocity=shared_doppler_velocity, method="nearest", drop=True)

            # Prefer exact common logT values; if none match exactly, fall back to overlap + nearest.
            responses_doppler_velocity = np.asarray(responses.doppler_velocity.values)
            vdem_doppler_velocity = np.asarray(vdems.doppler_velocity.values)
            exact_mask = np.any(
                np.isclose(vdem_doppler_velocity[:, np.newaxis], responses_doppler_velocity[np.newaxis, :], rtol=0, atol=1e-12),
                axis=1,
            )
            shared_doppler_velocity = vdems.doppler_velocity.where(exact_mask, drop=True)
            if shared_doppler_velocity.size != vdems.doppler_velocity.size:
                msg = "doppler_velocity axes have no overlap between response and VDEM"
                raise ValueError(msg)

            responses = responses.sel(doppler_velocity=shared_doppler_velocity, method="nearest", drop=True)
            vdems = vdems.sel(doppler_velocity=shared_doppler_velocity, method="nearest", drop=True)
 
    else:
        responses = responses.interp(doppler_velocity=vdems.doppler_velocity, method=doppler_velocity_method)

    return responses, vdems