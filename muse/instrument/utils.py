from pathlib import Path
from collections.abc import Sequence

import dask
import numpy as np
import xarray as xr

import astropy.units as u

from muse.log import logger
from muse.utils.documentation import format_docstring
from muse.utils.utils import add_history
from muse.variables import DEFAULTS_MUSE

__all__ = ["load_and_concat_responses", "read_response"]


@format_docstring("DEFAULTS_MUSE", gain="ccd_gain")
@u.quantity_input(gain=u.electron / u.DN)
def read_response(
    response_file: str | Path,
    *,
    logT: xr.DataArray | None = None,
    vdop: xr.DataArray | None = None,
    slit: xr.DataArray | None = None,
    logT_method: str = "nearest",
    vdop_method: str = "nearest",
    gain: u.Quantity = DEFAULTS_MUSE.ccd_gain,
) -> xr.Dataset:
    """
    Reads a response function into an `xarray.Dataset` interpolating if needed in vdop,
    and logT.

    Parameters
    ----------
    response_file : `str` | `pathlib.Path`
        Response function in Xarray readable format (netCDF file or Zarr store).
    logT : `xarray.DataArray`, optional
        Temperature axis to (re)sample onto.
    vdop : `xarray.DataArray`, optional
        Velocity axis to (re)sample onto.
    slit : `xarray.DataArray`, optional
        Number of slits array of integers.
    logT_method : `str`, optional
        Interpolation method for logT, by default "nearest".
    vdop_method : `str`, optional
        Interpolation method for vdop, by default "nearest".
    gain : `astropy.units.Quantity`, optional
        Camera gain, convertible to electron/DN, by default {gain}.

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
        If the ``logT``/``vdop`` axes are malformed, or the loaded dataset is
        missing the ``detector_response`` variable or the ``logT``/``vdop`` coordinates.
    """
    response_file = Path(response_file)

    for name, axis in (("logT", logT), ("vdop", vdop)):
        if axis is None:
            continue
        if len(axis.data) == 0:
            msg = f"{name} array must not be empty"
            raise ValueError(msg)
        if not np.all(np.isfinite(axis.data)):
            msg = f"{name} must contain only finite values"
            raise ValueError(msg)

    if response_file.is_dir() and (response_file / "zarr.json").exists():
        r = xr.open_zarr(response_file, consolidated=False)
    elif response_file.is_dir() and (response_file / ".zgroup").exists():
        r = xr.open_zarr(response_file)
    else:
        r = xr.open_dataset(response_file)

    legacy_names = {
        "SG_xpixel": "detector_x_pixel",
        "SG_wvl": "detector_wavelength",
        "SG_resp": "detector_response",
        "line_wvl": "line_wavelength",
    }
    for old_name, new_name in legacy_names.items():
        if old_name not in r.variables and old_name not in r.dims:
            continue
        r = r.drop_vars(new_name, errors="ignore").rename({old_name: new_name})

    if "detector_response" not in r.data_vars:
        msg = "Response dataset must contain 'detector_response' variable"
        raise ValueError(msg)
    for name in ("logT", "vdop"):
        if name not in r.coords and name not in r.dims:
            msg = f"Response must have {name} coordinate"
            raise ValueError(msg)

    r = _resample_axis(r, "logT", logT, logT_method)
    r = _resample_axis(r, "vdop", vdop, vdop_method)

    if slit is not None:
        r = r.sel(slit=np.arange(slit.max() + 1), drop=True, method="nearest")

    if "channel" not in r.dims and "line" not in r.dims:
        r = r.expand_dims("line")

    if "line_wavelength" not in r:
        fallback = r.attrs.get("LINE_WVL", r.attrs.get("MAIN_LINE_WVL"))
        if fallback is not None:
            r = r.assign_coords(line_wavelength=fallback)
        elif "channel" in r.coords:
            r = r.assign_coords(line_wavelength=r.channel)
        else:
            msg = "Response must define line_wavelength or LINE_WVL/MAIN_LINE_WVL metadata"
            raise ValueError(msg)

    gain_unit = u.electron / u.DN
    gain = gain.to(gain_unit)
    gain_dim = "channel" if "channel" in r.dims else "line"
    gain_values = np.broadcast_to(np.atleast_1d(gain.value), r.sizes[gain_dim])
    r = r.assign_coords(gain=(gain_dim, gain_values))
    r.gain.attrs["units"] = str(gain_unit)

    # The current response files carry no wavelength units; warn and assume Angstrom for now.
    _require_wavelength_units(r, "detector_wavelength")
    _require_wavelength_units(r, "line_wavelength")

    add_history(r, locals(), read_response)
    return r


def _require_wavelength_units(r: xr.Dataset, name: str) -> None:
    """
    Ensure ``r[name]`` carries wavelength units, assuming Angstrom when missing.

    Older response files store no units on ``SG_wvl``/``line_wvl``. For now a missing
    ``units`` attribute logs a warning and Angstrom is assumed; this is intended to
    become a hard error once all response files carry units.
    """
    if name not in r:
        return
    if "units" not in r[name].attrs:
        logger.warning(
            f"Response {name} is missing the 'units' attribute; assuming Angstrom. "
            f"This will raise an error in a future release once response files carry units."
        )
        r[name].attrs.update({"units": str(u.AA)})


def _resample_axis(r: xr.Dataset, name: str, axis: xr.DataArray | None, method: str) -> xr.Dataset:
    """
    Select or interpolate the response onto ``axis`` along ``name`` (``logT`` or
    ``vdop``).

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
            f"Run vdem.sel(logT=response.logT, vdop=response.vdop, drop=True, method='nearest') to match."
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
    r["detector_response"] = r.detector_response.fillna(0).clip(min=0)
    return r.assign_coords({name: axis})


def load_and_concat_responses(
    response_directory: str | Path,
    response_files: Sequence[str],
    *,
    channels: Sequence[int],
    logT: xr.DataArray | None = None,
    vdop: xr.DataArray | None = None,
    slit: xr.DataArray | None = None,
    logT_method: str = "nearest",
    vdop_method: str = "linear",
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
        One channel value per response file. The value is repeated for every
        line when a file contains multiple lines.
    logT : `xarray.DataArray`, optional
        Temperature axis to (re)sample onto. Passed to `muse.instrument.utils.read_response`.
    vdop : `xarray.DataArray`, optional
        Velocity axis to (re)sample onto. Passed to `muse.instrument.utils.read_response`.
    slit : `xarray.DataArray`, optional
        Number of slits array of integers. Passed to `muse.instrument.utils.read_response`.
    logT_method : `str`, optional
        Interpolation method for logT, by default "nearest".
        Passed to `muse.instrument.utils.read_response`.
    vdop_method : `str`, optional
        Interpolation method for vdop, by default "linear".
        Passed to `muse.instrument.utils.read_response`.

    Returns
    -------
    `xarray.Dataset`
        Concatenated response dataset with assigned channel coordinates.

    Raises
    ------
    ValueError
        If the length of ``channels`` does not match ``response_files``.
    """
    if len(channels) != len(response_files):
        msg = f"channels ({len(channels)}) must match the number of response_files ({len(response_files)})"
        raise ValueError(msg)

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        datasets = []
        for filename in response_files:
            dataset = read_response(
                Path(response_directory) / filename,
                logT=logT,
                vdop=vdop,
                slit=slit,
                logT_method=logT_method,
                vdop_method=vdop_method,
            ).drop_vars("effective_area", errors="ignore")
            unused_dims = [dim for dim in dataset.dims if dim not in dataset.detector_response.dims]
            datasets.append(dataset.drop_dims(unused_dims))
        response = xr.concat(datasets, dim="line", coords="different", compat="equals")
    line_channels = [
        channel for dataset, channel in zip(datasets, channels, strict=True) for _ in range(dataset.sizes["line"])
    ]
    return response.assign_coords(channel=("line", line_channels))
