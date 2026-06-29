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
    Reads a response function into an `xarray.Dataset` interpolating if needed
    in vdop, and logT.

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
        Allowed values are "nearest", "linear", "cubic" and "quadratic".
    vdop_method : `str`, optional
        Interpolation method for vdop, by default "nearest".
        Allowed values are "nearest", "linear", "cubic" and "quadratic".
    gain : `astropy.units.Quantity`, optional
        Camera gain, convertible to electron/DN, by default {gain}.

    Returns
    -------
    `xarray.Dataset`
        The combined response function dataset.

    Raises
    ------
    ValueError
        If ``response_file`` does not exist, an interpolation method is invalid, the
        ``logT``/``vdop`` axes are malformed, or the loaded dataset is
        missing the ``SG_resp`` variable or the ``logT``/``vdop`` coordinates.
    """
    _INTERP_METHODS = ("nearest", "linear", "cubic", "quadratic")
    response_file = Path(response_file)
    if not response_file.exists():
        msg = f"Response does not exist: {response_file}"
        raise ValueError(msg)
    for method_name, method in (("logT_method", logT_method), ("vdop_method", vdop_method)):
        if method not in _INTERP_METHODS:
            msg = f"Invalid {method_name}: {method}, allowed values are {_INTERP_METHODS}"
            raise ValueError(msg)

    for name, axis in (("logT", logT), ("vdop", vdop)):
        if axis is None:
            continue
        if len(axis.data) == 0:
            msg = f"{name} array must not be empty"
            raise ValueError(msg)
        if not np.all(np.isfinite(axis.data)):
            msg = f"{name} must contain only finite values"
            raise ValueError(msg)

    r = (
        xr.open_zarr(response_file)
        if response_file.suffix == ".zarr" and response_file.is_dir()
        else xr.load_dataset(response_file)
    )
    if "SG_resp" not in r.data_vars:
        msg = "Response dataset must contain 'SG_resp' variable"
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

    if "line_wvl" not in r:
        fallback = r.attrs.get("LINE_WVL", r.attrs.get("MAIN_LINE_WVL"))
        if fallback is not None:
            r = r.assign_coords(line_wvl=fallback)
        elif "channel" in r.coords:
            r = r.assign_coords(line_wvl=r.channel)
        else:
            msg = "Response must define line_wvl or LINE_WVL/MAIN_LINE_WVL metadata"
            raise ValueError(msg)

    gain = gain.to(u.electron / u.DN)
    gain_dim = "channel" if "channel" in r.dims else "line"
    r = r.assign_coords(gain=(gain_dim, np.atleast_1d(gain.to_value(u.electron / u.DN))))
    r.gain.attrs["units"] = str(u.electron / u.DN)

    # The current response files carry no wavelength units; warn and assume Angstrom for now.
    _require_wavelength_units(r, "SG_wvl")
    _require_wavelength_units(r, "line_wvl")

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
    Select or interpolate the response onto ``axis`` along ``name`` (``logT`` or ``vdop``).

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
    r["SG_resp"] = r.SG_resp.fillna(0).clip(min=0)
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
        Channel values to assign; length must equal ``len(response_files)``.
    logT : `xarray.DataArray`, optional
        Temperature axis to (re)sample onto. Passed to `muse.instrument.utils.read_response`.
    vdop : `xarray.DataArray`, optional
        Velocity axis to (re)sample onto. Passed to `muse.instrument.utils.read_response`.
    slit : `xarray.DataArray`, optional
        Number of slits array of integers. Passed to `muse.instrument.utils.read_response`.
    logT_method : `str`, optional
        Interpolation method for logT, by default "nearest".
        Allowed values are "nearest", "linear", "cubic" and "quadratic".
        Passed to `muse.instrument.utils.read_response`.
    vdop_method : `str`, optional
        Interpolation method for vdop, by default "linear".
        Allowed values are "nearest", "linear", "cubic" and "quadratic".
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
        datasets = [
            read_response(
                Path(response_directory) / f,
                logT=logT,
                vdop=vdop,
                slit=slit,
                logT_method=logT_method,
                vdop_method=vdop_method,
            ).drop_vars("effective_area", errors="ignore")
            for f in response_files
        ]
        response = xr.concat(datasets, dim="line", coords="different", compat="equals")
    return response.assign_coords(channel=("line", list(channels)))
