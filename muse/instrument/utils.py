from pathlib import Path

import dask
import numpy as np
import xarray as xr

import astropy.units as u

from muse.log import logger
from muse.utils.documentation import format_docstring
from muse.utils.utils import add_history
from muse.variables import DEFAULTS_MUSE


@format_docstring(
    "DEFAULTS_MUSE",
    gain="ccd_gain",
)
def read_response(
    respfile: str,
    *,
    logT: np.ndarray = None,
    vdop: np.ndarray = None,
    slit: np.ndarray = None,
    logTmethod: np.ndarray = "nearest",
    vdopmethod: np.ndarray = "nearest",
    gain=DEFAULTS_MUSE.ccd_gain.to_value(u.electron / u.DN),
    **kwargs: dict | None,
) -> xr.Dataset:
    """
    Reads a response function into an `xarray.Dataset` interpolating if needed
    in vdop, and logT.

    Parameters
    ----------
    respfile : `str`
        Response function in Xarray readable format.
    logT : `array-like`, optional
        Temperature axis
    vdop : `array-like`, optional
        Velocity axis
    slit : `array-like`, optional
        Number of slits array of integers.
    logTmethod: `str`
        Interpolation method for logT, by default "nearest".
    vdopmethod: `str`
        Interpolation method for vdop, by default "nearest".
    kwargs : `dict`
        Keyword arguments to pass to `xarray.Dataset.assign_coords`.
        This is currently only used for the `LINE` attribute.
    gain: `int`
        number of electron per DN, by default 10
    **kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    `xarray.Dataset`
        The response function dataset.
    """
    # At function start
    assert Path(respfile).exists() or respfile.endswith(".zarr"), f"Response file does not exist: {respfile}"

    # For interpolation methods
    assert logTmethod in ["nearest", "linear", "cubic", "quadratic"], f"Invalid logTmethod: {logTmethod}"
    assert vdopmethod in ["nearest", "linear", "cubic", "quadratic"], f"Invalid vdopmethod: {vdopmethod}"

    # For axes
    if logT is not None:
        assert hasattr(logT, "data"), "logT must be an xarray DataArray or similar"
        assert len(logT.data) > 0, "logT array must not be empty"
        assert np.all(np.isfinite(logT.data)), "logT must contain only finite values"

    if vdop is not None:
        assert hasattr(vdop, "data"), "vdop must be an xarray DataArray or similar"
        assert len(vdop.data) > 0, "vdop array must not be empty"
        assert np.all(np.isfinite(vdop.data)), "vdop must contain only finite values"

    if slit is not None:
        assert isinstance(slit, (np.ndarray, xr.DataArray)), "slit must be array-like"
        assert slit.max() >= 0, "slit indices must be non-negative"

    r = (
        xr.open_zarr(respfile)
        if respfile.rsplit(".", maxsplit=1)[-1] == "zarr" or Path(respfile).is_dir()
        else xr.load_dataset(respfile)
    )
    assert isinstance(r, xr.Dataset), "Response file must contain an xarray Dataset"
    assert "SG_resp" in r.data_vars, "Response dataset must contain 'SG_resp' variable"
    assert "logT" in r.coords or "logT" in r.dims, "Response must have logT coordinate"
    assert "vdop" in r.coords or "vdop" in r.dims, "Response must have vdop coordinate"

    if logT is not None:
        loc_max = np.argmin(np.abs(logT.data - r.logT.max().data))
        if logT.max() > logT[loc_max]:
            logger.info("Response function is smaller than the VDEM in temp")
            logger.info("Run vdem.sel(logT=response.logT, vdop=response.vdop, drop=True, method='nearest')")

            logT = logT.where(logT <= logT[loc_max], drop=True)
        loc_min = np.argmin(np.abs(logT.data - r.logT.min().data))
        if logT.min() < logT[loc_min]:
            logger.info("Response function is smaller than the VDEM in temp")
            logger.info("Run vdem.sel(logT=response.logT, vdop=response.vdop, drop=True, method='nearest')")
            logT = logT.where(logT >= logT[loc_min], drop=True)
        if logTmethod == "nearest":
            r = r.sel(logT=logT, drop=True, method="nearest")
        else:
            r = r.interp(logT=logT, method=logTmethod)
            r["SG_resp"] = r.SG_resp.fillna(0)
            r["SG_resp"] = r.SG_resp.where(r.SG_resp > 0, 0)
        r["logT"] = logT
        r = r.assign_coords(logT=logT)

    if vdop is not None:
        loc_max = np.argmin(np.abs(vdop.data - r.vdop.max().data))
        if vdop.max() > vdop[loc_max]:
            logger.info("Response function is smaller than the VDEM in vdop")
            logger.info("Run vdem.sel(logT=response.logT, vdop=response.vdop, drop=True, method='nearest')")
            vdop = vdop.where(vdop <= vdop[loc_max], drop=True)
        loc_min = np.argmin(np.abs(vdop.data - r.vdop.min().data))
        if vdop.min() < vdop[loc_min]:
            logger.info("Response function is smaller than the VDEM in vdop")
            logger.info("Run vdem.sel(logT=response.logT, vdop=response.vdop, drop=True, method='nearest')")
            vdop = vdop.where(vdop >= vdop[loc_min], drop=True)
        if vdopmethod == "nearest":
            r = r.sel(vdop=vdop, drop=True, method="nearest")
        else:
            r = r.interp(vdop=vdop, method=vdopmethod)
            r["SG_resp"] = r.SG_resp.fillna(0)
            r["SG_resp"] = r.SG_resp.where(r.SG_resp > 0, 0)

        r["vdop"] = vdop
        r = r.assign_coords(vdop=vdop)

    if slit is not None:
        r = r.sel(slit=np.arange(slit.max() + 1), drop=True, method="nearest")

    if "channel" not in r.dims and "line" not in r.dims:
        r = r.expand_dims("line")

    if "line_wvl" not in r:
        if r.attrs.get("LINE_WVL", r.attrs.get("MAIN_LINE_WVL")) is None and "channel" in r.dims:
            r["line_wvl"] = r.channel
        else:
            r["line_wvl"] = r.attrs.get("LINE_WVL", r.attrs.get("MAIN_LINE_WVL"))

    gain = np.array([10]) if gain is None else np.atleast_1d(gain)
    r = r.assign_coords(gain=("channel", gain)) if "channel" in r.dims else r.assign_coords(gain=("line", gain))

    # JMS this should be removed once we have the new response files with units in the attributes:
    if "SG_wvl" in r and "units" not in r.SG_wvl.attrs:
        r.SG_wvl.attrs.update({"units": str(u.AA)})
    if "units" not in r.line_wvl.attrs:
        r.line_wvl.attrs.update({"units": str(u.AA)})

    add_history(r, locals(), read_response)

    return r


def load_and_concat_responses(resp_dir, resp_files, logT, vdop, slit, logTmethod, channels):
    """
    Load multiple response functions and concatenate them along `line`.

    Parameters
    ----------
    resp_dir : str or Path
        Directory containing the response files.
    resp_files : list of str
        Filenames of response functions to load (in order).
    logT, vdop, slit, logTmethod : passed to `read_response`
    channels : list of int
        Channel values to assign (length must equal len(resp_files)).

    Returns
    -------
    xr.Dataset
        Concatenated response dataset with assigned channel coordinates.
    """
    datasets = []
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        for f in resp_files:
            ds = read_response(
                str(Path(resp_dir) / f),
                logT=logT,
                vdop=vdop,
                slit=slit,
                logTmethod=logTmethod,
                vdopmethod="linear",
            ).compute()
            if "effective_area" in ds.data_vars:
                ds = ds.drop_vars("effective_area")
            datasets.append(ds)

        response = xr.concat(datasets, dim="line", coords="different", compat="equals")
        response = response.assign_coords(channel=("line", channels))
    return response.compute()
