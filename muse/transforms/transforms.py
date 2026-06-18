import numpy as np
import xarray as xr

import astropy.units as u

from muse.log import logger
from muse.utils.documentation import format_docstring
from muse.utils.utils import add_history
from muse.variables import DEFAULTS_MUSE

__all__ = ["muse_fov", "reshape_x_to_slit_step"]

_CM_PER_ARCSEC_AT_1_AU = (u.arcsec.to(u.rad) * u.AU).to_value("cm")
_SPATIAL_EQUIVALENCY = [
    (
        u.arcsec,
        u.cm,
        lambda value: value * _CM_PER_ARCSEC_AT_1_AU,
        lambda value: value / _CM_PER_ARCSEC_AT_1_AU,
    )
]


def _coordinate_unit(ds: xr.Dataset, coord_name: str):
    if coord_name not in ds.coords:
        msg = f"{coord_name} coordinate is missing"
        raise ValueError(msg)
    if "units" not in ds[coord_name].attrs:
        msg = f"{coord_name} coordinate must define units"
        raise ValueError(msg)
    try:
        return u.Unit(ds[coord_name].attrs["units"])
    except (TypeError, ValueError) as exc:
        msg = f"{coord_name} coordinate units must be a valid astropy unit"
        raise ValueError(msg) from exc


def _coordinate_unit_to(ds: xr.Dataset, coord_name: str, target_unit):
    unit = _coordinate_unit(ds, coord_name)
    try:
        return unit.to(target_unit, equivalencies=_SPATIAL_EQUIVALENCY)
    except u.UnitConversionError as exc:
        msg = f"{coord_name} coordinate units must be convertible to {target_unit}"
        raise ValueError(msg) from exc


@format_docstring(
    "DEFAULTS_MUSE",
    dx_pix="dx_pixel_SG",
    dy_pix="dy_pixel_SG",
    nslits="number_of_slits_SG",
    nraster="steps_per_raster_SG",
)
def muse_fov(
    vdem: xr.Dataset,
    dx_pix=DEFAULTS_MUSE.dx_pixel_SG,
    dy_pix=DEFAULTS_MUSE.dy_pixel_SG,
    nslits=DEFAULTS_MUSE.number_of_slits_SG,
    nraster=DEFAULTS_MUSE.steps_per_raster_SG,
    restype: str = "match_res_tile",
    mode: str = "wrap",
    sub_interpolation: int = 2,
    rotate=False,  # NOQA: FBT002
):
    """
    Match the data to MUSE FOV allowing to:

    1) match the resolution, restype[:9] = "match_res"
    2) and tile the box in x axis, mode = "wrap" (np.pad option mode).
    3) or tile the box in x axis with zeros, mode = "constant". (or any other pad option)
    4) no tiling, restype[10:] = "notile".

    Parameters
    ----------
    vdem : `xarray.Dataset`
        VDEM dataset. The ``x`` and ``y`` coordinates must define units.
    dx_pix : `astropy.units.Quantity`, optional
        Pixel size in arcsec along the x axis, by default is {dx_pix}
    dy_pix : `astropy.units.Quantity`, optional
        Pixel size in arcsec along the y axis, by default is {dy_pix}
    nslits : `int`
        Number of slits, by default is {nslits}.
    nraster : `int`
        Number of raster steps, by default is {nraster}.
    restype : `str`, optional
        Type of tiling and resolution matching, by default is "match_res_tile".
    mode : `str`, optional
        This is the pad method used by `xarray.DataArray.pad`, by default is "wrap".
        Please check the the `xarray.DataArray.pad` documentation for all possible values.
    sub_interpolation: `int`
        Does a subgrid interpolation, by default 2,
    rotate: `bool`
        Rotates 90 degrees the FOV, by default False.

    Returns
    -------
    array - `xarray.Dataset`
            VDEM data matching MUSE data

    Notes
    -----
    Depending on the input grid, one of the following paths is taken:

    1. **Already at MUSE resolution** - if ``x`` has more than one point, its
       spacing already matches ``dx_pix`` and ``x`` spans exactly
       ``nslits * nraster`` points (and, when ``y`` has more than one point, its
       spacing matches ``dy_pix``), the input dataset is returned unchanged.
    2. **Single row or column** - if only one axis has more than one point, the
       match is decided on that axis alone (the other spacing cannot be measured)
       and, when it matches, the input dataset is returned unchanged.
    3. **Resample and tile** - otherwise each axis is resampled onto the MUSE
       pixel size (interpolated up, integer-factor averaged down, or
       sub-interpolated) and the ``x`` axis is then padded out with ``mode``
       (``restype`` without the ``"notile"`` suffix) or truncated to
       ``nslits * nraster``.
    4. **No tiling** - a ``restype`` ending in ``"notile"`` skips the padding in
       path 3 and keeps the resampled width.
    5. **Single pixel** - a 1x1 input has nothing to resample or tile and returns
       a copy with the coordinates relabeled onto the MUSE grid. This degenerate
       path may be removed in the future.
    """
    if not isinstance(dx_pix, u.Quantity):
        msg = "dx_pix must be an astropy.units.Quantity convertible to arcsec"
        raise TypeError(msg)
    if not isinstance(dy_pix, u.Quantity):
        msg = "dy_pix must be an astropy.units.Quantity convertible to arcsec"
        raise TypeError(msg)
    dx_pix = dx_pix.to("arcsec")
    dy_pix = dy_pix.to("arcsec")

    if not restype.startswith("match_res"):
        msg = f"Unsupported restype {restype!r}; only 'match_res*' is supported."
        raise ValueError(msg)

    if not rotate:
        sim_units_to_arcsec_x = _coordinate_unit_to(vdem, "x", u.arcsec)
        sim_units_to_arcsec_y = _coordinate_unit_to(vdem, "y", u.arcsec)

        if vdem.coords["x"].size > 1:
            dx_coord_diff = vdem.coords["x"][1] - vdem.coords["x"][0]
            dx_current = dx_coord_diff.data * sim_units_to_arcsec_x
            if hasattr(dx_current, "value"):
                dx_current = dx_current.value
            if (abs(dx_current - dx_pix.value) / dx_pix.value < 0.005) and (vdem.coords["x"].size == nslits * nraster):
                if vdem.coords["y"].size > 1:
                    dy_coord_diff = vdem.coords["y"][1] - vdem.coords["y"][0]
                    dy_current = dy_coord_diff.data * sim_units_to_arcsec_y
                    if abs(dy_current - dy_pix.value) / dy_pix.value < 0.005:
                        logger.info("vdem has already the MUSE pixel size")
                        return vdem
                else:
                    logger.info("vdem has already the MUSE pixel size")
                    return vdem
        elif vdem.coords["y"].size > 1:
            dy_coord_diff = vdem.coords["y"][1] - vdem.coords["y"][0]
            dy_current = dy_coord_diff.data * sim_units_to_arcsec_y
            if abs(dy_current - dy_pix.value) / dy_pix.value < 0.005:
                logger.info("vdem has already the MUSE pixel size")
                return vdem
    else:
        vdem = vdem.rename({"x": "ynew"})
        vdem = vdem.rename({"y": "x"})
        vdem = vdem.rename({"ynew": "y"})

    vdem_xr = vdem.copy(deep=True)

    sim_units_to_cm = _coordinate_unit_to(vdem_xr, "x", u.cm)
    if len(vdem_xr.x) > 1:
        nx = int(
            np.round(
                (vdem_xr.x[-1] - vdem_xr.x[0]) * sim_units_to_cm / (_CM_PER_ARCSEC_AT_1_AU * dx_pix.value),
            )
        )
        if nx > len(vdem_xr.x):
            vdem_xr = vdem_xr.interp(x=np.linspace(vdem_xr.x[0].values, vdem_xr.x[-1].values, nx))
        elif (int(len(vdem_xr.x) / nx)) > 3:
            factor = len(vdem_xr.x) / (
                (vdem_xr.x[-1].data - vdem_xr.x[0].data) * sim_units_to_cm / _CM_PER_ARCSEC_AT_1_AU / dx_pix.value
            )
            nx = int(np.round(len(vdem_xr.x) / factor))
            new_nx = nx * int(np.round(factor))
            vdem_xr = vdem_xr.interp(x=np.linspace(vdem_xr.x[0].values, vdem_xr.x[-1].values, new_nx))
            xc, t = (
                arr.flatten()
                for arr in np.meshgrid(
                    range(int(np.round(factor))), np.linspace(vdem_xr.x[0].values, vdem_xr.x[-1].values, nx)
                )
            )
            vdem_xr = vdem_xr.assign_coords(xt=("x", xc), xr=("x", t))
            vdem_xr = vdem_xr.set_index(x=("xt", "xr")).unstack("x")
            vdem_xr = vdem_xr.mean(dim="xt")
            vdem_xr = vdem_xr.rename({"xr": "x"})
        else:
            nx_int = int(len(vdem_xr.x) / nx * sub_interpolation)
            if sub_interpolation > 0:
                vdem_xr = vdem_xr.interp(x=np.linspace(vdem_xr.x[0].values, vdem_xr.x[-1].values, nx * nx_int))
            xc, t = (
                arr.flatten()
                for arr in np.meshgrid(
                    range(int(np.round(nx_int))), np.linspace(vdem_xr.x[0].values, vdem_xr.x[-1].values, nx)
                )
            )
            vdem_xr = vdem_xr.assign_coords(xt=("x", xc), xr=("x", t))
            vdem_xr = vdem_xr.set_index(x=("xt", "xr")).unstack("x")
            vdem_xr = vdem_xr.mean(dim="xt")
            vdem_xr = vdem_xr.rename({"xr": "x"})
    sim_units_to_cm = _coordinate_unit_to(vdem_xr, "y", u.cm)
    if len(vdem_xr.y) > 1:
        ny = int((vdem_xr.y[-1] - vdem_xr.y[0]) * sim_units_to_cm / (_CM_PER_ARCSEC_AT_1_AU * dy_pix.value))
        if ny > len(vdem_xr.y):
            vdem_xr = vdem_xr.interp(y=np.linspace(vdem_xr.y[0].values, vdem_xr.y[-1].values, ny))
        elif (int(len(vdem_xr.y) / ny)) > 3:
            factor = len(vdem_xr.y) / (
                (vdem_xr.y[-1].data - vdem_xr.y[0].data) * sim_units_to_cm / _CM_PER_ARCSEC_AT_1_AU / dy_pix.value
            )
            ny = int(np.round(len(vdem_xr.y) / factor))
            new_ny = ny * int(np.round(factor))
            vdem_xr = vdem_xr.interp(y=np.linspace(vdem_xr.y[0].values, vdem_xr.y[-1].values, new_ny))
            xc, t = (
                arr.flatten()
                for arr in np.meshgrid(
                    range(int(np.round(factor))), np.linspace(vdem_xr.y[0].values, vdem_xr.y[-1].values, ny)
                )
            )
            vdem_xr = vdem_xr.assign_coords(xt=("y", xc), xr=("y", t))
            vdem_xr = vdem_xr.set_index(y=("xt", "xr")).unstack("y")
            vdem_xr = vdem_xr.mean(dim="xt")
            vdem_xr = vdem_xr.rename({"xr": "y"})
        else:
            ny_int = int(len(vdem_xr.y) / ny * sub_interpolation)
            vdem_xr = vdem_xr.interp(y=np.linspace(vdem_xr.y[0].values, vdem_xr.y[-1].values, ny * ny_int))
            xc, t = (
                arr.flatten()
                for arr in np.meshgrid(range(ny_int), np.linspace(vdem_xr.y[0].values, vdem_xr.y[-1].values, ny))
            )
            vdem_xr = vdem_xr.assign_coords(xt=("y", xc), xr=("y", t))
            vdem_xr = vdem_xr.set_index(y=("xt", "xr")).unstack("y")
            vdem_xr = vdem_xr.mean(dim="xt")
            vdem_xr = vdem_xr.rename({"xr": "y"})
    if vdem_xr.x.size > 1:
        if nslits * nraster > nx:
            if restype[10:] != "notile":
                vdem_xr = vdem_xr.pad(x=(0, nslits * nraster - nx), mode=mode)
        else:
            vdem_xr = vdem_xr.isel(x=np.arange(nslits * nraster))

    vdem_xr.coords["x"] = np.arange(vdem_xr.x.size) * dx_pix.value
    vdem_xr.coords["y"] = np.arange(vdem_xr.y.size) * dy_pix.value
    vdem_xr.x.attrs["units"] = "arcsec"
    vdem_xr.y.attrs["units"] = "arcsec"
    for varss in vdem.data_vars:
        for atrs in vdem[varss].attrs:
            vdem_xr[varss].attrs[atrs] = vdem[varss].attrs[atrs]

    add_history(vdem_xr, locals(), muse_fov)

    return vdem_xr


@format_docstring(
    "DEFAULTS_MUSE",
    nslits="number_of_slits_SG",
    nraster="steps_per_raster_SG",
)
def reshape_x_to_slit_step(
    ds: xr.Dataset,
    nslits=DEFAULTS_MUSE.number_of_slits_SG,
    nraster=DEFAULTS_MUSE.steps_per_raster_SG,
):
    """
    For a given xarray data set (either vdem or spectra) we reshape from the x
    spatial axis to raster step and slits.

    TODO: Similar function should be done when we have time and we want to
    interpolate or integrate in MUSE time integration and step.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Data that contains x spatial axis. The ``x`` coordinate must define units.
    nslits : `int`
        Number of slits, by default is {nslits}.
    nraster : `int`
        Number of raster steps, by default is {nraster}.

    Returns
    -------
    `xarray.Dataset`
        vdem or spectra with raster and slit axis.
    """
    if "x" not in ds.coords:
        msg = "x coordinate is missing"
        raise ValueError(msg)
    x_unit = _coordinate_unit(ds, "x")
    ds_temp = ds.copy(deep=True)
    if "slit" in ds.coords:
        ds_temp = ds_temp.unstack("x")
    else:
        step_size = ds_temp.x[1] - ds_temp.x[0]
        step, slit = (arr.flatten() for arr in np.meshgrid(range(nraster), range(nslits)))
        ds_temp = ds_temp.assign_coords(slit=("x", slit), step=("x", step))
        ds_temp = ds_temp.set_index(x=("slit", "step")).unstack("x")
        ds_temp.attrs.update({"step_size": step_size.values, "step_size units": str(x_unit)})
    add_history(ds_temp, locals(), reshape_x_to_slit_step)
    return ds_temp
