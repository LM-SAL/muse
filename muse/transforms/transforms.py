import numpy as np
import xarray as xr

import astropy.units as u

from muse.log import logger
from muse.utils.documentation import format_docstring
from muse.utils.utils import add_history, require_unit, update_attrs
from muse.variables import DEFAULTS_MUSE

__all__ = ["match_fov", "reshape_slit_step_to_x", "reshape_x_to_slit_step"]

_CM_PER_ARCSEC_AT_1_AU = (u.arcsec.to(u.rad) * u.AU).to_value("cm")
_SPATIAL_EQUIVALENCY = [
    (
        u.arcsec,
        u.cm,
        lambda value: value * _CM_PER_ARCSEC_AT_1_AU,
        lambda value: value / _CM_PER_ARCSEC_AT_1_AU,
    )
]


def _coordinate_unit_to(ds: xr.Dataset, coord_name: str, target_unit):
    unit = require_unit(ds, coord_name, f"{coord_name} coordinate", coord_only=True)
    try:
        return unit.to(target_unit, equivalencies=_SPATIAL_EQUIVALENCY)
    except u.UnitConversionError as exc:
        msg = f"{coord_name} coordinate units must be convertible to {target_unit}"
        raise ValueError(msg) from exc


def _interp_keep_dtype(ds: xr.Dataset, axis: str, target) -> xr.Dataset:
    """
    Interpolate along ``axis`` and cast data variables back to their original dtype.

    Coordinates keep xarray's interpolation dtype, usually float64.
    """
    dtypes = {name: var.dtype for name, var in ds.data_vars.items()}
    out = ds.interp({axis: target})
    return out.assign({name: out[name].astype(dtype) for name, dtype in dtypes.items() if out[name].dtype != dtype})


def _resample_axis_to_pixel(ds: xr.Dataset, axis: str, pixel_arcsec: float, sub_interpolation: int) -> xr.Dataset:
    """
    Resample ``ds`` onto the MUSE pixel size along ``axis`` (``"x"`` or ``"y"``).

    Interpolates up when the target grid is finer, integer-factor averages down when it
    is much coarser, and otherwise sub-interpolates before averaging. A size-1 axis is
    returned unchanged. ``coord`` must be strictly monotonically increasing and roughly
    evenly spaced for block averaging.
    """
    coord = ds[axis]
    if coord.size <= 1:
        return ds
    if not np.all(np.diff(coord.values) > 0):
        msg = f"{axis} coordinate must be strictly monotonically increasing to resample onto the MUSE pixel grid"
        raise ValueError(msg)
    to_cm = _coordinate_unit_to(ds, axis, u.cm)

    def grid(n):
        return np.linspace(coord[0].values, coord[-1].values, n)

    span_pixels = (coord[-1].data - coord[0].data) * to_cm / (_CM_PER_ARCSEC_AT_1_AU * pixel_arcsec)
    n = int(np.round(span_pixels))
    if n > coord.size:
        return _interp_keep_dtype(ds, axis, grid(n))
    if coord.size // n > 3:
        # factor = coord.size / span_pixels; the rounded sample count n equals round(span_pixels).
        blocks = int(np.round(coord.size / span_pixels))
        ds = _interp_keep_dtype(ds, axis, grid(n * blocks))
    else:
        # sub_interpolation == 0 means "no sub-grid", but the meshgrid/unstack below still
        # needs >= 1 block; fall back to interpolating straight onto the n output pixels.
        blocks = max(1, int(coord.size / n * sub_interpolation))
        ds = _interp_keep_dtype(ds, axis, grid(n * blocks))
    block_index, centers = (arr.flatten() for arr in np.meshgrid(range(blocks), grid(n)))
    ds = ds.assign_coords(_block=(axis, block_index), _center=(axis, centers))
    return ds.set_index({axis: ("_block", "_center")}).unstack(axis).mean(dim="_block").rename({"_center": axis})


@format_docstring(
    "DEFAULTS_MUSE",
    dx_pix="dx_pixel_SG",
    dy_pix="dy_pixel_SG",
    nslits="number_of_slits_SG",
    nraster="steps_per_raster_SG",
)
def match_fov(
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

    # Shallow copy: every transform below returns a new object; the copy only isolates
    # coord/attr mutations on the degenerate size-1 paths where resampling is a no-op.
    vdem_xr = vdem.copy()

    vdem_xr = _resample_axis_to_pixel(vdem_xr, "x", dx_pix.value, sub_interpolation)
    vdem_xr = _resample_axis_to_pixel(vdem_xr, "y", dy_pix.value, sub_interpolation)

    nx = vdem_xr.x.size
    if nx > 1:
        if nslits * nraster > nx:
            if restype[10:] != "notile":
                if mode == "wrap":
                    # dask.array.pad silently clips a wrap pad wider than the axis; modular
                    # indexing tiles any width and works on both numpy and dask backends.
                    vdem_xr = vdem_xr.isel(x=np.arange(nslits * nraster) % nx)
                else:
                    vdem_xr = vdem_xr.pad(x=(0, nslits * nraster - nx), mode=mode)
        else:
            vdem_xr = vdem_xr.isel(x=np.arange(nslits * nraster))

    vdem_xr.coords["x"] = np.arange(vdem_xr.x.size) * dx_pix.value
    vdem_xr.coords["y"] = np.arange(vdem_xr.y.size) * dy_pix.value
    vdem_xr.x.attrs["units"] = "arcsec"
    vdem_xr.y.attrs["units"] = "arcsec"
    for var in vdem.data_vars:
        vdem_xr[var].attrs.update(vdem[var].attrs)

    update_attrs(vdem_xr, vdem)
    add_history(vdem_xr, locals(), match_fov)

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
    For a given xarray data set (either vdem or spectra) we reshape from the x spatial
    axis to raster step and slits.

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
    x_unit = require_unit(ds, "x", "x coordinate", coord_only=True)
    attrs = {}
    if "slit" in ds.coords:
        reshaped = ds.unstack("x")
    else:
        step_size = ds.x[1] - ds.x[0]
        step, slit = (arr.flatten() for arr in np.meshgrid(range(nraster), range(nslits)))
        reshaped = ds.assign_coords(slit=("x", slit), step=("x", step))
        reshaped = reshaped.set_index(x=("slit", "step")).unstack("x")
        attrs = {"step_size": step_size.values, "step_size units": str(x_unit)}
    update_attrs(reshaped, ds, **attrs)
    add_history(reshaped, locals(), reshape_x_to_slit_step)
    return reshaped


@format_docstring(
    "DEFAULTS_MUSE",
    nslits="number_of_slits_SG",
    nraster="steps_per_raster_SG",
)
def reshape_slit_step_to_x(
    ds: xr.Dataset,
    nslits=DEFAULTS_MUSE.number_of_slits_SG,
    nraster=DEFAULTS_MUSE.steps_per_raster_SG,
):
    """
    Inverse of `reshape_x_to_slit_step`: collapse the slit and raster step axes back
    onto a single x spatial axis.

    The x coordinate is rebuilt from the per-step spacing recorded in
    ``ds.attrs["step_size"]`` (set by `reshape_x_to_slit_step`); when that
    attribute is missing the default MUSE pixel size is used.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Data with ``slit`` and ``step`` coordinates.
    nslits : `int`
        Number of slits, by default is {nslits}.
    nraster : `int`
        Number of raster steps, by default is {nraster}.

    Returns
    -------
    `xarray.Dataset`
        vdem or spectra with a single x spatial axis.
    """
    for coord in ("slit", "step"):
        if coord not in ds.coords:
            msg = f"{coord} coordinate is missing"
            raise ValueError(msg)
    step_units = ds.attrs.get("step_size units", "arcsec")
    step_size = ds.attrs.get("step_size", DEFAULTS_MUSE.dx_pixel_SG.to_value(step_units))
    reshaped = ds.stack(x=("slit", "step"))
    reshaped = reshaped.drop_vars(["x", "slit", "step"])
    reshaped = reshaped.assign_coords(x=np.arange(reshaped.x.size) * step_size)
    reshaped.x.attrs["units"] = step_units
    update_attrs(reshaped, ds)
    add_history(reshaped, locals(), reshape_slit_step_to_x)
    return reshaped
