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


def _coordinate_unit_to(ds: xr.Dataset, coord_name: str, target_unit: u.UnitBase) -> float:
    unit = require_unit(ds, coord_name, f"{coord_name} coordinate", coord_only=True)
    try:
        return unit.to(target_unit, equivalencies=_SPATIAL_EQUIVALENCY)
    except u.UnitConversionError as exc:
        msg = f"{coord_name} coordinate units must be convertible to {target_unit}"
        raise ValueError(msg) from exc


def _spacing_matches(coord: xr.DataArray, to_arcsec: float, pixel_arcsec: float) -> bool:
    """
    Whether ``coord`` is already sampled at ``pixel_arcsec``, within 0.5%.

    Returns `False` for a size-1 axis, whose spacing cannot be measured. The coordinate
    data may be a plain array or an `astropy.units.Quantity`, so the scaled spacing is
    unwrapped before it is compared with a plain float.
    """
    if coord.size < 2:
        return False
    spacing = (coord[1] - coord[0]).data * to_arcsec
    spacing = getattr(spacing, "value", spacing)
    return bool(abs(spacing - pixel_arcsec) / pixel_arcsec < 0.005)


def _resample_weights(coord_values: np.ndarray, n: int, blocks: int) -> np.ndarray:
    """
    Matrix mapping samples at ``coord_values`` onto ``n`` output pixels.

    Row ``i`` holds the weights that linearly interpolate the input onto ``blocks``
    evenly spaced sub-samples of output pixel ``i`` and average them, so applying the
    matrix reproduces interpolate-then-block-average in a single product.
    """
    fine = np.linspace(coord_values[0], coord_values[-1], n * blocks)
    right = np.clip(np.searchsorted(coord_values, fine, side="right"), 1, coord_values.size - 1)
    left = right - 1
    frac = (fine - coord_values[left]) / (coord_values[right] - coord_values[left])
    weights = np.zeros((n * blocks, coord_values.size))
    rows = np.arange(n * blocks)
    weights[rows, left] = 1.0 - frac
    weights[rows, right] = frac
    return weights.reshape(n, blocks, coord_values.size).mean(axis=1)


def _resample_axis_to_pixel(ds: xr.Dataset, axis: str, pixel_arcsec: float, sub_interpolation: int) -> xr.Dataset:
    """
    Resample ``ds`` onto the MUSE pixel size along ``axis`` (``"x"`` or ``"y"``).

    Interpolates up when the target grid is finer, integer-factor averages down when it
    is much coarser, and otherwise sub-interpolates before averaging. A size-1 axis is
    returned unchanged. ``coord`` must be strictly monotonically increasing and roughly
    evenly spaced for block averaging.

    The interpolate-then-average pipeline is one linear map along ``axis``, applied per
    block with `xarray.apply_ufunc`: dask-backed inputs stay lazy and every output chunk
    depends only on its local input chunks, so chained x/y resampling streams instead of
    materializing the intermediate.

    Only data variables carrying ``axis`` are resampled; any auxiliary coordinate that
    depends on ``axis`` is dropped with the old grid (`match_fov` rebuilds the
    ``x``/``y`` coordinates and their units afterwards).
    """
    coord = ds[axis]
    if coord.size <= 1:
        return ds
    if not np.all(np.diff(coord.values) > 0):
        msg = f"{axis} coordinate must be strictly monotonically increasing to resample onto the MUSE pixel grid"
        raise ValueError(msg)
    to_cm = _coordinate_unit_to(ds, axis, u.cm)
    span_pixels = (coord[-1].data - coord[0].data) * to_cm / (_CM_PER_ARCSEC_AT_1_AU * pixel_arcsec)
    n = int(np.round(span_pixels))
    if n > coord.size:
        # Finer target grid: pure interpolation, no block averaging.
        blocks = 1
    elif coord.size // n > 3:
        # factor = coord.size / span_pixels; the rounded sample count n equals round(span_pixels).
        blocks = int(np.round(coord.size / span_pixels))
    else:
        # sub_interpolation == 0 means "no sub-grid", but the averaging still needs
        # >= 1 block; fall back to interpolating straight onto the n output pixels.
        blocks = max(1, int(coord.size / n * sub_interpolation))
    # No coords on the weights array: alignment must be by dimension only.
    weights = xr.DataArray(_resample_weights(coord.values, n, blocks), dims=("_center", axis))
    centers = np.linspace(coord[0].values, coord[-1].values, n)

    def apply_weights(block: np.ndarray, weight_block: np.ndarray, dtype=None) -> np.ndarray:
        return (block @ weight_block.T).astype(dtype)

    # apply_ufunc(dask="parallelized") applies the matrix per block, so every output
    # chunk depends only on its local input chunks. xr.dot builds a chunked einsum
    # graph instead, and chaining two of those (x then y) forms an all-to-all barrier
    # that holds the entire intermediate in memory at once (~7 GB on the tutorial VDEM).
    resampled = {
        name: xr.apply_ufunc(
            apply_weights,
            var,
            weights,
            kwargs={"dtype": var.dtype},
            input_core_dims=[[axis], ["_center", axis]],
            output_core_dims=[["_center"]],
            dask="parallelized",
            output_dtypes=[var.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True},
        ).rename({"_center": axis})
        for name, var in ds.data_vars.items()
        if axis in var.dims
    }
    return ds.drop_dims(axis).assign(resampled).assign_coords({axis: centers})


@format_docstring(
    "DEFAULTS_MUSE",
    dx_pix="dx_pixel_SG",
    dy_pix="dy_pixel_SG",
    nslits="number_of_slits_SG",
    nraster="steps_per_raster_SG",
    x_extent="fov_x_extent",
    mode="fov_mode",
    sub_interpolation="fov_sub_interpolation",
)
@u.quantity_input(dx_pix=u.arcsec, dy_pix=u.arcsec)
def match_fov(
    vdem: xr.Dataset,
    *,
    dx_pix: u.Quantity = DEFAULTS_MUSE.dx_pixel_SG,
    dy_pix: u.Quantity = DEFAULTS_MUSE.dy_pixel_SG,
    nslits: int = DEFAULTS_MUSE.number_of_slits_SG,
    nraster: int = DEFAULTS_MUSE.steps_per_raster_SG,
    x_extent: str = DEFAULTS_MUSE.fov_x_extent,
    mode: str = DEFAULTS_MUSE.fov_mode,
    sub_interpolation: int = DEFAULTS_MUSE.fov_sub_interpolation,
    rotate: bool = False,
) -> xr.Dataset:
    """
    Match a VDEM's spatial resolution and available extent to the MUSE FOV.

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
    x_extent : `str`, optional
        How to treat the x width relative to ``nslits * nraster``, by default
        {x_extent}. ``"tile"`` crops wider inputs and extends narrower ones
        using ``mode``. ``"crop"`` crops wider inputs but retains a narrower
        input at its available width. ``"keep"`` leaves the width untouched
        and only matches the spatial resolution, for building the FOV of
        instruments whose width is not tied to the MUSE raster. Spatial
        resolution matching is always performed.
    mode : `str`, optional
        Extension mode used when ``x_extent="tile"``, by default {mode}. See
        `xarray.DataArray.pad` for supported values.
        ``"wrap"`` repeats the input along x.
        ``"constant"`` fills the padded columns with zeros rather than the
        `xarray.DataArray.pad` default of NaN, so they read as "no emission here".
    sub_interpolation: `int`
        Does a subgrid interpolation, by default {sub_interpolation},
    rotate: `bool`
        Rotates 90 degrees the FOV, by default False.

    Returns
    -------
    `xarray.Dataset`
        VDEM data resampled to the requested pixel sizes and cropped or extended
        along x as described by ``x_extent``.

    Notes
    -----
    Depending on the input grid, one of the following paths is taken:

    1. **Already matched** - if rotation is disabled, the measurable spatial
       axes have the requested pixel spacing, and the x width needs no crop or
       extension, the input dataset is returned unchanged. After rotation,
       matched axes skip resampling but continue through output finalization.
    2. **Single row or column** - if only one axis has more than one point, the
       resolution match is decided on that axis alone because the other spacing
       cannot be measured. The x extent still follows ``x_extent``.
    3. **Resample** - otherwise each axis is resampled onto the requested
       pixel size (interpolated up, integer-factor averaged down, or
       sub-interpolated).
    4. **Match the x extent** - inputs wider than ``nslits * nraster`` are
       cropped. Narrower inputs are extended with ``mode`` only when
       ``x_extent="tile"``; ``"crop"`` retains their available width. Skipped
       entirely when ``x_extent="keep"``.
    5. **Single pixel** - a 1x1 input has no measurable spatial spacing. Its x
       extent is extended when ``x_extent="tile"`` and otherwise retained.
    """
    if x_extent not in ("tile", "crop", "keep"):
        msg = f"Unsupported x_extent {x_extent!r}; use 'tile', 'crop', or 'keep'"
        raise ValueError(msg)
    dx_pix = dx_pix.to("arcsec")
    dy_pix = dy_pix.to("arcsec")
    target_width = nslits * nraster

    if rotate:
        vdem = vdem.rename({"x": "ynew"})
        vdem = vdem.rename({"y": "x"})
        vdem = vdem.rename({"ynew": "y"})

    x_at_pixel = _spacing_matches(vdem.coords["x"], _coordinate_unit_to(vdem, "x", u.arcsec), dx_pix.value)
    y_at_pixel = _spacing_matches(vdem.coords["y"], _coordinate_unit_to(vdem, "y", u.arcsec), dy_pix.value)
    resolution_matches = (x_at_pixel and (y_at_pixel or vdem.coords["y"].size < 2)) or (
        y_at_pixel and vdem.coords["x"].size < 2
    )
    width_needs_no_change = (
        x_extent == "keep"
        or vdem.coords["x"].size == target_width
        or (x_extent == "crop" and vdem.coords["x"].size < target_width)
    )
    if not rotate and resolution_matches and width_needs_no_change:
        logger.info("vdem already has the requested pixel size and needs no x-extent change")
        return vdem

    # Shallow copy: every transform below returns a new object; the copy only isolates
    # coord/attr mutations on the degenerate size-1 paths where resampling is a no-op.
    vdem_xr = vdem.copy()

    if not resolution_matches:
        vdem_xr = _resample_axis_to_pixel(vdem_xr, "x", dx_pix.value, sub_interpolation)
        vdem_xr = _resample_axis_to_pixel(vdem_xr, "y", dy_pix.value, sub_interpolation)
        # The unstack inside the resampling moves the resampled axis last; restore each
        # variable's original dimension order so downstream plots keep their orientation.
        vdem_xr = vdem_xr.assign({name: vdem_xr[name].transpose(*vdem[name].dims) for name in vdem.data_vars})

    nx = vdem_xr.x.size
    if x_extent != "keep":
        if nx > target_width:
            vdem_xr = vdem_xr.isel(x=np.arange(target_width))
        elif x_extent == "tile" and nx < target_width:
            if mode == "wrap":
                # dask.array.pad silently clips a wrap pad wider than the axis; modular
                # indexing tiles any width and works on both numpy and dask backends.
                vdem_xr = vdem_xr.isel(x=np.arange(target_width) % nx)
            else:
                # xarray.pad defaults constant_values to NaN, which would poison every
                # downstream sum; a padded VDEM column means "no emission here", so fill
                # with zeros. Only "constant" accepts the argument.
                fill = {"constant_values": 0} if mode == "constant" else {}
                vdem_xr = vdem_xr.pad(x=(0, target_width - nx), mode=mode, **fill)

    vdem_xr.coords["x"] = np.arange(vdem_xr.x.size) * dx_pix.value
    vdem_xr.coords["y"] = np.arange(vdem_xr.y.size) * dy_pix.value
    vdem_xr.x.attrs["units"] = "arcsec"
    vdem_xr.y.attrs["units"] = "arcsec"
    for var in vdem.data_vars:
        vdem_xr[var].attrs.update(vdem[var].attrs)

    update_attrs(vdem_xr, vdem)
    add_history(vdem_xr, locals(), match_fov, sources=(vdem,))

    return vdem_xr


@format_docstring(
    "DEFAULTS_MUSE",
    nslits="number_of_slits_SG",
    nraster="steps_per_raster_SG",
)
def reshape_x_to_slit_step(
    ds: xr.Dataset,
    nslits: int = DEFAULTS_MUSE.number_of_slits_SG,
    nraster: int = DEFAULTS_MUSE.steps_per_raster_SG,
) -> xr.Dataset:
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
    add_history(reshaped, locals(), reshape_x_to_slit_step, sources=(ds,))
    return reshaped


@format_docstring(
    "DEFAULTS_MUSE",
    nslits="number_of_slits_SG",
    nraster="steps_per_raster_SG",
)
def reshape_slit_step_to_x(
    ds: xr.Dataset,
    nslits: int = DEFAULTS_MUSE.number_of_slits_SG,
    nraster: int = DEFAULTS_MUSE.steps_per_raster_SG,
) -> xr.Dataset:
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
    add_history(reshaped, locals(), reshape_slit_step_to_x, sources=(ds,))
    return reshaped
