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


def _coord_conversion_factor(ds: xr.Dataset, coord_name: str, target_unit: u.UnitBase) -> float:
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


def _resample_weights(coord_values: np.ndarray, n_pixels: int, samples_per_pixel: int) -> np.ndarray:
    """
    Matrix mapping samples at ``coord_values`` onto ``n_pixels`` output pixels.

    Row ``i`` holds the weights that linearly interpolate the input onto
    ``samples_per_pixel`` evenly spaced sub-samples of output pixel ``i`` and average
    them, so applying the matrix reproduces interpolate-then-block-average in a single
    product.
    """
    fine = np.linspace(coord_values[0], coord_values[-1], n_pixels * samples_per_pixel)
    right = np.clip(np.searchsorted(coord_values, fine, side="right"), 1, coord_values.size - 1)
    left = right - 1
    fraction = (fine - coord_values[left]) / (coord_values[right] - coord_values[left])
    weights = np.zeros((n_pixels * samples_per_pixel, coord_values.size))
    rows = np.arange(n_pixels * samples_per_pixel)
    weights[rows, left] = 1.0 - fraction
    weights[rows, right] = fraction
    return weights.reshape(n_pixels, samples_per_pixel, coord_values.size).mean(axis=1)


def _resample_axis_to_pixel(ds: xr.Dataset, axis: str, pixel_arcsec: float, oversampling: int) -> xr.Dataset:
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
    to_cm = _coord_conversion_factor(ds, axis, u.cm)
    span_pixels = (coord[-1].data - coord[0].data) * to_cm / (_CM_PER_ARCSEC_AT_1_AU * pixel_arcsec)
    n_pixels = int(np.round(span_pixels))
    if n_pixels > coord.size:
        # Finer target grid: pure interpolation, no block averaging.
        samples_per_pixel = 1
    elif coord.size // n_pixels > 3:
        # factor = coord.size / span_pixels; the rounded pixel count n_pixels equals round(span_pixels).
        samples_per_pixel = int(np.round(coord.size / span_pixels))
    else:
        # oversampling == 0 means "no sub-grid", but the averaging still needs
        # >= 1 sample per pixel; fall back to interpolating straight onto the n_pixels output pixels.
        samples_per_pixel = max(1, int(coord.size / n_pixels * oversampling))
    # No coords on the weights array: alignment must be by dimension only.
    weights = xr.DataArray(_resample_weights(coord.values, n_pixels, samples_per_pixel), dims=("_center", axis))
    centers = np.linspace(coord[0].values, coord[-1].values, n_pixels)

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


def _fit_axis(ds: xr.Dataset, axis: str, target_size: int, pad_mode: str, *, extend: bool) -> xr.Dataset:
    """
    Crop ``ds`` to ``target_size`` points along ``axis``; pad it with ``pad_mode`` if
    ``extend``.
    """
    current_size = ds.sizes[axis]
    if current_size > target_size:
        return ds.isel({axis: np.arange(target_size)})
    if current_size == target_size or not extend:
        return ds
    if pad_mode == "wrap":
        # dask.array.pad silently clips a wrap pad wider than the axis; modular
        # indexing tiles any width and works on both numpy and dask backends.
        return ds.isel({axis: np.arange(target_size) % current_size})
    # xarray.pad defaults constant_values to NaN, which would poison every
    # downstream sum; a padded VDEM pixel means "no emission here", so fill
    # with zeros. Only "constant" accepts the argument.
    fill = {"constant_values": 0} if pad_mode == "constant" else {}
    return ds.pad({axis: (0, target_size - current_size)}, mode=pad_mode, **fill)


def _quarter_turn(vdem: xr.Dataset) -> xr.Dataset:
    """
    ``vdem`` turned a quarter counter-clockwise: the point at (x, y) moves to (-y, x).
    """
    turned = vdem.rename({"x": "x_turned"}).rename({"y": "x"}).rename({"x_turned": "y"})
    # The new x runs against the old y: the samples are reversed, and their positions are -y
    # shifted to the same origin, which keeps them ascending and right on an uneven grid.
    old_y = turned.x
    flipped = turned.isel(x=slice(None, None, -1)).assign_coords(
        x=("x", old_y.values[0] + old_y.values[-1] - old_y.values[::-1], old_y.attrs)
    )
    # The rename keeps every dimension in its place, which swaps x and y in a variable along both:
    # give it the caller's order back, so that the array and a plot of it are not mirrored.
    both = [name for name in vdem.data_vars if {"x", "y"} <= set(vdem[name].dims)]
    return flipped.assign({name: flipped[name].transpose(*vdem[name].dims) for name in both})


@format_docstring(
    "DEFAULTS_MUSE",
    pixel_width="dx_pixel_SG",
    pixel_height="dy_pixel_SG",
    number_of_slits="number_of_slits_SG",
    steps_per_raster="steps_per_raster_SG",
    extent="fov_extent",
    pad_mode="fov_pad_mode",
    oversampling="fov_oversampling",
)
@u.quantity_input(pixel_width=u.arcsec, pixel_height=u.arcsec, rotation=u.deg)
def match_fov(
    vdem: xr.Dataset,
    *,
    pixel_width: u.Quantity = DEFAULTS_MUSE.dx_pixel_SG,
    pixel_height: u.Quantity = DEFAULTS_MUSE.dy_pixel_SG,
    number_of_slits: int = DEFAULTS_MUSE.number_of_slits_SG,
    steps_per_raster: int = DEFAULTS_MUSE.steps_per_raster_SG,
    extent: str = DEFAULTS_MUSE.fov_extent,
    pad_mode: str = DEFAULTS_MUSE.fov_pad_mode,
    oversampling: int = DEFAULTS_MUSE.fov_oversampling,
    rotation: u.Quantity = 0 * u.deg,
    y_pixels: int | None = None,
) -> xr.Dataset:
    """
    Match a VDEM's spatial resolution and available extent to the MUSE FOV.

    Parameters
    ----------
    vdem : `xarray.Dataset`
        VDEM dataset. The ``x`` and ``y`` coordinates must define units.
    pixel_width : `astropy.units.Quantity`, optional
        Pixel size in arcsec along the x axis, by default is {pixel_width}
    pixel_height : `astropy.units.Quantity`, optional
        Pixel size in arcsec along the y axis, by default is {pixel_height}
    number_of_slits : `int`
        Number of slits, by default is {number_of_slits}.
    steps_per_raster : `int`
        Number of raster steps, by default is {steps_per_raster}.
    extent : `str`, optional
        How to treat the x width relative to ``number_of_slits * steps_per_raster``, by default
        {extent}. ``"extend"`` crops wider inputs and extends narrower ones
        with ``pad_mode``. ``"crop"`` crops wider inputs but retains a narrower
        input at its available width. ``"keep"`` leaves the width untouched
        and only matches the spatial resolution, for building the FOV of
        instruments whose width is not tied to the MUSE raster. Spatial
        resolution matching is always performed.
    pad_mode : `str`, optional
        How an axis is extended when ``extent="extend"``, by default {pad_mode}. See
        `xarray.DataArray.pad` for supported values.
        ``"wrap"`` repeats the input along the axis.
        ``"constant"`` fills the padded columns with zeros rather than the
        `xarray.DataArray.pad` default of NaN, so they read as "no emission here".
    oversampling : `int`, optional
        Sub-samples per input sample that a coarse input is interpolated onto
        before it is averaged into pixels, by default {oversampling}; 0
        interpolates straight onto the pixels.
    rotation : `astropy.units.Quantity`, optional
        Turn of the field counter-clockwise before it is matched, a multiple of
        90 degrees, by default 0 degrees. A quarter turn moves the point at
        (x, y) to (-y, x).
    y_pixels : `int`, optional
        Number of pixels along y in the output, counted after the resampling.
        More are cropped to the first ``y_pixels``. Fewer are extended with
        ``pad_mode`` when ``extent="extend"``, as x is, and kept otherwise. A
        single row, one position along the slit, is always kept. By default
        `None`, which leaves y as it is.

    Returns
    -------
    `xarray.Dataset`
        VDEM data resampled to the requested pixel sizes and cropped or extended
        along x as described by ``extent``, and along y when ``y_pixels`` is given.

    Notes
    -----
    Depending on the input grid, one of the following paths is taken:

    1. **Already matched** - if there is no rotation, the measurable spatial
       axes have the requested pixel spacing, and neither the x width nor the
       rows need a crop or extension, the input dataset is returned unchanged.
       After a rotation, matched axes skip resampling but continue through
       output finalization.
    2. **Single row or column** - if only one axis has more than one point, the
       resolution match is decided on that axis alone because the other spacing
       cannot be measured. The x extent still follows ``extent``.
    3. **Resample** - otherwise each axis is resampled onto the requested
       pixel size (interpolated up, integer-factor averaged down, or
       oversampled and averaged).
    4. **Match the x extent** - inputs wider than ``number_of_slits * steps_per_raster`` are
       cropped. Narrower inputs are extended with ``pad_mode`` only when
       ``extent="extend"``; ``"crop"`` retains their available width. Skipped
       entirely when ``extent="keep"``.
    5. **Single pixel** - a 1x1 input has no measurable spatial spacing. Its x
       extent is extended when ``extent="extend"`` and otherwise retained.
    6. **Match the y extent** - with ``y_pixels``, an output of more than one
       row is cropped to its first ``y_pixels`` rows whatever ``extent`` is,
       and extended to ``y_pixels`` rows with ``pad_mode`` only when
       ``extent="extend"``. This follows the resampling, so the count is in
       output pixels.
    """
    if extent not in ("extend", "crop", "keep"):
        msg = f"Unsupported extent {extent!r}; use 'extend', 'crop', or 'keep'"
        raise ValueError(msg)
    quarter_turns = rotation.to_value(u.deg) / 90
    # An absolute tolerance, for the rounding of a conversion: 6 hourangle is 0.9999999999999999 turns.
    if not rotation.isscalar or not np.isclose(quarter_turns, round(quarter_turns), rtol=0):
        msg = f"rotation must be one multiple of 90 degrees, got {rotation}"
        raise ValueError(msg)
    quarter_turns = round(quarter_turns) % 4
    if y_pixels is not None and y_pixels < 1:
        msg = f"y_pixels must be at least 1, got {y_pixels}"
        raise ValueError(msg)
    pixel_width = pixel_width.to("arcsec")
    pixel_height = pixel_height.to("arcsec")
    target_width = number_of_slits * steps_per_raster

    for _ in range(quarter_turns):
        vdem = _quarter_turn(vdem)

    x_spacing_matches = _spacing_matches(
        vdem.coords["x"], _coord_conversion_factor(vdem, "x", u.arcsec), pixel_width.value
    )
    y_spacing_matches = _spacing_matches(
        vdem.coords["y"], _coord_conversion_factor(vdem, "y", u.arcsec), pixel_height.value
    )
    resolution_matches = (x_spacing_matches and (y_spacing_matches or vdem.coords["y"].size < 2)) or (
        y_spacing_matches and vdem.coords["x"].size < 2
    )
    width_needs_no_change = (
        extent == "keep"
        or vdem.coords["x"].size == target_width
        or (extent == "crop" and vdem.coords["x"].size < target_width)
    )
    # The input rows are the ones to count here: the early return is only taken when nothing is resampled.
    ny = vdem.coords["y"].size
    rows_need_no_change = y_pixels is None or ny in (1, y_pixels) or (ny < y_pixels and extent != "extend")
    if quarter_turns == 0 and resolution_matches and width_needs_no_change and rows_need_no_change:
        logger.info("vdem already has the requested pixel size and needs no extent change")
        return vdem

    # Shallow copy: every transform below returns a new object; the copy only isolates
    # coord/attr mutations on the degenerate size-1 paths where resampling is a no-op.
    matched = vdem.copy()

    if not resolution_matches:
        matched = _resample_axis_to_pixel(matched, "x", pixel_width.value, oversampling)
        matched = _resample_axis_to_pixel(matched, "y", pixel_height.value, oversampling)
        # The unstack inside the resampling moves the resampled axis last; restore each
        # variable's original dimension order so downstream plots keep their orientation.
        matched = matched.assign({name: matched[name].transpose(*vdem[name].dims) for name in vdem.data_vars})

    if extent != "keep":
        matched = _fit_axis(matched, "x", target_width, pad_mode, extend=extent == "extend")
    if y_pixels is not None and matched.y.size > 1:
        matched = _fit_axis(matched, "y", y_pixels, pad_mode, extend=extent == "extend")

    matched.coords["x"] = np.arange(matched.x.size) * pixel_width.value
    matched.coords["y"] = np.arange(matched.y.size) * pixel_height.value
    matched.x.attrs["units"] = "arcsec"
    matched.y.attrs["units"] = "arcsec"
    for var in vdem.data_vars:
        matched[var].attrs.update(vdem[var].attrs)

    update_attrs(matched, vdem)
    add_history(matched, locals(), match_fov, sources=(vdem,))

    return matched


@format_docstring(
    "DEFAULTS_MUSE",
    number_of_slits="number_of_slits_SG",
    steps_per_raster="steps_per_raster_SG",
)
def reshape_x_to_slit_step(
    ds: xr.Dataset,
    number_of_slits: int = DEFAULTS_MUSE.number_of_slits_SG,
    steps_per_raster: int = DEFAULTS_MUSE.steps_per_raster_SG,
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
    number_of_slits : `int`
        Number of slits, by default is {number_of_slits}.
    steps_per_raster : `int`
        Number of raster steps, by default is {steps_per_raster}.

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
        step, slit = (array.flatten() for array in np.meshgrid(range(steps_per_raster), range(number_of_slits)))
        reshaped = ds.assign_coords(slit=("x", slit), step=("x", step))
        reshaped = reshaped.set_index(x=("slit", "step")).unstack("x")
        attrs = {"step_size": str(u.Quantity(step_size.values, x_unit))}
    update_attrs(reshaped, ds, **attrs)
    add_history(reshaped, locals(), reshape_x_to_slit_step, sources=(ds,))
    return reshaped


def reshape_slit_step_to_x(ds: xr.Dataset) -> xr.Dataset:
    """
    Inverse of `reshape_x_to_slit_step`: collapse the slit and raster step axes back
    onto a single x spatial axis.

    The x coordinate is rebuilt from the per-step spacing recorded in
    ``ds.attrs["step_size"]``, such as ``"0.4 arcsec"`` (set by
    `reshape_x_to_slit_step`); when that attribute is missing the default MUSE
    pixel size is used.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Data with ``slit`` and ``step`` coordinates.

    Returns
    -------
    `xarray.Dataset`
        vdem or spectra with a single x spatial axis.
    """
    for coord in ("slit", "step"):
        if coord not in ds.coords:
            msg = f"{coord} coordinate is missing"
            raise ValueError(msg)
    step_size = u.Quantity(ds.attrs.get("step_size", DEFAULTS_MUSE.dx_pixel_SG))
    if step_size.unit == u.dimensionless_unscaled:
        msg = f"step_size {ds.attrs['step_size']!r} has no unit; it is one value with its unit, such as '0.4 arcsec'"
        raise ValueError(msg)
    reshaped = ds.stack(x=("slit", "step"))
    reshaped = reshaped.drop_vars(["x", "slit", "step"])
    reshaped = reshaped.assign_coords(x=np.arange(reshaped.x.size) * step_size.value)
    reshaped.x.attrs["units"] = str(step_size.unit)
    update_attrs(reshaped, ds)
    add_history(reshaped, locals(), reshape_slit_step_to_x, sources=(ds,))
    return reshaped
