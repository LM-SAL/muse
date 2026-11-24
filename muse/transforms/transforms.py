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

    Returns `False` for a size-1 axis, whose spacing cannot be measured.
    """
    if coord.size < 2:
        return False
    spacing = (coord[1] - coord[0]).data * to_arcsec
    return bool(abs(spacing - pixel_arcsec) / pixel_arcsec < 0.005)


def _resample_weights(coord_values: np.ndarray, n_pixels: int, samples_per_pixel: int) -> np.ndarray:
    """
    Matrix mapping samples at ``coord_values`` onto ``n_pixels`` output pixels.

    Row ``i`` linearly interpolates the input onto ``samples_per_pixel`` evenly spaced
    points in pixel ``i`` and averages them, so one product interpolates and averages.
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
    returned unchanged; ``coord`` must be strictly increasing and roughly evenly spaced.

    The pipeline is one linear map applied per block, so dask inputs stay lazy and
    chained x/y resampling streams. Only data variables along ``axis`` are resampled;
    the ``axis`` coordinate and the other coordinates along it are dropped (`match_fov`
    rebuilds ``x`` and ``y``).
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
        samples_per_pixel = 1
    elif coord.size // n_pixels > 3:
        samples_per_pixel = int(np.round(coord.size / span_pixels))
    else:
        # oversampling=0 still needs one sample per pixel: interpolate straight onto the pixels.
        samples_per_pixel = max(1, int(coord.size / n_pixels * oversampling))
    # No coords on the weights array: alignment must be by dimension only.
    weights = xr.DataArray(_resample_weights(coord.values, n_pixels, samples_per_pixel), dims=("_center", axis))

    def apply_weights(block: np.ndarray, weight_block: np.ndarray, dtype=None) -> np.ndarray:
        return (block @ weight_block.T).astype(dtype)

    # Not xr.dot: two chained chunked einsums hold the whole intermediate (~7 GB on the tutorial VDEM).
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
    return ds.drop_dims(axis).assign(resampled)


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
        # dask.array.pad clips a wrap wider than the axis; modular indexing tiles any width.
        return ds.isel({axis: np.arange(target_size) % current_size})
    # Zeros, not xarray's NaN: a padded pixel has no emission. Only "constant" takes the argument.
    fill = {"constant_values": 0} if pad_mode == "constant" else {}
    return ds.pad({axis: (0, target_size - current_size)}, mode=pad_mode, **fill)


def _quarter_turn(vdem: xr.Dataset) -> xr.Dataset:
    """
    ``vdem`` turned a quarter counter-clockwise: the point at (x, y) moves to (-y, x).
    """
    turned = vdem.rename({"x": "x_turned"}).rename({"y": "x"}).rename({"x_turned": "y"})
    # The new x is the old -y shifted to the same origin: ascending, and right on an uneven grid.
    old_y = turned.x
    flipped = turned.isel(x=slice(None, None, -1)).assign_coords(
        x=("x", old_y.values[0] + old_y.values[-1] - old_y.values[::-1], old_y.attrs)
    )
    # The rename swaps x and y in a variable along both; restore the order so plots are not mirrored.
    both = [name for name in vdem.data_vars if {"x", "y"} <= set(vdem[name].dims)]
    return flipped.assign({name: flipped[name].transpose(*vdem[name].dims) for name in both})


@format_docstring(
    "DEFAULTS_MUSE",
    pixel_width="pixel_width_SG",
    pixel_height="pixel_height_SG",
    number_of_slits="number_of_slits_SG",
    steps_per_raster="steps_per_raster_SG",
)
@u.quantity_input(pixel_width=u.arcsec, pixel_height=u.arcsec, rotation=u.deg)
def match_fov(
    vdem: xr.Dataset,
    *,
    pixel_width: u.Quantity = DEFAULTS_MUSE.pixel_width_SG,
    pixel_height: u.Quantity = DEFAULTS_MUSE.pixel_height_SG,
    number_of_slits: int = DEFAULTS_MUSE.number_of_slits_SG,
    steps_per_raster: int = DEFAULTS_MUSE.steps_per_raster_SG,
    extent: str = "extend",
    pad_mode: str = "wrap",
    oversampling: int = 2,
    rotation: u.Quantity = 0 * u.deg,
    y_pixels: int | None = None,
) -> xr.Dataset:
    """
    Match a VDEM's spatial resolution and extent to the MUSE FOV.

    Parameters
    ----------
    vdem : `xarray.Dataset`
        VDEM dataset. The ``x`` and ``y`` coordinates must define units; coordinates in
        cm are converted to arcsec at 1 AU.
    pixel_width : `astropy.units.Quantity`, optional
        Pixel size along x, by default {pixel_width}.
    pixel_height : `astropy.units.Quantity`, optional
        Pixel size along y, by default {pixel_height}.
    number_of_slits : `int`
        Number of slits, by default {number_of_slits}.
    steps_per_raster : `int`
        Number of raster steps, by default {steps_per_raster}.
    extent : `str`, optional
        How to treat the x width against ``number_of_slits * steps_per_raster``, by
        default ``"extend"``, which crops wider inputs and extends narrower ones with
        ``pad_mode``. ``"crop"`` crops wider inputs and keeps narrower ones. ``"keep"``
        leaves the width alone, for instruments whose width is not tied to the MUSE
        raster. The resolution is always matched.
    pad_mode : `str`, optional
        How an axis is extended, by default ``"wrap"``, which repeats the input. See
        `xarray.DataArray.pad` for the others; ``"constant"`` fills with zeros, not NaN.
    oversampling : `int`, optional
        Sub-samples per input sample that a coarse input is interpolated onto before it is
        averaged into pixels, by default 2; 0 interpolates straight onto the pixels.
    rotation : `astropy.units.Quantity`, optional
        Counter-clockwise rotation of the field before matching, a multiple of 90 degrees,
        by default 0 degrees. A quarter turn moves the point at (x, y) to (-y, x).
    y_pixels : `int`, optional
        Output pixels along y, counted after resampling. More are cropped to the first
        ``y_pixels``; fewer are extended with ``pad_mode`` when ``extent="extend"`` and
        kept otherwise. A single row is always kept. By default `None`, which leaves y as
        it is.

    Returns
    -------
    `xarray.Dataset`
        The VDEM resampled to the pixel sizes, and cropped or extended along x by
        ``extent`` and along y by ``y_pixels``.

    Notes
    -----
    An input whose measurable axes already have the pixel spacing is not resampled, and is
    returned unchanged when it needs no rotation, crop or extension. When one axis has a
    single point, the resolution is decided on the other alone; a size-1 axis is never
    resampled. Resampling interpolates up, averages down by an integer factor, or
    oversamples and averages, before the crop or extension.
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
    # Input rows: the early return is only taken when nothing is resampled.
    ny = vdem.coords["y"].size
    rows_need_no_change = y_pixels is None or ny in (1, y_pixels) or (ny < y_pixels and extent != "extend")
    if quarter_turns == 0 and resolution_matches and width_needs_no_change and rows_need_no_change:
        logger.info("vdem already has the requested pixel size and needs no extent change")
        return vdem

    # Shallow copy: guards the coord and attr writes below when nothing is resampled.
    matched = vdem.copy()

    if not resolution_matches:
        matched = _resample_axis_to_pixel(matched, "x", pixel_width.value, oversampling)
        matched = _resample_axis_to_pixel(matched, "y", pixel_height.value, oversampling)
        # The resampling moves its axis last; restore the order so plots keep their orientation.
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
    *,
    number_of_slits: int = DEFAULTS_MUSE.number_of_slits_SG,
    steps_per_raster: int = DEFAULTS_MUSE.steps_per_raster_SG,
) -> xr.Dataset:
    """
    Reshape the x axis of a VDEM or spectrum into ``slit`` and ``step`` axes.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Data along x. The ``x`` coordinate must define units.
    number_of_slits : `int`
        Number of slits, by default {number_of_slits}.
    steps_per_raster : `int`
        Number of raster steps, by default {steps_per_raster}.

    Returns
    -------
    `xarray.Dataset`
        ``ds`` with ``slit`` and ``step`` axes in place of ``x``.
    """
    x_unit = require_unit(ds, "x", "x coordinate", coord_only=True)
    step_size = ds.x[1] - ds.x[0]
    step, slit = (array.flatten() for array in np.meshgrid(range(steps_per_raster), range(number_of_slits)))
    reshaped = ds.assign_coords(slit=("x", slit), step=("x", step))
    reshaped = reshaped.set_index(x=("slit", "step")).unstack("x")
    update_attrs(reshaped, ds, step_size=str(u.Quantity(step_size.values, x_unit)))
    add_history(reshaped, locals(), reshape_x_to_slit_step, sources=(ds,))
    return reshaped


def reshape_slit_step_to_x(ds: xr.Dataset) -> xr.Dataset:
    """
    Collapse the ``slit`` and ``step`` axes onto x, undoing `reshape_x_to_slit_step`.

    The x spacing is ``ds.attrs["step_size"]``, such as ``"0.4 arcsec"``, which
    `reshape_x_to_slit_step` sets, or else the default MUSE pixel width.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Data with ``slit`` and ``step`` coordinates.

    Returns
    -------
    `xarray.Dataset`
        ``ds`` with a single x axis.
    """
    for coord in ("slit", "step"):
        if coord not in ds.coords:
            msg = f"{coord} coordinate is missing"
            raise ValueError(msg)
    step_size = u.Quantity(ds.attrs.get("step_size", DEFAULTS_MUSE.pixel_width_SG))
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
