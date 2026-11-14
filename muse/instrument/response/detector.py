"""
Map wavelength-space responses onto detectors.
"""

import numbers

import numpy as np
import xarray as xr

import astropy.units as u

from muse.utils.utils import _require_increasing_axis, add_history, coord_as_unit, require_unit
from muse.variables import DEFAULTS_MUSE

__all__ = ["map_response_to_ci_detector", "map_response_to_sg_detector"]

# Wavelength-space response units these mappings accept.
_ACCEPTED_SG_RESPONSE_UNITS = tuple(unit * u.cm**5 / (u.AA * u.s) for unit in (u.erg / u.sr, u.ph, u.DN))
_ACCEPTED_CI_RESPONSE_UNITS = tuple(
    unit * emission_measure / (u.AA * u.s)
    for unit in (u.erg / u.sr, u.ph, u.DN)
    for emission_measure in (u.cm**3, u.cm**5)
)


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


def _validate_wavelength_response(
    response: xr.Dataset, *, accepted_units: tuple[u.UnitBase, ...] = _ACCEPTED_SG_RESPONSE_UNITS
) -> tuple[u.UnitBase, xr.DataArray, xr.DataArray]:
    if not isinstance(response, xr.Dataset):
        msg = "response must be an xarray.Dataset"
        raise TypeError(msg)
    required = {"spectral_response", "wavelength_grid", "line_wavelength"}
    missing = sorted(required - set(response.variables))
    if missing:
        msg = f"response is missing required variables: {', '.join(missing)}"
        raise ValueError(msg)
    if "wavelength_bin" not in response.spectral_response.dims:
        msg = "response.spectral_response must include a wavelength_bin dimension"
        raise ValueError(msg)
    if "line" not in response.spectral_response.dims:
        msg = "response.spectral_response must include a line dimension"
        raise ValueError(msg)
    if response.wavelength_grid.dims != ("wavelength_bin",):
        msg = "response.wavelength_grid must be one-dimensional along wavelength_bin"
        raise ValueError(msg)
    if response.line_wavelength.dims != ("line",):
        msg = "response.line_wavelength must be one-dimensional along line"
        raise ValueError(msg)

    normalization = response.attrs.get("normalization")
    if not isinstance(normalization, numbers.Real) or not np.isfinite(normalization) or normalization <= 0:
        msg = "response normalization must be a finite, positive number"
        raise ValueError(msg)
    response_unit = require_unit(response, "spectral_response", "response.spectral_response")
    if not any(response_unit.is_equivalent(accepted) for accepted in accepted_units):
        accepted = ", ".join(str(unit) for unit in accepted_units)
        msg = f"response.spectral_response units must be convertible to one of {accepted}"
        raise ValueError(msg)
    wavelength_grid = coord_as_unit(response, "wavelength_grid", u.AA, "response.wavelength_grid")
    line_wavelength = coord_as_unit(response, "line_wavelength", u.AA, "response.line_wavelength")
    _require_increasing_axis(wavelength_grid, "response.wavelength_grid", positive=True)
    return response_unit, wavelength_grid, line_wavelength


@u.quantity_input(
    channel=u.AA,
    dispersion=u.AA / u.pix,
    slit_spacing=u.pix,
    wavelength_start=u.AA,
)
def map_response_to_sg_detector(
    response: xr.Dataset,
    *,
    channel: u.Quantity | None = None,
    number_of_slits: int | None = None,
    dispersion: u.Quantity | None = None,
    slit_spacing: u.Quantity | None = None,
    detector_pixels: int | None = None,
    wavelength_start: u.Quantity | None = None,
) -> xr.Dataset:
    """
    Map one wavelength-space response onto a multi-slit detector.

    Pass ``channel`` to use the MUSE SG calibration, or provide the detector
    geometry explicitly. The input should come from
    `muse.instrument.create_spectral_response` with effective area(s) already
    applied.

    .. warning::

        Radiometric units pass through, so convert them first
        with `muse.instrument.transform_response_units`.

    Parameters
    ----------
    response : `xarray.Dataset`
        Wavelength-space response containing ``spectral_response``,
        ``wavelength_grid``, and ``line_wavelength``.
    channel : `astropy.units.Quantity`, optional
        Nominal MUSE SG channel wavelength: 108, 171, or 284 Angstrom. When
        given, fills any omitted geometry parameters from
        `~muse.variables.DEFAULTS_MUSE` and records the channel on the output.
    number_of_slits : `int`, optional
        Number of simultaneous slits. Required when ``channel`` is omitted.
    dispersion : `astropy.units.Quantity`, optional
        Nominal wavelength width per detector pixel, also used for detector-bin
        integration. When ``channel`` is given, derive it from the channel's
        spectral order and the MUSE slit calibration by default. Otherwise it
        is required. Adjacent detector-pixel centers are separated by exactly
        ``dispersion``.
    slit_spacing : `astropy.units.Quantity`, optional
        Detector pixels between adjacent slits. Required for an explicit
        multi-slit geometry; unused for a single slit.
    detector_pixels : `int`, optional
        Number of spectral detector pixels. Required when ``channel`` is
        omitted.
    wavelength_start : `astropy.units.Quantity`, optional
        Wavelength at detector pixel zero for slit zero. Required when
        ``channel`` is omitted.

    Returns
    -------
    `xarray.Dataset`
        Detector response containing ``detector_response`` in the units of
        ``response.spectral_response`` integrated over the detector pixel
        width, so per Angstrom becomes per pixel, and
        ``detector_wavelength`` and ``line_wavelength`` coordinates
        in Angstrom.
    """
    response_unit, wavelength_grid, line_wavelength = _validate_wavelength_response(response)
    line_wavelength = np.asarray(line_wavelength)
    channel_value = None
    spectral_order = None
    if channel is not None:
        channel_value = _channel_as_angstrom(channel, DEFAULTS_MUSE.channel_spectral_order_SG.channel, "sg")
        spectral_order = DEFAULTS_MUSE.channel_spectral_order_SG.sel(channel=channel_value).item()
        number_of_slits = DEFAULTS_MUSE.number_of_slits_SG if number_of_slits is None else number_of_slits
        slit_spacing = DEFAULTS_MUSE.pixels_between_slits_SG if slit_spacing is None else slit_spacing
        detector_pixels = int(DEFAULTS_MUSE.pixels_SG.to_value(u.pix)) if detector_pixels is None else detector_pixels
        wavelength_start = (
            u.Quantity(DEFAULTS_MUSE.initial_wavelength_SG.sel(channel=channel_value).data)
            if wavelength_start is None
            else wavelength_start
        )
    else:
        missing = [
            name
            for name, value in (
                ("number_of_slits", number_of_slits),
                ("dispersion", dispersion),
                ("detector_pixels", detector_pixels),
                ("wavelength_start", wavelength_start),
            )
            if value is None
        ]
        if isinstance(number_of_slits, numbers.Integral) and number_of_slits > 1 and slit_spacing is None:
            missing.append("slit_spacing")
        if missing:
            msg = f"channel or explicit detector geometry is required; missing: {', '.join(missing)}"
            raise ValueError(msg)

    if "component_kind" in response.coords:
        component_kind = np.asarray(response.component_kind)
        valid_lines = np.isfinite(line_wavelength) & (line_wavelength > 0)
        missing_contaminants = ~valid_lines & (component_kind == "contaminants")
        physical_lines = valid_lines & (component_kind == "line")
        if missing_contaminants.any() and physical_lines.any():
            line_wavelength = np.where(
                missing_contaminants,
                line_wavelength[physical_lines][0],
                line_wavelength,
            )
    for name, value in (("number_of_slits", number_of_slits), ("detector_pixels", detector_pixels)):
        if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)) or value <= 0:
            msg = f"{name} must be a positive integer"
            raise ValueError(msg)

    for name, value in (
        ("slit_spacing", slit_spacing),
        ("wavelength_start", wavelength_start),
    ):
        if value is None:
            continue
        if not value.isscalar or not np.isfinite(value.value) or value.value <= 0:
            msg = f"{name} must be a finite, positive scalar"
            raise ValueError(msg)
    if dispersion is None:
        dispersion = 2 * DEFAULTS_MUSE.spectral_slit_separation_SG / slit_spacing / spectral_order
    if not dispersion.isscalar or not np.isfinite(dispersion.value) or dispersion.value <= 0:
        msg = "dispersion must be a finite, positive scalar"
        raise ValueError(msg)

    dispersion_value = dispersion.to_value(u.AA / u.pix)
    slit_offset = 0 if slit_spacing is None else slit_spacing.to_value(u.pix) * dispersion_value
    detector_start = wavelength_start.to_value(u.AA)
    detector_wavelength_values = detector_start + np.arange(detector_pixels) * dispersion_value
    detector_wavelength = xr.DataArray(
        detector_wavelength_values[:, np.newaxis] - np.arange(number_of_slits)[np.newaxis, :] * slit_offset,
        dims=("detector_x_pixel", "slit"),
        coords={"slit": np.arange(number_of_slits), "detector_x_pixel": np.arange(detector_pixels)},
        attrs={"units": str(u.AA)},
    )

    spectral_response = response.spectral_response.assign_coords(wavelength_grid=wavelength_grid)
    mapped = (
        spectral_response.swap_dims(wavelength_bin="wavelength_grid")
        .interp(wavelength_grid=detector_wavelength, kwargs={"fill_value": 0})
        .rename(wavelength_grid="detector_wavelength")
    )
    mapped = mapped * dispersion_value
    leading_dims = [dim for dim in mapped.dims if dim not in ("slit", "detector_x_pixel")]
    mapped = mapped.transpose(*leading_dims, "slit", "detector_x_pixel").assign_coords(
        detector_wavelength=detector_wavelength
    )
    mapped.attrs["units"] = str(response_unit * u.AA)
    mapped.detector_wavelength.attrs["units"] = str(u.AA)

    result = response.drop_dims("wavelength_bin")
    result = result.assign(detector_response=mapped).assign_coords(
        line_wavelength=("line", line_wavelength, {"units": str(u.AA)}),
    )
    if channel_value is not None:
        result = result.assign_coords(
            channel=("line", np.full(response.sizes["line"], channel_value), {"units": str(u.AA)})
        )
    add_history(result, locals(), map_response_to_sg_detector)
    return result


@u.quantity_input(channel=u.AA)
def map_response_to_ci_detector(response: xr.Dataset, channel: u.Quantity) -> xr.Dataset:
    """
    Integrate one wavelength-space response over an imaging band.

    Parameters
    ----------
    response : `xarray.Dataset`
        Wavelength-space response containing ``spectral_response``,
        ``wavelength_grid``, and ``line_wavelength``. The wavelength grid may
        be nonuniform.
    channel : `astropy.units.Quantity`
        Nominal channel wavelength recorded in the output coordinates.

    Returns
    -------
    `xarray.Dataset`
        Band-integrated response containing ``detector_response`` in the units
        of ``response.spectral_response`` integrated over wavelength, with
        ``channel`` and ``detector_wavelength`` coordinates along ``line``.
    """
    response_unit, wavelength_grid, line_wavelength = _validate_wavelength_response(
        response, accepted_units=_ACCEPTED_CI_RESPONSE_UNITS
    )
    line_wavelength = np.asarray(line_wavelength)
    channel_value = _channel_as_angstrom(channel, None, "ci")
    if wavelength_grid.size < 2:
        msg = "response.wavelength_grid must contain at least two points for integration"
        raise ValueError(msg)

    spectral_response = response.spectral_response.assign_coords(wavelength_grid=wavelength_grid)
    mapped = spectral_response.integrate("wavelength_grid")
    mapped.attrs = {**response.spectral_response.attrs, "units": str(response_unit * u.AA)}

    result = response.drop_dims("wavelength_bin")
    result = result.assign(detector_response=mapped).assign_coords(
        line_wavelength=("line", line_wavelength, {"units": str(u.AA)}),
        channel=("line", np.full(response.sizes["line"], channel_value), {"units": str(u.AA)}),
        detector_wavelength=(
            "line",
            np.full(response.sizes["line"], channel_value, dtype=float),
            {"units": str(u.AA)},
        ),
    )
    add_history(result, locals(), map_response_to_ci_detector)
    return result
