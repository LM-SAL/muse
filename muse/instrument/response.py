"""
Map wavelength-space responses onto the MUSE SG detector.
"""

import numbers

import numpy as np
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.utils.utils import add_history, coord_as_unit, require_unit
from muse.variables import DEFAULTS_MUSE

__all__ = ["map_response_to_sg_detector"]


@u.quantity_input(
    dispersion=u.AA / u.pix,
    slit_spacing=u.pix,
    wavelength_start=u.AA,
    pixel_width=u.arcsec,
    pixel_height=u.arcsec,
)
def map_response_to_sg_detector(
    response: xr.Dataset,
    channel: int,
    *,
    number_of_slits: int | None = None,
    dispersion: u.Quantity | None = None,
    slit_spacing: u.Quantity | None = None,
    detector_pixels: int | None = None,
    wavelength_start: u.Quantity | None = None,
    pixel_width: u.Quantity = DEFAULTS_MUSE.dx_pixel_SG,
    pixel_height: u.Quantity = DEFAULTS_MUSE.dy_pixel_SG,
) -> xr.Dataset:
    """
    Map one wavelength-space response onto the MUSE SG detector.

    One call maps one MUSE channel and spectral order. The input should come
    from `muse.instrument.spectral.create_spectral_response` with effective
    area already applied.

    Parameters
    ----------
    response : `xarray.Dataset`
        Wavelength-space response containing ``spectral_response``,
        ``wavelength_grid``, and ``line_wavelength``.
    channel : `int`
        MUSE SG channel: 108, 171, or 284 Angstrom.
    number_of_slits : `int`, optional
        Number of simultaneous slits. If `None`, use the MUSE default.
    dispersion : `astropy.units.Quantity`, optional
        Nominal wavelength width per detector pixel, also used for detector-bin
        integration. If `None`, derive it from the channel's spectral order and
        the MUSE slit calibration. For compatibility with existing MUSE
        response products, the detector grid spans ``detector_pixels *
        dispersion`` across ``detector_pixels`` samples.
    slit_spacing : `astropy.units.Quantity`, optional
        Detector pixels between adjacent slits. If `None`, use the MUSE
        default.
    detector_pixels : `int`, optional
        Number of spectral detector pixels. If `None`, use the MUSE default.
    wavelength_start : `astropy.units.Quantity`, optional
        Wavelength at detector pixel zero for slit zero. If `None`, use the
        channel calibration from `~muse.variables.DEFAULTS_MUSE`.
    pixel_width, pixel_height : `astropy.units.Quantity`, optional
        SG pixel angular size used to convert steradians to detector pixels,
        by default the MUSE SG pixel dimensions.

    Returns
    -------
    `xarray.Dataset`
        Detector response containing ``detector_response`` in photon-response
        units and ``detector_wavelength`` and ``line_wavelength`` coordinates
        in Angstrom. The generic
        ``doppler_velocity`` dimension is renamed to the legacy ``vdop`` name
        required by MUSE synthesis.
    """
    if not isinstance(response, xr.Dataset):
        msg = "response must be an xarray.Dataset"
        raise TypeError(msg)
    if not isinstance(channel, numbers.Integral) or isinstance(channel, (bool, np.bool_)):
        msg = "channel must be an integer"
        raise TypeError(msg)
    channel = int(channel)
    try:
        spectral_order = DEFAULTS_MUSE.channel_spectral_order.sel(channel=channel).item()
        default_wavelength_start = u.Quantity(DEFAULTS_MUSE.initial_wavelength_SG.sel(channel=channel).data)
    except KeyError:
        msg = f"unsupported MUSE SG channel {channel}"
        raise ValueError(msg) from None

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
    density_unit = normalization * u.erg * u.cm**5 / (u.AA * u.s * u.sr)
    response_unit = require_unit(
        response,
        "spectral_response",
        "response.spectral_response",
        convertible_to=density_unit,
    )
    wavelength_grid = coord_as_unit(response, "wavelength_grid", u.AA, "response.wavelength_grid")
    line_wavelength = coord_as_unit(response, "line_wavelength", u.AA, "response.line_wavelength")
    wavelength_values = np.asarray(wavelength_grid)
    if not np.all(np.isfinite(wavelength_values)) or np.any(np.diff(wavelength_values) <= 0):
        msg = "response.wavelength_grid must be finite and strictly increasing"
        raise ValueError(msg)

    if number_of_slits is None:
        number_of_slits = DEFAULTS_MUSE.number_of_slits_SG
    if detector_pixels is None:
        detector_pixels = int(DEFAULTS_MUSE.pixels_SG.to_value(u.pix))
    for name, value in (("number_of_slits", number_of_slits), ("detector_pixels", detector_pixels)):
        if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)) or value <= 0:
            msg = f"{name} must be a positive integer"
            raise ValueError(msg)

    if slit_spacing is None:
        slit_spacing = DEFAULTS_MUSE.pixels_between_slits
    if wavelength_start is None:
        wavelength_start = default_wavelength_start
    for name, value in (
        ("slit_spacing", slit_spacing),
        ("wavelength_start", wavelength_start),
        ("pixel_width", pixel_width),
        ("pixel_height", pixel_height),
    ):
        if not value.isscalar or not np.isfinite(value.value) or value.value <= 0:
            msg = f"{name} must be a finite, positive scalar"
            raise ValueError(msg)
    if dispersion is None:
        dispersion = 2 * DEFAULTS_MUSE.spectral_slit_separation_SG / slit_spacing / spectral_order
    if not dispersion.isscalar or not np.isfinite(dispersion.value) or dispersion.value <= 0:
        msg = "dispersion must be a finite, positive scalar"
        raise ValueError(msg)

    dispersion_value = dispersion.to_value(u.AA / u.pix)
    slit_offset = slit_spacing.to_value(u.pix) * dispersion_value
    detector_start = wavelength_start.to_value(u.AA)
    detector_wavelength_values = np.linspace(
        detector_start,
        detector_start + detector_pixels * dispersion_value,
        detector_pixels,
    )
    detector_wavelength = xr.DataArray(
        detector_wavelength_values[:, np.newaxis] - np.arange(number_of_slits)[np.newaxis, :] * slit_offset,
        dims=("detector_x_pixel", "slit"),
        coords={"slit": np.arange(number_of_slits), "detector_x_pixel": np.arange(detector_pixels)},
        attrs={"units": str(u.AA)},
    )

    spectral_response = response.spectral_response * response_unit.to(density_unit)
    photon_energy = xr.DataArray(
        (const.h * const.c / (wavelength_values * u.AA)).to_value(u.erg),
        dims="wavelength_bin",
    )
    pixel_solid_angle = pixel_width.to_value(u.rad) * pixel_height.to_value(u.rad)
    spectral_response = spectral_response * pixel_solid_angle / photon_energy
    spectral_response = spectral_response.assign_coords(wavelength_grid=wavelength_grid)
    mapped = (
        spectral_response.swap_dims(wavelength_bin="wavelength_grid")
        .interp(wavelength_grid=detector_wavelength)
        .fillna(0)
        .rename(wavelength_grid="detector_wavelength")
    )
    mapped = mapped * dispersion_value
    leading_dims = [dim for dim in mapped.dims if dim not in ("slit", "detector_x_pixel")]
    mapped = mapped.transpose(*leading_dims, "slit", "detector_x_pixel").assign_coords(
        detector_wavelength=detector_wavelength
    )
    mapped.attrs["units"] = str(normalization * u.ph * u.cm**5 / u.s)
    mapped.detector_wavelength.attrs["units"] = str(u.AA)

    line_wavelength = np.asarray(line_wavelength)
    finite_lines = np.isfinite(line_wavelength)
    if not finite_lines.all():
        physical_lines = finite_lines
        if "component_kind" in response.coords:
            physical_lines = finite_lines & (np.asarray(response.component_kind) == "line")
        if not physical_lines.any():
            msg = "response must contain a physical line wavelength"
            raise ValueError(msg)
        line_wavelength = np.where(physical_lines, line_wavelength, line_wavelength[physical_lines][0])

    result = response.drop_dims("wavelength_bin")
    result = result.assign(detector_response=mapped).assign_coords(
        line_wavelength=("line", line_wavelength, {"units": str(u.AA)}),
        channel=("line", np.full(response.sizes["line"], channel)),
    )
    if "doppler_velocity" in result.dims:
        result = result.rename(doppler_velocity="vdop")
    add_history(result, locals(), map_response_to_sg_detector)
    return result
