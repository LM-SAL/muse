"""
Radiometric unit conversions.
"""

import numpy as np
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.instrument.response import _channel_as_angstrom
from muse.utils.documentation import format_docstring
from muse.utils.utils import add_history, coord_as_unit, require_unit
from muse.variables import DEFAULTS_MUSE

__all__ = ["transform_response_units"]


def _unit_power(unit, base) -> float:
    """
    Return the exponent of ``base`` in ``unit``, or zero when ``base`` is absent.
    """
    unit = u.Unit(unit)
    return unit.powers[unit.bases.index(base)] if base in unit.bases else 0


@format_docstring(
    "DEFAULTS_MUSE",
    pixel_width_sg="dx_pixel_SG",
    pixel_height_sg="dy_pixel_SG",
    pixel_width_ci="dx_pixel_CI",
    pixel_height_ci="dy_pixel_CI",
)
@u.quantity_input(
    channel=u.AA,
    pixel_width=u.arcsec,
    pixel_height=u.arcsec,
    gain=u.electron / u.DN,
    pair_energy=u.eV / u.electron,
)
def transform_response_units(
    response: xr.Dataset,
    new_units: str,
    channel: u.Quantity | None = None,
    *,
    detector: str = "sg",
    pixel_width: u.Quantity | None = None,
    pixel_height: u.Quantity | None = None,
    gain: u.Quantity | None = None,
    pair_energy: u.Quantity | None = None,
) -> xr.Dataset:
    """
    Convert the radiometric units of a wavelength-space response.

    The conversion is: detector chain energy -> photon -> electron -> data
    number (DN), and converts a per-steradian response into a per-detector-pixel one,
    as required by ``new_units``.

    .. warning::

        Do this before `muse.instrument.map_response_to_sg_detector` or
        `muse.instrument.map_response_to_ci_detector`, which pass the units
        through.

    Parameters
    ----------
    response : `xarray.Dataset`
        Wavelength-space response containing ``spectral_response`` and
        ``wavelength_grid``, e.g. from
        `muse.instrument.create_spectral_response`.
    new_units : `str`
        Target units, as an `astropy.units.Unit` string, e.g.,
        ``"1e-27 cm5 ph / (Angstrom s)"``.
    channel : `astropy.units.Quantity`, optional
        Nominal MUSE channel wavelength. Selects the detector-specific
        per-channel ``gain`` and ``pair_energy`` calibrations, so it is
        required unless both are given explicitly, e.g. for a non-MUSE
        instrument.
    detector : {{"sg", "ci"}}, optional
        Detector whose default calibration to use, by default ``"sg"``.
    pixel_width, pixel_height : `astropy.units.Quantity`, optional
        Pixel angular size used to convert steradians to detector pixels. If
        `None`, use {pixel_width_sg} by {pixel_height_sg} for SG or
        {pixel_width_ci} by {pixel_height_ci} for CI.
    gain : `astropy.units.Quantity`, optional
        Camera gain, convertible to electron/DN. If `None`, use the channel's
        detector-specific value from `~muse.variables.DEFAULTS_MUSE`.
    pair_energy : `astropy.units.Quantity`, optional
        Mean energy that frees one electron-hole pair in silicon. If `None`,
        use the channel's detector-specific value from
        `~muse.variables.DEFAULTS_MUSE`.

    Returns
    -------
    `xarray.Dataset`
        Copy of ``response`` whose ``spectral_response`` is scaled to
        ``new_units``.
    """
    if not isinstance(response, xr.Dataset):
        msg = "response must be an xarray.Dataset"
        raise TypeError(msg)
    if detector not in ("sg", "ci"):
        msg = "detector must be 'sg' or 'ci'"
        raise ValueError(msg)
    old_unit = require_unit(response, "spectral_response", "response.spectral_response")
    target_unit = u.Unit(new_units)

    if detector == "sg":
        default_pixel_width = DEFAULTS_MUSE.dx_pixel_SG
        default_pixel_height = DEFAULTS_MUSE.dy_pixel_SG
        gains = DEFAULTS_MUSE.ccd_gain_sg
        pair_energies = DEFAULTS_MUSE.pair_creation_energy_sg
        channel_dimension = "channel"
    else:
        default_pixel_width = DEFAULTS_MUSE.dx_pixel_CI
        default_pixel_height = DEFAULTS_MUSE.dy_pixel_CI
        gains = DEFAULTS_MUSE.ccd_gain_ci
        pair_energies = DEFAULTS_MUSE.pair_creation_energy_ci
        channel_dimension = "ci_channel"
    pixel_width = default_pixel_width if pixel_width is None else pixel_width
    pixel_height = default_pixel_height if pixel_height is None else pixel_height

    if gain is None or pair_energy is None:
        if channel is None:
            msg = "channel is required unless both gain and pair_energy are given"
            raise ValueError(msg)
        channel_value = _channel_as_angstrom(channel, pair_energies[channel_dimension], detector)
        try:
            if gain is None:
                gain = u.Quantity(gains.sel({channel_dimension: channel_value}).data)
            if pair_energy is None:
                pair_energy = u.Quantity(pair_energies.sel({channel_dimension: channel_value}).data)
        except KeyError:
            msg = f"unsupported MUSE {detector.upper()} channel {channel}"
            raise ValueError(msg) from None
    wavelength = coord_as_unit(response, "wavelength_grid", u.AA, "response.wavelength_grid")

    photon_energy = (const.h * const.c / (np.asarray(wavelength) * u.AA)).to(u.erg) / u.ph
    electrons_per_photon = (photon_energy * u.ph / pair_energy).to(u.electron) / u.ph
    solid_angle = (pixel_width * pixel_height).to(u.sr)
    # Each `+` below is load-bearing: the chain is energy -> photon -> electron -> DN,
    # so a step carries the steps downstream of it. Asking for DN from a photon
    # response needs the electron step too, which independent per-base deltas
    # would skip, leaving an erg/electron residue that cannot reach the target.
    to_dn = _unit_power(target_unit, u.DN) - _unit_power(old_unit, u.DN)
    to_electron = _unit_power(target_unit, u.electron) - _unit_power(old_unit, u.electron) + to_dn
    to_photon = _unit_power(target_unit, u.ph) - _unit_power(old_unit, u.ph) + to_electron
    to_pixel = _unit_power(target_unit, u.sr) - _unit_power(old_unit, u.sr)
    factor = (
        solid_angle**to_pixel
        * (1 / photon_energy) ** to_photon
        * electrons_per_photon**to_electron
        * (1 / gain) ** to_dn
    )
    try:
        residual = (1.0 * old_unit * factor.unit).to_value(target_unit)
    except u.UnitConversionError as exc:
        msg = f"response.spectral_response units {old_unit} are not convertible to {target_unit}"
        raise ValueError(msg) from exc

    conversion = xr.DataArray(
        np.broadcast_to(factor.value * residual, wavelength.shape),
        dims=wavelength.dims,
    )
    converted = (response.spectral_response * conversion).assign_attrs(
        {**response.spectral_response.attrs, "units": str(target_unit)}
    )
    result = response.assign(spectral_response=converted)
    add_history(result, locals(), transform_response_units)
    return result
