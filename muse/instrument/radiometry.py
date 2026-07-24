"""
Radiometric unit conversions.
"""

import numpy as np
import xarray as xr

import astropy.constants as const
import astropy.units as u

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
    pixel_width="dx_pixel_SG",
    pixel_height="dy_pixel_SG",
    gain="ccd_gain",
)
@u.quantity_input(
    pixel_width=u.arcsec,
    pixel_height=u.arcsec,
    gain=u.electron / u.DN,
    pair_energy=u.eV / u.electron,
)
def transform_response_units(
    response: xr.Dataset,
    new_units: str,
    channel: int,
    *,
    pixel_width: u.Quantity = DEFAULTS_MUSE.dx_pixel_SG,
    pixel_height: u.Quantity = DEFAULTS_MUSE.dy_pixel_SG,
    gain: u.Quantity = DEFAULTS_MUSE.ccd_gain,
    pair_energy: u.Quantity | None = None,
) -> xr.Dataset:
    """
    Convert the radiometric units of a wavelength-space response.

    The conversion is: detector chain energy -> photon -> electron -> data
    number (DN), and converts a per-steradian response into a per-detector-pixel one,
    as required by ``new_units``.

    .. warning::

        Do this before `muse.instrument.map_response_to_sg_detector`,
        which passes the units through.

    Parameters
    ----------
    response : `xarray.Dataset`
        Wavelength-space response containing ``spectral_response`` and
        ``wavelength_grid``, e.g. from
        `muse.instrument.create_spectral_response`.
    new_units : `str`
        Target units, as an `astropy.units.Unit` string, e.g.,
        ``"1e-27 cm5 ph / (Angstrom s)"``.
    channel : `int`
        MUSE SG channel: 108, 171, or 284 Angstrom. Selects the per-channel
        ``pair_energy`` calibration.
    pixel_width, pixel_height : `astropy.units.Quantity`, optional
        SG pixel angular size used to convert steradians to detector pixels,
        by default {pixel_width} and {pixel_height}, respectively.
    gain : `astropy.units.Quantity`, optional
        Camera gain, convertible to electron/DN, by default {gain}.
    pair_energy : `astropy.units.Quantity`, optional
        Mean energy that frees one electron-hole pair in silicon. If `None`,
        use the channel's value from
        `~muse.variables.DEFAULTS_MUSE.pair_creation_energy`.

    Returns
    -------
    `xarray.Dataset`
        Copy of ``response`` whose ``spectral_response`` is scaled to
        ``new_units``.
    """
    old_unit = require_unit(response, "spectral_response", "response.spectral_response")
    target_unit = u.Unit(new_units)
    if pair_energy is None:
        try:
            pair_energy = u.Quantity(DEFAULTS_MUSE.pair_creation_energy.sel(channel=channel).data)
        except KeyError:
            msg = f"unsupported MUSE SG channel {channel}"
            raise ValueError(msg) from None
    wavelength = coord_as_unit(response, "wavelength_grid", u.AA, "response.wavelength_grid")

    photon_energy = (const.h * const.c / (np.asarray(wavelength) * u.AA)).to(u.erg) / u.ph
    electrons_per_photon = (photon_energy * u.ph / pair_energy).to(u.electron) / u.ph
    solid_angle = (pixel_width * pixel_height).to(u.sr)
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
