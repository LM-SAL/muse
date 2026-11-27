"""
Radiometric unit conversions.
"""

from fractions import Fraction

import numpy as np
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.instrument.response.detector import _channel_as_angstrom
from muse.utils.documentation import format_docstring
from muse.utils.utils import add_history, coord_as_unit, require_unit
from muse.variables import DEFAULTS_MUSE
from muse.variables_schema import InstrumentDefaults, _required

__all__ = ["transform_flux_units", "transform_response_units"]


def _unit_power(unit: u.UnitBase | str, base: u.UnitBase) -> float | Fraction:
    """
    The exponent of ``base`` in ``unit``, zero when absent.

    A `fractions.Fraction` exponent passes through so the unit arithmetic stays exact.
    """
    unit = u.Unit(unit)
    return unit.powers[unit.bases.index(base)] if base in unit.bases else 0


def _chain_steps(old_unit: u.UnitBase, new_unit: u.UnitBase) -> tuple[float, float, float]:
    """
    How often the photon, electron and DN steps of the detector chain are taken from
    ``old_unit`` to ``new_unit``; zero for a step that is not taken.
    """
    # A step carries the ones downstream of it: DN from photons needs the electron step too.
    dn_steps = _unit_power(new_unit, u.DN) - _unit_power(old_unit, u.DN)
    electron_steps = _unit_power(new_unit, u.electron) - _unit_power(old_unit, u.electron) + dn_steps
    photon_steps = _unit_power(new_unit, u.ph) - _unit_power(old_unit, u.ph) + electron_steps
    return photon_steps, electron_steps, dn_steps


def _detector_chain_factor(
    old_unit: u.UnitBase,
    new_unit: u.UnitBase,
    photon_energy: u.Quantity,
    pair_energy: u.Quantity | None,
    gain: u.Quantity | None,
) -> u.Quantity:
    """
    The factor that converts ``old_unit`` to ``new_unit`` along the detector chain
    energy -> photon -> electron -> data number (DN).

    ``pair_energy`` and ``gain`` are read only when their step is taken.
    """
    photon_steps, electron_steps, dn_steps = _chain_steps(old_unit, new_unit)
    factor = (1 / photon_energy) ** photon_steps
    for name, value, steps in (("pair_energy", pair_energy, electron_steps), ("gain", gain, dn_steps)):
        # A zero would give an infinite flux and a negative value a negative one, without an error.
        if steps and not np.all(np.isfinite(value.value) & (value.value > 0)):
            msg = f"{name} must be finite and positive, got {value}"
            raise ValueError(msg)
    if electron_steps:
        factor = factor * ((photon_energy * u.ph / pair_energy).to(u.electron) / u.ph) ** electron_steps
    if dn_steps:
        factor = factor * (1 / gain) ** dn_steps
    return factor


@format_docstring(
    "DEFAULTS_MUSE",
    pixel_width_sg="pixel_width_SG",
    pixel_height_sg="pixel_height_SG",
    pixel_width_ci="pixel_width_CI",
    pixel_height_ci="pixel_height_CI",
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
    *,
    channel: u.Quantity | None = None,
    detector: str = "SG",
    pixel_width: u.Quantity | None = None,
    pixel_height: u.Quantity | None = None,
    gain: u.Quantity | None = None,
    pair_energy: u.Quantity | None = None,
    instrument_defaults: InstrumentDefaults | None = None,
) -> xr.Dataset:
    """
    Convert the radiometric units of a wavelength-space response.

    Steps along the detector chain energy -> photon -> electron -> data number (DN), and
    from per steradian to per detector pixel, as ``new_units`` requires.

    .. warning::

        Do this before `muse.instrument.map_response_to_sg_detector` or
        `muse.instrument.map_response_to_ci_detector`, which pass the units
        through.

    Parameters
    ----------
    response : `xarray.Dataset`
        With ``spectral_response`` and ``wavelength_grid``, e.g. from
        `muse.instrument.create_spectral_response`.
    new_units : `str`
        Target units, as an `astropy.units.Unit` string, e.g.,
        ``"1e-27 cm5 ph / (Angstrom s)"``.
    channel : `astropy.units.Quantity`, optional
        Nominal MUSE channel wavelength, which selects the default ``gain`` and
        ``pair_energy``; required unless both are given.
    detector : {{"SG", "CI"}}, optional
        Detector whose default calibration to use, case-insensitive, by default ``"SG"``.
    pixel_width, pixel_height : `astropy.units.Quantity`, optional
        Pixel angular size. If `None`, {pixel_width_sg} by {pixel_height_sg} for SG or
        {pixel_width_ci} by {pixel_height_ci} for CI.
    gain : `astropy.units.Quantity`, optional
        Camera gain, convertible to electron/DN. If `None`, the channel's value from
        ``instrument_defaults``.
    pair_energy : `astropy.units.Quantity`, optional
        Mean energy that frees one electron-hole pair in silicon. If `None`, the
        channel's value from ``instrument_defaults``.
    instrument_defaults : `~muse.variables_schema.InstrumentDefaults`, optional
        The pixel sizes and calibrations to fall back on, by default
        `~muse.variables.DEFAULTS_MUSE`.

    Returns
    -------
    `xarray.Dataset`
        Copy of ``response`` whose ``spectral_response`` is scaled to
        ``new_units``.
    """
    if not isinstance(detector, str) or detector.lower() not in ("sg", "ci"):
        msg = "detector must be 'SG' or 'CI' (case-insensitive)"
        raise ValueError(msg)
    detector = detector.lower()
    old_unit = require_unit(response, "spectral_response", "response.spectral_response")
    new_unit = u.Unit(new_units)

    defaults = DEFAULTS_MUSE if instrument_defaults is None else instrument_defaults
    suffix = detector.upper()
    pixel_width = _required(defaults, f"pixel_width_{suffix}") if pixel_width is None else pixel_width
    pixel_height = _required(defaults, f"pixel_height_{suffix}") if pixel_height is None else pixel_height

    if gain is None or pair_energy is None:
        if channel is None:
            msg = "channel is required unless both gain and pair_energy are given"
            raise ValueError(msg)
        # Only the tables a value comes from: the CI ones may hold different channels.
        gains = _required(defaults, f"gain_{suffix}") if gain is None else None
        pair_energies = _required(defaults, f"pair_energy_{suffix}") if pair_energy is None else None
        channel_value = _channel_as_angstrom(channel, (pair_energies if gains is None else gains)["channel"], detector)
        try:
            if gain is None:
                gain = gains.sel(channel=channel_value).data
            if pair_energy is None:
                pair_energy = pair_energies.sel(channel=channel_value).data
        except KeyError:
            msg = f"unsupported MUSE {detector.upper()} channel {channel}"
            raise ValueError(msg) from None
    wavelength = coord_as_unit(response, "wavelength_grid", u.AA, "response.wavelength_grid")

    photon_energy = (const.h * const.c / (np.asarray(wavelength) * u.AA)).to(u.erg) / u.ph
    solid_angle = (pixel_width * pixel_height).to(u.sr)
    pixel_steps = _unit_power(new_unit, u.sr) - _unit_power(old_unit, u.sr)
    factor = solid_angle**pixel_steps * _detector_chain_factor(old_unit, new_unit, photon_energy, pair_energy, gain)
    try:
        residual = (1.0 * old_unit * factor.unit).to_value(new_unit)
    except u.UnitConversionError as exc:
        msg = f"response.spectral_response units {old_unit} are not convertible to {new_unit}"
        raise ValueError(msg) from exc

    conversion = xr.DataArray(
        np.broadcast_to(factor.value * residual, wavelength.shape),
        dims=wavelength.dims,
    )
    converted = (response.spectral_response * conversion).assign_attrs(
        {**response.spectral_response.attrs, "units": str(new_unit)}
    )
    result = response.assign(spectral_response=converted)
    add_history(result, locals(), transform_response_units)
    return result


@u.quantity_input(gain=u.electron / u.DN, pair_energy=u.eV / u.electron)
def transform_flux_units(
    spectrum: xr.Dataset,
    new_units: str,
    *,
    detector: str = "SG",
    gain: u.Quantity | None = None,
    pair_energy: u.Quantity | None = None,
    instrument_defaults: InstrumentDefaults | None = None,
) -> xr.Dataset:
    """
    Convert the flux of a spectrum between photons, electrons and data numbers (DN).

    Line by line, a photon frees ``h c / line_wavelength / pair_energy`` electrons and
    ``gain`` electrons make one DN. The rest of the unit, such as ``/ s``, can only change
    to an equivalent one, such as ``/ min``.

    Parameters
    ----------
    spectrum : `xarray.Dataset`
        E.g. from `muse.synthesis.vdem_synthesis`, with units on ``flux`` and on a
        ``line_wavelength`` coordinate along one dimension of ``flux``, such as ``line``
        or ``channel``. An entry with a NaN line wavelength, such as the continuum or a
        channel summed over lines of different wavelengths, has no photon energy, so a spectrum that holds one
        cannot be converted between photons and electrons or DN.
    new_units : `str`
        Target units, as an `astropy.units.Unit` string, e.g. ``"DN / s"``.
    detector : {"SG", "CI"}, optional
        Detector whose default calibration to use, case-insensitive, by default
        ``"SG"``.
    gain : `astropy.units.Quantity`, optional
        Camera gain, convertible to electron/DN, one value or one per line. If `None`,
        the ``gain`` coordinate of ``spectrum``, else each line's channel value from
        ``instrument_defaults``. Read only for a conversion to or from DN.
    pair_energy : `astropy.units.Quantity`, optional
        Mean energy that frees one electron-hole pair in silicon, one value or one per
        line. If `None`, the ``pair_energy`` coordinate of ``spectrum`` (a spectrum made
        from a response of `muse.instrument.read_response` has one), else each line's channel value
        from ``instrument_defaults``. Read only for a conversion between photons and
        electrons or DN.
    instrument_defaults : `~muse.variables_schema.InstrumentDefaults`, optional
        The calibrations to fall back on, by default `~muse.variables.DEFAULTS_MUSE`.

    Returns
    -------
    `xarray.Dataset`
        Copy of ``spectrum`` whose ``flux`` is in ``new_units``.

    Raises
    ------
    ValueError
        If ``flux`` or ``line_wavelength`` has no units, the detector chain cannot take
        ``flux`` to ``new_units``, or a default is needed and ``spectrum`` has no
        ``channel`` coordinate or a channel ``detector`` lacks.
    """
    if not isinstance(detector, str) or detector.lower() not in ("sg", "ci"):
        msg = "detector must be 'SG' or 'CI' (case-insensitive)"
        raise ValueError(msg)
    old_unit = require_unit(spectrum, "flux", "spectrum.flux")
    new_unit = u.Unit(new_units)
    wavelength = coord_as_unit(spectrum, "line_wavelength", u.AA, "spectrum.line_wavelength")
    if wavelength.ndim != 1 or wavelength.dims[0] not in spectrum.flux.dims:
        msg = "spectrum.line_wavelength must lie along one dimension of spectrum.flux"
        raise ValueError(msg)

    photon_steps, electron_steps, dn_steps = _chain_steps(old_unit, new_unit)
    if (photon_steps or electron_steps) and not np.isfinite(wavelength.values).all():
        dim = wavelength.dims[0]
        without = spectrum[dim].values[~np.isfinite(wavelength.values)].tolist()
        msg = (
            f"cannot convert between photons and electrons or DN: {dim} {without} have no line wavelength. "
            "Convert before summing the lines of a channel, or select the entries that have one."
        )
        raise ValueError(msg)
    if dn_steps and gain is None and "gain" in spectrum.coords:
        gain = u.Quantity(
            coord_as_unit(spectrum, "gain", u.electron / u.DN, "spectrum.gain").broadcast_like(wavelength).values,
            u.electron / u.DN,
        )
    if electron_steps and pair_energy is None and "pair_energy" in spectrum.coords:
        pair_energy = u.Quantity(
            coord_as_unit(spectrum, "pair_energy", u.eV / u.electron, "spectrum.pair_energy")
            .broadcast_like(wavelength)
            .values,
            u.eV / u.electron,
        )
    needs_gain, needs_pair_energy = bool(dn_steps) and gain is None, bool(electron_steps) and pair_energy is None
    if needs_gain or needs_pair_energy:
        if "channel" not in spectrum.coords:
            msg = "spectrum has no channel coordinate to select the default calibration; pass gain and pair_energy"
            raise ValueError(msg)
        defaults = DEFAULTS_MUSE if instrument_defaults is None else instrument_defaults
        try:
            if needs_gain:
                gain = _required(defaults, f"gain_{detector.upper()}").sel(channel=spectrum.channel).data
            if needs_pair_energy:
                pair_energy = _required(defaults, f"pair_energy_{detector.upper()}").sel(channel=spectrum.channel).data
        except KeyError:
            channels = np.unique(spectrum.channel.values).tolist()
            msg = f"unsupported MUSE {detector.upper()} channel among {channels}"
            raise ValueError(msg) from None

    photon_energy = (const.h * const.c / (wavelength.values * u.AA)).to(u.erg) / u.ph
    factor = _detector_chain_factor(old_unit, new_unit, photon_energy, pair_energy, gain)
    try:
        residual = (1.0 * old_unit * factor.unit).to_value(new_unit)
    except u.UnitConversionError as exc:
        msg = (
            f"spectrum.flux units {old_unit} are not convertible to {new_unit} "
            "along the detector chain photon -> electron -> DN"
        )
        raise ValueError(msg) from exc

    conversion = xr.DataArray(np.broadcast_to(factor.value * residual, wavelength.shape), dims=wavelength.dims)
    flux = (spectrum.flux * conversion).assign_attrs({**spectrum.flux.attrs, "units": str(new_unit)})
    result = spectrum.assign(flux=flux)
    add_history(result, locals(), transform_flux_units)
    return result
