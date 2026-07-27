import numpy as np
import pytest
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.instrument.radiometry import transform_response_units
from muse.variables import DEFAULTS_MUSE


def _spectral_response() -> xr.Dataset:
    wavelength = np.array([170.0, 171.0, 172.0])
    return xr.Dataset(
        {
            "spectral_response": (
                ("line", "wavelength_bin"),
                np.ones((1, wavelength.size)),
                {"units": "1e-27 erg cm5 / (Angstrom s sr)"},
            )
        },
        coords={
            "line": ["Fe IX 171.073"],
            "wavelength_grid": ("wavelength_bin", wavelength, {"units": "Angstrom"}),
        },
        attrs={"normalization": 1e-27},
    )


def test_transform_response_units_energy_to_photons():
    response = _spectral_response()
    original = response.copy(deep=True)
    pixel_width = 0.4 * u.arcsec
    pixel_height = 0.167 * u.arcsec

    converted = transform_response_units(
        response,
        "1e-27 cm5 ph / (Angstrom s)",
        171,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )

    wavelength = response.wavelength_grid.values * u.AA
    solid_angle = (pixel_width * pixel_height).to_value(u.sr)
    photon_energy = (const.h * const.c / wavelength).to_value(u.erg)
    np.testing.assert_allclose(converted.spectral_response.isel(line=0), solid_angle / photon_energy)
    assert u.Unit(converted.spectral_response.attrs["units"]) == u.Unit("1e-27 cm5 ph / (Angstrom s)")
    assert converted.attrs["HISTORY"][-1].startswith("transform_response_units(")
    xr.testing.assert_identical(response, original)


def test_transform_response_units_photons_to_data_numbers():
    response = _spectral_response()
    gain = 10 * u.electron / u.DN
    photons = transform_response_units(response, "1e-27 cm5 ph / (Angstrom s)", 171)

    data_numbers = transform_response_units(photons, "1e-27 cm5 DN / (Angstrom s)", 171, gain=gain)

    # Legacy conversion_ph2dn: 12398 eV Angstrom / wavelength / 3.65 eV / gain.
    wavelength = response.wavelength_grid.values
    expected = photons.spectral_response * (12398.0 / wavelength / 3.65 / gain.to_value(u.electron / u.DN))
    np.testing.assert_allclose(data_numbers.spectral_response, expected, rtol=1e-4)
    assert u.Unit(data_numbers.spectral_response.attrs["units"]) == u.Unit("1e-27 cm5 DN / (Angstrom s)")


def test_transform_response_units_uses_channel_gain(monkeypatch):
    monkeypatch.setattr(
        DEFAULTS_MUSE.ccd_gain,
        "data",
        np.array([8.0, 10.0, 12.0]) * u.electron / u.DN,
    )
    photons = transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s)", 284)

    default = transform_response_units(photons, "1e-27 cm5 DN / (Angstrom s)", 284)
    explicit = transform_response_units(
        photons,
        "1e-27 cm5 DN / (Angstrom s)",
        284,
        gain=12 * u.electron / u.DN,
    )

    np.testing.assert_allclose(default.spectral_response, explicit.spectral_response)


@pytest.mark.parametrize(
    "intermediate",
    [
        "1e-27 cm5 ph / (Angstrom s)",
        "1e-27 cm5 electron / (Angstrom s)",
        "1e-27 cm5 DN / (Angstrom s)",
    ],
)
def test_transform_response_units_round_trips(intermediate):
    # radiometry.py:104-107 says every "+" in the exponent cascade is load-bearing, but every
    # other test converts away from erg/sr and stops. Converting back exercises all four
    # exponents with the opposite sign and must return the original values.
    response = _spectral_response()
    original_units = response.spectral_response.attrs["units"]

    forward = transform_response_units(response, intermediate, 171)
    back = transform_response_units(forward, original_units, 171)

    np.testing.assert_allclose(
        back.spectral_response.values,
        response.spectral_response.values,
        rtol=1e-12,
    )
    assert u.Unit(back.spectral_response.attrs["units"]) == u.Unit(original_units)


def test_transform_response_units_rescales_without_a_stage_change():
    converted = transform_response_units(_spectral_response(), "erg cm5 / (Angstrom s sr)", 171)

    np.testing.assert_allclose(converted.spectral_response, 1e-27)
    assert u.Unit(converted.spectral_response.attrs["units"]) == u.Unit("erg cm5 / (Angstrom s sr)")


def test_transform_response_units_stays_lazy():
    response = _spectral_response().chunk({"wavelength_bin": 1})

    converted = transform_response_units(response, "1e-27 cm5 ph / (Angstrom s)", 171)

    assert converted.spectral_response.chunks is not None


def test_transform_response_units_rejects_incompatible_units():
    with pytest.raises(ValueError, match="not convertible"):
        transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s K)", 171)


def test_transform_response_units_rejects_unknown_channel():
    with pytest.raises(ValueError, match="unsupported MUSE SG channel"):
        transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s)", 195)


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("response_type", TypeError, "xarray.Dataset"),
        ("missing_variable", ValueError, r"response\.spectral_response is missing"),
        ("missing_units", ValueError, r"response\.spectral_response must define units"),
        ("invalid_units", ValueError, r"response\.spectral_response units must be a valid astropy unit"),
        ("missing_wavelength_units", ValueError, r"response\.wavelength_grid must define units"),
    ],
)
def test_transform_response_units_rejects_invalid_inputs(case, error, match):
    response = _spectral_response()
    if case == "response_type":
        response = None
    elif case == "missing_variable":
        response = response.drop_vars("spectral_response")
    elif case == "missing_units":
        response = response.assign(spectral_response=response.spectral_response.drop_attrs())
    elif case == "invalid_units":
        response = response.assign(spectral_response=response.spectral_response.assign_attrs(units="not-a-unit"))
    else:
        response = response.assign_coords(wavelength_grid=response.wavelength_grid.drop_attrs())

    with pytest.raises(error, match=match):
        transform_response_units(response, "1e-27 cm5 ph / (Angstrom s)", 171)


def test_transform_response_units_uses_the_channel_pair_energy():
    channel = 171
    pair_energy = u.Quantity(DEFAULTS_MUSE.pair_creation_energy.sel(channel=channel).data)
    response = transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s)", channel)

    default = transform_response_units(response, "1e-27 cm5 DN / (Angstrom s)", channel)
    doubled = transform_response_units(response, "1e-27 cm5 DN / (Angstrom s)", channel, pair_energy=2 * pair_energy)

    np.testing.assert_allclose(doubled.spectral_response, default.spectral_response / 2)
