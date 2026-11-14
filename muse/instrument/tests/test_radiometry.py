import attrs
import numpy as np
import pytest
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.instrument import radiometry as radiometry_module
from muse.instrument.radiometry import transform_flux_units, transform_response_units
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
        171 * u.AA,
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
    photons = transform_response_units(response, "1e-27 cm5 ph / (Angstrom s)", 171 * u.AA)

    data_numbers = transform_response_units(photons, "1e-27 cm5 DN / (Angstrom s)", 171 * u.AA, gain=gain)

    # Legacy conversion_ph2dn: 12398 eV Angstrom / wavelength / 3.65 eV / gain.
    wavelength = response.wavelength_grid.values
    expected = photons.spectral_response * (12398.0 / wavelength / 3.65 / gain.to_value(u.electron / u.DN))
    np.testing.assert_allclose(data_numbers.spectral_response, expected, rtol=1e-4)
    assert u.Unit(data_numbers.spectral_response.attrs["units"]) == u.Unit("1e-27 cm5 DN / (Angstrom s)")


def test_transform_response_units_accepts_explicit_calibration_without_channel():
    # A non-MUSE instrument supplies its own calibration; the MUSE channel list
    # is only consulted when a gain or pair_energy default is needed.
    response = _spectral_response()
    gain = 2.5 * u.electron / u.DN

    converted = transform_response_units(
        response,
        "1e-27 cm5 DN / (Angstrom s)",
        gain=gain,
        pair_energy=3.65 * u.eV / u.electron,
        pixel_width=0.6 * u.arcsec,
        pixel_height=0.6 * u.arcsec,
    )

    assert u.Unit(converted.spectral_response.attrs["units"]) == u.Unit("1e-27 cm5 DN / (Angstrom s)")
    with pytest.raises(ValueError, match="channel is required"):
        transform_response_units(response, "1e-27 cm5 DN / (Angstrom s)", gain=gain)


def test_transform_response_units_uses_channel_gain(monkeypatch):
    monkeypatch.setattr(
        DEFAULTS_MUSE.ccd_gain_SG,
        "data",
        np.array([8.0, 10.0, 12.0]) * u.electron / u.DN,
    )
    photons = transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s)", 284 * u.AA)

    default = transform_response_units(photons, "1e-27 cm5 DN / (Angstrom s)", 284 * u.AA)
    explicit = transform_response_units(
        photons,
        "1e-27 cm5 DN / (Angstrom s)",
        284 * u.AA,
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

    forward = transform_response_units(response, intermediate, 171 * u.AA)
    back = transform_response_units(forward, original_units, 171 * u.AA)

    np.testing.assert_allclose(
        back.spectral_response.values,
        response.spectral_response.values,
        rtol=1e-12,
    )
    assert u.Unit(back.spectral_response.attrs["units"]) == u.Unit(original_units)


def test_transform_response_units_rescales_without_a_stage_change():
    converted = transform_response_units(_spectral_response(), "erg cm5 / (Angstrom s sr)", 171 * u.AA)

    np.testing.assert_allclose(converted.spectral_response, 1e-27)
    assert u.Unit(converted.spectral_response.attrs["units"]) == u.Unit("erg cm5 / (Angstrom s sr)")


def test_transform_response_units_stays_lazy():
    response = _spectral_response().chunk({"wavelength_bin": 1})

    converted = transform_response_units(response, "1e-27 cm5 ph / (Angstrom s)", 171 * u.AA)

    assert converted.spectral_response.chunks is not None


def test_transform_response_units_rejects_incompatible_units():
    with pytest.raises(ValueError, match="not convertible"):
        transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s K)", 171 * u.AA)


def test_transform_response_units_rejects_unknown_channel():
    with pytest.raises(ValueError, match="unsupported MUSE SG channel"):
        transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s)", 195 * u.AA)


def test_transform_response_units_rejects_unknown_ci_channel():
    with pytest.raises(ValueError, match="unsupported MUSE CI channel"):
        transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s)", 171 * u.AA, detector="ci")


@pytest.mark.parametrize(("channel", "error"), [(171, TypeError), (1 * u.s, u.UnitsError)])
def test_transform_response_units_requires_channel_wavelength(channel, error):
    with pytest.raises(error, match="channel"):
        transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s)", channel)


@pytest.mark.parametrize("detector", ["other", None])
def test_transform_response_units_rejects_unknown_detector(detector):
    with pytest.raises(ValueError, match="detector must be"):
        transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s)", 171 * u.AA, detector=detector)


def test_transform_response_units_detector_case_insensitive():
    with pytest.raises(ValueError, match="unsupported MUSE CI channel"):
        transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s)", 171 * u.AA, detector="CI")


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
        transform_response_units(response, "1e-27 cm5 ph / (Angstrom s)", 171 * u.AA)


def test_transform_response_units_uses_the_channel_pair_energy():
    channel = 17.1 * u.nm
    pair_energy = u.Quantity(DEFAULTS_MUSE.pair_creation_energy_SG.sel(channel=171).data)
    response = transform_response_units(_spectral_response(), "1e-27 cm5 ph / (Angstrom s)", channel)

    default = transform_response_units(response, "1e-27 cm5 DN / (Angstrom s)", channel)
    doubled = transform_response_units(response, "1e-27 cm5 DN / (Angstrom s)", channel, pair_energy=2 * pair_energy)

    np.testing.assert_allclose(doubled.spectral_response, default.spectral_response / 2)


def test_transform_response_units_selects_detector_calibration(monkeypatch):
    ci_gain = xr.DataArray(
        np.array([20.0, 20.0]) * u.electron / u.DN,
        coords={"channel": [195, 304]},
        dims="channel",
    )
    ci_pair_energy = xr.DataArray(
        np.array([7.3, 7.3]) * u.eV / u.electron,
        coords={"channel": [195, 304]},
        dims="channel",
    )
    defaults = attrs.evolve(DEFAULTS_MUSE, ccd_gain_CI=ci_gain, pair_creation_energy_CI=ci_pair_energy)
    monkeypatch.setattr(radiometry_module, "DEFAULTS_MUSE", defaults)

    target_unit = "1e-27 cm5 DN / (Angstrom s)"
    sg = transform_response_units(_spectral_response(), target_unit, 171 * u.AA)
    ci = transform_response_units(_spectral_response(), target_unit, 195 * u.AA, detector="ci")

    sg_solid_angle = (defaults.dx_pixel_SG * defaults.dy_pixel_SG).to_value(u.sr)
    ci_solid_angle = (defaults.dx_pixel_CI * defaults.dy_pixel_CI).to_value(u.sr)
    sg_pair_energy = u.Quantity(defaults.pair_creation_energy_SG.sel(channel=171).data)
    sg_gain = u.Quantity(defaults.ccd_gain_SG.sel(channel=171).data)
    expected_ratio = (
        ci_solid_angle
        / sg_solid_angle
        * (sg_pair_energy / (7.3 * u.eV / u.electron)).to_value(u.dimensionless_unscaled)
        * (sg_gain / (20.0 * u.electron / u.DN)).to_value(u.dimensionless_unscaled)
    )
    np.testing.assert_allclose(ci.spectral_response / sg.spectral_response, expected_ratio)


def _spectrum(units="ph / s", *, dim="line", channel=(171, 284)):
    # Two lines, three pixels: every value is one, so the result is the conversion factor itself.
    flux = xr.DataArray(np.ones((2, 3)), dims=(dim, "detector_x_pixel"), attrs={"units": units})
    coords = {"line_wavelength": (dim, [171.073, 284.163], {"units": "Angstrom"})}
    if channel is not None:
        coords["channel"] = (dim, list(channel))
    return xr.Dataset({"flux": flux}, coords=coords)


def _electrons_per_photon(wavelength, pair_energy=3.65):
    # h c = 12398.42 eV Angstrom
    return (const.h * const.c).to_value(u.eV * u.AA) / np.asarray(wavelength) / pair_energy


def test_transform_flux_units_photons_to_data_numbers_line_by_line():
    spectrum = _spectrum()

    out = transform_flux_units(spectrum, "DN / s", gain=8 * u.electron / u.DN, pair_energy=3.65 * u.eV / u.electron)

    expected = _electrons_per_photon([171.073, 284.163]) / 8
    np.testing.assert_allclose(out.flux.values, np.repeat(expected[:, None], 3, axis=1), rtol=1e-12)
    assert u.Unit(out.flux.attrs["units"]) == u.DN / u.s
    assert spectrum.flux.attrs["units"] == "ph / s"  # the input is untouched
    assert out.attrs["HISTORY"][-1].startswith("transform_flux_units(")


def test_transform_flux_units_takes_the_calibration_of_each_channel_by_default():
    out = transform_flux_units(_spectrum(), "DN / s")

    explicit = transform_flux_units(
        _spectrum(), "DN / s", gain=[10, 10] * u.electron / u.DN, pair_energy=[3.65, 3.65] * u.eV / u.electron
    )
    np.testing.assert_allclose(out.flux.values, explicit.flux.values, rtol=1e-12)


def test_transform_flux_units_prefers_the_gain_coordinate_of_the_spectrum():
    spectrum = _spectrum().assign_coords(gain=("line", [5.0, 20.0], {"units": "electron / DN"}))

    out = transform_flux_units(spectrum, "DN / s")

    expected = _electrons_per_photon([171.073, 284.163]) / np.array([5.0, 20.0])
    np.testing.assert_allclose(out.flux.isel(detector_x_pixel=0).values, expected, rtol=1e-12)


@pytest.mark.parametrize("intermediate", ["electron / s", "DN / s"])
def test_transform_flux_units_round_trips(intermediate):
    spectrum = _spectrum()

    back = transform_flux_units(transform_flux_units(spectrum, intermediate), "ph / s")

    np.testing.assert_allclose(back.flux.values, spectrum.flux.values, rtol=1e-12)
    assert u.Unit(back.flux.attrs["units"]) == u.ph / u.s


def test_transform_flux_units_electrons_to_data_numbers_needs_only_the_gain():
    out = transform_flux_units(_spectrum("electron"), "DN", gain=4 * u.electron / u.DN)

    np.testing.assert_allclose(out.flux.values, 0.25)


def test_transform_flux_units_works_along_channel_as_sum_lines_per_channel_returns():
    out = transform_flux_units(_spectrum(dim="channel", channel=None).assign_coords(channel=[171, 284]), "electron / s")

    np.testing.assert_allclose(
        out.flux.isel(detector_x_pixel=0).values, _electrons_per_photon([171.073, 284.163]), rtol=1e-12
    )


def test_transform_flux_units_selects_the_detector_calibration(monkeypatch):
    ci_gain = xr.DataArray(np.array([2.0, 2.0]) * u.electron / u.DN, coords={"channel": [195, 304]}, dims="channel")
    monkeypatch.setattr(radiometry_module, "DEFAULTS_MUSE", attrs.evolve(DEFAULTS_MUSE, ccd_gain_CI=ci_gain))
    spectrum = _spectrum(channel=(195, 304))

    out = transform_flux_units(spectrum, "DN / s", detector="ci")

    np.testing.assert_allclose(
        out.flux.isel(detector_x_pixel=0).values, _electrons_per_photon([171.073, 284.163]) / 2, rtol=1e-12
    )
    with pytest.raises(ValueError, match="unsupported MUSE SG channel among"):
        transform_flux_units(spectrum, "DN / s")


@pytest.mark.parametrize(
    ("spectrum", "new_units", "kwargs", "match"),
    [
        (_spectrum(), "DN", {}, "are not convertible to DN along the detector chain"),
        (_spectrum(channel=None), "DN / s", {}, "no channel coordinate"),
        (_spectrum(), "DN / s", {"detector": "XX"}, "detector must be 'SG' or 'CI'"),
        (_spectrum().assign(flux=_spectrum().flux.assign_attrs(units=None)), "DN / s", {}, "spectrum.flux"),
        (_spectrum().drop_vars("line_wavelength"), "DN / s", {}, "line_wavelength"),
    ],
)
def test_transform_flux_units_rejects_invalid_inputs(spectrum, new_units, kwargs, match):
    with pytest.raises(ValueError, match=match):
        transform_flux_units(spectrum, new_units, **kwargs)
