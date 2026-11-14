import numpy as np
import pytest
import xarray as xr

import astropy.constants as const
import astropy.units as u
from astropy.time import Time

from muse.instrument.effective_area import (
    create_aia_effective_area,
    create_effective_area,
    effective_area_wavelength_range,
    read_effective_area,
    select_effective_area_curve,
)

WAVELENGTH = np.linspace(160.0, 180.0, 5) * u.AA


def test_create_effective_area_layout():
    effective_area = create_effective_area(
        np.arange(1.0, 6.0) * u.cm**2,
        WAVELENGTH,
        band=171 * u.AA,
        components={"mirror": np.ones(5) * u.dimensionless_unscaled},
        attrs={"coating": "test"},
    )
    assert effective_area.effective_area.dims == ("wavelength", "band")
    assert effective_area.effective_area.attrs["units"] == "cm2"
    assert effective_area.mirror.attrs["units"] == ""
    assert effective_area.wavelength.attrs["units"] == "Angstrom"
    assert effective_area.band.values.tolist() == [171.0]
    assert effective_area.attrs["coating"] == "test"
    assert "create_effective_area(" in effective_area.attrs["HISTORY"][-1]


def test_create_effective_area_accepts_dn_per_photon_and_rejects_other_units():
    effective_area = create_effective_area(np.ones(5) * u.cm**2 * u.DN / u.ph, WAVELENGTH, band=94 * u.AA)
    assert u.Unit(effective_area.effective_area.attrs["units"]) == u.cm**2 * u.DN / u.ph
    with pytest.raises(u.UnitsError, match="cm2"):
        create_effective_area(np.ones(5) * u.m, WAVELENGTH, band=94 * u.AA)
    with pytest.raises(TypeError, match="Quantity"):
        create_effective_area(np.ones(5), WAVELENGTH, band=94 * u.AA)
    with pytest.raises(ValueError, match="increasing"):
        create_effective_area(np.ones(5) * u.cm**2, WAVELENGTH[::-1], band=94 * u.AA)


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_read_effective_area_round_trip(tmp_path, suffix):
    effective_area = create_effective_area(np.arange(1.0, 6.0) * u.cm**2, WAVELENGTH, band=171 * u.AA)
    path = tmp_path / f"area{suffix}"
    (effective_area.to_zarr(path, consolidated=False) if suffix == ".zarr" else effective_area.to_netcdf(path))
    back = read_effective_area(path)
    xr.testing.assert_allclose(back.effective_area, effective_area.effective_area)
    curve = select_effective_area_curve(back)
    assert curve.dims == ("wavelength",)
    assert curve.attrs["units"] == "cm2"
    np.testing.assert_allclose(curve.wavelength.values, WAVELENGTH.value)
    assert "band" not in curve.coords


def test_read_effective_area_rejects_bad_files(tmp_path):
    xr.Dataset({"area": ("wavelength", np.ones(3))}).to_netcdf(tmp_path / "no_area.nc")
    with pytest.raises(ValueError, match="effective_area"):
        read_effective_area(tmp_path / "no_area.nc")
    bad = create_effective_area(np.ones(5) * u.cm**2, WAVELENGTH, band=171 * u.AA)
    bad.effective_area.attrs["units"] = "m"
    bad.to_netcdf(tmp_path / "bad_units.nc")
    with pytest.raises(ValueError, match="cm2"):
        read_effective_area(tmp_path / "bad_units.nc")


def test_select_effective_area_curve_takes_a_band():
    effective_area = create_effective_area(
        np.outer(np.arange(1.0, 6.0), [1.0, 2.0]) * u.cm**2, WAVELENGTH, band=[171, 284] * u.AA
    )
    with pytest.raises(ValueError, match="pass band"):
        select_effective_area_curve(effective_area)
    np.testing.assert_allclose(
        select_effective_area_curve(effective_area, band=284 * u.AA).values, 2 * np.arange(1.0, 6.0)
    )


def test_effective_area_wavelength_range_spans_every_area():
    first = create_effective_area(np.ones(5) * u.cm**2, WAVELENGTH, band=171 * u.AA)
    second = create_effective_area(np.ones(3) * u.cm**2, [190, 195, 200] * u.AA, band=195 * u.AA)
    np.testing.assert_allclose(effective_area_wavelength_range([first, second]).to_value(u.AA), [160.0, 200.0])
    with pytest.raises(ValueError, match="at least one"):
        effective_area_wavelength_range([])


class _FakeChannel:
    """
    The ``Channel`` of aiapy on five wavelengths: a 2 cm2 area, a 1 cm2 crosstalk, 0.6
    arcsec pixels and the gain of an 18 electron / DN camera.
    """

    wavelength = WAVELENGTH
    effective_area = np.full(5, 2.0) * u.cm**2
    crosstalk = np.ones(5) * u.cm**2
    primary_reflectance = secondary_reflectance = np.full(5, 0.5) * u.one
    entrance_filter_efficiency = focal_plane_filter_efficiency = np.ones(5) * u.one
    quantum_efficiency = contamination = np.ones(5) * u.one
    plate_scale = ((0.6 * u.arcsec) ** 2).to(u.sr) / u.pix
    telescope_number = 4
    gain = (const.h * const.c / WAVELENGTH).to(u.eV) / (3.65 * u.eV / u.electron) / (18 * u.electron / u.DN) / u.ph

    def __init__(self, channel) -> None:
        self.channel = channel

    def eve_correction(self, obstime):
        self.eve_obstime = obstime
        return 1.5 * u.one


@pytest.fixture
def fake_aiapy(monkeypatch):
    """
    The aiapy modules without the instrument files they download.
    """
    response = pytest.importorskip("aiapy.response")
    calibrate = pytest.importorskip("aiapy.calibrate")
    monkeypatch.setattr(response, "Channel", _FakeChannel)
    monkeypatch.setattr(calibrate, "degradation", lambda _channel, _obstime: [0.8] * u.one)


@pytest.mark.usefixtures("fake_aiapy")
def test_create_aia_effective_area_defaults_are_solarsoft_dn_evenorm(tmp_path):
    effective_area = create_aia_effective_area(304 * u.AA)
    # crosstalk and the launch-epoch EVE factor, no degradation, the gain folded in
    np.testing.assert_allclose(effective_area.effective_area.isel(band=0), 3.0 * 1.5 * _FakeChannel.gain.value)
    assert u.Unit(effective_area.effective_area.attrs["units"]) == u.cm**2 * u.DN / u.ph
    assert effective_area.band.values.tolist() == [304.0]
    np.testing.assert_allclose(effective_area.mirror, 0.25)
    assert u.Unit(effective_area.dn_per_photon.attrs["units"]) == u.DN / u.ph
    attrs = effective_area.attrs
    assert (attrs["degradation"], attrs["eve_correction"]) == (1.0, 1.5)
    assert (attrs["obstime"], attrs["include_crosstalk"], attrs["units"]) == ("2010-05-01T00:00:00.000", 1, "DN")
    # the detector, as generate_responses reads it
    assert u.allclose(u.Quantity(attrs["pixel_width"]), 0.6 * u.arcsec)
    assert u.allclose(u.Quantity(attrs["gain"]), 18 * u.electron / u.DN)
    assert u.Quantity(attrs["pair_energy"]) == 3.65 * u.eV / u.electron
    assert "create_aia_effective_area(" in attrs["HISTORY"][-1]
    effective_area.to_netcdf(tmp_path / "aia.nc")
    assert read_effective_area(tmp_path / "aia.nc").attrs["gain"] == attrs["gain"]


@pytest.mark.usefixtures("fake_aiapy")
def test_create_aia_effective_area_switches():
    effective_area = create_aia_effective_area(
        304 * u.AA, obstime="2014-01-01", include_crosstalk=False, include_degradation=True, units="ph"
    )
    np.testing.assert_allclose(effective_area.effective_area, 2.0 * 0.8 * 1.5)
    assert effective_area.effective_area.attrs["units"] == "cm2"
    np.testing.assert_array_equal(effective_area.crosstalk, 0.0)
    assert effective_area.attrs["obstime"] == "2014-01-01T00:00:00.000"
    with pytest.raises(ValueError, match="units"):
        create_aia_effective_area(304 * u.AA, units="electrons")


@pytest.mark.remote_data
def test_create_aia_effective_area_matches_aiapy_wavelength_response():
    """
    With an epoch and the degradation, the area is aiapy's own wavelength response.
    """
    response = pytest.importorskip("aiapy.response")
    # aiapy takes a Time only; create_aia_effective_area also a string
    expected = response.Channel(335 * u.AA).wavelength_response(obstime=Time("2014-01-01"), include_eve_correction=True)
    effective_area = create_aia_effective_area(335 * u.AA, obstime="2014-01-01", include_degradation=True)
    np.testing.assert_allclose(
        effective_area.effective_area.isel(band=0), expected.to_value(u.cm**2 * u.DN / u.ph), rtol=1e-12
    )
