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
    save_effective_area,
)

WAVELENGTH = np.linspace(160.0, 180.0, 5) * u.AA


def _area(**kwargs):
    return create_effective_area(np.arange(1.0, 6.0) * u.cm**2, WAVELENGTH, channel=171 * u.AA, **kwargs)


def test_create_effective_area_layout():
    effective_area = _area(
        components={"mirror": np.ones(5) * u.dimensionless_unscaled, "contamination": [0.9] * u.one},
        attrs={"coating": "test"},
    )
    assert effective_area.effective_area.dims == ("wavelength",)
    assert effective_area.effective_area.attrs["units"] == "cm2"
    assert effective_area.mirror.attrs["units"] == ""
    assert effective_area.wavelength.attrs["units"] == "Angstrom"
    assert effective_area.channel.ndim == 0
    assert float(effective_area.channel) == 171.0
    assert effective_area.channel.attrs["units"] == "Angstrom"
    # One value of a component holds at every wavelength.
    np.testing.assert_array_equal(effective_area.contamination, 0.9)
    assert effective_area.attrs["coating"] == "test"
    assert "create_effective_area(" in effective_area.attrs["HISTORY"][-1]


def test_create_effective_area_accepts_dn_per_photon_and_rejects_other_units():
    effective_area = create_effective_area(np.ones(5) * u.cm**2 * u.DN / u.ph, WAVELENGTH, channel=94 * u.AA)
    assert u.Unit(effective_area.effective_area.attrs["units"]) == u.cm**2 * u.DN / u.ph
    with pytest.raises(u.UnitsError, match="cm2"):
        create_effective_area(np.ones(5) * u.m, WAVELENGTH, channel=94 * u.AA)
    with pytest.raises(TypeError, match="Quantity"):
        create_effective_area(np.ones(5), WAVELENGTH, channel=94 * u.AA)
    with pytest.raises(ValueError, match="increasing"):
        create_effective_area(np.ones(5) * u.cm**2, WAVELENGTH[::-1], channel=94 * u.AA)


def test_create_effective_area_takes_one_curve_of_one_channel():
    # A (channel, wavelength) table used to be reshaped into a scrambled curve.
    with pytest.raises(ValueError, match="area must have one value per wavelength"):
        create_effective_area(np.ones((2, 5)) * u.cm**2, WAVELENGTH, channel=171 * u.AA)
    with pytest.raises(ValueError, match="mirror must have one value per wavelength"):
        _area(components={"mirror": np.ones(3) * u.one})
    with pytest.raises(ValueError, match="holds one channel"):
        create_effective_area(np.ones(5) * u.cm**2, WAVELENGTH, channel=[171, 284] * u.AA)


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_save_and_read_effective_area_round_trip(tmp_path, suffix):
    effective_area = _area(components={"mirror": np.full(5, 0.5) * u.one})
    path = tmp_path / f"area{suffix}"

    save_effective_area(effective_area, path)
    back = read_effective_area(path)

    xr.testing.assert_allclose(back.effective_area, effective_area.effective_area)
    xr.testing.assert_allclose(back.mirror, effective_area.mirror)
    assert back.effective_area.dims == ("wavelength",)
    assert back.effective_area.attrs["units"] == "cm2"
    assert float(back.channel) == 171.0
    with pytest.raises(ValueError, match="Refusing to overwrite existing effective area"):
        save_effective_area(effective_area, path)


def test_save_effective_area_rejects_another_suffix_and_layout(tmp_path):
    with pytest.raises(ValueError, match="effective_area_file must end in"):
        save_effective_area(_area(), tmp_path / "area.txt")
    with pytest.raises(ValueError, match="channel coordinate"):
        save_effective_area(_area().drop_vars("channel"), tmp_path / "area.nc")
    assert not list(tmp_path.iterdir())


def test_read_effective_area_converts_wavelength_and_channel_to_angstrom(tmp_path):
    # A file made by hand in nm: the numbers must not be taken as Angstrom later.
    in_nm = _area().assign_coords(
        wavelength=("wavelength", WAVELENGTH.to_value(u.nm), {"units": "nm"}), channel=((), 17.1, {"units": "nm"})
    )
    in_nm.to_netcdf(tmp_path / "nm.nc")

    back = read_effective_area(tmp_path / "nm.nc")

    np.testing.assert_allclose(back.wavelength, WAVELENGTH.value)
    assert back.wavelength.attrs["units"] == "Angstrom"
    np.testing.assert_allclose(float(back.channel), 171.0)
    assert back.channel.attrs["units"] == "Angstrom"
    xr.testing.assert_allclose(back.effective_area, _area().effective_area)


def test_read_effective_area_rejects_bad_files(tmp_path):
    xr.Dataset({"area": ("wavelength", np.ones(3))}).to_netcdf(tmp_path / "no_area.nc")
    with pytest.raises(ValueError, match="effective_area and a channel coordinate"):
        read_effective_area(tmp_path / "no_area.nc")

    bad_units = _area()
    bad_units.effective_area.attrs["units"] = "m"
    bad_units.to_netcdf(tmp_path / "bad_units.nc")
    with pytest.raises(ValueError, match="cm2"):
        read_effective_area(tmp_path / "bad_units.nc")

    # A table over several channels is not a file of one channel.
    table = xr.Dataset(
        {"effective_area": (("wavelength", "channel"), np.ones((5, 2)), {"units": "cm2"})},
        coords={"wavelength": ("wavelength", WAVELENGTH.value, {"units": "Angstrom"}), "channel": [171.0, 284.0]},
    )
    table.to_netcdf(tmp_path / "table.nc")
    with pytest.raises(ValueError, match="along wavelength alone"):
        read_effective_area(tmp_path / "table.nc")

    no_wavelength_unit = _area()
    del no_wavelength_unit.wavelength.attrs["units"]
    no_wavelength_unit.to_netcdf(tmp_path / "no_unit.nc")
    with pytest.raises(ValueError, match="wavelength"):
        read_effective_area(tmp_path / "no_unit.nc")


def test_effective_area_wavelength_range_spans_every_area():
    second = create_effective_area(np.ones(3) * u.cm**2, [190, 195, 200] * u.AA, channel=195 * u.AA)
    np.testing.assert_allclose(effective_area_wavelength_range([_area(), second]).to_value(u.AA), [160.0, 200.0])
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
    np.testing.assert_allclose(effective_area.effective_area, 3.0 * 1.5 * _FakeChannel.gain.value)
    assert u.Unit(effective_area.effective_area.attrs["units"]) == u.cm**2 * u.DN / u.ph
    assert float(effective_area.channel) == 304.0
    np.testing.assert_allclose(effective_area.mirror, 0.25)
    assert u.Unit(effective_area.dn_per_photon.attrs["units"]) == u.DN / u.ph
    attrs = effective_area.attrs
    assert (attrs["degradation"], attrs["eve_correction"]) == (1.0, 1.5)
    assert (attrs["obstime"], attrs["include_crosstalk"], attrs["include_gain"]) == ("2010-05-01T00:00:00.000", 1, 1)
    # the detector, as the file describes it
    assert u.allclose(u.Quantity(attrs["pixel_width"]), 0.6 * u.arcsec)
    assert u.allclose(u.Quantity(attrs["gain"]), 18 * u.electron / u.DN)
    assert u.Quantity(attrs["pair_energy"]) == 3.65 * u.eV / u.electron
    assert "create_aia_effective_area(" in attrs["HISTORY"][-1]
    save_effective_area(effective_area, tmp_path / "aia.nc")
    assert read_effective_area(tmp_path / "aia.nc").attrs["gain"] == attrs["gain"]


@pytest.mark.usefixtures("fake_aiapy")
def test_create_aia_effective_area_switches():
    effective_area = create_aia_effective_area(
        304 * u.AA, obstime="2014-01-01", include_crosstalk=False, include_degradation=True, include_gain=False
    )
    np.testing.assert_allclose(effective_area.effective_area, 2.0 * 0.8 * 1.5)
    assert effective_area.effective_area.attrs["units"] == "cm2"
    np.testing.assert_array_equal(effective_area.crosstalk, 0.0)
    assert effective_area.attrs["obstime"] == "2014-01-01T00:00:00.000"


@pytest.mark.usefixtures("fake_aiapy")
def test_create_aia_effective_area_takes_a_component_of_one_value(monkeypatch):
    # aiapy gives the contamination of the 1600, 1700 and 4500 channels as one value.
    monkeypatch.setattr(_FakeChannel, "contamination", [0.5] * u.one)

    effective_area = create_aia_effective_area(304 * u.AA)

    np.testing.assert_array_equal(effective_area.contamination, 0.5)
    assert effective_area.contamination.dims == ("wavelength",)


@pytest.mark.remote_data
def test_create_aia_effective_area_matches_aiapy_wavelength_response():
    """
    With an epoch and the degradation, the area is aiapy's own wavelength response.
    """
    response = pytest.importorskip("aiapy.response")
    # aiapy takes a Time only; create_aia_effective_area also a string
    expected = response.Channel(335 * u.AA).wavelength_response(obstime=Time("2014-01-01"), include_eve_correction=True)
    effective_area = create_aia_effective_area(335 * u.AA, obstime="2014-01-01", include_degradation=True)
    np.testing.assert_allclose(effective_area.effective_area, expected.to_value(u.cm**2 * u.DN / u.ph), rtol=1e-12)
