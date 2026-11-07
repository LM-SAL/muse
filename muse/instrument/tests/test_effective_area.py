import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument.effective_area import create_effective_area, effective_area_curve, read_effective_area

WAVELENGTH = np.linspace(160.0, 180.0, 5) * u.AA


def test_create_effective_area_layout():
    area = create_effective_area(
        [1.0, 2.0, 3.0, 4.0, 5.0], WAVELENGTH, 171 * u.AA, mirror=np.ones(5), attrs={"coating": "test"}
    )
    assert area.effective_area.dims == ("wavelength", "band")
    assert area.effective_area.attrs["units"] == "cm2"
    assert area.mirror.attrs["units"] == "cm2"
    assert area.wavelength.attrs["units"] == "Angstrom"
    assert area.band.values.tolist() == [171.0]
    assert area.attrs["coating"] == "test"
    assert "create_effective_area(" in area.attrs["HISTORY"][-1]


def test_create_effective_area_accepts_dn_per_photon_and_rejects_other_units():
    area = create_effective_area(np.ones(5) * u.cm**2 * u.DN / u.ph, WAVELENGTH, 94 * u.AA)
    assert u.Unit(area.effective_area.attrs["units"]) == u.cm**2 * u.DN / u.ph
    with pytest.raises(ValueError, match="cm2"):
        create_effective_area(np.ones(5) * u.m, WAVELENGTH, 94 * u.AA)
    with pytest.raises(ValueError, match="increasing"):
        create_effective_area(np.ones(5), WAVELENGTH[::-1], 94 * u.AA)


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_read_effective_area_round_trip(tmp_path, suffix):
    area = create_effective_area(np.arange(1.0, 6.0), WAVELENGTH, 171 * u.AA)
    path = tmp_path / f"area{suffix}"
    (area.to_zarr(path, consolidated=False) if suffix == ".zarr" else area.to_netcdf(path))
    back = read_effective_area(path)
    xr.testing.assert_allclose(back.effective_area, area.effective_area)
    curve = effective_area_curve(back)
    assert curve.dims == ("wavelength",)
    assert curve.attrs["units"] == "cm2"
    np.testing.assert_allclose(curve.wavelength.values, WAVELENGTH.value)
    assert "band" not in curve.coords


def test_read_effective_area_rejects_bad_files(tmp_path):
    xr.Dataset({"area": ("wavelength", np.ones(3))}).to_netcdf(tmp_path / "no_area.nc")
    with pytest.raises(ValueError, match="effective_area"):
        read_effective_area(tmp_path / "no_area.nc")
    bad = create_effective_area(np.ones(5), WAVELENGTH, 171 * u.AA)
    bad.effective_area.attrs["units"] = "m"
    bad.to_netcdf(tmp_path / "bad_units.nc")
    with pytest.raises(ValueError, match="cm2"):
        read_effective_area(tmp_path / "bad_units.nc")


def test_effective_area_curve_selects_a_band():
    area = create_effective_area(np.outer(np.arange(1.0, 6.0), [1.0, 2.0]), WAVELENGTH, [171, 284] * u.AA)
    with pytest.raises(ValueError, match="pass band"):
        effective_area_curve(area)
    np.testing.assert_allclose(effective_area_curve(area, 284 * u.AA).values, 2 * np.arange(1.0, 6.0))
