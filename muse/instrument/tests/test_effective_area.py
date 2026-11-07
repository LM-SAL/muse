import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument.effective_area import (
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
    with pytest.raises(ValueError, match="cm2"):
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
