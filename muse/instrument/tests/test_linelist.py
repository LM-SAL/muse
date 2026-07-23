import os
import warnings

import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument import linelist
from muse.instrument.linelist import create_chianti_line_list


def test_rejects_missing_density_and_pressure():
    temperature = xr.DataArray([1e6] * u.K, dims="logT")
    with pytest.raises(ValueError, match="Specify density or pressure"):
        create_chianti_line_list(temperature)


def test_rejects_both_density_and_pressure():
    temperature = xr.DataArray([1e6] * u.K, dims="logT")
    grid = xr.DataArray([1e9] / u.cm**3, dims="density")
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_chianti_line_list(temperature, density=grid, pressure=grid)


@pytest.mark.parametrize(
    ("temperature", "error", "error_type"),
    [
        (np.array([1e6]) * u.K, "xarray.DataArray", TypeError),
        (xr.DataArray([1e6] * u.K, dims="temperature"), "one-dimensional logT", ValueError),
        (xr.DataArray([] * u.K, dims="logT"), "not be empty", ValueError),
        (xr.DataArray([1e6], dims="logT"), "astropy.units.Quantity", TypeError),
        (xr.DataArray([np.nan] * u.K, dims="logT"), "finite", ValueError),
        (xr.DataArray([0.0] * u.K, dims="logT"), "positive", ValueError),
        (xr.DataArray([1e6] * u.m, dims="logT"), "convertible", ValueError),
    ],
)
def test_rejects_invalid_temperature(temperature, error, error_type):
    pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
    with pytest.raises(error_type, match=error):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=[170, 172] * u.AA, ion_list=["fe_9"])


@pytest.mark.parametrize(
    ("pressure", "error", "error_type"),
    [
        ([3e15] * u.K / u.cm**3, "xarray.DataArray", TypeError),
        (xr.DataArray([[3e15]] * u.K / u.cm**3, dims=("pressure", "sample")), "one-dimensional", ValueError),
        (xr.DataArray([3e15], dims="pressure"), "astropy.units.Quantity", TypeError),
        (xr.DataArray([np.nan] * u.K / u.cm**3, dims="pressure"), "finite", ValueError),
        (xr.DataArray([0.0] * u.K / u.cm**3, dims="pressure"), "positive", ValueError),
        (xr.DataArray([3e15] * u.m, dims="pressure"), "convertible", ValueError),
        (xr.DataArray([3e15] * u.K / u.cm**3, dims="logT"), "must not be named", ValueError),
    ],
)
def test_rejects_invalid_plasma_grid(pressure, error, error_type):
    temperature = xr.DataArray([1e6] * u.K, dims="logT")
    with pytest.raises(error_type, match=error):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=[170, 172] * u.AA, ion_list=["fe_9"])


@pytest.mark.parametrize(
    ("wavelength_range", "error", "error_type"),
    [
        (None, "astropy.units.Quantity", TypeError),
        ([170] * u.AA, "exactly two", ValueError),
        ([170, np.nan] * u.AA, "finite", ValueError),
        ([172, 170] * u.AA, "in increasing order", ValueError),
        ([170, 172] * u.s, "convertible", ValueError),
    ],
)
def test_rejects_invalid_wavelength_range(wavelength_range, error, error_type):
    temperature = xr.DataArray([1e6] * u.K, dims="logT")
    pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
    with pytest.raises(error_type, match=error):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=wavelength_range, ion_list=["fe_9"])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"element_list": ["fe"], "ion_list": ["fe_9"]}, "mutually exclusive"),
        ({"minimum_abundance": 1e-5, "ion_list": ["fe_9"]}, "minimum_abundance"),
        ({"ion_list": []}, "non-empty"),
        ({"ion_list": ["Fe IX"]}, "invalid ion_list"),
    ],
)
def test_rejects_invalid_species_selection(kwargs, error):
    temperature = xr.DataArray([1e6] * u.K, dims="logT")
    pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
    with pytest.raises(ValueError, match=error):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=[170, 172] * u.AA, **kwargs)


@pytest.mark.parametrize(
    ("minimum_abundance", "error", "error_type"),
    [
        (None, "Specify minimum_abundance", ValueError),
        (True, "real number", TypeError),
        (0, "finite and positive", ValueError),
        (np.inf, "finite and positive", ValueError),
    ],
)
def test_rejects_invalid_minimum_abundance(minimum_abundance, error, error_type):
    temperature = xr.DataArray([1e6] * u.K, dims="logT")
    pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
    with pytest.raises(error_type, match=error):
        create_chianti_line_list(
            temperature,
            pressure=pressure,
            wavelength_range=[170, 172] * u.AA,
            minimum_abundance=minimum_abundance,
        )


def test_converts_units_for_chianti(monkeypatch):
    captured = {}

    class FakeChianti:
        @staticmethod
        def bunch(temperature, density, wavelength_range, **kwargs):
            captured.update(temperature=temperature, density=density, wavelength_range=wavelength_range, **kwargs)
            return type("FakeBunch", (), {"AbundanceName": "chianti/sun_coronal_2021_chianti.abund"})()

    monkeypatch.setattr(linelist, "_initialize_chianti", lambda: ("test", FakeChianti))
    monkeypatch.setattr(
        linelist,
        "_chianti_bunch_to_dataset",
        lambda *_args, **_kwargs: xr.Dataset(coords={"trans_index": [0]}),
    )

    line_list = create_chianti_line_list(
        xr.DataArray([1] * u.MK, dims="logT"),
        pressure=xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure"),
        abundance="sun_coronal_2021_chianti",
        wavelength_range=[17, 17.2] * u.nm,
        minimum_abundance=np.float64(1e-5),
    )

    np.testing.assert_allclose(captured["temperature"], [1e6])
    np.testing.assert_allclose(captured["density"], [3e9])
    np.testing.assert_allclose(captured["wavelength_range"], [170, 172])
    assert type(captured["minAbund"]) is float
    assert line_list.attrs["abundance"] == "sun_coronal_2021_chianti"


def test_no_lines_raises(monkeypatch):
    class FakeChianti:
        @staticmethod
        def bunch(*_args, **_kwargs):
            return object()

    monkeypatch.setattr(linelist, "_initialize_chianti", lambda: ("test", FakeChianti))
    monkeypatch.setattr(
        linelist,
        "_chianti_bunch_to_dataset",
        lambda *_args, **_kwargs: xr.Dataset(coords={"trans_index": []}),
    )
    temperature = xr.DataArray([1e6] * u.K, dims="logT")
    pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
    with pytest.raises(ValueError, match="no lines"):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=[170, 172] * u.AA, ion_list=["fe_9"])


def test_missing_xuvtop_raises(monkeypatch):
    monkeypatch.delenv("XUVTOP", raising=False)
    temperature = xr.DataArray([1e6] * u.K, dims="logT")
    pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
    with pytest.raises(OSError, match="XUVTOP"):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=[170, 172] * u.AA, ion_list=["fe_9"])


@pytest.mark.remote_data
def test_create_chianti_line_list_live(monkeypatch):
    assert os.environ.get("XUVTOP")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        import ChiantiPy.tools.data as chdata  # noqa: PLC0415

    monkeypatch.delattr(chdata, "Defaults")
    temperature = xr.DataArray(10 ** np.arange(5.6, 6.2, 0.2) * u.K, dims="logT")
    pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
    line_list = create_chianti_line_list(
        abundance="sun_coronal_2021_chianti",
        wavelength_range=[170, 172] * u.AA,
        temperature=temperature,
        pressure=pressure,
        ion_list=["fe_9"],
    )
    assert line_list.sizes["trans_index"] > 0
    assert "Fe IX 171.073" in line_list.full_name.values
    assert (line_list.wavelength > 170).all()
    assert (line_list.wavelength < 172).all()
    assert {"ion_name", "atomic_number", "spectroscopic_name", "logT_peak"} <= set(line_list.data_vars)
    assert set(line_list.ion_name.values) == {"fe_9"}
    assert line_list.attrs["abundance"] == "sun_coronal_2021_chianti"
    assert line_list.attrs["ion_list"] == ["fe_9"]
    assert "create_chianti_line_list(" in line_list.attrs["HISTORY"][0]


@pytest.mark.remote_data
def test_create_chianti_line_list_live_density():
    assert os.environ.get("XUVTOP")
    temperature = xr.DataArray(10 ** np.arange(5.6, 6.2, 0.2) * u.K, dims="logT")
    density = xr.DataArray([1e8, 1e9] / u.cm**3, dims="density")
    line_list = create_chianti_line_list(
        temperature=temperature,
        density=density,
        wavelength_range=[170, 172] * u.AA,
        ion_list=["fe_9"],
    )
    assert line_list.gofnt.dims == ("logT", "logD", "trans_index")
    np.testing.assert_allclose(line_list.logD.values, [8.0, 9.0])
    assert line_list.logT.attrs["units"] == "dex(K)"
    assert line_list.logD.attrs["units"] == "dex(1 / cm3)"
    assert line_list.sizes["trans_index"] > 0
    assert "Fe IX 171.073" in line_list.full_name.values
