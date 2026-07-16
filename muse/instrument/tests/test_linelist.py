import os
import warnings

import numpy as np
import pytest
import xarray as xr

from muse.instrument.linelist import create_chianti_line_list


def test_rejects_missing_density_and_pressure():
    temperature = xr.DataArray([1e6], dims="logT")
    with pytest.raises(ValueError, match="Specify density or pressure"):
        create_chianti_line_list(temperature)


def test_rejects_both_density_and_pressure():
    temperature = xr.DataArray([1e6], dims="logT")
    grid = xr.DataArray([1e9], dims="density")
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_chianti_line_list(temperature, density=grid, pressure=grid)


@pytest.mark.parametrize(
    ("temperature", "error", "error_type"),
    [
        (np.array([1e6]), "xarray.DataArray", TypeError),
        (xr.DataArray([1e6], dims="temperature"), "one-dimensional logT", ValueError),
        (xr.DataArray([], dims="logT"), "not be empty", ValueError),
        (xr.DataArray(["hot"], dims="logT"), "numeric", TypeError),
        (xr.DataArray([np.nan], dims="logT"), "finite", ValueError),
        (xr.DataArray([0.0], dims="logT"), "positive", ValueError),
    ],
)
def test_rejects_invalid_temperature(temperature, error, error_type):
    pressure = xr.DataArray([3e15], dims="pressure")
    with pytest.raises(error_type, match=error):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=(170, 172))


@pytest.mark.parametrize(
    ("pressure", "error", "error_type"),
    [
        ([3e15], "xarray.DataArray", TypeError),
        (xr.DataArray([[3e15]], dims=("pressure", "sample")), "one-dimensional", ValueError),
        (xr.DataArray([np.nan], dims="pressure"), "finite", ValueError),
        (xr.DataArray([0.0], dims="pressure"), "positive", ValueError),
    ],
)
def test_rejects_invalid_plasma_grid(pressure, error, error_type):
    temperature = xr.DataArray([1e6], dims="logT")
    with pytest.raises(error_type, match=error):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=(170, 172))


@pytest.mark.parametrize(
    ("wavelength_range", "error"),
    [
        (None, "exactly two"),
        ((170, np.nan), "finite"),
        ((172, 170), "in increasing order"),
    ],
)
def test_rejects_invalid_wavelength_range(wavelength_range, error):
    temperature = xr.DataArray([1e6], dims="logT")
    pressure = xr.DataArray([3e15], dims="pressure")
    with pytest.raises(ValueError, match=error):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=wavelength_range)


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
    temperature = xr.DataArray([1e6], dims="logT")
    pressure = xr.DataArray([3e15], dims="pressure")
    with pytest.raises(ValueError, match=error):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=(170, 172), **kwargs)


def test_missing_xuvtop_raises(monkeypatch):
    pytest.importorskip("ChiantiPy")
    monkeypatch.delenv("XUVTOP", raising=False)
    temperature = xr.DataArray([1e6], dims="logT")
    pressure = xr.DataArray([3e15], dims="pressure")
    with pytest.raises(OSError, match="XUVTOP"):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=(170, 172))


@pytest.mark.skipif(os.environ.get("XUVTOP") is None, reason="CHIANTI database (XUVTOP) not available")
def test_create_chianti_line_list_live(monkeypatch):
    pytest.importorskip("ChiantiPy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        import ChiantiPy.tools.data as chdata  # noqa: PLC0415

    monkeypatch.delattr(chdata, "Defaults")
    temperature = 10 ** xr.DataArray(np.arange(5.6, 6.2, 0.2), dims="logT")
    pressure = xr.DataArray([3e15], dims="pressure")
    line_list = create_chianti_line_list(
        abundance="sun_coronal_2021_chianti",
        wavelength_range=(170, 172),
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
