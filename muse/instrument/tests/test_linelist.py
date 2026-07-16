import os
import warnings

import numpy as np
import pytest
import xarray as xr

from muse.instrument.linelist import (
    _save_compressed_netcdf,
    chianti_line_list,
    get_line_list,
    line_list_cache_path,
)


@pytest.fixture
def fake_linelist():
    return xr.Dataset(
        {
            "wavelength": xr.DataArray([171.073, 174.531], dims="trans_index"),
            "gofnt": xr.DataArray(
                np.ones((3, 1, 1, 2)),
                dims=("logT", "pressure", "abundance", "trans_index"),
                coords={"logT": [5.8, 5.9, 6.0], "pressure": [3e15]},
            ),
            "full_name": xr.DataArray(np.array(["Fe IX 171.073", "Fe X 174.531"], dtype=object), dims="trans_index"),
        }
    )


def test_cache_path_naming(tmp_path):
    full = line_list_cache_path(tmp_path, "sun_coronal_2021_chianti", (100.5, 110.9))
    assert full == tmp_path / "ll_wvl100.5_110.9_sun_coronal_2021_chianti.ncdf"
    path = line_list_cache_path(tmp_path, "abund", (100, 110), density_dependent=True)
    assert path.name == "ll_wvl_eDens100_110_abund.ncdf"
    elements = line_list_cache_path(tmp_path, "abund", (100, 110), element_list=["Fe", "O"])
    assert elements.name == "ll_wvl100_110_abund_elements-fe-o.ncdf"
    ions = line_list_cache_path(tmp_path, "abund", (100, 110), ion_list=["fe_9", "Fe_10"])
    assert ions.name == "ll_wvl100_110_abund_ions-fe_10-fe_9.ncdf"
    assert ions != line_list_cache_path(tmp_path, "abund", (100, 110))


def test_get_line_list_uses_restricted_cache(tmp_path, fake_linelist, monkeypatch):
    captured = {}

    def fake_chianti_line_list(**kwargs):
        captured.update(kwargs)
        return fake_linelist

    monkeypatch.setattr("muse.instrument.linelist.chianti_line_list", fake_chianti_line_list)
    kwargs = {
        "output_dir": tmp_path,
        "abundance": "abund",
        "wavelength_range": (170, 175),
        "temperature": 10**fake_linelist.logT,
        "pressure": fake_linelist.pressure,
        "ion_list": ["fe_9", "Fe_10"],
    }
    generated = get_line_list(**kwargs)
    cached = get_line_list(**kwargs, compute_if_missing=False)

    assert captured["ion_list"] == ["fe_10", "fe_9"]
    assert line_list_cache_path(tmp_path, "abund", (170, 175), ion_list=kwargs["ion_list"]).exists()
    assert not line_list_cache_path(tmp_path, "abund", (170, 175)).exists()
    xr.testing.assert_identical(cached, generated)


def test_get_line_list_loads_cache(tmp_path, fake_linelist):
    cache_path = line_list_cache_path(tmp_path, "abund", (170, 175))
    _save_compressed_netcdf(cache_path, fake_linelist)
    loaded = get_line_list(
        output_dir=tmp_path,
        abundance="abund",
        wavelength_range=(170, 175),
        temperature=10**fake_linelist.logT,
        pressure=fake_linelist.pressure,
        compute_if_missing=False,
    )
    xr.testing.assert_allclose(loaded.gofnt, fake_linelist.gofnt)
    assert list(loaded.full_name.values) == list(fake_linelist.full_name.values)


def test_get_line_list_explicit_path(tmp_path, fake_linelist):
    explicit = tmp_path / "custom_name.ncdf"
    _save_compressed_netcdf(explicit, fake_linelist)
    loaded = get_line_list(
        output_dir=tmp_path,
        abundance="ignored",
        wavelength_range=(0, 1),
        temperature=10**fake_linelist.logT,
        line_list_file=explicit,
    )
    xr.testing.assert_allclose(loaded.gofnt, fake_linelist.gofnt)
    with pytest.raises(ValueError, match="explicit line_list_file"):
        get_line_list(
            output_dir=tmp_path,
            abundance="ignored",
            wavelength_range=(0, 1),
            temperature=10**fake_linelist.logT,
            line_list_file=explicit,
            ion_list=["fe_9"],
        )


def test_get_line_list_missing_raises(tmp_path, fake_linelist):
    with pytest.raises(FileNotFoundError, match="line-list cache"):
        get_line_list(
            output_dir=tmp_path,
            abundance="abund",
            wavelength_range=(170, 175),
            temperature=10**fake_linelist.logT,
            pressure=fake_linelist.pressure,
            compute_if_missing=False,
        )


def test_rejects_both_density_and_pressure():
    temperature = xr.DataArray([1e6], dims="logT")
    grid = xr.DataArray([1e9], dims="density")
    with pytest.raises(ValueError, match="exactly one"):
        chianti_line_list(temperature, density=grid, pressure=grid)
    with pytest.raises(ValueError, match="exactly one"):
        chianti_line_list(temperature)


def test_rejects_missing_wavelength_range():
    temperature = xr.DataArray([1e6], dims="logT")
    pressure = xr.DataArray([3e15], dims="pressure")
    with pytest.raises(ValueError, match="wavelength_range"):
        chianti_line_list(temperature, pressure=pressure)


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
        chianti_line_list(temperature, pressure=pressure, wavelength_range=(170, 172), **kwargs)


@pytest.mark.parametrize(("workers", "error"), [(0, ValueError), (True, TypeError), (1.5, TypeError)])
def test_rejects_invalid_workers(workers, error):
    temperature = xr.DataArray([1e6], dims="logT")
    pressure = xr.DataArray([3e15], dims="pressure")
    with pytest.raises(error, match="workers"):
        chianti_line_list(temperature, pressure=pressure, wavelength_range=(170, 172), workers=workers)


def test_missing_xuvtop_raises(monkeypatch):
    pytest.importorskip("ChiantiPy")
    monkeypatch.delenv("XUVTOP", raising=False)
    temperature = xr.DataArray([1e6], dims="logT")
    pressure = xr.DataArray([3e15], dims="pressure")
    with pytest.raises(OSError, match="XUVTOP"):
        chianti_line_list(temperature, pressure=pressure, wavelength_range=(170, 172))


@pytest.mark.skipif(os.environ.get("XUVTOP") is None, reason="CHIANTI database (XUVTOP) not available")
def test_line_list_live(tmp_path, monkeypatch):
    pytest.importorskip("ChiantiPy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        import ChiantiPy.tools.data as chdata  # noqa: PLC0415

    monkeypatch.delattr(chdata, "Defaults")
    temperature = 10 ** xr.DataArray(np.arange(5.6, 6.2, 0.2), dims="logT")
    pressure = xr.DataArray([3e15], dims="pressure")
    line_list = get_line_list(
        output_dir=tmp_path,
        abundance="sun_coronal_2021_chianti",
        wavelength_range=(170, 172),
        temperature=temperature,
        pressure=pressure,
        minimum_abundance=1e-5,
        workers=2,
    )
    assert line_list.sizes["trans_index"] > 0
    assert "Fe IX 171.073" in line_list.full_name.values
    assert (line_list.wavelength > 170).all()
    assert (line_list.wavelength < 172).all()
    assert {"ion_name", "atomic_number", "spectroscopic_name", "logT_peak"} <= set(line_list.data_vars)

    cached = get_line_list(
        output_dir=tmp_path,
        abundance="sun_coronal_2021_chianti",
        wavelength_range=(170, 172),
        temperature=temperature,
        pressure=pressure,
        compute_if_missing=False,
    )
    xr.testing.assert_allclose(cached.gofnt, line_list.gofnt)

    restricted = get_line_list(
        output_dir=tmp_path,
        abundance="sun_coronal_2021_chianti",
        wavelength_range=(170, 172),
        temperature=temperature,
        pressure=pressure,
        ion_list=["fe_9"],
    )
    assert set(restricted.ion_name.values) == {"fe_9"}
    assert line_list_cache_path(
        tmp_path,
        "sun_coronal_2021_chianti",
        (170, 172),
        ion_list=["fe_9"],
    ).exists()
