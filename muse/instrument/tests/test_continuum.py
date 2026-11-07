"""
Tests for the continuum calculators and the fold onto bins.
"""

import sys
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.instrument import continuum as continuum_module
from muse.instrument.continuum import (
    _bin_average,
    bin_continuum,
    create_chianti_continuum,
    create_fiasco_continuum,
)
from muse.instrument.spectral import create_log_wavelength_edges
from muse.tests.helpers import synthetic_effective_area

NORMALIZATION = 1e-27
TEMPERATURE = xr.DataArray(10 ** np.array([5.5, 6.0, 6.5]) * u.K, dims="logT")
PRESSURE = xr.DataArray([3e14, 3e15] * u.K / u.cm**3, dims="pressure")
GRID = np.linspace(170.0, 172.0, 21) * u.AA


def _bin_width(response):
    return response.wavelength_bounds.isel(bound=1) - response.wavelength_bounds.isel(bound=0)


def _stepped_effective_area():
    """
    An area with a factor-of-three step at 171.8 A and a 1 %/A slope.
    """
    wavelength = np.linspace(165.0, 177.0, 121)
    values = np.where(wavelength < 171.8, 1.0, 3.0) * (1 + 0.01 * (wavelength - 165.0))
    return synthetic_effective_area(values, wavelength), wavelength, values


def _continuum_cube():
    """
    A sloped continuum on the stepped area's grid, so a Doppler shift changes the
    response.
    """
    _, wavelength, _ = _stepped_effective_area()
    density = (1.0 + 0.2 * (wavelength - 165.0)) * np.array([1.0, 2.0, 3.0])[:, None]
    response = xr.DataArray(
        np.stack([density, 0.5 * density, 0.1 * density]),
        dims=("line", "logT", "wavelength_bin"),
        coords={"line": ["free-free", "free-bound", "two-photon"], "logT": [5.5, 6.0, 6.5]},
        attrs={"units": "1e-27 erg cm3 / (Angstrom s sr)"},
    )
    return xr.Dataset(
        {"spectral_response": response},
        coords={"wavelength_grid": ("wavelength_bin", wavelength, {"units": "Angstrom"})},
        attrs={"normalization": NORMALIZATION},
    )


def test_linear_density_is_exact():
    x = np.linspace(0.0, 10.0, 101)
    values = xr.DataArray(2.0 + 0.5 * x, dims="wavelength_bin")
    edges = np.array([0.0, 1.3, 2.0, 7.7, 10.0])
    expected = [(2.0 * (b - a) + 0.25 * (b**2 - a**2)) / (b - a) for a, b in itertools.pairwise(edges)]
    np.testing.assert_allclose(_bin_average(values, x, edges).values, expected, rtol=1e-12)


def test_zero_outside_the_samples_and_conservative():
    x = np.linspace(0.0, 10.0, 101)
    y = np.exp(-0.5 * ((x - 4.0) / 1.5) ** 2)
    edges = np.array([-2.0, 0.0, 3.3, 6.1, 10.0, 13.0])
    out = _bin_average(xr.DataArray(y, dims="wavelength_bin"), x, edges)
    assert out.values[0] == 0.0
    assert out.values[-1] == 0.0
    np.testing.assert_allclose((out.values * np.diff(edges)).sum(), np.trapezoid(y, x), rtol=1e-12)


def test_extra_dims_and_attrs_are_kept():
    values = xr.DataArray(
        np.outer([1.0, 2.0], np.ones(11)),
        dims=("logT", "wavelength_bin"),
        coords={"logT": [5.0, 6.0]},
        attrs={"units": "u"},
    )
    out = _bin_average(values, np.linspace(0.0, 1.0, 11), np.array([0.0, 0.5, 1.0]))
    assert out.dims == ("logT", "wavelength_bin")
    assert out.attrs == {"units": "u"}
    np.testing.assert_allclose(out.values, [[1.0, 1.0], [2.0, 2.0]])


def test_total_matches_the_fine_integral():
    area, wavelength, values = _stepped_effective_area()
    edges = create_log_wavelength_edges([170, 172] * u.AA, 0.005)
    out = bin_continuum(_continuum_cube(), edges, effective_area=area)
    assert out.spectral_response.dims == ("line", "logT", "wavelength_bin")
    assert out.spectral_response.attrs["units"] == "1e-27 cm5 erg / (Angstrom s sr)"
    assert out.spectral_response.attrs["effective_area_at"] == "bin average"
    assert out.spectral_response.attrs["bin_method"] == "average"
    np.testing.assert_allclose(out.wavelength_grid.values, 0.5 * (edges[1:] + edges[:-1]).value)
    np.testing.assert_allclose(_bin_width(out).values, np.diff(edges.value))
    assert "bin_continuum(" in out.attrs["HISTORY"][-1]
    total = (out.spectral_response * _bin_width(out)).sum("wavelength_bin")
    x = np.linspace(edges[0].value, edges[-1].value, 200001)
    for i_line, factor in enumerate([1.0, 0.5, 0.1]):
        for i_t, scale in enumerate([1.0, 2.0, 3.0]):
            expected = factor * scale * np.trapezoid((1.0 + 0.2 * (x - 165.0)) * np.interp(x, wavelength, values), x)
            # the product of density and area is sampled on the 0.1 A grid and joined by
            # straight lines; the reference integrates the exact quadratic product
            np.testing.assert_allclose(float(total.isel(line=i_line, logT=i_t)), expected, rtol=2e-4)


def test_doppler_shift():
    area, _, _ = _stepped_effective_area()
    edges = create_log_wavelength_edges([170, 172] * u.AA, 0.005)
    velocities = np.array([-200.0, 0.0, 200.0]) * u.km / u.s
    out = bin_continuum(_continuum_cube(), edges, effective_area=area, doppler_velocity=velocities)
    assert out.spectral_response.dims == ("line", "doppler_velocity", "logT", "wavelength_bin")
    assert out.doppler_velocity.attrs["units"] == "km / s"
    still = bin_continuum(_continuum_cube(), edges, effective_area=area)
    xr.testing.assert_allclose(
        out.spectral_response.sel(doppler_velocity=0.0, drop=True), still.spectral_response, rtol=1e-12
    )
    total = (out.spectral_response * _bin_width(out)).sum("wavelength_bin")
    # the density rises with wavelength: light observed at lambda was emitted at
    # lambda / (1 + v / c), so a redshift shows the fainter short-wavelength emission
    assert (total.sel(doppler_velocity=200.0) < total.sel(doppler_velocity=0.0)).all()
    assert (total.sel(doppler_velocity=-200.0) > total.sel(doppler_velocity=0.0)).all()
    expected = -200.0 / const.c.to_value(u.km / u.s) * 171.0 * 0.2 / (1.0 + 0.2 * 6.0)
    ratio = (total.sel(doppler_velocity=200.0) / total.sel(doppler_velocity=0.0) - 1.0).values
    np.testing.assert_allclose(ratio, expected, rtol=0.1)


def test_rejects_bad_inputs():
    area, _, _ = _stepped_effective_area()
    with pytest.raises(ValueError, match="increasing"):
        bin_continuum(_continuum_cube(), np.array([170.0, 171.0, 170.5]) * u.AA, effective_area=area)
    with pytest.raises(ValueError, match="wavelength coordinate"):
        bin_continuum(
            _continuum_cube(),
            np.array([170.0, 171.0]) * u.AA,
            effective_area=xr.DataArray(2.0, attrs={"units": "cm2"}),
        )
    with pytest.raises(ValueError, match="spectral_response"):
        bin_continuum(xr.Dataset(), np.array([170.0, 171.0]) * u.AA, effective_area=area)


def _fake_intensity(temperatures, wavelengths, scale):
    return scale * np.outer(temperatures / 1e6, wavelengths / 171.0)


@pytest.fixture
def fake_chianti(monkeypatch):
    """
    ChiantiPy replaced by closed-form continua: H and He only, He II two-photon scaling
    with the density.
    """
    calls = []

    class Continuum:
        def __init__(self, ion_name, temperature, **_kwargs) -> None:
            if ion_name not in ("h_1", "h_2", "he_1", "he_2", "he_3"):
                msg = "no data"
                raise KeyError(msg)
            self.ion_name, self.temperature = ion_name, np.asarray(temperature, dtype=float)

        def freeFree(self, wavelength, **_kwargs):
            self.FreeFree = {"intensity": _fake_intensity(self.temperature, wavelength, 1.0)}

        def freeBound(self, wavelength, **_kwargs):
            if self.ion_name == "h_1":
                self.FreeBound = {"errorMessage": "no free-bound data"}
                return
            self.FreeBound = {"intensity": _fake_intensity(self.temperature, wavelength, 0.5)}

    class Ion:
        def __init__(self, ion_name, temperature, eDensity, **_kwargs) -> None:
            calls.append((ion_name, np.asarray(eDensity, dtype=float)))
            self.ion_name = ion_name
            self.temperature, self.eDensity = np.asarray(temperature, dtype=float), np.asarray(eDensity, dtype=float)

        def twoPhoton(self, wavelength):
            if self.ion_name != "he_2":
                self.TwoPhoton = {"errorMessage": "not hydrogenic"}
                return
            self.TwoPhoton = {
                "intensity": _fake_intensity(self.temperature, wavelength, 1.0) * (self.eDensity / 1e9)[:, None]
            }

    core = SimpleNamespace(continuum=Continuum, ion=Ion)
    constants = SimpleNamespace(El=["h", "he", "li"])
    data = SimpleNamespace(AbundanceDefault={"abundance": np.array([1.0, 0.085, 1e-9])})
    io = SimpleNamespace(
        abundanceRead=lambda _name: {"abundance": np.array([1.0, 0.085, 1e-9])}, versionRead=lambda: "fake 11"
    )
    tools = SimpleNamespace(constants=constants, data=data, io=io)
    for name, module in {
        "ChiantiPy": SimpleNamespace(core=core, tools=tools, __version__="fake"),
        "ChiantiPy.core": core,
        "ChiantiPy.tools": tools,
        "ChiantiPy.tools.constants": constants,
        "ChiantiPy.tools.data": data,
        "ChiantiPy.tools.io": io,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(continuum_module, "_initialize_chianti", lambda: ("fake", core))
    return calls


@pytest.mark.usefixtures("fake_chianti")
def test_chianti_continuum_layout_and_density_dependence(fake_chianti):
    continuum = create_chianti_continuum(
        TEMPERATURE, GRID, pressure=PRESSURE, abundance="sun_coronal_2021_chianti", minimum_abundance=0.01
    )
    response = continuum.spectral_response
    assert response.dims == ("line", "pressure", "logT", "wavelength_bin")
    assert list(continuum.line.values) == ["free-free", "free-bound", "two-photon"]
    assert response.attrs["units"] == "1e-27 cm3 erg / (Angstrom s sr)"
    assert continuum.logT.attrs["units"] == "dex(K)"
    assert continuum.wavelength_grid.attrs["bounds"] == "wavelength_bounds"
    assert np.isnan(continuum.line_wavelength.values).all()
    assert continuum.attrs["Chianti"] == "fake 11"
    assert continuum.attrs["abundance"] == "sun_coronal_2021_chianti"
    assert "create_chianti_continuum(" in continuum.attrs["HISTORY"][-1]
    temperatures, wavelengths = TEMPERATURE.values, GRID.value
    # five ions of H and He contribute free-free, four free-bound (h_1 lacks it), on both pressures
    expected_ff = 5 * _fake_intensity(temperatures, wavelengths, 1.0) / NORMALIZATION
    expected_fb = 4 * _fake_intensity(temperatures, wavelengths, 0.5) / NORMALIZATION
    for i_pressure in range(2):
        np.testing.assert_allclose(
            response.sel(line="free-free").isel(pressure=i_pressure).values, expected_ff, rtol=1e-12
        )
        np.testing.assert_allclose(
            response.sel(line="free-bound").isel(pressure=i_pressure).values, expected_fb, rtol=1e-12
        )
    # two-photon from He II only, at the density pressure / temperature of each pressure
    for i_pressure, pressure in enumerate(PRESSURE.values):
        density = pressure / temperatures
        expected_tp = _fake_intensity(temperatures, wavelengths, 1.0) * (density / 1e9)[:, None] / NORMALIZATION
        np.testing.assert_allclose(
            response.sel(line="two-photon").isel(pressure=i_pressure).values, expected_tp, rtol=1e-12
        )
    # hydrogen- and helium-like stages only: h_1, he_1 and he_2, each once per pressure
    assert sorted({name for name, _ in fake_chianti}) == ["h_1", "he_1", "he_2"]
    assert len(fake_chianti) == 6


@pytest.mark.usefixtures("fake_chianti")
def test_chianti_continuum_density_grid_and_single_temperature():
    density = xr.DataArray([1e9, 1e10] * u.cm**-3, dims="density")
    one = create_chianti_continuum(TEMPERATURE.isel(logT=[1]), GRID, density=density, minimum_abundance=0.01)
    assert one.spectral_response.dims == ("line", "logT", "logD", "wavelength_bin")
    np.testing.assert_allclose(one.logD.values, [9.0, 10.0])
    assert one.logD.attrs["units"] == "dex(1 / cm3)"
    assert one.sizes["logT"] == 1
    two_photon = one.spectral_response.sel(line="two-photon").isel(logT=0)
    np.testing.assert_allclose(two_photon.isel(logD=1).values, 10 * two_photon.isel(logD=0).values, rtol=1e-12)


@pytest.mark.usefixtures("fake_chianti")
def test_chianti_continuum_requires_a_contributing_ion():
    with pytest.raises(RuntimeError, match="free-free"):
        create_chianti_continuum(TEMPERATURE, GRID, pressure=PRESSURE, minimum_abundance=10.0)


def test_continuum_rejects_bad_grids():
    with pytest.raises(ValueError, match="density or pressure"):
        create_chianti_continuum(TEMPERATURE, GRID)
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_chianti_continuum(
            TEMPERATURE, GRID, pressure=PRESSURE, density=xr.DataArray([1e9] * u.cm**-3, dims="density")
        )
    with pytest.raises(ValueError, match="increasing"):
        create_chianti_continuum(TEMPERATURE, np.array([172.0, 171.0]) * u.AA, pressure=PRESSURE)


@pytest.fixture
def fake_fiasco(monkeypatch):
    """
    Fiasco replaced by closed-form continua that are not per steradian, as fiasco's are.
    """
    fiasco = pytest.importorskip("fiasco")

    class Ion:
        def __init__(self, atomic_number, charge_state, temperature) -> None:
            self.atomic_number, self.charge_state, self.temperature = atomic_number, charge_state, temperature

    class Element:
        def __init__(self, atomic_number, temperature, **_kwargs) -> None:
            if atomic_number > 2:
                msg = "not in the database"
                raise KeyError(msg)
            self.abundance = {1: 1.0, 2: 0.085}[atomic_number]
            self.ions = [Ion(atomic_number, charge, temperature) for charge in range(atomic_number + 1)]

        def __iter__(self):
            return iter(self.ions)

    class IonCollection:
        def __init__(self, *ions) -> None:
            self.ions = ions

        def _intensity(self, wavelength, scale):
            temperatures = self.ions[0].temperature.to_value(u.K)
            return (
                len(self.ions)
                * _fake_intensity(temperatures, wavelength.to_value(u.AA), scale)
                * u.erg
                * u.cm**3
                / (u.s * u.AA)
            )

        def free_free(self, wavelength):
            return 4 * np.pi * self._intensity(wavelength, 1.0)

        def free_bound(self, wavelength):
            return 4 * np.pi * self._intensity(wavelength, 0.5)

        def two_photon(self, wavelength, electron_density):
            density = np.atleast_1d(electron_density.to_value(u.cm**-3))
            base = self._intensity(wavelength, 1.0)
            return 4 * np.pi * base[:, None, :] * (density / 1e9)[None, :, None]

    monkeypatch.setattr(fiasco, "Element", Element)
    monkeypatch.setattr(fiasco, "IonCollection", IonCollection)
    monkeypatch.setattr(fiasco, "__version__", "fake")


@pytest.mark.usefixtures("fake_fiasco")
def test_fiasco_continuum_matches_the_chianti_layout_per_steradian():
    continuum = create_fiasco_continuum(TEMPERATURE, GRID, pressure=PRESSURE, minimum_abundance=0.01)
    response = continuum.spectral_response
    assert response.dims == ("line", "pressure", "logT", "wavelength_bin")
    assert continuum.attrs["fiasco"] == "fake"
    temperatures, wavelengths = TEMPERATURE.values, GRID.value
    # five ions of H and He, divided by 4 pi; two-photon from the three hydrogen- and helium-like ones
    np.testing.assert_allclose(
        response.sel(line="free-free").isel(pressure=0).values,
        5 * _fake_intensity(temperatures, wavelengths, 1.0) / NORMALIZATION,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        response.sel(line="free-bound").isel(pressure=1).values,
        5 * _fake_intensity(temperatures, wavelengths, 0.5) / NORMALIZATION,
        rtol=1e-12,
    )
    for i_pressure, pressure in enumerate(PRESSURE.values):
        density = pressure / temperatures
        expected = 3 * _fake_intensity(temperatures, wavelengths, 1.0) * (density / 1e9)[:, None] / NORMALIZATION
        np.testing.assert_allclose(
            response.sel(line="two-photon").isel(pressure=i_pressure).values, expected, rtol=1e-12
        )


@pytest.mark.usefixtures("fake_fiasco")
def test_fiasco_continuum_density_grid():
    density = xr.DataArray([1e9, 1e10] * u.cm**-3, dims="density")
    continuum = create_fiasco_continuum(TEMPERATURE, GRID, density=density, minimum_abundance=0.01)
    assert continuum.spectral_response.dims == ("line", "logT", "logD", "wavelength_bin")
    two_photon = continuum.spectral_response.sel(line="two-photon")
    np.testing.assert_allclose(two_photon.isel(logD=1).values, 10 * two_photon.isel(logD=0).values, rtol=1e-12)


@pytest.mark.usefixtures("fake_fiasco")
def test_fiasco_continuum_requires_an_ion():
    with pytest.raises(RuntimeError, match="no ion"):
        create_fiasco_continuum(TEMPERATURE, GRID, pressure=PRESSURE, minimum_abundance=10.0)


@pytest.mark.chianti
def test_chianti_continuum_live_hydrogen_and_helium():
    pytest.importorskip("ChiantiPy")
    continuum = create_chianti_continuum(
        TEMPERATURE, GRID, pressure=PRESSURE, abundance="sun_coronal_2021_chianti", minimum_abundance=0.05
    )
    free_free = continuum.spectral_response.sel(line="free-free")
    assert (free_free > 0).all()
    # hotter plasma radiates more free-free emission at these wavelengths
    assert (free_free.isel(logT=2) > free_free.isel(logT=0)).all()
    assert continuum.attrs["Chianti"].startswith("11")


def test_fiasco_continuum_live_hydrogen_and_helium():
    fiasco = pytest.importorskip("fiasco")
    from pathlib import Path  # noqa: PLC0415

    if not Path(fiasco.defaults["hdf5_dbase_root"]).is_file():
        pytest.skip("fiasco's HDF5 database is not available")
    continuum = create_fiasco_continuum(
        TEMPERATURE, GRID, pressure=PRESSURE, abundance="sun_coronal_2021_chianti", minimum_abundance=0.05
    )
    free_free = continuum.spectral_response.sel(line="free-free")
    assert (free_free > 0).all()
    assert (free_free.isel(logT=2) > free_free.isel(logT=0)).all()


@pytest.mark.chianti
def test_both_backends_agree_live():
    pytest.importorskip("ChiantiPy")
    fiasco = pytest.importorskip("fiasco")
    from pathlib import Path  # noqa: PLC0415

    if not Path(fiasco.defaults["hdf5_dbase_root"]).is_file():
        pytest.skip("fiasco's HDF5 database is not available")
    kwargs = {"pressure": PRESSURE, "abundance": "sun_coronal_2021_chianti", "minimum_abundance": 0.05}
    chianti = create_chianti_continuum(TEMPERATURE, GRID, **kwargs).spectral_response
    fiasco_continuum = create_fiasco_continuum(TEMPERATURE, GRID, **kwargs).spectral_response
    # the same hydrogen and helium ions, the same database: per steradian both agree closely
    for component in ("free-free", "free-bound"):
        xr.testing.assert_allclose(fiasco_continuum.sel(line=component), chianti.sel(line=component), rtol=0.01)
