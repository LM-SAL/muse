import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument import linelist
from muse.instrument.linelist import (
    apply_line_boosts,
    create_chianti_line_list,
    create_fiasco_line_list,
    rank_line_list,
    select_transitions,
)
from muse.tests.helpers import synthetic_effective_area, synthetic_line_list


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
        minimum_abundance=np.float64(1e-6),
    )

    np.testing.assert_allclose(captured["temperature"], [1e6])
    np.testing.assert_allclose(captured["density"], [3e9])
    np.testing.assert_allclose(captured["wavelength_range"], [170, 172])
    assert type(captured["minAbund"]) is float
    assert captured["minAbund"] == 1e-6
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
    pytest.importorskip("ChiantiPy")
    monkeypatch.delenv("XUVTOP", raising=False)
    temperature = xr.DataArray([1e6] * u.K, dims="logT")
    pressure = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
    with pytest.raises(OSError, match="XUVTOP"):
        create_chianti_line_list(temperature, pressure=pressure, wavelength_range=[170, 172] * u.AA, ion_list=["fe_9"])


@pytest.mark.chianti
def test_create_chianti_line_list_live_minimum_abundance_scalar_grid(monkeypatch):
    pytest.importorskip("ChiantiPy")
    assert os.environ.get("XUVTOP")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        chdata = pytest.importorskip("ChiantiPy.tools.data")

    monkeypatch.delattr(chdata, "Defaults")
    temperature = xr.DataArray([1e6] * u.K, dims="logT")
    density = xr.DataArray([1e9] / u.cm**3, dims="density")
    line_list = create_chianti_line_list(
        abundance="sun_coronal_2021_chianti",
        wavelength_range=[170, 172] * u.AA,
        temperature=temperature,
        density=density,
        minimum_abundance=1e-6,
    )
    assert line_list.gofnt.sizes == {"logT": 1, "logD": 1, "trans_index": line_list.sizes["trans_index"]}
    assert line_list.sizes["trans_index"] > 0
    assert "Fe IX 171.073" in line_list.full_name.values
    assert (line_list.wavelength >= 170).all()
    assert (line_list.wavelength <= 172).all()
    assert {"ion_name", "atomic_number", "spectroscopic_name", "logT_peak"} <= set(line_list.data_vars)
    assert "fe_9" in line_list.ion_name.values
    assert line_list.attrs["abundance"] == "sun_coronal_2021_chianti"
    assert line_list.attrs["minimum_abundance"] == 1e-6
    assert "create_chianti_line_list(" in line_list.attrs["HISTORY"][0]


@pytest.mark.chianti
def test_create_chianti_line_list_live_density():
    pytest.importorskip("ChiantiPy")
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


try:
    import fiasco
except ImportError:  # pragma: no cover
    fiasco = None

needs_fiasco_line_lists = pytest.mark.skipif(
    fiasco is None or not hasattr(fiasco.IonCollection, "build_line_list"),
    reason="fiasco with IonCollection.build_line_list is not installed",
)


@pytest.fixture
def temperature():
    return xr.DataArray(10 ** np.linspace(5.5, 6.5, 5) * u.K, dims="logT")


@pytest.fixture
def controlled_ions(monkeypatch):
    """
    Keep fiasco's real line-list builder, replacing only the database-backed ion data.
    """

    class Ion:
        def __init__(self, name, temperature, *, abundance="controlled_abundance", **_kwargs) -> None:
            self.temperature = temperature.to(u.K)
            self.abundance = 1e-4
            self._ion_name = name
            self.ion_name = "Fe 9"
            self.ion_name_roman = "Fe IX"
            self.atomic_number = 26
            self.atomic_symbol = "Fe"
            self.ionization_potential = 233 * u.eV
            self._dset_names = {"abundance": abundance, "ionization_fraction": "controlled"}
            self._elvlc = SimpleNamespace(version="controlled CHIANTI")
            wavelength = np.array([171.073, 174.0]) * u.AA
            self.transitions = SimpleNamespace(
                wavelength=wavelength,
                is_bound_bound=np.array([True, True]),
                delta_energy=wavelength.to(u.erg, equivalencies=u.spectral()),
                A=np.array([1.0, 0.1]) / u.s,
                lower_level=np.array([1, 1]),
                upper_level=np.array([2, 3]),
                lower_label=np.array(["lower", "lower"]),
                upper_label=np.array(["upper", "upper"]),
                is_observed=np.array([True, False]),
            )

        def contribution_function(self, density, **_kwargs):
            peak = np.exp(-(((np.log10(self.temperature.to_value(u.K)) - 5.9) / 0.2) ** 2))
            values = peak * density.to_value(u.cm**-3) / 1e9
            return values[:, None, None] * np.array([1.0, 0.1])[None, None, :] * 1e-25 * u.cm**3 * u.erg / u.s

    monkeypatch.setattr(fiasco, "Ion", Ion)
    monkeypatch.setattr(fiasco, "proton_electron_ratio", lambda temperature, **_kwargs: np.ones(temperature.size))
    monkeypatch.setattr(fiasco, "list_ions", lambda **_kwargs: ["fe_9"])


@needs_fiasco_line_lists
@pytest.mark.usefixtures("controlled_ions")
def test_fiasco_line_list_has_the_chianti_layout(temperature):
    pressure = xr.DataArray([3e14, 1e15] * u.K / u.cm**3, dims="pressure")
    line_list = create_fiasco_line_list(temperature, pressure=pressure, wavelength_range=[170, 175] * u.AA)
    assert {
        "ion_name",
        "wavelength",
        "lower_level_label",
        "upper_level_label",
        "lower_level_index",
        "upper_level_index",
        "spectroscopic_name",
        "atomic_number",
        "observed",
        "gofnt",
        "logT_peak",
        "full_name",
    } <= set(line_list.data_vars)
    assert line_list.gofnt.dims == ("pressure", "logT", "trans_index")
    assert line_list.full_name.values.tolist() == ["Fe IX 171.073", "Fe IX 174.0"]
    assert line_list.ion_name.values.tolist() == ["fe_9", "fe_9"]
    assert line_list.observed.values.tolist() == [True, False]
    assert line_list.wavelength.attrs["units"] == "Angstrom"
    assert line_list.gofnt.attrs["units"] == "erg cm3 / (s sr)"
    assert line_list.logT.attrs["units"] == "dex(K)"
    np.testing.assert_allclose(line_list.logT.values, np.linspace(5.5, 6.5, 5))
    # the contribution function per steradian, on the density pressure / temperature
    density = pressure.values[0] / temperature.values[0]
    expected = np.exp(-(((5.5 - 5.9) / 0.2) ** 2)) * density / 1e9 * 1e-25 / (4 * np.pi)
    np.testing.assert_allclose(line_list.gofnt.isel(pressure=0, logT=0, trans_index=0).item(), expected, rtol=1e-12)
    assert line_list.attrs["Chianti"] == "controlled CHIANTI"
    assert line_list.attrs["abundance"] == "controlled_abundance"
    assert "create_fiasco_line_list(" in line_list.attrs["HISTORY"][-1]
    assert float(line_list.logT_peak.isel(pressure=0, trans_index=0)) == pytest.approx(6.0, abs=0.26)


@needs_fiasco_line_lists
@pytest.mark.usefixtures("controlled_ions")
def test_fiasco_line_list_density_grid_and_range_units(temperature):
    density = xr.DataArray([1e9] * u.cm**-3, dims="density")
    line_list = create_fiasco_line_list(
        temperature, density=density, wavelength_range=[17, 17.2] * u.nm, abundance="sun_coronal_2021_chianti"
    )
    assert line_list.gofnt.dims == ("logT", "logD", "trans_index")
    np.testing.assert_allclose(line_list.logD.values, [9.0])
    assert line_list.logD.attrs["units"] == "dex(1 / cm3)"
    assert line_list.wavelength.values.tolist() == [171.073]
    assert line_list.attrs["abundance"] == "sun_coronal_2021_chianti"


@needs_fiasco_line_lists
@pytest.mark.usefixtures("controlled_ions")
def test_fiasco_line_list_rejects_bad_inputs(temperature, monkeypatch):
    pressure = xr.DataArray([3e14] * u.K / u.cm**3, dims="pressure")
    with pytest.raises(ValueError, match="minimum_abundance"):
        create_fiasco_line_list(
            temperature, pressure=pressure, wavelength_range=[170, 175] * u.AA, minimum_abundance=-1
        )
    with pytest.raises(ValueError, match="ion_list"):
        create_fiasco_line_list(temperature, pressure=pressure, wavelength_range=[170, 175] * u.AA, ion_list=["Fe IX"])
    monkeypatch.setattr(linelist, "_fiasco_line_table", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="no lines"):
        create_fiasco_line_list(temperature, pressure=pressure, wavelength_range=[170, 175] * u.AA)


@pytest.mark.skipif(fiasco is None, reason="fiasco is not installed")
def test_fiasco_line_list_needs_the_builder(temperature, monkeypatch):
    monkeypatch.delattr(fiasco.IonCollection, "build_line_list", raising=False)
    with pytest.raises(ImportError, match=r"IonCollection\.build_line_list"):
        create_fiasco_line_list(
            temperature,
            pressure=xr.DataArray([3e14] * u.K / u.cm**3, dims="pressure"),
            wavelength_range=[170, 175] * u.AA,
        )


def _chunk_sum(chunk, offset):
    return sum(chunk) + offset


def test_over_ion_chunks_runs_every_chunk():
    pytest.importorskip("joblib")
    serial = linelist._over_ion_chunks(_chunk_sum, [1, 2, 3, 4, 5], 1, 10)
    parallel = linelist._over_ion_chunks(_chunk_sum, [1, 2, 3, 4, 5], 2, 10)
    assert serial == [25]
    assert sorted(parallel) == [10 + 1 + 2 + 3, 10 + 4 + 5]


def test_apply_line_boosts():
    line_list = xr.Dataset(
        {
            "ion_name": ("trans_index", np.array(["he_2", "he_2", "fe_9"])),
            "lower_level_index": ("trans_index", np.array([1, 1, 1])),
            "upper_level_index": ("trans_index", np.array([3, 4, 13])),
            "wavelength": ("trans_index", np.array([303.786, 303.785, 171.073])),
            "gofnt": (("logT", "trans_index"), np.ones((2, 3)), {"units": "erg cm3 / (s sr)"}),
        }
    )
    boosted = apply_line_boosts(line_list, {("he_2", 1, 3): 20})
    assert boosted.gofnt.values[:, 0].tolist() == [20.0, 20.0]
    assert (boosted.gofnt.values[:, 1:] == 1).all()
    assert boosted.gofnt.attrs == line_list.gofnt.attrs
    assert boosted.attrs["line_boosts"] == '{"he_2 1-3": 20}'
    assert (line_list.gofnt.values == 1).all()
    with pytest.raises(ValueError, match="0 transitions match he_2 1-5"):
        apply_line_boosts(line_list, {("he_2", 1, 5): 20})


def test_rank_line_list_orders_by_peak_times_area():
    line_list = synthetic_line_list(3)  # peaks 1.0, 0.75, 0.5 at 170.6, 171.0, 171.4 A
    by_peak = rank_line_list(line_list)
    assert by_peak.full_name.values.tolist() == line_list.full_name.values.tolist()
    np.testing.assert_allclose(by_peak.brightness.values, [1e-25, 0.75e-25, 0.5e-25])
    area = synthetic_effective_area([1.0, 0.0, 10.0], [170.5, 171.0, 171.5])
    by_area = rank_line_list(line_list, area)
    assert by_area.full_name.values.tolist() == [
        line_list.full_name[2].item(),
        line_list.full_name[0].item(),
        line_list.full_name[1].item(),
    ]
    assert by_area.brightness.values[-1] == 0.0
    by_sum = rank_line_list(line_list, statistic="sum")
    np.testing.assert_allclose(by_sum.brightness.values, line_list.gofnt.sum("logT").values)
    with pytest.raises(ValueError, match="statistic"):
        rank_line_list(line_list, statistic="median")


def test_select_transitions():
    line_list = xr.Dataset(
        {
            "ion_name": ("trans_index", np.array(["he_2", "he_2", "fe_9"])),
            "lower_level_index": ("trans_index", np.array([1, 1, 1])),
            "upper_level_index": ("trans_index", np.array([3, 4, 13])),
            "wavelength": ("trans_index", np.array([303.786, 303.785, 171.073])),
            "gofnt": (("logT", "trans_index"), np.ones((2, 3)), {"units": "erg cm3 / (s sr)"}),
        }
    )
    doublet = select_transitions(line_list, [("he_2", 1, 3), ("he_2", 1, 4)])
    assert doublet.wavelength.values.tolist() == [303.786, 303.785]
    assert doublet.attrs["line_selection"] == '["he_2 1-3", "he_2 1-4"]'
    with pytest.raises(ValueError, match="0 transitions match fe_9 1-5"):
        select_transitions(line_list, [("fe_9", 1, 5)])
