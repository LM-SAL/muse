"""
Tests of the response pipeline on synthetic inputs: a line list file, an effective-area
file and a closed-form continuum stand in for CHIANTI, so every test runs offline in
seconds.
"""

import numpy as np
import pytest
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.instrument import pipeline
from muse.instrument.continuum import _continuum_dataset
from muse.instrument.effective_area import create_effective_area
from muse.instrument.linelist import _broadcast_plasma_grid, _validate_positive_data_array
from muse.instrument.pipeline import (
    generate_continuum_responses,
    generate_responses,
    load_or_create_continuum,
    load_or_create_line_list,
)
from muse.instrument.response_io import load_and_concat_responses, read_response, read_spectral_response
from muse.instrument.spectral import create_log_wavelength_edges
from muse.tests.helpers import synthetic_line_list
from muse.variables import DEFAULTS_MUSE

pytest.importorskip("numexpr")
pytest.importorskip("periodictable")

BAND = 171
MAIN_LINE = "Fe IX 171.073"
LINE_NAMES = ["Fe X 155.000", "Fe VIII 165.000", MAIN_LINE, "Fe X 174.500", "Fe XI 180.000", "Fe XII 188.000"]
TEMPERATURE = xr.DataArray(10 ** np.array([5.5, 6.0, 6.5]) * u.K, dims="logT")
PRESSURE = xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure")
DOPPLER_VELOCITY = np.array([-500.0, 0.0, 500.0]) * u.km / u.s
CI_UNITS_KWARGS = {"gain": 10 * u.electron / u.DN, "pair_energy": 3.65 * u.eV / u.electron}
NUMBER_OF_SLITS = int(DEFAULTS_MUSE.number_of_slits_SG)
NUMBER_OF_PIXELS = int(DEFAULTS_MUSE.pixels_SG.to_value(u.pix))
GEOMETRY_TAG = f"nslits{NUMBER_OF_SLITS}_sep390"


def _area_file(directory, *, unit=u.cm**2, wavelength=(150.0, 192.0)):
    """
    A 171-band effective area spanning the whole waveband (153.5-188.5 A) with a slope.
    """
    wavelengths = np.linspace(*wavelength, 100)
    effective_area = create_effective_area(
        10.0 * (1 + 0.01 * (wavelengths - 150.0)) * unit, wavelengths * u.AA, band=BAND * u.AA
    )
    path = directory / f"effective_area_band_{BAND}_TEST.zarr"
    effective_area.to_zarr(path, consolidated=False)
    return path


def _line_list_file(directory, *, plasma="pressure"):
    """
    Six lines across the waveband on one pressure (or two densities), the main line the
    brightest.
    """
    line_list = synthetic_line_list(
        6, wavelength=[155.0, 165.0, 171.073, 174.5, 180.0, 188.0], logT=np.array([5.5, 6.0, 6.5])
    )
    line_list = line_list.assign(
        full_name=("trans_index", LINE_NAMES),
        ion_name=("trans_index", ["fe_10", "fe_8", "fe_9", "fe_10", "fe_11", "fe_12"]),
        lower_level_index=("trans_index", [1] * 6),
        upper_level_index=("trans_index", [2, 3, 13, 4, 5, 6]),
    )
    if plasma == "pressure":
        line_list["gofnt"] = line_list.gofnt.expand_dims(pressure=[3e15]).transpose("pressure", "logT", "trans_index")
    else:
        line_list["gofnt"] = line_list.gofnt.expand_dims(logD=[9.0, 10.0]).transpose("logT", "logD", "trans_index")
    path = directory / f"line_list_{plasma}.nc"
    line_list.to_netcdf(path)
    return path


def fake_continuum(
    temperature, wavelength_grid, *, density=None, pressure=None, abundance=None, minimum_abundance=None, **_kwargs
):
    """
    A flat continuum: free-free 1e-24, free-bound half of it, two-photon a tenth, per
    plasma point.
    """
    temperature = _validate_positive_data_array(temperature, "temperature", u.K, dimension="logT")
    grid = wavelength_grid.to_value(u.AA)
    by_density = density is not None
    plasma = _validate_positive_data_array(
        density if by_density else pressure, "plasma", u.cm**-3 if by_density else u.K / u.cm**3
    )
    temperature_bc, _, extra_coords = _broadcast_plasma_grid(temperature, plasma, by_density=by_density)
    base = 1e-24 * np.ones((temperature.size, grid.size))
    return _continuum_dataset(
        base,
        0.5 * base,
        np.stack([0.1 * base] * plasma.size),
        temperature=temperature,
        temperature_bc=temperature_bc,
        extra_coords=extra_coords,
        grid=grid,
        attrs={"abundance": abundance, "minimum_abundance": minimum_abundance},
    )


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("inputs")
    return {
        "area": _area_file(directory),
        "line_list": _line_list_file(directory),
        "density_line_list": _line_list_file(directory, plasma="density"),
    }


@pytest.fixture
def _fake_continuum(monkeypatch):
    monkeypatch.setitem(pipeline.CONTINUUM_BACKENDS, "fiasco", fake_continuum)


def _run(inputs, output_directory, **overrides):
    kwargs = {
        "output_directory": output_directory,
        "author": "TEST",
        "effective_area_note": "testea",
        "sg_effective_areas": [inputs["area"]],
        "ci_effective_areas": [inputs["area"]],
        "abundances": ["sun_coronal_2021_chianti"],
        "main_lines_by_band": {BAND: [MAIN_LINE]},
        "pressure": PRESSURE,
        "temperature": TEMPERATURE,
        "doppler_velocity": DOPPLER_VELOCITY,
        "maximum_contaminants": 3,
        "line_list_file": inputs["line_list"],
        "ci_units_kwargs": CI_UNITS_KWARGS,
    }
    kwargs.update(overrides)
    return generate_responses(**kwargs)


@pytest.fixture(scope="module")
def products(inputs, tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(pipeline.CONTINUUM_BACKENDS, "fiasco", fake_continuum)
        return _run(inputs, tmp_path_factory.mktemp("run"))


def _only_product(products, pattern):
    matches = [path for path in products if pattern in path.name]
    assert len(matches) == 1, f"expected one product matching {pattern!r}, got {[p.name for p in matches]}"
    return matches[0]


def test_every_product_is_written_once(products):
    stem = "sun_coronal_2021_chianti_testea_pressure_3e+15"
    expected = [
        f"respfunc_171_FeIX171.073_waveband_{stem}_TEST_",
        f"respfunc_171_FeIX171.073_{GEOMETRY_TAG}_{stem}_TEST_",
        f"respfunc_171_nomainlines_waveband_{stem}_TEST_",
        f"respfunc_171_L_{GEOMETRY_TAG}_{stem}_TEST_",
        f"respfunc_ci_171_waveband_{stem}_TEST_",
        f"respfunc_ci_171_{stem}_TEST_",
        f"respfunc_ci_171_waveband_{stem}_DN_TEST_",
        f"respfunc_ci_171_{stem}_DN_TEST_",
    ]
    assert len(products) == len(expected)
    for prefix in expected:
        assert sum(path.name.startswith(prefix) and path.suffix == ".zarr" for path in products) == 1
    assert all(path.is_dir() for path in products)


def test_main_line_response(products):
    response = read_response(_only_product(products, "FeIX171.073_nslits"))
    assert response.sizes["slit"] == NUMBER_OF_SLITS
    assert response.sizes["detector_x_pixel"] == NUMBER_OF_PIXELS
    assert response.sizes["pressure"] == 1
    assert response.sizes["doppler_velocity"] == 3
    assert list(response.line.values) == [MAIN_LINE]
    assert (response.channel == BAND).all()
    np.testing.assert_allclose(response.line_wavelength.values, [171.073])
    assert response.detector_response.attrs["units"] == "1e-27 cm5 ph / s"
    assert response.attrs["author"] == "TEST"
    assert response.attrs["abundance"] == "sun_coronal_2021_chianti"
    assert "effective_area" in response.data_vars
    assert np.isfinite(response.detector_response.values).all()
    assert float(response.detector_response.max()) > 0


def test_contaminant_products(products):
    detector_response = read_response(_only_product(products, "_L_nslits"))
    assert list(detector_response.line.values) == ["171 remaining 3 lines"]
    # referenced to the band's main line on the detector, to the brightest contaminant in the cube
    np.testing.assert_allclose(detector_response.line_wavelength.values, [171.073])
    cube = read_spectral_response(_only_product(products, "nomainlines_waveband"))
    assert cube.line_wavelength.values[0] in (174.5, 165.0, 180.0, 155.0, 188.0)
    assert float(detector_response.detector_response.max()) > 0


def test_waveband_cube_is_on_the_sg_grid(products):
    cube = read_spectral_response(_only_product(products, "FeIX171.073_waveband"))
    assert cube.spectral_response.attrs["units"] == "1e-27 cm5 ph / (Angstrom s)"
    assert cube.sizes["pressure"] == 1
    np.testing.assert_allclose(np.diff(cube.wavelength_grid.values), 0.0049, rtol=1e-6)
    assert "effective_area" in cube.data_vars


def test_downstream_concatenation(products):
    main = _only_product(products, "FeIX171.073_nslits")
    contaminants = _only_product(products, "_L_nslits")
    response = load_and_concat_responses(
        main.parent,
        [main.name, contaminants.name],
        channels=[BAND, BAND],
        doppler_velocity=xr.DataArray(DOPPLER_VELOCITY.value, dims="doppler_velocity", attrs={"units": "km / s"}),
    )
    assert response.sizes["line"] == 2
    assert (response.channel == BAND).all()
    assert (response.detector_response.max(dim=[d for d in response.detector_response.dims if d != "line"]) > 0).all()


def test_cube_holds_the_lines_and_the_continuum_on_log_bins(products):
    cube = read_spectral_response(
        _only_product(products, "ci_171_waveband_sun_coronal_2021_chianti_testea_pressure_3e+15_TEST")
    )
    edges = create_log_wavelength_edges([150, 192] * u.AA, 0.01)
    assert cube.sizes["wavelength_bin"] == edges.size - 1
    np.testing.assert_allclose(cube.wavelength_bounds.isel(bound=0).values, edges[:-1].value)
    assert list(cube.line.values) == [
        "171 remaining 3 lines",
        "171 continuum free-free",
        "171 continuum free-bound",
        "171 continuum two-photon",
    ]
    assert cube.spectral_response.attrs["units"] == "1e-27 cm5 ph / (Angstrom s)"
    assert cube.spectral_response.attrs["bin_method"] == "average"
    assert cube.attrs["continuum"].startswith("fiasco")
    assert np.isfinite(cube.spectral_response.values).all()
    assert (cube.spectral_response.max(dim=[d for d in cube.spectral_response.dims if d != "line"]) > 0).all()


def test_summed_response_is_the_cube_sum(products):
    cube = read_spectral_response(
        _only_product(products, "ci_171_waveband_sun_coronal_2021_chianti_testea_pressure_3e+15_TEST")
    )
    summed = read_response(_only_product(products, "ci_171_sun_coronal_2021_chianti_testea_pressure_3e+15_TEST"))
    width = cube.wavelength_bounds.isel(bound=1) - cube.wavelength_bounds.isel(bound=0)
    expected = (cube.spectral_response * width).sum(["wavelength_bin", "line"])
    assert list(summed.line.values) == ["171 remaining 3 lines + continuum"]
    assert summed.detector_response.attrs["units"] == "1e-27 cm5 ph / s"
    assert (summed.channel == BAND).all()
    np.testing.assert_allclose(
        summed.detector_response.isel(line=0).transpose(*expected.dims).values, expected.values, rtol=1e-12
    )


def test_dn_product_follows_the_photon_energy(products):
    photons = read_spectral_response(
        _only_product(products, "ci_171_waveband_sun_coronal_2021_chianti_testea_pressure_3e+15_TEST")
    )
    dn = read_spectral_response(
        _only_product(products, "ci_171_waveband_sun_coronal_2021_chianti_testea_pressure_3e+15_DN_TEST")
    )
    assert dn.spectral_response.attrs["units"] == "1e-27 cm5 DN / (Angstrom s)"
    photon_energy = (const.h * const.c / (photons.wavelength_grid.values * u.AA)).to(u.eV)
    expected = (photon_energy / (3.65 * u.eV)).value / 10
    ratio = (dn.spectral_response / photons.spectral_response).where(photons.spectral_response > 0)
    for i_line in range(ratio.sizes["line"]):
        values = ratio.isel(line=i_line).values
        mask = np.isfinite(values)
        np.testing.assert_allclose(values[mask], np.broadcast_to(expected, values.shape)[mask], rtol=1e-12)


@pytest.mark.usefixtures("_fake_continuum")
def test_an_area_in_dn_gives_dn_products_only(inputs, tmp_path):
    area = _area_file(tmp_path, unit=u.cm**2 * u.DN / u.ph)
    with pytest.raises(ValueError, match="only DN products"):
        _run(inputs, tmp_path / "ph", sg_effective_areas=[], ci_effective_areas=[area], ci_units=("ph",))
    written = _run(inputs, tmp_path / "dn", sg_effective_areas=[], ci_effective_areas=[area], include_continuum=False)
    assert [path.name.split("_testea_")[1].split("_TEST")[0] for path in written] == [
        "pressure_3e+15_DN",
        "pressure_3e+15_DN",
    ]
    summed = read_response(written[1])
    assert summed.detector_response.attrs["units"] == "1e-27 cm5 DN / s"
    assert list(summed.line.values) == ["171 remaining 3 lines"]


@pytest.mark.usefixtures("_fake_continuum")
def test_mains_only_matches_the_full_run(inputs, products, tmp_path):
    written = _run(inputs, tmp_path, ci_effective_areas=[], maximum_contaminants=0)
    assert [
        path.name.split("_waveband_")[0] if "_waveband_" in path.name else path.name.split("_nslits")[0]
        for path in written
    ] == [
        "respfunc_171_FeIX171.073",
        "respfunc_171_FeIX171.073",
    ]
    full = read_response(_only_product(products, "FeIX171.073_nslits"))
    mains = read_response(written[1])
    xr.testing.assert_allclose(full.detector_response, mains.detector_response, rtol=1e-12)


@pytest.mark.usefixtures("_fake_continuum")
def test_density_grid_is_one_run(inputs, tmp_path):
    written = _run(
        inputs,
        tmp_path,
        pressure=None,
        density=xr.DataArray([1e9, 1e10] * u.cm**-3, dims="density"),
        line_list_file=inputs["density_line_list"],
    )
    assert len(written) == 8
    assert all("_density_" in path.name or "_density_DN_" in path.name for path in written)
    response = read_response(_only_product(written, "FeIX171.073_nslits"))
    assert response.sizes["logD"] == 2
    cube = read_spectral_response(
        _only_product(written, "ci_171_waveband_sun_coronal_2021_chianti_testea_density_TEST")
    )
    assert cube.sizes["logD"] == 2


@pytest.mark.usefixtures("_fake_continuum")
def test_boosts_and_selection_reach_the_products(inputs, tmp_path):
    written = _run(
        inputs,
        tmp_path,
        ci_effective_areas=[],
        maximum_contaminants=0,
        line_boosts={("fe_9", 1, 13): 2.0},
        line_selection=[("fe_9", 1, 13), ("fe_10", 1, 4)],
    )
    response = read_response(written[1])
    assert response.attrs["line_boosts"] == '{"fe_9 1-13": 2.0}'
    assert response.attrs["line_selection"] == '["fe_9 1-13", "fe_10 1-4"]'


@pytest.mark.usefixtures("_fake_continuum")
def test_refuses_to_overwrite(inputs, tmp_path):
    _run(inputs, tmp_path, ci_effective_areas=[], maximum_contaminants=0)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _run(inputs, tmp_path, ci_effective_areas=[], maximum_contaminants=0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"ci_units": ("electrons",)}, "ci_units"),
        ({"line_list_backend": "idl"}, "line_list_backend"),
        ({"continuum_backend": "idl"}, "continuum_backend"),
        ({"density": xr.DataArray([1e9] * u.cm**-3, dims="density")}, "mutually exclusive"),
        ({"author": ""}, "author"),
        ({"sg_effective_areas": [], "ci_effective_areas": []}, "at least one"),
    ],
)
def test_rejects_bad_options(inputs, tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        _run(inputs, tmp_path, **overrides)


def test_unknown_band_needs_explicit_settings(inputs, tmp_path):
    wavelength = np.linspace(90.0, 100.0, 20)
    create_effective_area(np.ones(20) * u.cm**2, wavelength * u.AA, band=94 * u.AA).to_zarr(
        tmp_path / "area_94.zarr", consolidated=False
    )
    with pytest.raises(ValueError, match="spectral_order_by_band"):
        _run(
            inputs,
            tmp_path,
            sg_effective_areas=[tmp_path / "area_94.zarr"],
            ci_effective_areas=[],
            main_lines_by_band={94: [MAIN_LINE]},
        )


def test_line_list_is_cached_and_reloaded(inputs, tmp_path, monkeypatch):
    calls = []

    def backend(_temperature, **kwargs):
        calls.append(kwargs["abundance"])
        return xr.load_dataset(inputs["line_list"])

    monkeypatch.setitem(pipeline.LINE_LIST_BACKENDS, "chianti", backend)
    kwargs = {
        "abundance": "a",
        "wavelength_range": [150, 192] * u.AA,
        "temperature": TEMPERATURE,
        "pressure": PRESSURE,
    }
    first = load_or_create_line_list(tmp_path, **kwargs)
    second = load_or_create_line_list(tmp_path, **kwargs)
    assert calls == ["a"]
    assert (tmp_path / "line_list_chianti_150-192A_a.nc").is_file()
    xr.testing.assert_identical(first.drop_attrs(), second.drop_attrs())
    with pytest.raises(FileNotFoundError, match="preparation"):
        load_or_create_line_list(tmp_path, compute_if_missing=False, **{**kwargs, "abundance": "b"})
    with pytest.raises(ValueError, match="backend"):
        load_or_create_line_list(tmp_path, backend="idl", **kwargs)
    assert (
        pipeline._line_list_cache_path(tmp_path, "fiasco", "a", [25, 900] * u.AA, density_dependent=True).name
        == "line_list_fiasco_25-900A_a_density.nc"
    )


@pytest.mark.parametrize(
    ("grid", "initial", "requested", "expected_calls"),
    [
        ("density", [1e8, 1e9], [1e8, 1e9], 1),
        ("density", [1e8, 1e9], [1e10], 2),
        ("density", [1e8, 1e9], [1.0001e8], 2),
        ("density", [1e8, 1e9], [1e9], 1),
        ("density", [1e8, 1e9], [1e9, 1e8], 1),
        ("pressure", [3e15, 5e15], [5e15], 1),
        ("pressure", [3e15, 5e15], [7e15], 2),
    ],
    ids=["same-grid", "new-density", "nearby-density", "subset", "reordered", "pressure-subset", "new-pressure"],
)
def test_continuum_cache_matches_the_request(tmp_path, monkeypatch, grid, initial, requested, expected_calls):
    calls = []

    def backend(temperature, wavelength_grid, *, density=None, pressure=None, minimum_abundance=None, **_kwargs):
        plasma = density if density is not None else pressure
        calls.append(plasma.values.copy())
        temperature = _validate_positive_data_array(temperature, "temperature", u.K, dimension="logT")
        plasma = _validate_positive_data_array(plasma, "plasma", u.cm**-3 if density is not None else u.K / u.cm**3)
        temperature_bc, _, extra = _broadcast_plasma_grid(temperature, plasma, by_density=density is not None)
        values = np.broadcast_to(plasma.data[:, None, None], (plasma.size, temperature.size, wavelength_grid.size))
        base = np.zeros((temperature.size, wavelength_grid.size))
        return _continuum_dataset(
            base,
            base,
            values,
            temperature=temperature,
            temperature_bc=temperature_bc,
            extra_coords=extra,
            grid=wavelength_grid.to_value(u.AA),
            attrs={"minimum_abundance": minimum_abundance},
        )

    monkeypatch.setitem(pipeline.CONTINUUM_BACKENDS, "chianti", backend)
    unit = u.cm**-3 if grid == "density" else u.K / u.cm**3
    kwargs = {
        "backend": "chianti",
        "abundance": "test",
        "wavelength_grid": [170.0, 171.0, 172.0] * u.AA,
        "temperature": TEMPERATURE.isel(logT=[0, 1]),
        "minimum_abundance": 1e-5,
    }
    load_or_create_continuum(tmp_path, **{grid: xr.DataArray(initial * unit, dims=grid)}, **kwargs)
    actual = load_or_create_continuum(tmp_path, **{grid: xr.DataArray(requested * unit, dims=grid)}, **kwargs)
    assert len(calls) == expected_calls
    coordinate = "logD" if grid == "density" else "pressure"
    expected_coord = np.log10(requested) if grid == "density" else requested
    np.testing.assert_allclose(actual[coordinate].values, expected_coord)
    two_photon = actual.spectral_response.sel(line="two-photon").transpose(coordinate, "logT", "wavelength_bin").values
    np.testing.assert_allclose(
        two_photon, np.broadcast_to(np.array(requested)[:, None, None], two_photon.shape) / 1e-27
    )
    with pytest.raises(FileNotFoundError, match="preparation"):
        load_or_create_continuum(
            tmp_path,
            compute_if_missing=False,
            **{grid: xr.DataArray(requested * unit, dims=grid)},
            **{**kwargs, "abundance": "other"},
        )


@pytest.mark.usefixtures("_fake_continuum")
def test_generate_continuum_responses(inputs, tmp_path):
    written = generate_continuum_responses(
        output_directory=tmp_path,
        author="TEST",
        effective_area_note="testea",
        sg_effective_areas=[inputs["area"]],
        pressure=PRESSURE,
        temperature=TEMPERATURE,
        doppler_velocity=DOPPLER_VELOCITY,
        wavelength_step=0.1 * u.AA,
    )
    assert len(written) == 1
    assert written[0].name.startswith(
        f"respfunc_171_continuum_fiasco_{GEOMETRY_TAG}_sun_coronal_2021_chianti_testea_pressure_3e+15_TEST_"
    )
    response = read_response(written[0])
    assert list(response.line.values) == ["free-free", "free-bound", "two-photon"]
    assert response.sizes["slit"] == NUMBER_OF_SLITS
    assert response.sizes["doppler_velocity"] == 3
    assert response.detector_response.attrs["units"] == "1e-27 cm5 ph / s"
    assert response.attrs["continuum_backend"] == "fiasco"
    assert float(response.detector_response.sel(line="free-free").max()) > 0
    assert (tmp_path / "cache" / "continuum_fiasco_153-188A_n350_sun_coronal_2021_chianti.nc").is_file()
    several = generate_continuum_responses(
        output_directory=tmp_path / "several",
        author="TEST",
        effective_area_note="testea",
        sg_effective_areas=[inputs["area"]],
        temperature=TEMPERATURE,
        doppler_velocity=DOPPLER_VELOCITY,
        wavelength_step=0.1 * u.AA,
    )
    assert "_pressure_TEST_" in several[0].name


@pytest.mark.chianti
def test_generate_responses_live_chianti(tmp_path):
    """
    The whole chain on real CHIANTI data: a narrow 171 band, the abundant elements, one
    pressure, no continuum.
    """
    pytest.importorskip("ChiantiPy")
    # 13 A around the main line keep the CHIANTI line list, which spans the area, to a few minutes
    area = _area_file(tmp_path, wavelength=(165.0, 178.0))
    written = generate_responses(
        output_directory=tmp_path / "run",
        author="TEST",
        effective_area_note="testea",
        sg_effective_areas=[area],
        ci_effective_areas=[area],
        abundances=["sun_coronal_2021_chianti"],
        main_lines_by_band={BAND: [MAIN_LINE]},
        pressure=PRESSURE,
        temperature=TEMPERATURE,
        doppler_velocity=DOPPLER_VELOCITY,
        maximum_contaminants=3,
        minimum_abundance=3e-5,
        include_continuum=False,
        ci_units=("ph",),
        ci_units_kwargs=CI_UNITS_KWARGS,
    )
    assert len(written) == 6
    main = read_response(_only_product(written, "FeIX171.073_nslits"))
    assert float(main.detector_response.max()) > 0
    # Fe IX 171.073 peaks near logT 5.9: the middle temperature of the grid is the brightest
    by_temperature = main.detector_response.sel(doppler_velocity=0).sum(["slit", "detector_x_pixel"]).squeeze().values
    assert np.argmax(by_temperature) == 1
    cube = read_spectral_response(_only_product(written, "ci_171_waveband"))
    assert list(cube.line.values) == ["171 remaining 3 lines"]
    assert float(cube.spectral_response.max()) > 0
    assert (tmp_path / "run" / "cache" / "line_list_chianti_165-178A_sun_coronal_2021_chianti.nc").is_file()


def test_limited_threads_caps_the_pools_and_restores_them():
    numexpr = pytest.importorskip("numexpr")
    from threadpoolctl import threadpool_info

    before = numexpr.set_num_threads(3)
    numexpr.set_num_threads(before)
    with pipeline._limited_threads(2):
        assert numexpr.set_num_threads(2) == 2
        assert all(pool["num_threads"] <= 2 for pool in threadpool_info())
    assert numexpr.set_num_threads(before) == before
    with pytest.raises(ValueError, match="threads"), pipeline._limited_threads(0):
        pass
