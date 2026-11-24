"""
Tests of the response generators on synthetic inputs: a line list file and a flat
continuum stand in for CHIANTI, and a sloped effective area for the instrument.

Only the test marked ``chianti`` reads the database.
"""

import numpy as np
import pytest
import xarray as xr
from attrs import evolve

import astropy.units as u

from muse.instrument import (
    create_effective_area,
    create_log_wavelength_edges,
    generate_ci_responses,
    generate_sg_continuum_responses,
    generate_sg_responses,
    read_response,
    read_responses,
    read_spectral_response,
    save_effective_area,
)
from muse.instrument.atomic import cache
from muse.instrument.atomic.backends import _MINIMUM_ABUNDANCE
from muse.instrument.response import products as products_module
from muse.instrument.response.io import _open_dataset_file
from muse.tests.helpers import fake_continuum, fake_line_list
from muse.variables import DEFAULTS_MUSE

CHANNEL = 171
MAIN_LINE = "Fe IX 171.073"
LINE_NAMES = ["Fe X 155.000", "Fe VIII 165.000", MAIN_LINE, "Fe X 174.500", "Fe XI 180.000", "Fe XII 188.000"]
ABUNDANCE = "sun_coronal_2021_chianti"
OTHER_ABUNDANCE = "sun_photospheric_2021_asplund"
TEMPERATURE = 10 ** np.array([5.5, 6.0, 6.5]) * u.K
PRESSURE = [3e15] * u.K / u.cm**3
DOPPLER_VELOCITY = np.array([-500.0, 0.0, 500.0]) * u.km / u.s
# 171 is no channel of the MUSE imager: the test area describes its detector, as an AIA file does.
DETECTOR = {"gain": "10.0 electron / DN", "pair_energy": "3.65 eV / electron"}
# What every product name of the default test call carries after its prefix.
TAIL = "testea_pressure_3e+15"


def _waveband(channel):
    """
    The ends of an SG waveband of ``DEFAULTS_MUSE`` in Angstrom: the half width over the
    spectral order of the channel, on either side of it.
    """
    order = int(DEFAULTS_MUSE.channel_spectral_order_SG.sel(channel=channel))
    return channel + np.array([-1, 1]) * DEFAULTS_MUSE.waveband_half_width_SG.to_value(u.AA) / order


def _area_file(directory, *, channel=CHANNEL, wavelength=(150.0, 192.0), name=None, attrs=DETECTOR):
    """
    An effective area with a slope, by default wider than the whole 171 waveband.
    """
    wavelengths = np.linspace(*wavelength, 100)
    effective_area = create_effective_area(
        10.0 * (1 + 0.01 * (wavelengths - 150.0)) * u.cm**2, wavelengths * u.AA, channel=channel * u.AA, attrs=attrs
    )
    path = directory / (name or f"effective_area_channel_{channel}_TEST.zarr")
    save_effective_area(effective_area, path)
    return path


def _line_list(plasma="pressure", *, pressure=(3e15,), scale=(1.0,)):
    """
    Six lines across the waveband on one pressure (or two densities), each fainter than
    the one before; the main line is the third.

    Given several values of ``pressure``, each holds the lines times its ``scale``.
    """
    line_list = fake_line_list(
        6, wavelength=[155.0, 165.0, 171.073, 174.5, 180.0, 188.0], logT=np.array([5.5, 6.0, 6.5])
    )
    line_list = line_list.assign(
        full_name=("transition", LINE_NAMES),
        ion_name=("transition", ["fe_10", "fe_8", "fe_9", "fe_10", "fe_11", "fe_12"]),
        lower_level_index=("transition", [1] * 6),
        upper_level_index=("transition", [2, 3, 13, 4, 5, 6]),
    ).assign_attrs(fiasco_version="0.0.test", chianti_version="11.0.2", history=["create_fiasco_line_list()"])
    if plasma == "pressure":
        factor = xr.DataArray(np.asarray(scale, dtype=float), dims="pressure", coords={"pressure": list(pressure)})
        gofnt = (factor * line_list.gofnt).transpose("pressure", "logT", "transition")
        line_list["gofnt"] = gofnt.assign_attrs(line_list.gofnt.attrs)
        line_list.pressure.attrs["units"] = "K / cm3"  # as the line-list builders label it
    else:
        line_list["gofnt"] = line_list.gofnt.expand_dims(logD=[9.0, 10.0]).transpose("logT", "logD", "transition")
    return line_list


def _line_list_file(directory, *, plasma="pressure"):
    path = directory / f"line_list_{plasma}.nc"
    _line_list(plasma).to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("inputs")
    return {
        "area": _area_file(directory),
        "line_list": _line_list_file(directory),
        "density_line_list": _line_list_file(directory, plasma="density"),
    }


@pytest.fixture
def _fake_continuum_backend(monkeypatch):
    monkeypatch.setitem(cache._CONTINUUM_BACKENDS, "fiasco", fake_continuum)


def _options(inputs, output_directory, **overrides):
    return {
        "output_directory": output_directory,
        "author": "TEST",
        "effective_area_label": "testea",
        "effective_area_files": [inputs["area"]],
        "abundances": [ABUNDANCE],
        "pressure": PRESSURE,
        "temperature": TEMPERATURE,
        "doppler_velocity": DOPPLER_VELOCITY,
        "line_list_file": inputs["line_list"],
        **overrides,
    }


def _sg(inputs, output_directory, **overrides):
    options = {"main_lines_by_channel": {CHANNEL: [MAIN_LINE]}, "maximum_contaminants": 3, **overrides}
    return generate_sg_responses(**_options(inputs, output_directory, **options))


def _ci(inputs, output_directory, **overrides):
    return generate_ci_responses(**_options(inputs, output_directory, **{"maximum_lines": 3, **overrides}))


def _sg_continuum(inputs, output_directory, **overrides):
    options = _options(inputs, output_directory, **{"wavelength_step": 0.1 * u.AA, **overrides})
    del options["line_list_file"]
    return generate_sg_continuum_responses(**options)


@pytest.fixture(scope="module")
def sg_products(inputs, tmp_path_factory):
    return _sg(inputs, tmp_path_factory.mktemp("sg"))


@pytest.fixture(scope="module")
def ci_products(inputs, tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(cache._CONTINUUM_BACKENDS, "fiasco", fake_continuum)
        return _ci(inputs, tmp_path_factory.mktemp("ci"))


def _only(products, pattern):
    matches = [path for path in products if pattern in path.name]
    assert len(matches) == 1, f"expected one product matching {pattern!r}, got {[path.name for path in matches]}"
    return matches[0]


def _prefixes(products):
    """
    The names without their suffix.
    """
    return [path.name.removesuffix(".zarr.zip") for path in products]


def _main_response(products):
    return read_response(_only(products, "FeIX171.073_detector")).detector_response


def _not_called(*_args, **_kwargs):
    msg = "work was done that the call did not need"
    raise AssertionError(msg)


def test_every_sg_product_is_written_once(sg_products):
    assert _prefixes(sg_products) == [
        f"response_sg_171_FeIX171.073_waveband_{TAIL}",
        f"response_sg_171_FeIX171.073_detector_{TAIL}",
        f"response_sg_171_contaminants_waveband_{TAIL}",
        f"response_sg_171_contaminants_detector_{TAIL}",
    ]
    # nothing else in the directory: no staging file is left
    assert sorted(path.name for path in sg_products[0].parent.iterdir()) == sorted(path.name for path in sg_products)


def test_every_ci_product_is_written_once(ci_products):
    assert _prefixes(ci_products) == [
        f"response_ci_171_waveband_{TAIL}",
        f"response_ci_171_waveband_{TAIL}_DN",
        f"response_ci_171_{TAIL}_DN",
    ]


def test_main_line_response(sg_products):
    response = read_response(_only(sg_products, "FeIX171.073_detector"))

    assert response.sizes["doppler_velocity"] == 3
    assert response.detector_response.attrs["units"] == "1e-27 cm5 ph / s"
    assert "effective_area" in response.data_vars


def test_products_record_what_made_them(sg_products, ci_products):
    response = read_response(_only(sg_products, "FeIX171.073_detector"))

    assert response.attrs["author"] == "TEST"
    assert response.attrs["effective_area_label"] == "testea"
    assert response.attrs["abundance"] == ABUNDANCE
    assert response.attrs["effective_area_file"] == "effective_area_channel_171_TEST.zarr"
    assert response.attrs["maximum_contaminants"] == 3
    # Made from a given line list: its file is recorded, by name alone. Neither a backend nor
    # a minimum abundance is claimed for it, as neither was applied in this call.
    assert response.attrs["line_list_file"] == "line_list_pressure.nc"
    assert "line_list_backend" not in response.attrs
    assert "minimum_abundance" not in response.attrs
    # the database and the software of the line list, and the whole chain of calls up to this read
    assert (response.attrs["chianti_version"], response.attrs["fiasco_version"]) == ("11.0.2", "0.0.test")
    # the line list and the effective area first: a product keeps how its inputs were made
    assert [entry.split("(")[0] for entry in response.attrs["history"]] == [
        "create_fiasco_line_list",
        "create_effective_area",
        "create_spectral_response",
        "transform_response_units",
        "map_response_to_sg_detector",
        "generate_sg_responses",
        "read_response",
    ]
    assert "maximum_contaminants=3" in response.attrs["history"][-2]

    cube = read_spectral_response(_only(ci_products, f"ci_171_waveband_{TAIL}.zarr"))
    # An imager cube records its continuum: that it is included, its backend and minimum
    # abundance, and the calls that made it, which enter the history before the unit conversion.
    assert (cube.attrs["include_continuum"], cube.attrs["continuum_backend"]) == (1, "fiasco")
    assert cube.attrs["minimum_abundance"] == _MINIMUM_ABUNDANCE
    assert cube.attrs["continuum_fiasco_version"] == "0.0.test"
    assert [entry.split("(")[0] for entry in cube.attrs["history"]] == [
        "create_fiasco_line_list",
        "create_effective_area",
        "create_spectral_response",
        "create_continuum_response",
        "transform_response_units",
        "generate_ci_responses",
    ]


def test_contaminant_products(sg_products):
    detector_response = read_response(_only(sg_products, "contaminants_detector"))
    cube = read_spectral_response(_only(sg_products, "contaminants_waveband"))

    assert list(detector_response.line.values) == ["171 contaminants"]
    assert detector_response.attrs["number_of_lines"] == 3
    # referenced to the channel's main line on the detector, to the brightest contaminant in the cube
    np.testing.assert_allclose(detector_response.line_wavelength.values, [171.073])
    assert cube.line_wavelength.values[0] in (174.5, 165.0, 180.0, 155.0, 188.0)
    assert float(detector_response.detector_response.max()) > 0
    assert "effective_area" in cube.data_vars


def test_the_sg_grid_has_one_length_whatever_the_round_off():
    """
    Stepped up to the upper end, the grids of 108 and 171 at 0.01 Angstrom differ by one
    sample, though both wavebands are equally wide.
    """
    step = 0.01 * u.AA
    grids = [products_module._sg_waveband_grid(channel, DEFAULTS_MUSE, step) for channel in (108, 171)]
    assert grids[0].size == grids[1].size
    np.testing.assert_allclose(grids[0][[0, -1]].value, _waveband(108))
    # the last sample is the one nearest the upper end of the waveband
    last = products_module._sg_waveband_grid(171, DEFAULTS_MUSE)[-1]
    assert abs(last - _waveband(171)[1] * u.AA) < DEFAULTS_MUSE.wavelength_step_SG / 2


def test_cube_holds_the_lines_and_the_continuum_on_log_bins(ci_products):
    cube = read_spectral_response(_only(ci_products, f"ci_171_waveband_{TAIL}.zarr"))

    edges = create_log_wavelength_edges([150, 192] * u.AA, 0.01)
    assert cube.spectral_response.dims == ("line", "doppler_velocity", "logT", "pressure", "wavelength_bin")
    assert cube.sizes["wavelength_bin"] == edges.size - 1
    np.testing.assert_allclose(cube.wavelength_bounds.isel(bound=0).values, edges[:-1].value)
    assert list(cube.line.values) == [
        "171 lines",
        "171 continuum free-free",
        "171 continuum free-bound",
        "171 continuum two-photon",
    ]
    assert cube.spectral_response.attrs["units"] == "1e-27 cm5 ph / (Angstrom s)"
    assert cube.spectral_response.attrs["bin_method"] == "average"
    assert np.isfinite(cube.spectral_response.values).all()
    assert (cube.spectral_response.max(dim=[d for d in cube.spectral_response.dims if d != "line"]) > 0).all()


def test_imager_response_is_the_dn_cube_sum(ci_products):
    cube = read_spectral_response(_only(ci_products, f"ci_171_waveband_{TAIL}_DN"))
    summed = read_response(_only(ci_products, f"ci_171_{TAIL}_DN"))

    width = cube.wavelength_bounds.isel(bound=1) - cube.wavelength_bounds.isel(bound=0)
    expected = (cube.spectral_response * width).sum(["wavelength_bin", "line"])
    assert list(summed.line.values) == ["171 lines + continuum"]
    assert summed.detector_response.dims == ("line", "doppler_velocity", "logT", "pressure")
    assert summed.attrs["number_of_lines"] == 3
    assert summed.detector_response.attrs["units"] == "1e-27 cm5 DN / s"
    assert (summed.channel == CHANNEL).all()
    np.testing.assert_allclose(
        summed.detector_response.isel(line=0).transpose(*expected.dims).values, expected.values, rtol=1e-12
    )


@pytest.mark.usefixtures("_fake_continuum_backend")
def test_each_imager_file_is_converted_with_its_own_detector(inputs, ci_products, tmp_path):
    twice_the_gain = {"gain": "20.0 electron / DN", "pair_energy": "3.65 eV / electron"}
    reference = read_response(_only(ci_products, f"response_ci_171_{TAIL}_DN")).detector_response

    # one file, given as it often is: a single path as a string
    gain = _ci(
        inputs,
        tmp_path / "gain",
        effective_area_files=str(_area_file(tmp_path, name="gain.zarr", attrs=twice_the_gain)),
    )

    # twice the electrons per DN halve the DN, and the product reads back with its own gain
    doubled = read_response(_only(gain, f"response_ci_171_{TAIL}_DN"))
    assert doubled.gain.values.tolist() == [20.0]
    xr.testing.assert_allclose(doubled.detector_response.drop_vars("gain"), (reference / 2).drop_vars("gain"))


def test_settings_keyed_by_a_wavelength_count_as_those_keyed_by_a_number(inputs, sg_products, tmp_path):
    """
    A key given as a wavelength, such as ``17.1 * u.nm``, selects the channel as the
    number 171 does, for the main lines and for the instrumental widths.
    """
    # one line alone may be given as its name
    by_wavelength = _sg(
        inputs, tmp_path / "wavelength", main_lines_by_channel={17.1 * u.nm: "Fe X 174.500"}, maximum_contaminants=0
    )

    # the names of channel 171
    assert _prefixes(by_wavelength) == [
        f"response_sg_171_FeX174.500_waveband_{TAIL}",
        f"response_sg_171_FeX174.500_detector_{TAIL}",
    ]
    wider = _sg(inputs, tmp_path / "wider", instrumental_fwhm_by_channel={171 * u.AA: 0.1 * u.AA})
    assert float(_main_response(wider).max()) < float(_main_response(sg_products).max())


def test_a_pressure_is_taken_by_value_from_a_longer_line_list(inputs, sg_products, tmp_path):
    """
    A call for one pressure reads a line list made for several and takes its own by
    value, not by position.
    """
    path = tmp_path / "two_pressures.nc"
    # Another pressure first and ten times brighter: taking the first would show.
    _line_list(pressure=[3e14, 3e15], scale=[10, 1]).to_netcdf(path)
    options = {"maximum_contaminants": 0, "line_list_file": path}

    written = _sg(inputs, tmp_path / "run", **options)

    xr.testing.assert_allclose(_main_response(written), _main_response(sg_products))


def test_a_line_list_file_that_does_not_fit_stops_the_call(inputs, tmp_path):
    """
    Of another abundance the lines would give products that carry the wrong name.
    """
    photospheric = tmp_path / "photospheric.nc"
    xr.load_dataset(inputs["line_list"]).assign_attrs(abundance=OTHER_ABUNDANCE).to_netcdf(photospheric)
    with pytest.raises(ValueError, match=rf"photospheric\.nc is of the abundance {OTHER_ABUNDANCE}, not {ABUNDANCE}"):
        _sg(inputs, tmp_path / "run", line_list_file=photospheric)

    # on pressures, where densities are requested
    density = [1e9] * u.cm**-3
    with pytest.raises(ValueError, match=r"line_list_pressure\.nc has no logD axis"):
        _sg(inputs, tmp_path / "run", pressure=None, density=density)

    assert not (tmp_path / "run").exists()


@pytest.mark.usefixtures("_fake_continuum_backend")
def test_a_density_grid_goes_into_one_product_whole(inputs, tmp_path):
    options = {"pressure": None, "density": [1e9, 1e10] * u.cm**-3}

    written = [
        *_sg(inputs, tmp_path / "sg", line_list_file=inputs["density_line_list"], **options),
        *_ci(inputs, tmp_path / "ci", line_list_file=inputs["density_line_list"], **options),
        *_sg_continuum(inputs, tmp_path / "continuum", **options),
    ]

    assert len(written) == 8
    assert all(path.name.endswith(("_density.zarr.zip", "_density_DN.zarr.zip")) for path in written)
    cube = read_spectral_response(_only(written, "ci_171_waveband_testea_density.zarr"))
    assert cube.sizes["logD"] == 2
    # one order of the dimensions for the line and the continuum products alike
    order = ("line", "doppler_velocity", "logT", "logD", "slit", "detector_x_pixel")
    assert _main_response(written).dims == order
    assert read_response(_only(written, "continuum_fiasco")).detector_response.dims == order


def test_boosts_and_selection_reach_the_products(inputs, sg_products, tmp_path):
    written = _sg(
        inputs,
        tmp_path,
        maximum_contaminants=0,
        transition_boosts={("fe_9", 1, 13): 2.0},
        transition_selection=[("fe_9", 1, 13), ("fe_10", 1, 4)],
    )

    response = read_response(written[1])
    assert response.attrs["transition_boosts"] == '{"fe_9 1-13": 2.0}'
    assert response.attrs["transition_selection"] == '["fe_9 1-13", "fe_10 1-4"]'
    xr.testing.assert_allclose(response.detector_response, 2 * _main_response(sg_products))


@pytest.fixture
def line_list_calls(inputs, monkeypatch):
    """
    A line-list backend that gives the test line list and records the wavelength range
    of each call.
    """
    calls = []

    def backend(_temperature, **kwargs):
        calls.append(kwargs["wavelength_range"].value.tolist())
        return xr.load_dataset(inputs["line_list"])

    monkeypatch.setitem(cache._LINE_LIST_BACKENDS, "fiasco", backend)
    return calls


def test_calls_with_one_wavelength_range_share_one_cached_line_list(inputs, tmp_path, line_list_calls):
    options = {
        "maximum_contaminants": 0,
        "line_list_file": None,
        "cache_directory": tmp_path / "cache",
        "line_list_wavelength_range": [100, 300] * u.AA,
    }

    first = _sg(inputs, tmp_path / "first", **options)
    _sg(inputs, tmp_path / "second", **options)

    assert line_list_calls == [[100.0, 300.0]]
    cached = tmp_path / "cache" / f"line_list_fiasco_100-300A_{ABUNDANCE}.zarr.zip"
    assert cached.is_file()
    # a cache file is a line list like any other, and can be given as one
    given = _sg(inputs, tmp_path / "given", maximum_contaminants=0, line_list_file=cached)
    xr.testing.assert_allclose(_main_response(given), _main_response(first))
    # made by a backend: that is recorded with its minimum abundance, and no file
    attrs = read_response(first[1]).attrs
    assert (attrs["line_list_backend"], attrs["minimum_abundance"]) == ("fiasco", _MINIMUM_ABUNDANCE)
    assert "line_list_file" not in attrs


def test_the_default_line_list_spans_what_the_products_use(inputs, tmp_path, line_list_calls):
    """
    The waveband for the SG, which is far less than a file may span; the whole area for
    the imager, which sums over it.
    """
    options = {"line_list_file": None, "cache_directory": tmp_path / "cache"}

    _sg(inputs, tmp_path / "sg", maximum_contaminants=0, **options)
    _ci(inputs, tmp_path / "ci", include_continuum=False, **options)

    assert line_list_calls == [_waveband(CHANNEL).tolist(), [150.0, 192.0]]


def test_a_missing_cache_can_be_an_error(inputs, tmp_path, line_list_calls):
    with pytest.raises(FileNotFoundError, match="create_if_missing is False"):
        _sg(inputs, tmp_path, line_list_file=None, create_if_missing=False)
    assert not line_list_calls


def test_a_product_that_is_already_there_stops_the_call_unless_told_to_skip(inputs, tmp_path, monkeypatch):
    first = _sg(inputs, tmp_path, maximum_contaminants=0)
    with pytest.raises(ValueError, match="is already there; pass skip_existing=True"):
        _sg(inputs, tmp_path, maximum_contaminants=0)
    # As a call that stopped halfway leaves it: one product is missing.
    first[-1].unlink()

    again = _sg(inputs, tmp_path, maximum_contaminants=0, skip_existing=True)

    # The product that is there is kept as it is; the missing one is made.
    assert again == first
    # A finished call reads no line list and builds no response.
    monkeypatch.setattr(products_module, "_open_dataset_file", _not_called)
    monkeypatch.setattr(products_module, "create_spectral_response", _not_called)
    assert _sg(inputs, tmp_path, maximum_contaminants=0, skip_existing=True) == first


def test_a_resumed_call_keeps_only_products_made_with_its_settings(inputs, tmp_path):
    path = tmp_path / "two_pressures.nc"
    _line_list(pressure=[3e15, 3.4e15], scale=[1, 2]).to_netcdf(path)
    pressures = [3e15, 3.4e15] * u.K / u.cm**3
    options = {"line_list_file": path, "skip_existing": True}
    first = _sg(inputs, tmp_path / "run", pressure=pressures, **options)

    # The pressure is in the name of a product and the author shapes none: both may differ.
    assert _sg(inputs, tmp_path / "run", pressure=pressures[:1], author="OTHER", **options) == first[:4]
    other = "made with other settings than this call's"
    for overrides in (
        {"temperature": TEMPERATURE[:2]},
        {"instrumental_fwhm_by_channel": {CHANNEL: 0.1 * u.AA}},
        # another main line renames the main products, not the contaminants
        {"main_lines_by_channel": {CHANNEL: ["Fe X 174.500"]}},
        # the same area under the same name, with another gain or other values
        {"effective_area_files": [_area_file(tmp_path / "gain20", attrs={**DETECTOR, "gain": "20.0 electron / DN"})]},
        {"effective_area_files": [_area_file(tmp_path / "narrower", wavelength=(151.0, 192.0))]},
    ):
        with pytest.raises(ValueError, match=other):
            _sg(inputs, tmp_path / "run", pressure=pressures, **{**options, **overrides})
    # the line-list file rewritten under its name; last, as every product differs from then on
    _line_list(pressure=[3e15, 3.4e15], scale=[1, 3]).to_netcdf(path)
    with pytest.raises(ValueError, match=other):
        _sg(inputs, tmp_path / "run", pressure=pressures, **options)


def test_products_of_an_earlier_muse_stop_the_call(inputs, tmp_path):
    (tmp_path / f"respfunc_171_FeIX171.073_waveband_{ABUNDANCE}_testea_pressure_3e+15_TEST_2026-01-01.zarr.zip").touch()

    with pytest.raises(ValueError, match="holds products of an earlier muse"):
        _sg(inputs, tmp_path, skip_existing=True)


SCALE_OF_ABUNDANCE = {ABUNDANCE: 1.0, OTHER_ABUNDANCE: 5.0}


@pytest.fixture
def _backends_that_scale(monkeypatch):
    """
    A line list that grows with the abundance, by the factors above, and in proportion
    to the pressure, and a continuum that grows with the abundance; of its parts, the
    synthetic continuum makes the two-photon one alone grow with the pressure.
    """

    def line_list_backend(_temperature, *, pressure, abundance, **_kwargs):
        values = pressure.to_value(u.K / u.cm**3)
        return _line_list(pressure=values, scale=SCALE_OF_ABUNDANCE[abundance] * values / 3e15)

    def continuum_backend(temperature, wavelength_grid, *, abundance, **kwargs):
        continuum = fake_continuum(temperature, wavelength_grid, abundance=abundance, **kwargs)
        scaled = (continuum.spectral_response * SCALE_OF_ABUNDANCE[abundance]).assign_attrs(
            continuum.spectral_response.attrs
        )
        return continuum.assign(spectral_response=scaled)

    monkeypatch.setitem(cache._LINE_LIST_BACKENDS, "fiasco", line_list_backend)
    monkeypatch.setitem(cache._CONTINUUM_BACKENDS, "fiasco", continuum_backend)


@pytest.mark.usefixtures("_backends_that_scale")
def test_every_product_holds_the_abundance_and_the_pressure_of_its_name(inputs, tmp_path):
    """
    Every generator pairs a line list or a continuum with a name in a loop over abundances
    and pressures: each product must hold the data of its own pair.
    """
    options = {
        "abundances": list(SCALE_OF_ABUNDANCE),
        "pressure": [3e15, 6e15] * u.K / u.cm**3,
        "cache_directory": tmp_path / "cache",
    }
    pairs = [(abundance, tag) for abundance in SCALE_OF_ABUNDANCE for tag in ("3e+15", "6e+15")]
    expected = [SCALE_OF_ABUNDANCE[abundance] * (2.0 if tag == "6e+15" else 1.0) for abundance, tag in pairs]

    def peaks(written, pattern, **selection):
        """
        The peak of each pair's product, over that of the first pair.
        """
        values = []
        for abundance, tag in pairs:
            of_abundance = [path for path in written if path.parent.name == abundance]
            product = _open_dataset_file(_only(of_abundance, f"{pattern}_testea_pressure_{tag}."))
            assert (product.attrs["abundance"], product.pressure.values.tolist()) == (abundance, [float(tag)])
            variable = product["detector_response" if "detector_response" in product else "spectral_response"]
            values.append(float(variable.sel(**selection).max()))
        return np.array(values) / values[0]

    sg = _sg(inputs, tmp_path / "sg", line_list_file=None, **options)
    ci = _ci(inputs, tmp_path / "ci", line_list_file=None, **options)
    continuum = _sg_continuum(inputs, tmp_path / "continuum", **options)

    assert (len(sg), len(ci), len(continuum)) == (16, 12, 4)
    np.testing.assert_allclose(peaks(sg, "FeIX171.073_detector"), expected)
    # In an imager cube the lines and the two-photon continuum grow with both.
    np.testing.assert_allclose(peaks(ci, "ci_171_waveband", line="171 lines"), expected)
    np.testing.assert_allclose(peaks(ci, "ci_171_waveband", line="171 continuum two-photon"), expected)
    np.testing.assert_allclose(peaks(continuum, "continuum_fiasco_detector", line="two-photon"), expected)


def test_two_products_with_one_name_stop_the_call_before_any_work(inputs, tmp_path, monkeypatch):
    # what the generators read a line list with
    monkeypatch.setattr(products_module, "_open_dataset_file", _not_called)

    # pressures that agree to six significant digits
    with pytest.raises(ValueError, match=r"product name\(s\) would be used twice"):
        _sg(inputs, tmp_path / "run", pressure=[3e15, 3.0000001e15] * u.K / u.cm**3)
    assert not (tmp_path / "run").exists()


def test_a_repeated_abundance_is_taken_once(inputs, sg_products, tmp_path):
    written = _sg(inputs, tmp_path, abundances=[ABUNDANCE, ABUNDANCE])

    assert _prefixes(written) == _prefixes(sg_products)


@pytest.mark.parametrize(
    ("generate", "overrides", "match"),
    [
        (_ci, {"maximum_lines": 0}, "maximum_lines must be at least 1"),
        (_ci, {"maximum_lines": 2.5}, "maximum_lines must be an integer"),
        (_ci, {"bin_fractional_width": 0.0}, "bin_fractional_width must be positive"),
        (_ci, {"continuum_backend": "idl"}, "continuum_backend must be one of"),
        (_sg, {"maximum_contaminants": 2.5}, "maximum_contaminants must be an integer"),
        (_sg, {"line_list_backend": "idl"}, "line_list_backend must be one of"),
        (_sg, {"abundances": [ABUNDANCE, OTHER_ABUNDANCE], "line_list_file": "a.nc"}, "one abundance"),
        (_sg, {"abundances": []}, "at least one abundance"),
        (_sg, {"effective_area_files": []}, "at least one effective-area file"),
        # a name where a transition is meant
        (_sg, {"transition_boosts": {MAIN_LINE: 2.0}}, "a transition is"),
        (_sg, {"transition_selection": [("fe_9", 1)]}, "a transition is"),
        # per-channel settings: an unknown channel, names that differ by spaces alone, a width without a unit
        (_sg, {"main_lines_by_channel": {94: [MAIN_LINE]}}, "the SG channels are 108, 171, 284"),
        (_sg, {"main_lines_by_channel": {171: [MAIN_LINE, "FeIX 171.073"]}}, "would get one file name"),
        (_sg, {"instrumental_fwhm_by_channel": {171: 0.02}}, "must be one wavelength, such as 0.04"),
        # what goes into a file name: not empty, no slash, which makes a directory, and no underscore in the label
        (_sg, {"effective_area_label": "Al_Zr"}, "effective_area_label is part of a file name"),
        # with a file, as without one the cache refuses the name too
        (_sg, {"abundances": [".."], "line_list_file": "a.nc"}, "abundance is part of a file name"),
        (_sg, {"main_lines_by_channel": {171: ["Fe IX/../171.073"]}}, "would make a directory"),
        # a line-list range that does not cover the products, and one that is no pair of wavelengths
        (_sg, {"line_list_wavelength_range": [160, 170] * u.AA}, "does not cover what the products need"),
        (_sg, {"line_list_wavelength_range": [100, 200, 300] * u.AA}, "must hold a lower and an upper wavelength"),
    ],
)
def test_bad_options_stop_the_call_before_any_work(inputs, tmp_path, line_list_calls, generate, overrides, match):
    with pytest.raises((ValueError, TypeError), match=match):
        generate(inputs, tmp_path / "run", **{"line_list_file": None, **overrides})
    assert not line_list_calls
    assert not list(tmp_path.iterdir())


def test_files_a_generator_cannot_use_stop_the_call_before_any_work(inputs, tmp_path, line_list_calls):
    other_channel = _area_file(tmp_path, channel=94)
    no_detector = _area_file(tmp_path, name="no_detector.zarr", attrs={})

    # the spectrograph: the MUSE channels alone
    for generate, options in ((_sg, {"line_list_file": None}), (_sg_continuum, {})):
        with pytest.raises(ValueError, match="is of channel 94 Angstrom; SG products exist for the MUSE channels 108"):
            generate(inputs, tmp_path / "run", effective_area_files=[other_channel], **options)
    # an imager: any channel, but one that is not of MUSE must describe its detector
    with pytest.raises(
        ValueError, match=r"no channel of the MUSE imager \(195, 304\), so it must give gain and pair_energy"
    ):
        _ci(inputs, tmp_path / "run", effective_area_files=[no_detector], line_list_file=None)

    assert not line_list_calls
    assert not (tmp_path / "run").exists()


@pytest.mark.usefixtures("_fake_continuum_backend")
def test_generate_sg_continuum_responses(inputs, sg_products, tmp_path):
    # coarser than the lines, with the same ends
    velocities = DOPPLER_VELOCITY[[0, -1]]
    written = _sg_continuum(inputs, tmp_path, doppler_velocity=velocities)
    main = _only(sg_products, "FeIX171.073_detector")

    response = read_response(written[0])
    assert response.sizes["doppler_velocity"] == 2
    # the dimensions in the order of the line products
    assert response.detector_response.dims == read_response(main).detector_response.dims
    assert response.attrs["continuum_backend"] == "fiasco"
    # the first entry is how its effective area was made, the last this read
    assert response.attrs["history"][0].startswith("create_effective_area(")
    assert response.attrs["history"][-2].startswith("generate_sg_continuum_responses(")
    # on the waveband, in the steps of 0.1 Angstrom that _sg_continuum asks for
    lower, upper = _waveband(CHANNEL)
    cached = f"continuum_fiasco_{lower:g}-{upper:g}A_n{round((upper - lower) / 0.1) + 1}_{ABUNDANCE}.zarr.zip"
    assert (tmp_path / "cache" / cached).is_file()
    together = read_responses([main, written[0]], doppler_velocity=DOPPLER_VELOCITY)
    assert list(together.line.values) == [MAIN_LINE, "free-free", "free-bound", "two-photon"]

    # the detector a file describes counts here too: pixels twice as wide see twice the solid angle
    wide = _area_file(tmp_path, attrs={"pixel_width": str(2 * DEFAULTS_MUSE.pixel_width_SG)})
    twice = _sg_continuum(inputs, tmp_path / "wide", effective_area_files=[wide], doppler_velocity=velocities)
    xr.testing.assert_allclose(read_response(twice[0]).detector_response, 2 * response.detector_response)
    lines = _sg(inputs, tmp_path / "wide_lines", effective_area_files=[wide], maximum_contaminants=0)
    xr.testing.assert_allclose(_main_response(lines), 2 * _main_response(sg_products))


@pytest.mark.usefixtures("_fake_continuum_backend")
def test_the_defaults_of_the_continuum_generator(inputs, tmp_path):
    """
    Every default abundance and pressure gets one product, as for the lines, on the
    default continuum velocities.
    """
    written = generate_sg_continuum_responses(
        output_directory=tmp_path,
        author="TEST",
        effective_area_label="testea",
        effective_area_files=[inputs["area"]],
        temperature=TEMPERATURE,
        wavelength_step=0.1 * u.AA,
    )

    assert [(path.parent.name, path.name.split("detector_")[1]) for path in written] == [
        (abundance, f"testea_pressure_{pressure:g}.zarr.zip")
        for abundance in products_module._ABUNDANCES
        for pressure in products_module._PRESSURE.value
    ]
    response = read_response(written[0])
    assert response.sizes["pressure"] == 1
    velocities = products_module._CONTINUUM_DOPPLER_VELOCITY
    np.testing.assert_array_equal(response.doppler_velocity.values, velocities.value)


@pytest.mark.chianti
def test_the_generators_on_live_chianti(tmp_path):
    """
    The whole chain on real CHIANTI data: a narrow 171 effective area, the abundant
    elements, one pressure, no continuum.
    """
    pytest.importorskip("ChiantiPy")
    files = {"area": _area_file(tmp_path, wavelength=(165.0, 178.0)), "line_list": None}
    options = {
        "minimum_abundance": 3e-5,
        "line_list_backend": "chianti",
        # The 171 waveband, which the SG call asks for whatever the area spans: given to the
        # imager call too, so that the two share one line list.
        "line_list_wavelength_range": _waveband(CHANNEL) * u.AA,
    }

    sg = _sg(files, tmp_path / "run", **options)
    ci = _ci(files, tmp_path / "run", include_continuum=False, **options)

    assert (len(sg), len(ci)) == (4, 3)
    main = read_response(_only(sg, "FeIX171.073_detector"))
    assert float(main.detector_response.max()) > 0
    # Fe IX 171.073 peaks near logT 5.9: the middle temperature of the grid is the brightest
    by_temperature = main.detector_response.sel(doppler_velocity=0).sum(["slit", "detector_x_pixel"]).squeeze().values
    assert np.argmax(by_temperature) == 1
    cube = read_spectral_response(_only(ci, f"ci_171_waveband_{TAIL}.zarr"))
    assert list(cube.line.values) == ["171 lines"]
    assert float(cube.spectral_response.max()) > 0
    # one line list, made by the first call and read by the second
    lower, upper = _waveband(CHANNEL)
    assert [path.name for path in (tmp_path / "run" / "cache").iterdir()] == [
        f"line_list_chianti_{lower:g}-{upper:g}A_{ABUNDANCE}.zarr.zip"
    ]


def test_the_sg_products_follow_the_instrument_defaults_they_are_given(inputs, sg_products, tmp_path):
    """
    Every instrument value the SG generators look up comes from ``instrument_defaults``:

    the detector geometry, the spectral order, the waveband, the FWHM, the main lines
    and the pixel size. Nothing falls back to ``DEFAULTS_MUSE``, and the products say
    the defaults were others.
    """
    by_channel = {"coords": {"channel": [108, 171, 284]}, "dims": "channel"}
    defaults = evolve(
        DEFAULTS_MUSE,
        spectral_slit_separation_SG=xr.DataArray([380.0, 385.0, 400.0] * u.mAA, **by_channel),
        number_of_slits_SG=5,
        pixels_between_slits_SG=20 * u.pix,
        detector_x_pixels_SG=256,
        wavelength_start_SG=xr.DataArray([107.0, 165.0, 280.0] * u.AA, **by_channel),
        channel_spectral_order_SG=xr.DataArray([2, 1, 1], **by_channel),
        waveband_half_width_SG=10 * u.AA,
        wavelength_step_SG=20 * u.mAA,
        instrumental_fwhm_SG=xr.DataArray([0.08, 0.08, 0.08] * u.AA, **by_channel),
        main_lines_by_channel_SG={108: "Fe XIX 108.355", 171: "Fe X 174.500", 284: "Fe XV 284.163"},
        pixel_width_SG=0.8 * u.arcsec,
        pixel_height_SG=0.3 * u.arcsec,
    )

    lines = _sg(
        inputs, tmp_path / "lines", main_lines_by_channel=None, maximum_contaminants=1, instrument_defaults=defaults
    )
    with pytest.MonkeyPatch.context() as continuum:
        continuum.setitem(cache._CONTINUUM_BACKENDS, "fiasco", fake_continuum)
        continua = _sg_continuum(inputs, tmp_path / "continuum", instrument_defaults=defaults)

    assert sorted(_prefixes(lines)) == [
        f"response_sg_171_FeX174.500_detector_{TAIL}",
        f"response_sg_171_FeX174.500_waveband_{TAIL}",
        f"response_sg_171_contaminants_detector_{TAIL}",
        f"response_sg_171_contaminants_waveband_{TAIL}",
    ]
    assert _prefixes(continua) == [f"response_sg_171_continuum_fiasco_detector_{TAIL}"]
    cube = read_spectral_response(_only(lines, "FeX174.500_waveband"))
    # 10 Angstrom on either side of 171 at its first order, in steps of 20 mAngstrom
    np.testing.assert_allclose(cube.wavelength_grid.values[[0, -1]], [161.0, 181.0])
    assert cube.wavelength_grid.size == 1001
    assert cube.attrs["instrumental_fwhm"] == "0.08 Angstrom"
    # the brightest other line of this waveband; that of the MUSE one is at 155 Angstrom
    contaminants = read_spectral_response(_only(lines, "contaminants_waveband"))
    profile = contaminants.spectral_response.sum(["line", "doppler_velocity", "logT", "pressure"])
    np.testing.assert_allclose(
        float(contaminants.wavelength_grid[int(profile.argmax("wavelength_bin"))]), 165.0, atol=0.1
    )
    # the continuum is computed over this waveband, as the name of its cache states
    assert len(list((tmp_path / "continuum" / "cache").glob("continuum_fiasco_161-181A_*"))) == 1
    for product in (_only(lines, "FeX174.500_detector"), _only(lines, "contaminants_detector"), continua[0]):
        mapped = read_response(product)
        assert (mapped.sizes["slit"], mapped.sizes["detector_x_pixel"]) == (5, 256)
        wavelength = mapped.detector_wavelength
        np.testing.assert_allclose(float(wavelength.isel(slit=0, detector_x_pixel=0)), 165.0)
        np.testing.assert_allclose(wavelength.isel(slit=0).diff("detector_x_pixel"), 0.385 / 20)
        np.testing.assert_allclose(-wavelength.isel(detector_x_pixel=0).diff("slit"), 0.385)
        assert (mapped.attrs["pixel_width"], mapped.attrs["pixel_height"]) == ("0.8 arcsec", "0.3 arcsec")
        assert mapped.attrs["spectral_slit_separation"] == "385 mAngstrom"
        assert mapped.attrs["instrument_defaults"].startswith("custom ")
    assert read_response(sg_products[1]).attrs["instrument_defaults"].startswith("DEFAULTS_MUSE ")


def test_the_ci_products_follow_the_instrument_defaults_they_are_given(inputs, tmp_path):
    """
    A channel of the MUSE imager takes its gain and its pixel size from
    ``instrument_defaults`` when its file states neither.
    """
    area = _area_file(tmp_path, channel=195, wavelength=(180.0, 192.0), attrs=None)
    imager = {"coords": {"channel": [195, 304]}, "dims": "channel"}
    defaults = evolve(
        DEFAULTS_MUSE,
        gain_CI=xr.DataArray([20.0, 10.0] * u.electron / u.DN, **imager),
        pixel_width_CI=3 * DEFAULTS_MUSE.pixel_width_CI,
    )
    options = {"effective_area_files": [area], "include_continuum": False}

    default = _only(_ci(inputs, tmp_path / "default", **options), f"ci_195_{TAIL}_DN")
    custom = _only(_ci(inputs, tmp_path / "custom", instrument_defaults=defaults, **options), f"ci_195_{TAIL}_DN")

    # three times the pixel area over twice the gain
    np.testing.assert_allclose(
        read_response(custom).detector_response, 1.5 * read_response(default).detector_response, rtol=1e-12
    )
    assert read_response(custom).attrs["instrument_defaults"].startswith("custom ")
    # without a continuum none is claimed, and so no minimum abundance for the given line list
    attrs = read_response(default).attrs
    assert (attrs["include_continuum"], "continuum_backend" in attrs, "minimum_abundance" in attrs) == (0, False, False)


def test_imager_files_that_state_their_detector_need_no_imager_in_the_defaults(inputs, tmp_path):
    no_gains = evolve(DEFAULTS_MUSE, gain_CI=None)
    products = _ci(inputs, tmp_path / "both", include_continuum=False, instrument_defaults=no_gains)
    # a file that states its gain alone takes the pair energy of its channel from the defaults
    gain_alone = _area_file(tmp_path, channel=195, wavelength=(180.0, 192.0), attrs={"gain": DETECTOR["gain"]})
    dn = _ci(
        inputs,
        tmp_path / "gain",
        effective_area_files=[gain_alone],
        include_continuum=False,
        instrument_defaults=no_gains,
    )

    assert len(products) == 3
    pair_energy = u.Quantity(read_response(_only(dn, f"ci_195_{TAIL}_DN")).attrs["pair_energy"])
    assert pair_energy == DEFAULTS_MUSE.pair_energy_CI.sel(channel=195).data


def test_custom_instrument_defaults_neither_fall_back_nor_mix(inputs, tmp_path):
    """
    A value the custom defaults lack is refused rather than taken from
    ``DEFAULTS_MUSE``, and products made with the MUSE defaults are not kept for a call
    with others.
    """
    custom = evolve(DEFAULTS_MUSE, pixel_width_SG=0.8 * u.arcsec)
    mixed = r"made with DEFAULTS_MUSE [0-9a-f]{12} instrument defaults and this call uses custom [0-9a-f]{12} ones"
    with pytest.raises(ValueError, match="gives the pixel_width"):
        _sg(inputs, tmp_path / "lacking", instrument_defaults=evolve(DEFAULTS_MUSE, pixel_width_SG=None))
    with pytest.raises(ValueError, match="instrument_defaults has no spectral_slit_separation_SG"):
        _sg(inputs, tmp_path / "lacking", instrument_defaults=evolve(DEFAULTS_MUSE, spectral_slit_separation_SG=None))

    _sg(inputs, tmp_path / "mixed", maximum_contaminants=0)
    with pytest.raises(ValueError, match=mixed):
        _sg(inputs, tmp_path / "mixed", maximum_contaminants=0, skip_existing=True, instrument_defaults=custom)
