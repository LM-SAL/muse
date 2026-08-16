from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument import linelist as linelist_module
from muse.instrument import map_response_to_ci_detector, map_response_to_sg_detector
from muse.instrument.linelist import create_chianti_line_list
from muse.instrument.radiometry import transform_response_units
from muse.instrument.response_io import read_response
from muse.instrument.spectral import create_spectral_response
from muse.synthesis import calculate_moments, vdem_synthesis, wavelength_to_doppler
from muse.variables import DEFAULTS_MUSE


def _spectral_response(*, contaminants: bool = False) -> xr.Dataset:
    wavelength = np.arange(169.5, 171.51, 0.01)
    line = ["Fe IX 171.073"]
    line_wavelength = [171.073]
    component_kind = ["line"]
    if contaminants:
        line.append("contaminants")
        line_wavelength.append(np.nan)
        component_kind.append("contaminants")

    values = np.broadcast_to(
        wavelength,
        (len(line), 1, 1, wavelength.size),
    ).copy()
    return xr.Dataset(
        {
            "spectral_response": (
                ("line", "logT", "doppler_velocity", "wavelength_bin"),
                values,
                {"units": "1e-27 erg cm5 / (Angstrom s sr)"},
            )
        },
        coords={
            "line": line,
            "logT": [6.0],
            "doppler_velocity": ("doppler_velocity", [0.0], {"units": "km / s"}),
            "wavelength_grid": ("wavelength_bin", wavelength, {"units": "Angstrom"}),
            "line_wavelength": ("line", line_wavelength, {"units": "Angstrom"}),
            "component_kind": ("line", component_kind),
        },
        attrs={"normalization": 1e-27},
    )


def _ci_spectral_response() -> xr.Dataset:
    response = _spectral_response().isel(wavelength_bin=slice(0, 3))
    values = np.broadcast_to([1.0, 2.0, 4.0], response.spectral_response.shape).copy()
    return response.assign(
        spectral_response=(response.spectral_response.dims, values, response.spectral_response.attrs),
    ).assign_coords(
        wavelength_grid=("wavelength_bin", [190.0, 191.0, 194.0], {"units": "Angstrom"}),
        line_wavelength=("line", [195.0], {"units": "Angstrom"}),
    )


def test_map_response_to_sg_detector_geometry_and_units():
    response = _spectral_response()
    original = response.copy(deep=True)

    mapped = map_response_to_sg_detector(
        response,
        number_of_slits=2,
        dispersion=0.1 * u.AA / u.pix,
        slit_spacing=2 * u.pix,
        detector_pixels=4,
        wavelength_start=170 * u.AA,
    )

    expected_wavelength = (170.0 + np.arange(4) * 0.1)[np.newaxis, :] - np.array([[0.0], [0.2]])
    wavelength = response.wavelength_grid.values
    expected_response = np.stack([np.interp(row, wavelength, wavelength) for row in expected_wavelength]) * 0.1

    assert mapped.detector_response.dims == ("line", "logT", "doppler_velocity", "slit", "detector_x_pixel")
    assert mapped.detector_wavelength.dims == ("detector_x_pixel", "slit")
    np.testing.assert_allclose(
        mapped.detector_wavelength.transpose("slit", "detector_x_pixel"),
        expected_wavelength,
    )
    np.testing.assert_allclose(np.diff(mapped.detector_wavelength.isel(slit=0)), 0.1)
    np.testing.assert_allclose(mapped.detector_response.isel(line=0, logT=0, doppler_velocity=0), expected_response)
    assert u.Unit(mapped.detector_response.attrs["units"]) == u.Unit("1e-27 erg cm5 / (s sr)")
    assert mapped.detector_wavelength.attrs["units"] == "Angstrom"
    assert mapped.line_wavelength.attrs["units"] == "Angstrom"
    assert mapped.line_wavelength.item() == pytest.approx(171.073)
    assert "channel" not in mapped.coords
    assert "spectral_response" not in mapped
    assert "wavelength_bin" not in mapped.dims
    assert mapped.attrs["HISTORY"][-1].startswith("map_response_to_sg_detector(")
    xr.testing.assert_identical(response, original)


def test_map_response_to_sg_detector_keeps_chunked_input_lazy():
    response = _spectral_response()
    kwargs = {
        "number_of_slits": 2,
        "dispersion": 0.1 * u.AA / u.pix,
        "slit_spacing": 2 * u.pix,
        "detector_pixels": 4,
        "wavelength_start": 170 * u.AA,
    }

    lazy = map_response_to_sg_detector(response.chunk({"doppler_velocity": 1}), **kwargs)
    eager = map_response_to_sg_detector(response, **kwargs)

    assert isinstance(lazy.detector_response.data, da.Array)
    xr.testing.assert_allclose(lazy.compute(), eager)


def test_map_response_to_sg_detector_integrates_over_detector_pixels():
    response = _spectral_response()
    response["spectral_response"] = (
        response.spectral_response.dims,
        np.ones(response.spectral_response.shape),
        response.spectral_response.attrs,
    )

    mapped = map_response_to_sg_detector(
        response,
        number_of_slits=1,
        dispersion=0.01 * u.AA / u.pix,
        detector_pixels=100,
        wavelength_start=170 * u.AA,
    )

    # A flat response of 1 per Angstrom over 100 pixels of 0.01 Angstrom each.
    np.testing.assert_allclose(mapped.detector_response.sum("detector_x_pixel"), 1.0, rtol=1e-12)


def test_map_response_to_sg_detector_uses_muse_defaults():
    mapped = map_response_to_sg_detector(_spectral_response(), channel=17.1 * u.nm)

    assert mapped.sizes["slit"] == DEFAULTS_MUSE.number_of_slits_SG
    assert mapped.sizes["detector_x_pixel"] == DEFAULTS_MUSE.pixels_SG.to_value(u.pix)
    expected_start = u.Quantity(DEFAULTS_MUSE.initial_wavelength_SG.sel(channel=171).data).to_value(u.AA)
    assert mapped.detector_wavelength.isel(slit=0, detector_x_pixel=0).item() == pytest.approx(expected_start)
    dispersion = (
        2
        * DEFAULTS_MUSE.spectral_slit_separation_SG
        / DEFAULTS_MUSE.pixels_between_slits_SG
        / DEFAULTS_MUSE.channel_spectral_order_SG.sel(channel=171).item()
    ).to_value(u.AA / u.pix)
    assert mapped.detector_wavelength.isel(slit=0, detector_x_pixel=-1).item() == pytest.approx(
        expected_start + (DEFAULTS_MUSE.pixels_SG.to_value(u.pix) - 1) * dispersion
    )
    assert mapped.channel.item() == 171
    assert mapped.channel.attrs["units"] == "Angstrom"


def test_map_response_to_sg_detector_gives_contaminants_a_line_reference():
    mapped = map_response_to_sg_detector(
        _spectral_response(contaminants=True),
        channel=171 * u.AA,
        number_of_slits=1,
        detector_pixels=1,
        wavelength_start=171 * u.AA,
    )

    np.testing.assert_allclose(mapped.line_wavelength, [171.073, 171.073])
    np.testing.assert_array_equal(mapped.component_kind, ["line", "contaminants"])


@pytest.mark.parametrize("component_kind", [None, ["line", "line"]])
def test_map_response_to_sg_detector_preserves_invalid_physical_line_wavelength(component_kind):
    response = _spectral_response(contaminants=True)
    if component_kind is None:
        response = response.drop_vars("component_kind")
    else:
        response = response.assign_coords(component_kind=("line", component_kind))

    mapped = map_response_to_sg_detector(response, channel=171 * u.AA, number_of_slits=1, detector_pixels=1)

    np.testing.assert_allclose(mapped.line_wavelength, [171.073, np.nan])


def test_map_response_to_sg_detector_preserves_input_nan():
    response = _spectral_response()
    response.spectral_response.data[..., response.sizes["wavelength_bin"] // 2] = np.nan

    mapped = map_response_to_sg_detector(
        response,
        number_of_slits=1,
        dispersion=0.01 * u.AA / u.pix,
        detector_pixels=response.sizes["wavelength_bin"],
        wavelength_start=response.wavelength_grid[0].item() * u.AA,
    )

    assert bool(mapped.detector_response.isnull().any())
    assert mapped.detector_response.isel(detector_x_pixel=-1).item() == 0


def test_map_response_to_sg_detector_requires_effective_area():
    response = _spectral_response()
    response.spectral_response.attrs["units"] = "1e-27 erg cm3 / (Angstrom s sr)"

    with pytest.raises(ValueError, match="convertible"):
        map_response_to_sg_detector(response, channel=171 * u.AA)


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("response_type", TypeError, "xarray.Dataset"),
        ("channel", ValueError, "unsupported MUSE SG channel"),
        ("channel_unitless", TypeError, "no 'unit' attribute"),
        ("channel_units", u.UnitsError, "convertible to 'Angstrom'"),
        ("channel_shape", ValueError, "scalar wavelength"),
        ("channel_nonfinite", ValueError, "finite, positive wavelength"),
        ("schema", ValueError, "missing required variables"),
        ("normalization", ValueError, "normalization"),
        ("wavelength_grid", ValueError, "strictly increasing"),
        ("wavelength_grid_empty", ValueError, "wavelength_grid"),
        ("wavelength_grid_positive", ValueError, "positive"),
        ("slit_spacing", ValueError, "slit_spacing"),
        ("geometry", ValueError, "number_of_slits"),
    ],
)
def test_map_response_to_sg_detector_rejects_invalid_inputs(case, error, match):
    response = _spectral_response()
    channel = 171 * u.AA
    kwargs = {}
    if case == "response_type":
        response = None
    elif case == "channel":
        channel = 195 * u.AA
    elif case == "channel_unitless":
        channel = 171
    elif case == "channel_units":
        channel = 1 * u.s
    elif case == "channel_shape":
        channel = [171] * u.AA
    elif case == "channel_nonfinite":
        channel = np.nan * u.AA
    elif case == "schema":
        response = response.drop_vars("line_wavelength")
    elif case == "normalization":
        response.attrs["normalization"] = 0
    elif case == "wavelength_grid":
        response = response.assign_coords(wavelength_grid=response.wavelength_grid[::-1])
    elif case == "wavelength_grid_empty":
        response = response.isel(wavelength_bin=slice(0, 0))
    elif case == "wavelength_grid_positive":
        response = response.assign_coords(wavelength_grid=response.wavelength_grid - 200)
    elif case == "slit_spacing":
        kwargs["slit_spacing"] = 0 * u.pix
    else:
        kwargs["number_of_slits"] = 0

    with pytest.raises(error, match=match):
        map_response_to_sg_detector(response, channel=channel, **kwargs)


def test_map_response_to_sg_detector_requires_explicit_geometry_without_channel():
    with pytest.raises(
        ValueError,
        match="missing: number_of_slits, dispersion, detector_pixels, wavelength_start",
    ):
        map_response_to_sg_detector(_spectral_response())

    with pytest.raises(ValueError, match="missing: slit_spacing"):
        map_response_to_sg_detector(
            _spectral_response(),
            number_of_slits=2,
            dispersion=0.1 * u.AA / u.pix,
            detector_pixels=4,
            wavelength_start=170 * u.AA,
        )


def test_map_response_to_ci_detector_integrates_nonuniform_grid_without_detector_axis():
    response = _ci_spectral_response().chunk({"wavelength_bin": 2})
    original = response.copy(deep=True)

    with xr.set_options(keep_attrs=False):
        mapped = map_response_to_ci_detector(response, 195 * u.AA)

    assert isinstance(mapped.detector_response.data, da.Array)
    assert mapped.detector_response.dims == ("line", "logT", "doppler_velocity")
    assert "wavelength_bin" not in mapped.dims
    assert "detector_x_pixel" not in mapped.dims
    assert mapped.detector_wavelength.dims == ("line",)
    assert mapped.detector_wavelength.item() == 195
    assert mapped.detector_wavelength.attrs["units"] == "Angstrom"
    assert mapped.line_wavelength.item() == 195
    assert mapped.channel.item() == 195
    assert mapped.channel.attrs["units"] == "Angstrom"
    assert mapped.attrs["normalization"] == 1e-27
    assert mapped.attrs["HISTORY"][-1].startswith("map_response_to_ci_detector(")
    assert u.Unit(mapped.detector_response.attrs["units"]) == u.Unit("1e-27 erg cm5 / (s sr)")
    np.testing.assert_allclose(mapped.detector_response.compute(), 10.5)
    xr.testing.assert_identical(response, original)


def test_map_response_to_ci_detector_accepts_any_channel():
    # The channel is only recorded in the output coordinates, so non-MUSE
    # imagers (e.g. AIA 171) pass their own band centre.
    response = _ci_spectral_response()

    mapped = map_response_to_ci_detector(response, 171 * u.AA)

    assert mapped.channel.item() == 171
    assert mapped.detector_wavelength.item() == 171
    # Basic channel validation still applies.
    with pytest.raises(ValueError, match="finite, positive"):
        map_response_to_ci_detector(response, -171 * u.AA)


def test_map_response_to_ci_detector_accepts_cm3_units():
    # e.g. an AIA line list carries gofnt per cm3 instead of the MUSE cm5 bases.
    response = _ci_spectral_response()
    response = response.assign(
        spectral_response=response.spectral_response.assign_attrs(units="1e-27 cm3 erg / (Angstrom s sr)"),
    )

    mapped = map_response_to_ci_detector(response, 171 * u.AA)

    assert u.Unit(mapped.detector_response.attrs["units"]) == u.Unit("1e-27 cm3 erg / (s sr)")


@pytest.mark.parametrize("units", ["s", "erg cm3 / (s sr)"])
def test_map_response_to_ci_detector_rejects_invalid_units(units):
    response = _ci_spectral_response()
    response = response.assign(
        spectral_response=response.spectral_response.assign_attrs(units=units),
    )

    with pytest.raises(ValueError, match="convertible"):
        map_response_to_ci_detector(response, 171 * u.AA)


def test_ci_response_maps_directly_into_synthesis():
    spectral = _ci_spectral_response().assign_coords(
        line_wavelength=("line", [304.0], {"units": "Angstrom"}),
    )
    spectral = transform_response_units(spectral, "1e-27 cm5 ph / (Angstrom s)", 304 * u.AA, detector="ci")
    response = map_response_to_ci_detector(spectral, 304 * u.AA)
    raster = xr.Dataset(
        {
            "vdem": (
                ("logT", "doppler_velocity"),
                np.ones((1, 1)),
                {"units": "cm-5"},
            )
        },
        coords={
            "logT": response.logT,
            "doppler_velocity": response.doppler_velocity,
        },
    )

    synthesized = vdem_synthesis(raster, response, sum_over=("logT", "doppler_velocity"))

    np.testing.assert_allclose(
        synthesized.flux,
        response.detector_response.sum(("logT", "doppler_velocity")),
    )
    assert synthesized.channel.item() == 304
    assert synthesized.detector_wavelength.item() == 304
    assert u.Unit(synthesized.flux.attrs["units"]) == u.Unit("1e-27 ph / s")


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("response_type", TypeError, "xarray.Dataset"),
        ("channel_unitless", TypeError, "no 'unit' attribute"),
        ("channel_units", u.UnitsError, "convertible to 'Angstrom'"),
        ("schema", ValueError, "missing required variables"),
        ("normalization", ValueError, "normalization"),
        ("wavelength_grid", ValueError, "strictly increasing"),
        ("wavelength_grid_short", ValueError, "at least two points"),
    ],
)
def test_map_response_to_ci_detector_rejects_invalid_inputs(case, error, match):
    response = _ci_spectral_response()
    channel = 195 * u.AA
    if case == "response_type":
        response = None
    elif case == "channel_unitless":
        channel = 195
    elif case == "channel_units":
        channel = 1 * u.s
    elif case == "schema":
        response = response.drop_vars("line_wavelength")
    elif case == "normalization":
        response.attrs["normalization"] = 0
    elif case == "wavelength_grid":
        response = response.assign_coords(wavelength_grid=response.wavelength_grid[::-1])
    else:
        response = response.isel(wavelength_bin=slice(0, 1))

    with pytest.raises(error, match=match):
        map_response_to_ci_detector(response, channel)


@pytest.mark.filterwarnings(
    "ignore:numpy.ndarray size changed:RuntimeWarning",
    "ignore:Setting the shape on a NumPy array has been deprecated in NumPy:DeprecationWarning: ",
)
def test_public_response_workflow_composes_through_moment_analysis(monkeypatch, tmp_path):
    generated_line_list = xr.Dataset(
        {
            "wavelength": ("trans_index", [171.073], {"units": "Angstrom"}),
            "atomic_number": ("trans_index", [26]),
            "gofnt": (
                ("logT", "trans_index"),
                [[1e-25]],
                {"units": "erg cm3 / (s sr)"},
            ),
            "full_name": ("trans_index", ["Fe IX 171.073"]),
        },
        coords={"logT": [6.0]},
    )

    monkeypatch.setattr(linelist_module, "_initialize_chianti", lambda: ("test", object()))
    monkeypatch.setattr(
        linelist_module,
        "_compute_bunch",
        lambda *_args, **_kwargs: SimpleNamespace(AbundanceName="test.abund"),
    )
    monkeypatch.setattr(
        linelist_module,
        "_chianti_bunch_to_dataset",
        lambda *_args, **_kwargs: generated_line_list.copy(deep=True),
    )
    line_list = create_chianti_line_list(
        xr.DataArray([1e6] * u.K, dims="logT"),
        pressure=xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure"),
        wavelength_range=[170.0, 172.0] * u.AA,
        ion_list=["fe_9"],
    )
    effective_area = xr.DataArray(
        [10.0, 10.0],
        dims="wavelength",
        coords={"wavelength": ("wavelength", [170.0, 172.0], {"units": "Angstrom"})},
        attrs={"units": "cm2"},
    )
    spectral = create_spectral_response(
        line_list,
        np.linspace(170.5, 171.5, 101) * u.AA,
        main_lines=["Fe IX 171.073"],
        doppler_velocity=[0.0] * u.km / u.s,
        effective_area=effective_area,
    )
    assert "slit" not in spectral.dims
    spectral = transform_response_units(spectral, "1e-27 cm5 ph / (Angstrom s)", 171 * u.AA)
    response = map_response_to_sg_detector(
        spectral,
        channel=171 * u.AA,
        number_of_slits=2,
        dispersion=0.01 * u.AA / u.pix,
        slit_spacing=2 * u.pix,
        detector_pixels=5,
        wavelength_start=171.05 * u.AA,
    )
    raster = xr.Dataset(
        {
            "vdem": (
                ("logT", "doppler_velocity", "slit"),
                np.ones((1, 1, 2)),
                {"units": "1e27 / cm5"},
            )
        },
        coords={"logT": response.logT, "doppler_velocity": response.doppler_velocity, "slit": response.slit},
    )
    path = tmp_path / "response.nc"
    response.to_netcdf(path)
    loaded_response = read_response(path).load()
    loaded_response.close()

    synthesized = vdem_synthesis(raster, loaded_response)

    assert u.Unit(response.detector_response.attrs["units"]) == u.Unit("1e-27 ph cm5 / s")
    assert u.Unit(synthesized.flux.attrs["units"]) == u.Unit("ph / s")
    assert synthesized.flux.dims == ("line", "detector_x_pixel")
    assert np.isfinite(synthesized.flux).all()
    assert bool((synthesized.flux > 0).any())
    assert synthesized.line_wavelength.item() == pytest.approx(171.073)

    # Moment analysis needs the per-slit detector_wavelength coordinate, so keep the
    # slit dimension out of the contraction (the tutorial-05/06 workflow).
    spectrum = vdem_synthesis(raster, loaded_response, sum_over=("logT", "doppler_velocity"))
    moments = calculate_moments(wavelength_to_doppler(spectrum))

    assert moments["0th"].dims == ("slit", "line")
    assert u.Unit(moments["0th"].attrs["units"]) == u.Unit("ph / s")
    assert moments["1st"].attrs["units"] == "km / s"
    assert moments["2nd"].attrs["units"] == "km / s"
    for name in ("0th", "1st", "2nd"):
        assert np.isfinite(moments[name]).all()
    assert moments.attrs["HISTORY"] == [
        (
            "create_chianti_line_list(temperature=[1000000.0], density=None, pressure=[3000000000000000.0], "
            "abundance=test, wavelength_range=(170.0, 172.0), minimum_abundance=None, element_list=None, "
            "ion_list=['fe_9'])"
        ),
        (
            f"create_spectral_response(line_list=line_list, wavelength_grid={np.linspace(170.5, 171.5, 101)}, "
            "main_lines=['Fe IX 171.073'], instrumental_width=0.0, doppler_velocity=[0.], nonthermal_velocity=None, "
            "effective_area=effective_area, include_contaminants=False)"
        ),
        (
            "transform_response_units(response=response, new_units=1e-27 cm5 ph / (Angstrom s), channel=171.0, "
            "detector=sg, pixel_width=0.4, pixel_height=0.167, gain=10.0, pair_energy=3.65)"
        ),
        (
            "map_response_to_sg_detector(response=response, channel=171.0, number_of_slits=2, dispersion=0.01, "
            "slit_spacing=2.0, detector_pixels=5, wavelength_start=171.05)"
        ),
        (
            f"read_response(response_file={path}, logT=None, doppler_velocity=None, slit=None, logT_method=nearest, "
            "doppler_velocity_method=nearest, gain=[10.], chunked=False)"
        ),
        (
            "vdem_synthesis(raster=raster, response=response, sum_over=('logT', 'doppler_velocity'), "
            "cuda_device=None, backend=numpy)"
        ),
        "wavelength_to_doppler(response=response)",
        (
            "calculate_moments(spectrum=spectrum, moment_dim=detector_x_pixel, integration_name=flux, "
            "doppler_name=doppler_velocity, vmax=None, vmask=None)"
        ),
    ]
