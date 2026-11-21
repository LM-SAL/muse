import dask.array as dask_array
import numpy as np
import pytest
import xarray as xr

import astropy.units as u

import muse.synthesis.utils as synthesis_utils
from muse.synthesis.synthesis import vdem_synthesis
from muse.variables import DEFAULTS_MUSE


def _spectrum(response, raster):
    # Keep slit so detector_wavelength survives, then add the Doppler coordinate moments require.
    spectrum = vdem_synthesis(raster, response, sum_over=("logT", "doppler_velocity"))
    return synthesis_utils.wavelength_to_doppler(spectrum)


def _tiny_response():
    line_wavelength = np.array([108.355, 171.073])
    detector_wavelength = np.array([[108.0, 108.355, 108.7], [170.6, 171.073, 171.5]])
    ds = xr.Dataset()
    ds = ds.assign_coords(
        line_wavelength=("line", line_wavelength),
        detector_wavelength=(("line", "detector_x_pixel"), detector_wavelength),
    )
    ds.line_wavelength.attrs["units"] = "Angstrom"
    ds.detector_wavelength.attrs["units"] = "Angstrom"
    return ds


def _tiny_moment_spectrum():
    return xr.Dataset(
        data_vars={
            "flux": (
                ("detector_x_pixel",),
                [1.0, 2.0, 10.0, 4.0, 1.0],
                {"units": "ph / s"},
            ),
        },
        coords={
            "detector_x_pixel": [0, 1, 2, 3, 4],
            "detector_doppler_velocity": (
                ("detector_x_pixel",),
                [-200.0, -100.0, 0.0, 100.0, 200.0],
                {"units": "km/s"},
            ),
        },
    )


def test_wavelength_doppler_round_trip() -> None:
    resp = _tiny_response()
    with_vel = synthesis_utils.wavelength_to_doppler(resp)

    assert u.Unit(with_vel.detector_doppler_velocity.attrs["units"]) == u.km / u.s
    assert "detector_doppler_velocity" not in resp.coords  # input must not be mutated

    back = synthesis_utils.doppler_to_wavelength(with_vel.drop_vars("detector_wavelength"))
    np.testing.assert_allclose(back.detector_wavelength.values, resp.detector_wavelength.values, rtol=1e-10)
    assert u.Unit(back.detector_wavelength.attrs["units"]) == u.AA


def test_doppler_to_wavelength_converts_the_coordinate_it_is_given() -> None:
    axis = ("doppler_velocity", [-300.0, 0.0, 300.0], {"units": "km / s"})
    mapped = _tiny_response().assign_coords(doppler_velocity=axis)
    with pytest.raises(ValueError, match="already has a detector_wavelength"):
        synthesis_utils.doppler_to_wavelength(mapped, doppler_name="doppler_velocity")
    resp = mapped.drop_vars("detector_wavelength")

    out = synthesis_utils.doppler_to_wavelength(resp, doppler_name="doppler_velocity")

    assert out.detector_wavelength.sizes == {"line": 2, "doppler_velocity": 3}
    np.testing.assert_allclose(out.detector_wavelength.sel(doppler_velocity=0.0), resp.line_wavelength)
    np.testing.assert_allclose(
        out.detector_wavelength.isel(line=1, doppler_velocity=2), 171.073 * (1 + 300 / 299792.458)
    )


def test_wavelength_to_doppler_normalizes_units() -> None:
    # Same physical wavelengths expressed in nm must give the same Doppler shift.
    resp = _tiny_response()
    resp_nm = resp.assign_coords(detector_wavelength=resp.detector_wavelength / 10.0)
    resp_nm.detector_wavelength.attrs["units"] = "nm"

    angstrom = synthesis_utils.wavelength_to_doppler(resp)
    nanometer = synthesis_utils.wavelength_to_doppler(resp_nm)
    np.testing.assert_allclose(
        angstrom.detector_doppler_velocity.values,
        nanometer.detector_doppler_velocity.values,
        rtol=1e-9,
        atol=1e-6,
    )


def test_wavelength_to_doppler_requires_units() -> None:
    resp = _tiny_response()
    del resp.detector_wavelength.attrs["units"]
    with pytest.raises(ValueError, match=r"data\.detector_wavelength must define units"):
        synthesis_utils.wavelength_to_doppler(resp)


def test_calculate_moments_structure(response, raster) -> None:
    spectrum = _spectrum(response, raster)
    moments = synthesis_utils.calculate_moments(spectrum)

    assert set(moments.data_vars) == {"0th", "1st", "2nd"}
    assert {"y", "slit", "step", "line"} <= set(moments.coords)
    assert dict(moments.sizes) == {
        "y": 32,
        "slit": DEFAULTS_MUSE.number_of_slits_SG,
        "step": DEFAULTS_MUSE.steps_per_raster_SG,
        "line": 7,
    }
    assert np.isfinite(moments.to_dataarray()).all()
    assert u.Unit(moments["1st"].attrs["units"]) == u.km / u.s
    assert u.Unit(moments["2nd"].attrs["units"]) == u.km / u.s


def test_moments_pipeline_keeps_dask_inputs_lazy(response, raster) -> None:
    # Spectra opened with chunks="auto" (see example 06) must stay lazy through
    # wavelength_to_doppler and calculate_moments and match the eager results.
    eager_spectrum = _spectrum(response, raster)
    lazy_spectrum = synthesis_utils.wavelength_to_doppler(
        vdem_synthesis(
            raster.chunk({"logT": 4}),
            response,
            sum_over=("logT", "doppler_velocity"),
        )
    )
    assert isinstance(lazy_spectrum.flux.data, dask_array.Array)

    lazy_moments = synthesis_utils.calculate_moments(lazy_spectrum)
    eager_moments = synthesis_utils.calculate_moments(eager_spectrum)
    for name in ("0th", "1st", "2nd"):
        assert isinstance(lazy_moments[name].data, dask_array.Array)
        # atol absorbs chunk-summation float noise where a moment is itself ~0 km/s.
        np.testing.assert_allclose(
            lazy_moments[name].compute().values, eager_moments[name].values, rtol=1e-9, atol=1e-9, equal_nan=True
        )


def test_calculate_moments_vmax_keeps_nan_doppler_flux() -> None:
    spectrum = xr.Dataset(
        data_vars={"flux": (("detector_x_pixel",), [5.0, 3.0], {"units": "ph / s"})},
        coords={
            "detector_x_pixel": [0, 1],
            "detector_doppler_velocity": (("detector_x_pixel",), [np.nan, 100.0], {"units": "km/s"}),
        },
    )

    # 50 km/s given in m/s: the 100 km/s pixel is cut, the one without a velocity is kept.
    moments = synthesis_utils.calculate_moments(spectrum, maximum_doppler_velocity=5e4 * u.m / u.s)

    assert moments["0th"].item() == 5.0


def test_calculate_moments_rejects_unrelated_doppler_dim() -> None:
    spectrum = xr.Dataset(
        data_vars={"flux": (("detector_x_pixel",), [5.0, 3.0], {"units": "ph / s"})},
        coords={
            "detector_x_pixel": [0, 1],
            "detector_doppler_velocity": (("other",), [0.0, 100.0], {"units": "km/s"}),
        },
    )

    with pytest.raises(ValueError, match=r"dimensions must be a subset"):
        synthesis_utils.calculate_moments(spectrum)


def test_calculate_moments_with_and_without_masks() -> None:
    spectrum = _tiny_moment_spectrum()
    flux, velocity = spectrum.flux.values, spectrum.detector_doppler_velocity.values

    moments = synthesis_utils.calculate_moments(spectrum)
    first = np.average(velocity, weights=flux)
    assert moments["0th"].item() == flux.sum()
    np.testing.assert_allclose(moments["1st"].item(), first)
    np.testing.assert_allclose(moments["2nd"].item(), np.sqrt(np.average((velocity - first) ** 2, weights=flux)))
    assert synthesis_utils.calculate_moments(spectrum, maximum_doppler_velocity=150 * u.km / u.s)["0th"].item() == 16.0
    moments = synthesis_utils.calculate_moments(spectrum, peak_half_width=1 * u.pix)
    assert moments["0th"].item() == 10.0
    assert moments["1st"].item() == 0.0
    assert moments["2nd"].item() == 0.0


def test_calculate_moments_requires_doppler_velocity(response, raster) -> None:
    spectrum = vdem_synthesis(raster, response, sum_over=("logT", "doppler_velocity"))  # no Doppler coordinate
    with pytest.raises(ValueError, match=r"run wavelength_to_doppler first"):
        synthesis_utils.calculate_moments(spectrum)


def _tiny_vdem_inputs():
    # x (axis 0) and y (axis 1) lengths differ so the test also pins the x-then-y axis order.
    return {
        "temperature": np.full((2, 3, 2), 1e5) * u.K,
        "velocity": np.zeros((2, 3, 2)) * u.km / u.s,
        "ne_nh": np.ones((2, 3, 2)) * u.cm**-6,
        "cell_length": np.array([2.0, 3.0]) * u.cm,
        "x": np.array([10.0, 11.0]) * u.cm,
        "y": np.array([20.0, 21.0, 22.0]) * u.cm,
        "velocity_grid": np.array([-1.0, 0.0, 1.0]) * u.km / u.s,
        "logT_grid": np.array([4.5, 5.0, 5.5]) * u.dex(u.K),
    }


def _expected_tiny_vdem():
    # Uniform column: only the boundary cell (prev=100) has a temperature gradient, so it
    # spreads over the low-T bins; every (x, y) column is identical.
    expected = np.zeros((3, 3, 2, 3))
    expected[0, 1] = 2e-27
    expected[1, 1] = 1e-27
    return expected


def test_create_simple_vdem_tiny_cube() -> None:
    result = synthesis_utils.create_simple_vdem(**_tiny_vdem_inputs())

    assert result.vdem.dims == ("logT", "doppler_velocity", "x", "y")
    np.testing.assert_array_equal(result.logT.values, [4.5, 5.0, 5.5])
    np.testing.assert_array_equal(result.doppler_velocity.values, [-1.0, -0.0, 1.0])
    np.testing.assert_array_equal(result.x.values, [10.0, 11.0])
    np.testing.assert_array_equal(result.y.values, [20.0, 21.0, 22.0])
    np.testing.assert_allclose(result.vdem.values, _expected_tiny_vdem(), rtol=1e-12)
    units = {name: result[name].attrs["units"] for name in ("vdem", "logT", "doppler_velocity", "x", "y")}
    assert units == {"vdem": "1e27 / cm5", "logT": "dex(K)", "doppler_velocity": "km/s", "x": "cm", "y": "cm"}
    assert result.attrs["history"][0].startswith("create_simple_vdem(")


def test_create_simple_vdem_converts_its_inputs() -> None:
    # The same cube in other units gives the same VDEM, so no input is read as a bare number.
    inputs = _tiny_vdem_inputs()
    inputs["velocity"] = np.full((2, 3, 2), 1.0) * u.km / u.s
    expected = synthesis_utils.create_simple_vdem(**inputs)
    other_units = {
        "temperature": u.MK,
        "velocity": u.m / u.s,
        "ne_nh": u.m**-6,
        "cell_length": u.m,
        "x": u.m,
        "y": u.m,
        "velocity_grid": u.m / u.s,
        "logT_grid": u.K,  # a linear temperature grid is taken as its logarithm
    }

    result = synthesis_utils.create_simple_vdem(**{name: inputs[name].to(unit) for name, unit in other_units.items()})

    # atol=0: VDEM values are of order 1e-27, far below the default absolute tolerance.
    xr.testing.assert_allclose(result.vdem, expected.vdem, rtol=1e-12, atol=0)


def test_create_simple_vdem_velocity_bin_edges_are_half_open() -> None:
    # Velocity exactly on a bin edge must land in the upper bin ([edge_lo, edge_hi) convention).
    inputs = _tiny_vdem_inputs()
    # Bin centers [-1, 0, 1] with dv=1 give edges [-1.5, -0.5, 0.5, 1.5]; -0.5 is the -1|0 edge.
    inputs["velocity"] = np.full((2, 3, 2), -0.5) * u.km / u.s

    result = synthesis_utils.create_simple_vdem(**inputs)
    # doppler_velocity axis is -velocity_grid[::-1] = [-1, 0, 1]; emission must sit in doppler_velocity=0, not doppler_velocity=-1.
    emission_per_vdop = result.vdem.sum(dim=("logT", "x", "y")).values
    assert emission_per_vdop[1] > 0  # doppler_velocity == 0 bin
    assert emission_per_vdop[0] == 0  # doppler_velocity == -1 bin stays empty


def test_create_simple_vdem_flow_towards_observer_is_blueshifted() -> None:
    # +1 km/s is towards the observer, so all emission must sit at doppler_velocity == -1. The
    # grid is asymmetric so that a missing reversal of the data or the axis also moves it.
    inputs = _tiny_vdem_inputs()
    inputs["velocity"] = np.full((2, 3, 2), 1.0) * u.km / u.s
    inputs["velocity_grid"] = np.array([-1.0, 0.0, 1.0, 2.0]) * u.km / u.s

    emission = synthesis_utils.create_simple_vdem(**inputs).vdem.sum(dim=("logT", "x", "y"))
    assert emission.sel(doppler_velocity=-1.0).item() > 0
    assert emission.sel(doppler_velocity=-1.0).item() == emission.sum().item()


def test_create_simple_vdem_internal_x_slices_are_exact() -> None:
    # Multiple x indices exercise the per-slice worker tasks; each must be exact.
    inputs = _tiny_vdem_inputs()
    n_x = 3
    shape = (n_x, 3, 2)
    inputs["temperature"] = np.full(shape, 1e5) * u.K
    inputs["velocity"] = np.zeros(shape) * u.km / u.s
    x_scale = np.arange(1.0, n_x + 1)
    inputs["ne_nh"] = np.broadcast_to(x_scale[:, np.newaxis, np.newaxis], shape) * u.cm**-6
    inputs["x"] = np.arange(n_x) * u.cm

    result = synthesis_utils.create_simple_vdem(**inputs)

    expected = _expected_tiny_vdem()[:, :, :1, :] * x_scale[np.newaxis, np.newaxis, :, np.newaxis]
    np.testing.assert_allclose(result.vdem.values, expected, rtol=1e-12)


def test_create_simple_vdem_integration_axis_matches_transposed_input() -> None:
    # Feeding cubes with the LOS axis moved first plus the matching integration_axis must reproduce
    # the default-axis result; ne_nh varies along every axis so a mix-up cannot cancel.
    inputs = _tiny_vdem_inputs()
    inputs["ne_nh"] = np.linspace(1.0, 2.0, 12).reshape(2, 3, 2) * u.cm**-6
    default = synthesis_utils.create_simple_vdem(**inputs)

    moved = dict(inputs)
    for cube in ("temperature", "velocity", "ne_nh"):
        moved[cube] = np.moveaxis(inputs[cube], 2, 0)
    result = synthesis_utils.create_simple_vdem(**moved, integration_axis=0)
    np.testing.assert_array_equal(result.vdem.values, default.vdem.values)


def test_create_simple_vdem_temperature_outside_axis_range() -> None:
    # A cube entirely below the temperature axis contributes nothing; one entirely
    # above spans every bin through the boundary-cell gradient (prev = 100 K), so
    # each bin integrates the full log-T overlap of its column.
    inputs = _tiny_vdem_inputs()
    inputs["temperature"] = np.full((2, 3, 2), 10.0) * u.K  # logT = 1, below the lowest bin edge
    assert synthesis_utils.create_simple_vdem(**inputs).vdem.values.sum() == 0.0

    inputs["temperature"] = np.full((2, 3, 2), 1e9) * u.K  # logT = 9, above the highest bin edge
    result = synthesis_utils.create_simple_vdem(**inputs)
    # Six (x, y) columns whose boundary cell (cell_length 2, ne_nh 1) fully overlaps each bin.
    np.testing.assert_allclose(result.vdem.values.sum(axis=(1, 2, 3)), np.full(3, 1.2e-26), rtol=1e-12)


def test_create_simple_vdem_temperature_on_bin_edge_lands_in_lower_bin() -> None:
    # Bin centers [4.5, 5.0, 5.5] have edges at 4.75/5.25; an isothermal cube exactly on
    # the 4.5|5.0 edge is reached from below (boundary prev = 100 K), so the whole
    # log-T overlap sits in the lower bin and the upper bins stay empty.
    inputs = _tiny_vdem_inputs()
    inputs["temperature"] = np.full((2, 3, 2), 10**4.75) * u.K

    result = synthesis_utils.create_simple_vdem(**inputs)
    np.testing.assert_allclose(result.vdem.values.sum(axis=(1, 2, 3)), [1.2e-26, 0.0, 0.0], rtol=1e-12)


def test_create_simple_vdem_warns_on_non_finite_input(caplog: pytest.LogCaptureFixture) -> None:
    inputs = _tiny_vdem_inputs()
    inputs["ne_nh"] = inputs["ne_nh"].copy()
    inputs["ne_nh"][0, 0, 0] = np.nan

    result = synthesis_utils.create_simple_vdem(**inputs)
    assert "ne_nh contains non-finite" in caplog.text
    assert "vdem" in result  # warning, not an error: the VDEM is still produced


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("temperature", np.ones((1, 2)) * u.K, "3D array"),
        ("velocity", np.zeros((1, 1, 3)) * u.km / u.s, "must match temperature"),
        ("cell_length", np.array([1.0, 2.0, 3.0]) * u.cm, "line-of-sight"),
        ("x", np.array([1.0]) * u.cm, "non-LOS"),
    ],
)
def test_create_simple_vdem_validates_inputs(key, value, match) -> None:
    inputs = _tiny_vdem_inputs()
    inputs[key] = value

    with pytest.raises(ValueError, match=match):
        synthesis_utils.create_simple_vdem(**inputs)


def _rate_spectrum(units="ph / s"):
    flux = xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]]), dims=("line", "detector_x_pixel"), attrs={"units": units})
    return xr.Dataset({"flux": flux}, attrs={"history": ["vdem_synthesis()"]})


def test_apply_exposure_time_turns_a_rate_into_counts() -> None:
    # Half a minute of a rate per millisecond: the time is converted to the unit of the rate, and
    # recorded in seconds whatever unit it is given in.
    spectrum = _rate_spectrum("DN / ms")

    out = synthesis_utils.apply_exposure_time(spectrum, 0.5 * u.min)

    rate = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(out.flux.values, 30000 * rate)
    assert u.Unit(out.flux.attrs["units"]) == u.DN
    assert float(out.exposure_time) == 30.0
    assert out.exposure_time.attrs["units"] == "s"
    assert out.attrs["history"][-1].startswith("apply_exposure_time(")
    # The input is left as it was.
    np.testing.assert_array_equal(spectrum.flux.values, rate)
    assert spectrum.flux.attrs["units"] == "DN / ms"
    assert "exposure_time" not in spectrum.coords
    assert spectrum.attrs["history"] == ["vdem_synthesis()"]


def test_remove_exposure_time_undoes_apply_exposure_time() -> None:
    spectrum = _rate_spectrum()

    back = synthesis_utils.remove_exposure_time(synthesis_utils.apply_exposure_time(spectrum, 300 * u.ms))

    np.testing.assert_allclose(back.flux.values, spectrum.flux.values)
    assert u.Unit(back.flux.attrs["units"]) == u.ph / u.s
    assert "exposure_time" not in back.coords


@pytest.mark.parametrize(
    ("spectrum", "exposure_time", "match"),
    [
        (_rate_spectrum("ph"), 1 * u.s, "not a rate"),
        (
            _rate_spectrum().assign_coords(exposure_time=((), 2.0, {"units": "s"})),
            1 * u.s,
            "already has an exposure time",
        ),
        (_rate_spectrum(), [1, 2] * u.s, "must be a scalar"),
        (_rate_spectrum(), 0 * u.s, "must be finite and positive"),
    ],
)
def test_apply_exposure_time_rejects_what_it_cannot_integrate(spectrum, exposure_time, match) -> None:
    with pytest.raises(ValueError, match=match):
        synthesis_utils.apply_exposure_time(spectrum, exposure_time)


def test_remove_exposure_time_requires_a_recorded_time() -> None:
    with pytest.raises(ValueError, match="no exposure_time coordinate"):
        synthesis_utils.remove_exposure_time(_rate_spectrum("ph"))


def test_sum_lines_per_channel_adds_the_lines_of_each_channel(response) -> None:
    # Without its first line, the first and the last line of a channel differ in wavelength.
    response = response.isel(line=slice(1, None))
    response = response.assign_coords(position=("line", np.arange(response.sizes["line"])))

    out = synthesis_utils.sum_lines_per_channel(response)

    channels = list(dict.fromkeys(response.channel.values.tolist()))
    assert out.channel.values.tolist() == channels
    assert out.detector_response.dims == tuple(
        "channel" if dim == "line" else dim for dim in response.detector_response.dims
    )
    assert out.detector_response.attrs["units"] == response.detector_response.attrs["units"]
    for channel in channels:
        lines = np.flatnonzero(response.channel.values == channel)
        expected = response.detector_response.isel(line=lines).sum("line")
        np.testing.assert_allclose(out.detector_response.sel(channel=channel).values, expected.values)
        # The first line of the channel gives its coordinates.
        assert int(out.position.sel(channel=channel)) == lines[0]
    # A wavelength only where every line of the channel has the same one.
    np.testing.assert_equal(out.line_wavelength.values, [np.nan, 171.073, 284.163])
    assert out.line_wavelength.attrs["units"] == response.line_wavelength.attrs["units"]
    assert out.attrs["history"][-1].startswith("sum_lines_per_channel(")


def test_sum_lines_per_channel_keeps_the_order_in_which_channels_first_appear(response) -> None:
    reversed_lines = response.isel(line=slice(None, None, -1))

    out = synthesis_utils.sum_lines_per_channel(reversed_lines)

    assert out.channel.values.tolist() == list(dict.fromkeys(reversed_lines.channel.values.tolist()))
    xr.testing.assert_allclose(
        out.detector_response.sortby("channel"),
        synthesis_utils.sum_lines_per_channel(response).detector_response.sortby("channel"),
    )


def test_sum_lines_per_channel_sums_the_slits_on_request(response, raster) -> None:
    # Keep slit in the spectrum, so that the sum over slits is this function's.
    spectrum = vdem_synthesis(raster, response, sum_over=("logT", "doppler_velocity"))

    out = synthesis_utils.sum_lines_per_channel(spectrum, sum_slits=True)

    assert "slit" not in out.dims
    assert "detector_wavelength" not in out.coords  # it is per slit, so it goes with the slits
    expected = synthesis_utils.sum_lines_per_channel(spectrum).flux.sum("slit")
    xr.testing.assert_allclose(out.flux, expected)
    assert u.Unit(out.flux.attrs["units"]) == u.Unit(spectrum.flux.attrs["units"])
    # Data without slits has nothing more to sum.
    one_slit = spectrum.isel(slit=0, drop=True)
    xr.testing.assert_allclose(
        synthesis_utils.sum_lines_per_channel(one_slit, sum_slits=True).flux,
        synthesis_utils.sum_lines_per_channel(spectrum).flux.isel(slit=0, drop=True),
    )


def test_sum_lines_per_channel_keeps_a_nan_in_the_sum(response) -> None:
    holed = response.copy(deep=True)
    holed.detector_response[{"line": 0, "logT": 0}] = np.nan

    out = synthesis_utils.sum_lines_per_channel(holed)

    first_channel = out.detector_response.isel(channel=0)
    assert bool(np.isnan(first_channel.isel(logT=0)).all())
    assert bool(np.isfinite(first_channel.isel(logT=1)).all())


def test_sum_lines_per_channel_requires_a_channel_along_line(response) -> None:
    with pytest.raises(ValueError, match="channel coordinate along it"):
        synthesis_utils.sum_lines_per_channel(response.drop_vars("channel"))
    with pytest.raises(ValueError, match="channel coordinate along it"):
        synthesis_utils.sum_lines_per_channel(response.drop_vars("channel").assign_coords(channel=171))
