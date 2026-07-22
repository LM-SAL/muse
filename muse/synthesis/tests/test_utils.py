import numpy as np
import pytest
import xarray as xr

import astropy.units as u

import muse.synthesis.utils as synthesis_utils
from muse.synthesis.synthesis import vdem_synthesis
from muse.tests.helpers import assert_dataset_structure
from muse.transforms.transforms import reshape_x_to_slit_step


def _spectrum(response, vdem):
    # Keep slit so detector_wavelength survives, then add the Doppler coordinate moments require.
    reshaped = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    spectrum = vdem_synthesis(reshaped, response, sum_over=("logT", "vdop"))
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
                [1.0, 2.0, 10.0, 2.0, 1.0],
                {"units": "ph / s"},
            ),
        },
        coords={
            "detector_x_pixel": [0, 1, 2, 3, 4],
            "doppler_velocity": (
                ("detector_x_pixel",),
                [-200.0, -100.0, 0.0, 100.0, 200.0],
                {"units": "km/s"},
            ),
        },
    )


def test_wavelength_doppler_round_trip() -> None:
    resp = _tiny_response()
    with_vel = synthesis_utils.wavelength_to_doppler(resp)

    assert u.Unit(with_vel.doppler_velocity.attrs["units"]) == u.km / u.s
    assert "doppler_velocity" not in resp.coords  # input must not be mutated

    back = synthesis_utils.doppler_to_wavelength(with_vel)
    np.testing.assert_allclose(back.detector_wavelength.values, resp.detector_wavelength.values, rtol=1e-10)
    assert u.Unit(back.detector_wavelength.attrs["units"]) == u.AA


def test_wavelength_to_doppler_normalizes_units() -> None:
    # Same physical wavelengths expressed in nm must give the same Doppler shift.
    resp = _tiny_response()
    resp_nm = resp.assign_coords(detector_wavelength=resp.detector_wavelength / 10.0)
    resp_nm.detector_wavelength.attrs["units"] = "nm"

    angstrom = synthesis_utils.wavelength_to_doppler(resp)
    nanometer = synthesis_utils.wavelength_to_doppler(resp_nm)
    np.testing.assert_allclose(
        angstrom.doppler_velocity.values,
        nanometer.doppler_velocity.values,
        rtol=1e-9,
        atol=1e-6,
    )


def test_wavelength_to_doppler_requires_units() -> None:
    resp = _tiny_response()
    del resp.detector_wavelength.attrs["units"]
    with pytest.raises(ValueError, match=r"response\.detector_wavelength must define units"):
        synthesis_utils.wavelength_to_doppler(resp)


def test_calculate_moments_structure(response, vdem) -> None:
    spectrum = _spectrum(response, vdem)
    moments = synthesis_utils.calculate_moments(spectrum)

    assert_dataset_structure(
        moments,
        data_vars=("0th", "1st", "2nd"),
        coords=("y", "slit", "step", "line"),
        sizes={"y": 32, "slit": 35, "step": 11, "line": 7},
        finite_vars=("0th", "1st", "2nd"),
    )
    assert u.Unit(moments["1st"].attrs["units"]) == u.km / u.s
    assert u.Unit(moments["2nd"].attrs["units"]) == u.km / u.s


def test_calculate_moments_preserves_flux_units_with_vmax() -> None:
    spectrum = _tiny_moment_spectrum()

    moments = synthesis_utils.calculate_moments(spectrum, vmax=50)

    assert moments["0th"].attrs["units"] == "ph / s"


def test_calculate_moments_vmask_keeps_peak_window() -> None:
    spectrum = _tiny_moment_spectrum()

    moments = synthesis_utils.calculate_moments(spectrum, vmax=300, vmask=1)

    assert moments["0th"].item() == 10.0
    assert moments["1st"].item() == 0.0
    assert moments["2nd"].item() == 0.0


def test_calculate_moments_does_not_mutate_input(response, vdem) -> None:
    spectrum = _spectrum(response, vdem)
    before_attrs = dict(spectrum.attrs)
    before_dopp = spectrum.doppler_velocity.values.copy()
    synthesis_utils.calculate_moments(spectrum)
    # provenance/normalization must land on the result, not leak back into the caller
    assert dict(spectrum.attrs) == before_attrs
    np.testing.assert_array_equal(spectrum.doppler_velocity.values, before_dopp)


def test_calculate_moments_rejects_bad_moment_dim(response, vdem) -> None:
    spectrum = _spectrum(response, vdem)
    with pytest.raises(ValueError, match=r"'nope' not found in array dimensions"):
        synthesis_utils.calculate_moments(spectrum, moment_dim="nope")


def test_calculate_moments_requires_doppler_velocity(response, vdem) -> None:
    reshaped = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    spectrum = vdem_synthesis(reshaped, response, sum_over=("logT", "vdop"))  # no Doppler coordinate
    with pytest.raises(ValueError, match=r"run wavelength_to_doppler first"):
        synthesis_utils.calculate_moments(spectrum)


def _tiny_vdem_inputs():
    # x (axis 0) and y (axis 1) lengths differ so the test also pins the x-then-y axis order.
    return {
        "temperature": np.full((2, 3, 2), 1e5),
        "velocity": np.zeros((2, 3, 2)),
        "ne_nh": np.ones((2, 3, 2)),
        "cell_length": np.array([2.0, 3.0]),
        "x": np.array([10.0, 11.0]),
        "y": np.array([20.0, 21.0, 22.0]),
        "velocity_axis": np.array([-1.0, 0.0, 1.0]),
        "log_temperature_axis": np.array([4.5, 5.0, 5.5]),
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
    assert isinstance(result.vdem.data, np.ndarray)

    assert_dataset_structure(
        result,
        data_vars=("vdem",),
        coords=("logT", "vdop", "x", "y"),
        sizes={"logT": 3, "vdop": 3, "x": 2, "y": 3},
        finite_vars=("vdem",),
    )
    assert result.vdem.dims == ("logT", "vdop", "x", "y")
    np.testing.assert_array_equal(result.logT.values, [4.5, 5.0, 5.5])
    np.testing.assert_array_equal(result.vdop.values, [-1.0, -0.0, 1.0])
    np.testing.assert_array_equal(result.x.values, [10.0, 11.0])
    np.testing.assert_array_equal(result.y.values, [20.0, 21.0, 22.0])
    np.testing.assert_allclose(result.vdem.values, _expected_tiny_vdem(), rtol=1e-12)
    units = {name: result[name].attrs["units"] for name in ("vdem", "logT", "vdop", "x", "y")}
    assert units == {"vdem": "1e27 / cm5", "logT": "dex(K)", "vdop": "km/s", "x": "cm", "y": "cm"}
    for unit in units.values():
        u.Unit(unit)  # Every unit string must parse as an astropy unit
    assert result.attrs["HISTORY"][0].startswith("create_simple_vdem(")


def test_create_simple_vdem_velocity_bin_edges_are_half_open() -> None:
    # Velocity exactly on a bin edge must land in the upper bin ([edge_lo, edge_hi) convention).
    inputs = _tiny_vdem_inputs()
    # Bin centers [-1, 0, 1] with dv=1 give edges [-1.5, -0.5, 0.5, 1.5]; -0.5 is the -1|0 edge.
    inputs["velocity"] = np.full((2, 3, 2), -0.5)

    result = synthesis_utils.create_simple_vdem(**inputs)
    # vdop axis is -velocity_axis[::-1] = [-1, 0, 1]; emission must sit in vdop=0, not vdop=-1.
    emission_per_vdop = result.vdem.sum(dim=("logT", "x", "y")).values
    assert emission_per_vdop[1] > 0  # vdop == 0 bin
    assert emission_per_vdop[0] == 0  # vdop == -1 bin stays empty


def test_create_simple_vdem_internal_x_blocks_are_exact() -> None:
    inputs = _tiny_vdem_inputs()
    n_x = synthesis_utils._VDEM_X_BLOCK_SIZE + 1
    shape = (n_x, 3, 2)
    inputs["temperature"] = np.broadcast_to(inputs["temperature"][:1], shape).copy()
    inputs["velocity"] = np.broadcast_to(inputs["velocity"][:1], shape).copy()
    x_scale = np.arange(1.0, n_x + 1)
    inputs["ne_nh"] = np.broadcast_to(x_scale[:, np.newaxis, np.newaxis], shape).copy()
    inputs["x"] = np.arange(n_x)

    result = synthesis_utils.create_simple_vdem(**inputs)

    expected = _expected_tiny_vdem()[:, :, :1, :] * x_scale[np.newaxis, np.newaxis, :, np.newaxis]
    np.testing.assert_allclose(result.vdem.values, expected, rtol=1e-12)


@pytest.mark.parametrize("integration_axis", [0, 1])
def test_create_simple_vdem_integration_axis_matches_transposed_input(integration_axis: int) -> None:
    # Feeding cubes with the LOS axis moved elsewhere plus the matching integration_axis must
    # reproduce the default-axis result; ne_nh varies along every axis so a mix-up cannot cancel.
    inputs = _tiny_vdem_inputs()
    inputs["ne_nh"] = np.linspace(1.0, 2.0, 12).reshape(2, 3, 2)
    default = synthesis_utils.create_simple_vdem(**inputs)

    moved = dict(inputs)
    for cube in ("temperature", "velocity", "ne_nh"):
        moved[cube] = np.moveaxis(inputs[cube], 2, integration_axis)
    result = synthesis_utils.create_simple_vdem(**moved, integration_axis=integration_axis)
    np.testing.assert_array_equal(result.vdem.values, default.vdem.values)


def test_create_simple_vdem_dense_ne_nh_no_overflow() -> None:
    # Dense voxels: ne_nh = 1e39 exceeds float32 max (~3.4e38); the float64 pipeline must
    # carry it through the 1e27 units normalization and the per-voxel LOS product intact.
    inputs = _tiny_vdem_inputs()
    inputs["ne_nh"] = np.full((2, 3, 2), 1e39)
    inputs["cell_length"] = np.full(2, 1e7)

    result = synthesis_utils.create_simple_vdem(**inputs)
    # Same geometry as the tiny cube, rescaled: ne_nh 1 -> 1e39, contributing cell_length 2 -> 1e7.
    expected = _expected_tiny_vdem() * 1e39 * 1e7 / 2.0
    np.testing.assert_allclose(result.vdem.values, expected, rtol=1e-12)


@pytest.mark.parametrize("field", ["ne_nh", "temperature", "velocity"])
def test_create_simple_vdem_warns_on_non_finite_input(field: str, caplog: pytest.LogCaptureFixture) -> None:
    inputs = _tiny_vdem_inputs()
    inputs[field] = inputs[field].copy()
    inputs[field][0, 0, 0] = np.nan

    result = synthesis_utils.create_simple_vdem(**inputs)
    assert f"{field} contains non-finite" in caplog.text
    assert "vdem" in result  # warning, not an error: the VDEM is still produced


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("temperature", np.ones((1, 2)), "3D array"),
        ("velocity", np.zeros((1, 1, 3)), "must match temperature"),
        ("cell_length", np.array([1.0, 2.0, 3.0]), "line-of-sight"),
        ("x", np.array([1.0]), "non-LOS"),
    ],
)
def test_create_simple_vdem_validates_inputs(key, value, match) -> None:
    inputs = _tiny_vdem_inputs()
    inputs[key] = value

    with pytest.raises(ValueError, match=match):
        synthesis_utils.create_simple_vdem(**inputs)
