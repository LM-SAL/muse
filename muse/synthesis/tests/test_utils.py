import numpy as np
import pytest
import xarray as xr

import astropy.units as u

import muse.synthesis.utils as synthesis_utils
from muse.synthesis.synthesis import vdem_synthesis
from muse.tests.helpers import assert_dataset_structure
from muse.transforms.transforms import reshape_x_to_slit_step


def _spectrum(response, vdem):
    # Keep slit so SG_wvl survives, then add the dopp_vel coordinate moments require.
    reshaped = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    spectrum = vdem_synthesis(reshaped, response, sum_over=("logT", "vdop"))
    return synthesis_utils.wavelength_to_doppler(spectrum)


def _tiny_response():
    line_wvl = np.array([108.355, 171.073])
    sg_wvl = np.array([[108.0, 108.355, 108.7], [170.6, 171.073, 171.5]])
    ds = xr.Dataset()
    ds = ds.assign_coords(line_wvl=("line", line_wvl), SG_wvl=(("line", "SG_xpixel"), sg_wvl))
    ds.line_wvl.attrs["units"] = "Angstrom"
    ds.SG_wvl.attrs["units"] = "Angstrom"
    return ds


def _tiny_moment_spectrum():
    return xr.Dataset(
        data_vars={
            "flux": (
                ("SG_xpixel",),
                [1.0, 2.0, 10.0, 2.0, 1.0],
                {"units": "ph / s"},
            ),
        },
        coords={
            "SG_xpixel": [0, 1, 2, 3, 4],
            "dopp_vel": (("SG_xpixel",), [-200.0, -100.0, 0.0, 100.0, 200.0], {"units": "km/s"}),
        },
    )


def test_wavelength_doppler_round_trip() -> None:
    resp = _tiny_response()
    with_vel = synthesis_utils.wavelength_to_doppler(resp)

    assert u.Unit(with_vel.dopp_vel.attrs["units"]) == u.km / u.s
    assert "dopp_vel" not in resp.coords  # input must not be mutated

    back = synthesis_utils.doppler_to_wavelength(with_vel)
    np.testing.assert_allclose(back.SG_wvl.values, resp.SG_wvl.values, rtol=1e-10)
    assert u.Unit(back.SG_wvl.attrs["units"]) == u.AA


def test_wavelength_to_doppler_normalizes_units() -> None:
    # Same physical wavelengths expressed in nm must give the same Doppler shift.
    resp = _tiny_response()
    resp_nm = resp.assign_coords(SG_wvl=resp.SG_wvl / 10.0)
    resp_nm.SG_wvl.attrs["units"] = "nm"

    angstrom = synthesis_utils.wavelength_to_doppler(resp)
    nanometer = synthesis_utils.wavelength_to_doppler(resp_nm)
    np.testing.assert_allclose(angstrom.dopp_vel.values, nanometer.dopp_vel.values, rtol=1e-9, atol=1e-6)


def test_wavelength_to_doppler_requires_units() -> None:
    resp = _tiny_response()
    del resp.SG_wvl.attrs["units"]
    with pytest.raises(ValueError, match=r"response\.SG_wvl must define units"):
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


def test_calculate_moments_requires_vmask_with_vdop_reference() -> None:
    spectrum = _tiny_moment_spectrum()

    with pytest.raises(ValueError, match="vmask must be provided"):
        synthesis_utils.calculate_moments(spectrum, vmax=300, vdop_reference=xr.Dataset())


def test_calculate_moments_does_not_mutate_input(response, vdem) -> None:
    spectrum = _spectrum(response, vdem)
    before_attrs = dict(spectrum.attrs)
    before_dopp = spectrum.dopp_vel.values.copy()
    synthesis_utils.calculate_moments(spectrum)
    # provenance/normalization must land on the result, not leak back into the caller
    assert dict(spectrum.attrs) == before_attrs
    np.testing.assert_array_equal(spectrum.dopp_vel.values, before_dopp)


def test_calculate_moments_rejects_bad_moment_dim(response, vdem) -> None:
    spectrum = _spectrum(response, vdem)
    with pytest.raises(ValueError, match=r"'nope' not found in array dimensions"):
        synthesis_utils.calculate_moments(spectrum, moment_dim="nope")


def test_calculate_moments_requires_dopp_vel(response, vdem) -> None:
    reshaped = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    spectrum = vdem_synthesis(reshaped, response, sum_over=("logT", "vdop"))  # no dopp_vel coordinate
    with pytest.raises(ValueError, match=r"run wavelength_to_doppler first"):
        synthesis_utils.calculate_moments(spectrum)


def _tiny_vdem_inputs():
    # x (axis 0) and y (axis 1) lengths differ so the test also pins the x-then-y axis order.
    return {
        "temperature": np.full((2, 3, 2), 1e5, dtype=np.float32),
        "velocity": np.zeros((2, 3, 2), dtype=np.float32),
        "ne_nh": np.ones((2, 3, 2), dtype=np.float32),
        "cell_length": np.array([2.0, 3.0], dtype=np.float32),
        "x": np.array([10.0, 11.0], dtype=np.float32),
        "y": np.array([20.0, 21.0, 22.0], dtype=np.float32),
        "velocity_axis": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "log_temperature_axis": np.array([4.5, 5.0, 5.5], dtype=np.float32),
    }


def _expected_tiny_vdem():
    # Uniform column: only the boundary cell (prev=100) has a temperature gradient, so it
    # spreads over the low-T bins; every (x, y) column is identical.
    expected = np.zeros((3, 3, 2, 3), dtype=np.float32)
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
    np.testing.assert_allclose(result.vdem.values, _expected_tiny_vdem(), rtol=5e-6)  # float32 rounding
    assert result.vdem.attrs["units"] == "1e27 / cm5"
    assert result.x.attrs["units"] == "cm"
    assert result.y.attrs["units"] == "cm"
    assert result.vdop.attrs["units"] == "km/s"
    for var in (result.vdem, result.logT, result.vdop, result.x, result.y):
        u.Unit(var.attrs["units"])  # every unit string must parse as an astropy unit
    assert result.attrs["HISTORY"][0].startswith("create_simple_vdem(")


def test_create_simple_vdem_velocity_bin_edges_are_half_open() -> None:
    # Velocity exactly on a bin edge must land in the upper bin ([edge_lo, edge_hi) convention).
    inputs = _tiny_vdem_inputs()
    # Bin centers [-1, 0, 1] with dv=1 give edges [-1.5, -0.5, 0.5, 1.5]; -0.5 is the -1|0 edge.
    inputs["velocity"] = np.full((2, 3, 2), -0.5, dtype=np.float32)

    result = synthesis_utils.create_simple_vdem(**inputs)
    # vdop axis is -velocity_axis[::-1] = [-1, 0, 1]; emission must sit in vdop=0, not vdop=-1.
    emission_per_vdop = result.vdem.sum(dim=("logT", "x", "y")).values
    assert emission_per_vdop[1] > 0  # vdop == 0 bin
    assert emission_per_vdop[0] == 0  # vdop == -1 bin stays empty


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("temperature", np.ones((1, 2), dtype=np.float32), "3D array"),
        ("velocity", np.zeros((1, 1, 3), dtype=np.float32), "must match temperature"),
        ("cell_length", np.array([1.0, 2.0, 3.0], dtype=np.float32), "line-of-sight"),
        ("x", np.array([1.0], dtype=np.float32), "non-LOS"),
    ],
)
def test_create_simple_vdem_validates_inputs(key, value, match) -> None:
    inputs = _tiny_vdem_inputs()
    inputs[key] = value

    with pytest.raises(ValueError, match=match):
        synthesis_utils.create_simple_vdem(**inputs)
