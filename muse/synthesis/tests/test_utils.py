import numpy as np
import pytest

import astropy.units as u

import muse.synthesis.utils as synthesis_utils
from muse.tests.helpers import assert_dataset_structure


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
    assert result.attrs["HISTORY"][0].startswith("create_simple_vdem(")


def test_create_simple_vdem_units_parse_with_astropy() -> None:
    result = synthesis_utils.create_simple_vdem(**_tiny_vdem_inputs())

    for var in (result.vdem, result.logT, result.vdop, result.x, result.y):
        u.Unit(var.attrs["units"])


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
