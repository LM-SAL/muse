import jax
import numpy as np
import pytest

import astropy.units as u

import muse.synthesis.utils as synthesis_utils
from muse.tests.helpers import assert_dataset_structure
from muse.utils.utils import _use_jax_backend


def _gpu_devices():
    try:
        return _use_jax_backend(0)
    except ValueError:
        return False


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
    assert isinstance(result.vdem.data, jax.Array)

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
    np.testing.assert_allclose(result.vdem.values, _expected_tiny_vdem(), rtol=1e-6)
    assert result.vdem.attrs["units"] == "1e27 / cm5"
    assert result.x.attrs["units"] == "cm"
    assert result.y.attrs["units"] == "cm"
    assert result.vdop.attrs["units"] == "km/s"
    assert result.attrs["HISTORY"][0].startswith("create_simple_vdem(")


def test_create_simple_vdem_units_parse_with_astropy() -> None:
    result = synthesis_utils.create_simple_vdem(**_tiny_vdem_inputs())

    for var in (result.vdem, result.logT, result.vdop, result.x, result.y):
        u.Unit(var.attrs["units"])


def test_create_simple_vdem_defaults_to_cpu(monkeypatch) -> None:
    calls = []
    original = synthesis_utils.numpy_to_jax

    def spy_numpy_to_jax(numpy_array, cuda_device=None):
        calls.append(cuda_device)
        return original(numpy_array, cuda_device=cuda_device)

    monkeypatch.setattr(synthesis_utils, "numpy_to_jax", spy_numpy_to_jax)

    synthesis_utils.create_simple_vdem(**_tiny_vdem_inputs())

    assert calls
    assert set(calls) == {None}


def test_create_simple_vdem_numpy_backend(monkeypatch) -> None:
    monkeypatch.setattr(synthesis_utils, "_use_jax_backend", lambda *_args, **_kwargs: False)

    result = synthesis_utils.create_simple_vdem(**_tiny_vdem_inputs())

    assert isinstance(result.vdem.data, np.ndarray)
    np.testing.assert_allclose(result.vdem.values, _expected_tiny_vdem(), rtol=5e-6)


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


@pytest.mark.cuda
def test_create_simple_vdem_cuda_tiny_cube() -> None:
    if not _gpu_devices():
        pytest.skip("requires a CUDA GPU")

    result = synthesis_utils.create_simple_vdem(**_tiny_vdem_inputs(), cuda_device=0)

    np.testing.assert_allclose(result.vdem.values, _expected_tiny_vdem(), rtol=1e-6)
