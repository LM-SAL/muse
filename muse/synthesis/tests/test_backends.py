import importlib.util

import numpy as np
import pytest

from muse.synthesis._backends import (
    _available_memory_bytes,
    _resolve_backend,
    _resolve_device,
    numpy_to_torch,
    torch_to_numpy,
)


@pytest.mark.parametrize(
    ("device", "backend", "match"),
    [
        (None, "cupy", "Unknown backend"),
        ("cpu", "numpy", "numpy backend does not support device"),
    ],
)
def test_resolve_backend_rejects(device, backend, match) -> None:
    with pytest.raises(ValueError, match=match):
        _resolve_backend(device, backend)


def test_resolve_backend_accelerator_not_installed_raises(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda *_: None)
    with pytest.raises(ValueError, match="Torch is not installed"):
        _resolve_backend(backend="torch")


def test_resolve_device_defaults_to_the_cpu() -> None:
    torch = pytest.importorskip("torch")
    assert _resolve_device(None) == torch.device("cpu")
    assert _resolve_device("cpu") == torch.device("cpu")
    assert _resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_auto_picks_an_available_device() -> None:
    torch = pytest.importorskip("torch")
    device = _resolve_device("auto")
    assert device.type in ("cuda", "mps", "cpu")
    if device.type == "cuda":
        assert torch.cuda.is_available()


def test_available_memory_bytes_on_the_cpu_is_positive() -> None:
    torch = pytest.importorskip("torch")
    assert _available_memory_bytes(torch.device("cpu")) > 0


def test_torch_numpy_round_trip() -> None:
    pytest.importorskip("torch")
    array = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(torch_to_numpy(numpy_to_torch(array)), array)


def test_numpy_to_torch_caps_precision_at_float32() -> None:
    torch = pytest.importorskip("torch")
    assert numpy_to_torch(np.ones(3, dtype=np.float64)).dtype == torch.float32  # Downcast
    assert numpy_to_torch(np.ones(3, dtype=np.float32)).dtype == torch.float32  # Unchanged
    assert numpy_to_torch(np.ones(3, dtype=np.float16)).dtype == torch.float16  # Narrower kept


@pytest.mark.cuda
def test_torch_numpy_round_trip_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    array = np.arange(6.0).reshape(2, 3)
    tensor = numpy_to_torch(array, device="cuda:0")
    assert tensor.is_cuda
    np.testing.assert_array_equal(torch_to_numpy(tensor), array)
