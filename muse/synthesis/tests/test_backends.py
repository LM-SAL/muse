import importlib.util

import numpy as np
import pytest

from muse.synthesis._backends import (
    _available_memory_bytes,
    _resolve_backend,
    _resolve_device,
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


def test_resolve_backend_without_psutil_points_at_the_extra(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *_: None if name == "psutil" else object())
    with pytest.raises(ValueError, match=r"pip install muse\[torch\]"):
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


def test_torch_to_numpy_round_trip() -> None:
    torch = pytest.importorskip("torch")
    array = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(torch_to_numpy(torch.from_numpy(array)), array)


@pytest.mark.cuda
def test_torch_to_numpy_from_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    array = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(torch_to_numpy(torch.from_numpy(array).to("cuda:0")), array)
