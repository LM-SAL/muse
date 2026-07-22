import importlib.util

import numpy as np
import pytest
import torch

from muse.synthesis._backends import _resolve_backend, numpy_to_torch, torch_to_numpy


@pytest.mark.parametrize(
    ("cuda_device", "backend", "expected"),
    [
        (None, "numpy", "numpy"),
        (None, "torch", "torch"),
    ],
)
def test_resolve_backend_decision(cuda_device, backend, expected) -> None:
    assert _resolve_backend(cuda_device, backend) == expected


@pytest.mark.parametrize(
    ("cuda_device", "backend", "match"),
    [
        (None, "cupy", "Unknown backend"),
        (0, "numpy", "numpy backend does not support cuda_device"),
        (-1, "torch", "is not valid"),
    ],
)
def test_resolve_backend_rejects(cuda_device, backend, match) -> None:
    with pytest.raises(ValueError, match=match):
        _resolve_backend(cuda_device, backend)


def test_resolve_backend_accelerator_not_installed_raises(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda *_: None)
    with pytest.raises(ValueError, match="Torch is not installed"):
        _resolve_backend(backend="torch")


def test_torch_numpy_round_trip() -> None:
    array = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(torch_to_numpy(numpy_to_torch(array)), array)


def test_numpy_to_torch_caps_precision_at_float32() -> None:
    assert numpy_to_torch(np.ones(3, dtype=np.float64)).dtype == torch.float32  # Downcast
    assert numpy_to_torch(np.ones(3, dtype=np.float32)).dtype == torch.float32  # Unchanged
    assert numpy_to_torch(np.ones(3, dtype=np.float16)).dtype == torch.float16  # Narrower kept


@pytest.mark.cuda
def test_torch_numpy_round_trip_cuda() -> None:
    array = np.arange(6.0).reshape(2, 3)
    tensor = numpy_to_torch(array, cuda_device=0)
    assert tensor.is_cuda
    np.testing.assert_array_equal(torch_to_numpy(tensor), array)
