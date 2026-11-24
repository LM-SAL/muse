import importlib.util

import pytest

from muse.synthesis._backends import _resolve_backend, _resolve_device


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


def test_resolve_device_resolves_a_name_and_auto() -> None:
    torch = pytest.importorskip("torch")
    assert _resolve_device("meta") == torch.device("meta")
    # "auto" gives a device on every machine; which one depends on the hardware.
    assert isinstance(_resolve_device("auto"), torch.device)
