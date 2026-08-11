import sys
from unittest.mock import Mock

import pytest

from muse.data import _REGISTRY, fetch_example_data


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="not a known example data file"):
        fetch_example_data("nope.nc")


def test_missing_pooch_raises_import_error(monkeypatch):
    # pooch is imported lazily inside the function; a None entry in sys.modules
    # makes that import raise ImportError.
    monkeypatch.setitem(sys.modules, "pooch", None)
    with pytest.raises(ImportError, match="pip install pooch"):
        fetch_example_data("muse_synthetic_spectra.nc")


def test_registry_entries_well_formed():
    for url, known_hash, subdir in _REGISTRY.values():
        assert url.startswith("https://")
        assert known_hash.startswith("sha256:")
        assert len(known_hash) == len("sha256:") + 64
        assert subdir


def test_incomplete_local_vdem_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    local = tmp_path / "muse_example_vdem.zarr"
    local.mkdir()
    cached = tmp_path / "cache" / "muse_example_vdem" / local.name
    pooch = Mock()
    pooch.os_cache.return_value = tmp_path / "cache"
    monkeypatch.setitem(sys.modules, "pooch", pooch)

    assert fetch_example_data(local.name) == cached
    pooch.retrieve.assert_called_once()


def test_completed_local_vdem_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    local = tmp_path / "muse_example_vdem.zarr"
    local.mkdir()
    (local / ".muse-complete").touch()
    pooch = Mock()
    pooch.os_cache.return_value = tmp_path / "cache"
    monkeypatch.setitem(sys.modules, "pooch", pooch)

    assert fetch_example_data(local.name) == local
    pooch.retrieve.assert_not_called()
