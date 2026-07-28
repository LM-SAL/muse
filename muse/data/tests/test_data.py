import pytest

from muse.data import _REGISTRY, fetch_example_data


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="not a known example data file"):
        fetch_example_data("nope.nc")


def test_registry_entries_well_formed():
    for url, known_hash, subdir in _REGISTRY.values():
        assert url.startswith("https://")
        if "dropbox.com" in url:
            assert url.endswith("dl=1")
        assert known_hash.startswith("sha256:")
        assert len(known_hash) == len("sha256:") + 64
        assert subdir


def test_local_vdem_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    local = tmp_path / "muse_example_vdem.zarr"
    local.mkdir()
    assert fetch_example_data("muse_example_vdem.zarr") == local
