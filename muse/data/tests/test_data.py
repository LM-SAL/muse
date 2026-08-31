import pytest

from muse.data import _REGISTRY, fetch_example_data


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="not a known example data file"):
        fetch_example_data("nope.nc")


def test_registry_entries_well_formed():
    for url, known_hash, subdir in _REGISTRY.values():
        assert url.startswith("https://")
        assert known_hash.startswith("sha256:")
        assert len(known_hash) == len("sha256:") + 64
        assert subdir


@pytest.mark.parametrize("name", ["muse_example_vdem.zarr", "muse_synthetic_spectra.nc"])
def test_local_tutorial_output_wins(name, tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    local_path = tmp_path / name
    local_path.touch()

    assert fetch_example_data(name) == local_path
