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
