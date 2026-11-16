import pytest

from muse.data import fetch_example_data


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="not a known example data file"):
        fetch_example_data("nope.nc")


def test_local_tutorial_output_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    local_path = tmp_path / "muse_synthetic_spectra.nc"
    local_path.touch()

    assert fetch_example_data("muse_synthetic_spectra.nc") == local_path
