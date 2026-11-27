import zipfile

import pytest
import xarray as xr

from muse.data import fetch_example_data, open_example_dataset, save_example_dataset


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="not a known example data file"):
        fetch_example_data("nope.nc")


def test_local_tutorial_output_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    local_path = tmp_path / "muse_synthetic_spectra.nc"
    local_path.touch()

    assert fetch_example_data("muse_synthetic_spectra.nc") == local_path


def test_gallery_zip_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    name = "muse_synthetic_spectra.zarr.zip"
    original = xr.Dataset({"flux": ("line", [1.5, 2.5], {"units": "ph / s"})}, coords={"line": ["Fe IX", "Fe XV"]})
    updated = original.assign_attrs(history="second run").chunk({"line": 1})
    save_example_dataset(original, name)
    path = save_example_dataset(updated, name)

    assert path == tmp_path / name
    with zipfile.ZipFile(path) as archive:
        assert "zarr.json" in archive.namelist()
        assert len(archive.namelist()) == len(set(archive.namelist()))
    loaded = open_example_dataset(name)
    assert loaded.flux.chunks is None
    xr.testing.assert_identical(loaded, updated)
    with open_example_dataset(name, chunked=True) as reopened:
        assert reopened.flux.chunks == ((1, 1),)
        xr.testing.assert_identical(reopened.compute(), updated)
