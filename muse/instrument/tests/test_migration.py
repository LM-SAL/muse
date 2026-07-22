import pytest
import xarray as xr

from muse.instrument import migration
from muse.tests.helpers import fake_response_file

pytestmark = [
    pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning"),
    pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning"),
    pytest.mark.filterwarnings("ignore:numpy.ndarray size changed:RuntimeWarning"),
    # NumPy 2.5 warns inside the NetCDF backend while writing the legacy fixture.
    pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning"),
]


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_migrate_response_returns_and_verifies_schema(tmp_path, suffix) -> None:
    source_response = fake_response_file()
    source = tmp_path / "legacy.nc"
    destination = tmp_path / f"canonical{suffix}"
    source_response.to_netcdf(source)

    before, after = migration.migrate_response(source, destination)

    assert "SG_resp" in before
    assert "detector_response" in after
    with migration.response_utils._open_response_file(destination) as migrated:
        expected = migration.response_utils._canonicalize_response_names(source_response)
        xr.testing.assert_identical(migrated.load(), expected)
    with xr.open_dataset(source) as unchanged:
        assert "SG_resp" in unchanged


def test_migrate_response_leaves_no_destination_when_verification_fails(tmp_path, monkeypatch) -> None:
    source = tmp_path / "legacy.nc"
    destination = tmp_path / "canonical.zarr"
    fake_response_file().to_netcdf(source)

    def fail_verification(_expected, _actual) -> None:
        msg = "verification failed"
        raise ValueError(msg)

    monkeypatch.setattr(migration, "_verify_values", fail_verification)

    with pytest.raises(ValueError, match="verification failed"):
        migration.migrate_response(source, destination)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == [source]


def test_verify_values_rejects_metadata_and_coordinate_role_changes() -> None:
    expected = xr.Dataset(
        {"detector_response": ("line", [1.0])},
        coords={"line_wavelength": ("line", [171.073], {"units": "Angstrom"})},
        attrs={"normalization": 1e-27},
    )

    for actual in (expected.assign_attrs(normalization=1.0), expected.reset_coords("line_wavelength")):
        with pytest.raises(ValueError, match="does not match"):
            migration._verify_values(expected, actual)
