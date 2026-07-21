from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr

from muse.instrument import utils as response_utils

__all__ = ["migrate_response"]


def migrate_response(source: str | Path, destination: str | Path) -> tuple[str, str]:
    """
    Migrate a legacy response to canonical names at a new destination.

    The migrated response is staged beside the destination and becomes visible
    only after every stored value has been verified against the source.

    Parameters
    ----------
    source : `str` or `pathlib.Path`
        Existing response file or Zarr store.
    destination : `str` or `pathlib.Path`
        New ``.zarr`` or NetCDF destination.

    Returns
    -------
    before : `str`
        Human-readable source schema.
    after : `str`
        Human-readable migrated schema.
    """
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        msg = f"Response does not exist: {source}"
        raise ValueError(msg)
    if destination.exists():
        msg = f"Refusing to overwrite existing response: {destination}"
        raise ValueError(msg)
    if not destination.parent.is_dir():
        msg = f"Destination directory does not exist: {destination.parent}"
        raise ValueError(msg)

    with response_utils._open_response_file(source, chunked=True) as opened:
        before = _schema(opened)
        canonical = response_utils._canonicalize_response_names(opened)
        with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary_directory:
            staged = Path(temporary_directory) / destination.name
            response_utils.save_response(canonical, staged)
            with response_utils._open_response_file(staged, chunked=True) as written:
                _verify_values(canonical, written)
                after = _schema(written)
            if destination.exists():
                msg = f"Refusing to overwrite existing response: {destination}"
                raise ValueError(msg)
            staged.replace(destination)
    return before, after


def _schema(response: xr.Dataset) -> str:
    dimensions = ", ".join(f"{name}={size}" for name, size in response.sizes.items())
    data_variables = ", ".join(f"{name}{response[name].dims}" for name in response.data_vars)
    coordinates = ", ".join(f"{name}{response[name].dims}" for name in response.coords)
    return f"dimensions: {dimensions}\ndata variables: {data_variables}\ncoordinates: {coordinates}"


def _verify_values(expected: xr.Dataset, actual: xr.Dataset) -> None:
    try:
        xr.testing.assert_identical(expected, actual)
    except AssertionError as exc:
        msg = "Migrated response does not match the canonical source"
        raise ValueError(msg) from exc
