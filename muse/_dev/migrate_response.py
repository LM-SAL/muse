import sys
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Sequence

import dask.array as da
import numpy as np
import xarray as xr

from muse.instrument import utils as response_utils

__all__ = ["main", "migrate_response"]


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
    if dict(expected.sizes) != dict(actual.sizes) or set(expected.variables) != set(actual.variables):
        msg = "Migrated response schema does not match the canonical source schema"
        raise ValueError(msg)
    for name in expected.variables:
        left = expected[name]
        right = actual[name]
        if left.dims != right.dims or left.shape != right.shape:
            msg = f"Migrated variable schema does not match for {name!r}"
            raise ValueError(msg)
        left_data = da.asarray(left.data)
        right_data = da.asarray(right.data)
        if np.issubdtype(left.dtype, np.inexact) and np.issubdtype(right.dtype, np.inexact):
            equal = da.allclose(left_data, right_data, rtol=0, atol=0, equal_nan=True)
        else:
            equal = da.all(left_data == right_data)
        if not bool(equal.compute()):
            msg = f"Migrated values differ for {name!r}"
            raise ValueError(msg)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the response migration command.
    """
    parser = argparse.ArgumentParser(description="Migrate a MUSE response to the canonical schema.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)

    before, after = migrate_response(args.source, args.destination)
    sys.stdout.write(f"Before migration\n{before}\n\nAfter migration\n{after}\n\nNumerical verification passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
