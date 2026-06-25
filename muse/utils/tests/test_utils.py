import jax
import numpy as np
import pytest
import torch
import xarray as xr

import astropy.units as u

from muse.utils.utils import (
    add_history,
    jax_to_numpy,
    numpy_to_jax,
    numpy_to_torch,
    torch_to_numpy,
    update_attrs,
)


def _jax_gpu_devices():
    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


def _record(ds, gain=2.0, shift=None, *, flag=True, weights=None, label="muse"):
    """Record a call on ``ds`` so its keyword inputs are stored as attributes."""
    add_history(ds, locals(), _record)
    return ds


def test_add_history_records_call_and_version() -> None:
    ds = xr.Dataset({"a": ("x", [1, 2, 3])})

    def demo(ds, n=2):
        add_history(ds, locals(), demo)
        return ds

    demo(ds, n=5)
    assert ds.attrs["HISTORY"] == ["demo(ds=ds, n=5)"]
    assert "version" in ds.attrs


def test_add_history_stores_serializable_keyword_inputs() -> None:
    ds = _record(xr.Dataset({"a": ("x", [1, 2, 3])}), shift=3 * u.km / u.s, weights=np.ones(20))
    assert "ds" not in ds.attrs  # Keyword inputs with defaults are stored, the required positional `ds` is not
    assert ds.attrs["gain"] == 2.0
    assert ds.attrs["flag"] == 1  # bool -> int (netCDF4 attrs have no bool type)
    assert ds.attrs["shift"] == "3.0 km / s"  # Quantity -> "<value> <unit>"
    assert ds.attrs["label"] == "muse"
    assert "weights" not in ds.attrs  # Large array dropped, not serializable as an attr


@pytest.mark.parametrize("backend", ["netcdf4", "zarr"])
def test_add_history_stored_attrs_round_trip(tmp_path, backend) -> None:
    ds = _record(xr.Dataset({"a": ("x", [1, 2, 3])}), shift=3 * u.km / u.s)

    if backend == "netcdf4":
        pytest.importorskip("netCDF4")
        path = tmp_path / "out.nc"
        ds.to_netcdf(path, engine="netcdf4")
        loaded = xr.open_dataset(path, engine="netcdf4")
    else:
        pytest.importorskip("zarr")
        path = tmp_path / "out.zarr"
        ds.to_zarr(path, zarr_format=3, consolidated=False)
        loaded = xr.open_zarr(path, consolidated=False)

    # The coerced keyword inputs survive a real write/read cycle with both backends.
    assert loaded.attrs["gain"] == 2.0
    assert loaded.attrs["flag"] == 1
    assert loaded.attrs["shift"] == "3.0 km / s"
    assert loaded.attrs["label"] == "muse"


def test_add_history_warns_on_unserializable_keyword_input(caplog) -> None:
    ds = _record(xr.Dataset(), weights=np.ones(20))

    assert "weights" not in ds.attrs
    assert "weights" in caplog.text
    assert "serializable" in caplog.text


def test_add_history_records_bare_name_without_locals() -> None:
    def demo(ds, n=1): ...

    ds = xr.Dataset()
    add_history(ds, demo)
    add_history(ds, "calibrate")
    assert ds.attrs["HISTORY"] == ["demo", "calibrate"]


def test_add_history_appends_to_existing() -> None:
    ds = xr.Dataset({"a": ("x", [1])}, attrs={"HISTORY": ["first()"]})

    def demo(ds):
        add_history(ds, locals(), demo)

    demo(ds)
    assert ds.attrs["HISTORY"] == ["first()", "demo(ds=ds)"]


def test_add_history_records_function_name_without_locals() -> None:
    ds = xr.Dataset(
        {"a": ("x", [1])},
        attrs={"HISTORY": ["build"], "date created": "01-Jan-2026"},
    )
    add_history(ds, "calibrate")
    assert ds.attrs["HISTORY"] == ["build", "calibrate"]
    assert "date modified" in ds.attrs


def test_update_attrs_copies_source_attrs_and_applies_updates() -> None:
    source = xr.Dataset({"a": ("x", [1])}, attrs={"HISTORY": ["load"], "instrument": "MUSE"})
    ds = xr.Dataset({"b": ("x", [2])})

    update_attrs(ds, source, level=2)
    add_history(ds, "demo")
    assert ds.attrs["HISTORY"] == ["load", "demo"]
    assert ds.attrs["instrument"] == "MUSE"
    assert ds.attrs["level"] == 2


def test_update_attrs_merges_history_from_multiple_sources() -> None:
    ds = xr.Dataset(attrs={"HISTORY": ["start"]})
    source = xr.Dataset(attrs={"HISTORY": ["source"], "instrument": "MUSE"})
    update_attrs(ds, source)
    assert ds.attrs["HISTORY"] == ["start", "source"]
    assert ds.attrs["instrument"] == "MUSE"


def test_jax_numpy_round_trip() -> None:
    array = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(jax_to_numpy(numpy_to_jax(array)), array)


def test_numpy_to_jax_caps_precision_at_float32() -> None:
    assert numpy_to_jax(np.ones(3, dtype=np.float64)).dtype.name == "float32"  # Downcast
    assert numpy_to_jax(np.ones(3, dtype=np.float32)).dtype.name == "float32"  # Unchanged
    assert numpy_to_jax(np.ones(3, dtype=np.float16)).dtype.name == "float16"  # Narrower kept


@pytest.mark.cuda
def test_jax_numpy_round_trip_cuda() -> None:
    gpu_devices = _jax_gpu_devices()
    if not gpu_devices:
        pytest.skip("requires a CUDA GPU")

    array = np.arange(6.0).reshape(2, 3)
    jax_array = numpy_to_jax(array, cuda_device=0)
    assert jax_array.device == gpu_devices[0]
    np.testing.assert_array_equal(jax_to_numpy(jax_array), array)


def test_torch_numpy_round_trip() -> None:
    array = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(torch_to_numpy(numpy_to_torch(array)), array)


def test_numpy_to_torch_caps_precision_at_float32() -> None:
    assert numpy_to_torch(np.ones(3, dtype=np.float64)).dtype == torch.float32  # Downcast
    assert numpy_to_torch(np.ones(3, dtype=np.float32)).dtype == torch.float32  # Unchanged
    assert numpy_to_torch(np.ones(3, dtype=np.float16)).dtype == torch.float16  # Narrower kept


@pytest.mark.cuda
def test_torch_numpy_round_trip_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")

    array = np.arange(6.0).reshape(2, 3)
    tensor = numpy_to_torch(array, cuda_device=0)
    assert tensor.is_cuda
    np.testing.assert_array_equal(torch_to_numpy(tensor), array)
