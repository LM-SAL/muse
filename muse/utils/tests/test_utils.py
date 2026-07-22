import importlib.util

import dask
import dask.array as da
import numpy as np
import pytest
import torch
import xarray as xr

import astropy.units as u

from muse.utils.utils import (
    _resolve_backend,
    add_history,
    jax_to_numpy,
    numpy_to_jax,
    numpy_to_torch,
    torch_to_numpy,
    update_attrs,
)


@pytest.mark.parametrize(
    ("cuda_device", "backend", "expected"),
    [
        (None, "numpy", "numpy"),
        (None, "jax", "jax"),
        (None, "torch", "torch"),
    ],
)
def test_resolve_backend_decision(cuda_device, backend, expected) -> None:
    assert _resolve_backend(cuda_device, backend) == expected


@pytest.mark.parametrize(
    ("cuda_device", "backend", "match"),
    [
        (None, "cupy", "Unknown backend"),
        (0, "numpy", "numpy backend does not support cuda_device"),
        (-1, "jax", "is not valid"),
    ],
)
def test_resolve_backend_rejects(cuda_device, backend, match) -> None:
    with pytest.raises(ValueError, match=match):
        _resolve_backend(cuda_device, backend)


def test_resolve_backend_accelerator_not_installed_raises(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda *_: None)
    with pytest.raises(ValueError, match="JAX is not installed"):
        _resolve_backend(backend="jax")
    with pytest.raises(ValueError, match="Torch is not installed"):
        _resolve_backend(backend="torch")


def _record(ds, gain=2.0, shift=None, *, flag=True, weights=None, label="muse"):
    """
    Record a call on ``ds`` so its keyword inputs are stored as attributes.
    """
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


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed:RuntimeWarning")
@pytest.mark.parametrize("backend", ["netcdf4", "zarr"])
def test_add_history_stored_attrs_round_trip(tmp_path, backend) -> None:
    ds = _record(xr.Dataset({"a": ("x", [1, 2, 3])}), shift=3 * u.km / u.s)
    if backend == "netcdf4":
        path = tmp_path / "out.nc"
        ds.to_netcdf(path, engine="netcdf4")
        loaded = xr.open_dataset(path, engine="netcdf4")
    else:
        path = tmp_path / "out.zarr"
        ds.to_zarr(path, zarr_format=3, consolidated=False)
        loaded = xr.open_zarr(path, consolidated=False)

    # The coerced keyword inputs must survive a
    # real write/read cycle with both backends.
    assert loaded.attrs["gain"] == 2.0
    assert loaded.attrs["flag"] == 1
    assert loaded.attrs["shift"] == "3.0 km / s"
    assert loaded.attrs["label"] == "muse"


def test_add_history_warns_on_unserializable_keyword_input(caplog) -> None:
    ds = _record(xr.Dataset(), weights=object())

    assert "weights" not in ds.attrs
    assert "weights" in caplog.text
    assert "serializable" in caplog.text


def test_add_history_silently_drops_array_keyword_inputs(caplog) -> None:
    ds = _record(xr.Dataset(), weights=np.ones(20))

    assert "weights" not in ds.attrs
    assert "weights" not in caplog.text


def test_add_history_does_not_compute_dask_arrays(caplog) -> None:
    def explode():
        msg = "dask array was computed"
        raise AssertionError(msg)

    def demo(ds, lazy_array=None, lazy_dataarray=None):
        add_history(ds, locals(), demo)

    lazy_array = da.from_delayed(dask.delayed(explode)(), shape=(), dtype=float)
    lazy_dataarray = xr.DataArray(lazy_array)
    ds = xr.Dataset()

    demo(ds, lazy_array=lazy_array, lazy_dataarray=lazy_dataarray)

    assert ds.attrs["HISTORY"] == ["demo(ds=ds, lazy_array=lazy_array, lazy_dataarray=lazy_dataarray)"]
    assert "lazy_array" not in ds.attrs
    assert "lazy_dataarray" not in ds.attrs
    assert "serializable" not in caplog.text


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


def test_update_attrs_copies_source_attrs_but_never_provenance() -> None:
    source = xr.Dataset(
        {"a": ("x", [1])},
        attrs={"HISTORY": ["load"], "instrument": "MUSE", "date created": "01-Jan-2026", "version": "0.1"},
    )
    ds = xr.Dataset({"b": ("x", [2])})

    update_attrs(ds, source, level=2)
    assert ds.attrs == {"instrument": "MUSE", "level": 2}


def test_update_attrs_does_not_mutate_source_attrs() -> None:
    source = xr.Dataset(attrs={"HISTORY": ["load"], "instrument": "MUSE"})
    ds = xr.Dataset()
    update_attrs(ds, source, instrument="other")
    assert source.attrs == {"HISTORY": ["load"], "instrument": "MUSE"}


def test_add_history_inherits_history_from_sources() -> None:
    ds = xr.Dataset(attrs={"HISTORY": ["start"]})
    raster = xr.Dataset(attrs={"HISTORY": ["make_raster()"]})
    response = xr.Dataset(attrs={"HISTORY": ["make_response()"]})
    add_history(ds, "demo", sources=(raster, response))
    assert ds.attrs["HISTORY"] == ["start", "make_raster()", "make_response()", "demo"]


def test_add_history_does_not_duplicate_inherited_history() -> None:
    source = xr.Dataset(attrs={"HISTORY": ["load"]})
    # A result derived through xarray operations may already carry its source history.
    ds = xr.Dataset(attrs={"HISTORY": ["load"]})
    add_history(ds, "demo", sources=(source,))
    assert ds.attrs["HISTORY"] == ["load", "demo"]


def test_add_history_nested_source_histories_merge_in_any_order() -> None:
    loaded = xr.Dataset(attrs={"HISTORY": ["load"]})
    transformed = xr.Dataset(attrs={"HISTORY": ["load", "transform"]})
    for sources in [(loaded, transformed), (transformed, loaded)]:
        ds = xr.Dataset()
        add_history(ds, "demo", sources=sources)
        assert ds.attrs["HISTORY"] == ["load", "transform", "demo"]


def test_add_history_initializes_history_from_sources() -> None:
    ds = xr.Dataset()
    raster = xr.Dataset(attrs={"HISTORY": ["make_raster()"]})
    response = xr.Dataset(attrs={"HISTORY": ["make_response()"]})
    add_history(ds, "demo", sources=(raster, response))
    assert ds.attrs["HISTORY"] == ["make_raster()", "make_response()", "demo"]


@pytest.mark.parametrize("key", ["HISTORY", "date created", "date modified", "version"])
def test_update_attrs_rejects_provenance_updates(key: str) -> None:
    ds = xr.Dataset()
    value = ["fake"] if key == "HISTORY" else "fake"
    with pytest.raises(ValueError, match="owned by add_history"):
        update_attrs(ds, **{key: value})


def test_jax_numpy_round_trip() -> None:
    array = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(jax_to_numpy(numpy_to_jax(array)), array)


def test_numpy_to_jax_caps_precision_at_float32() -> None:
    assert numpy_to_jax(np.ones(3, dtype=np.float64)).dtype.name == "float32"  # Downcast
    assert numpy_to_jax(np.ones(3, dtype=np.float32)).dtype.name == "float32"  # Unchanged
    assert numpy_to_jax(np.ones(3, dtype=np.float16)).dtype.name == "float16"  # Narrower kept


@pytest.mark.cuda
def test_jax_numpy_round_trip_cuda() -> None:
    import jax  # NOQA: PLC0415

    array = np.arange(6.0).reshape(2, 3)
    jax_array = numpy_to_jax(array, cuda_device=0)
    assert jax_array.device == jax.devices("gpu")[0]
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
    array = np.arange(6.0).reshape(2, 3)
    tensor = numpy_to_torch(array, cuda_device=0)
    assert tensor.is_cuda
    np.testing.assert_array_equal(torch_to_numpy(tensor), array)
