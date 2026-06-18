import numpy as np
import xarray as xr

from muse.utils.utils import add_history, numpy_to_torch, torch_to_numpy, update_attrs


def test_add_history_records_call_and_version() -> None:
    ds = xr.Dataset({"a": ("x", [1, 2, 3])})

    def demo(ds, n=2):
        add_history(ds, locals(), demo)
        return ds

    demo(ds, n=5)
    assert ds.attrs["HISTORY"] == ["demo(ds=ds, n=5)"]
    assert "version" in ds.attrs


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


def test_torch_numpy_round_trip() -> None:
    array = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(torch_to_numpy(numpy_to_torch(array)), array)
