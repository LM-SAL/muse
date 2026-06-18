import numpy as np
import xarray as xr

from muse.utils.utils import add_history, numpy_to_torch, torch_to_numpy


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


def test_torch_numpy_round_trip() -> None:
    array = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(torch_to_numpy(numpy_to_torch(array)), array)
