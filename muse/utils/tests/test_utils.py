import re

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.utils.utils import add_history, coord_as_unit, update_attrs
from muse.version import version

STAMP = re.compile(r" \[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) muse (\S+)\]$")


def _calls(ds):
    """
    The history entries of ``ds`` without their stamps.
    """
    return [STAMP.sub("", entry) for entry in ds.attrs["history"]]


@pytest.mark.parametrize(
    ("source_unit", "target_unit", "values", "expected"),
    [
        ("m/s", u.km / u.s, [1000.0, 2000.0], [1.0, 2.0]),
        ("dex(mK)", u.dex(u.K), [6.0, 7.0], [3.0, 4.0]),
    ],
)
def test_coord_as_unit_converts_coordinate_values(source_unit, target_unit, values, expected) -> None:
    ds = xr.Dataset(coords={"axis": ("axis", values, {"units": source_unit})})

    converted = coord_as_unit(ds, "axis", target_unit, "axis")

    np.testing.assert_allclose(converted, expected)
    assert converted.attrs["units"] == str(target_unit)


def _record(ds, gain=2.0, shift=None, *, flag=True, weights=None, label="muse"):
    """
    Record a call on ``ds`` so its keyword inputs are stored as attributes.
    """
    add_history(ds, locals(), _record)
    return ds


def test_add_history_records_the_call_with_its_time_and_version() -> None:
    ds = xr.Dataset({"a": ("x", [1, 2, 3])})

    def demo(ds, n=2):
        add_history(ds, locals(), demo)
        return ds

    demo(ds, n=5)
    (entry,) = ds.attrs["history"]
    assert STAMP.search(entry).groups() == (ds.attrs["date_created"], version)
    assert _calls(ds) == ["demo(ds=<Dataset of a>, n=5)"]
    assert ds.attrs["muse_version"] == version


def test_a_history_entry_keeps_units_and_summarises_arrays() -> None:
    grid = xr.DataArray(np.arange(5.0), dims="x", attrs={"units": "K"})
    quantity_grid = xr.DataArray([3e14, 1e15] * u.K / u.cm**3, dims="pressure")

    ds = _record(xr.Dataset(), gain=quantity_grid, shift=[1.0, 2.0] * u.km / u.s, weights=grid)

    assert _calls(ds) == [
        (
            "_record(ds=<Dataset>, gain=<2 values, 300000000000000.0 to 1000000000000000.0 K / cm3>, "
            "shift=<2 values, 1.0 to 2.0 km / s>, flag=True, weights=<5 values, 0.0 to 4.0 K>, label=muse)"
        )
    ]


def test_options_of_how_the_code_ran_stay_out_of_the_attributes() -> None:
    def demo(ds, gain=1.0, workers=1):
        add_history(ds, locals(), demo)

    ds = xr.Dataset()
    demo(ds, gain=2.0, workers=4)

    assert ds.attrs["gain"] == 2.0
    assert "workers" not in ds.attrs
    assert _calls(ds) == ["demo(ds=<Dataset>, gain=2.0, workers=4)"]


def test_a_reader_adds_its_entry_but_keeps_the_provenance_of_the_file() -> None:
    made = {"history": ["made()"], "date_created": "2026-01-01T00:00:00Z", "muse_version": "0.1"}
    ds = xr.Dataset(attrs=made)

    add_history(ds, {}, "read", reading=True)

    assert _calls(ds) == ["made()", "read()"]
    assert {key: ds.attrs[key] for key in ("date_created", "muse_version")} == {
        "date_created": "2026-01-01T00:00:00Z",
        "muse_version": "0.1",
    }
    assert "date_modified" not in ds.attrs


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

    lazy_array = da.from_delayed(dask.delayed(explode)(), shape=(5,), dtype=float)
    lazy_dataarray = xr.DataArray(lazy_array)
    ds = xr.Dataset()

    demo(ds, lazy_array=lazy_array, lazy_dataarray=lazy_dataarray)

    assert _calls(ds) == ["demo(ds=<Dataset>, lazy_array=<array of shape (5,)>, lazy_dataarray=<array of shape (5,)>)"]
    assert "lazy_array" not in ds.attrs
    assert "lazy_dataarray" not in ds.attrs
    assert "serializable" not in caplog.text


def test_add_history_records_bare_name_without_inputs() -> None:
    def demo(ds, n=1): ...

    ds = xr.Dataset()
    add_history(ds, {}, demo)
    add_history(ds, {}, "calibrate")
    assert _calls(ds) == ["demo()", "calibrate()"]
    # An empty local_vars records no inputs, so the defaulted parameter never reaches attrs.
    assert "n" not in ds.attrs


def test_add_history_ignores_inputs_for_a_string_label() -> None:
    # A string label has no signature to inspect, so inputs are skipped even when passed.
    ds = xr.Dataset()

    add_history(ds, {"x": 1, "gain": 10.0}, "calibrate")

    assert _calls(ds) == ["calibrate()"]
    assert not {"x", "gain"} & set(ds.attrs)


def test_add_history_appends_to_existing() -> None:
    made = {"history": ["first()"], "date_created": "2026-01-01T00:00:00Z", "muse_version": "0.1"}
    ds = xr.Dataset({"a": ("x", [1])}, attrs=made)

    def demo(ds):
        add_history(ds, locals(), demo)

    demo(ds)
    assert _calls(ds) == ["first()", "demo(ds=<Dataset of a>)"]
    # A dataset that has its date of creation gets a date of modification; the date of
    # creation and the version that made it stay.
    assert STAMP.search(ds.attrs["history"][-1]).group(1) == ds.attrs["date_modified"]
    assert (ds.attrs["date_created"], ds.attrs["muse_version"]) == ("2026-01-01T00:00:00Z", "0.1")


def test_update_attrs_copies_source_attrs_but_never_provenance() -> None:
    source = xr.Dataset(
        {"a": ("x", [1])},
        attrs={
            "history": ["load"],
            "instrument": "MUSE",
            "date_created": "2026-01-01T00:00:00Z",
            "muse_version": "0.1",
        },
    )
    ds = xr.Dataset({"b": ("x", [2])})

    update_attrs(ds, source, level=2)
    assert ds.attrs == {"instrument": "MUSE", "level": 2}


def test_a_dataset_with_the_attributes_of_an_old_file_keeps_their_lineage() -> None:
    # as a result made by an xarray operation on a file of muse 0.3 has them
    ds = xr.Dataset(attrs={"HISTORY": ["load"], "date created": "01-Mar-2026", "version": "0.3.0", "workers": 4})

    add_history(ds, {}, "demo")

    assert _calls(ds) == ["load", "demo()"]
    assert (ds.attrs["date_created"], ds.attrs["muse_version"]) == ("2026-03-01", "0.3.0")
    assert not {"HISTORY", "date created", "version", "workers"} & set(ds.attrs)


def test_a_version_attribute_of_another_tool_is_kept() -> None:
    ds = xr.Dataset()

    update_attrs(ds, xr.Dataset(attrs={"version": "table v3"}))

    assert ds.attrs == {"version": "table v3"}


def test_the_two_outputs_of_one_call_are_one_step_even_stamped_apart() -> None:
    first = xr.Dataset(attrs={"history": ["load", "align() [2026-01-01T00:00:00Z muse 0.4.0]"]})
    second = xr.Dataset(attrs={"history": ["load", "align() [2026-01-01T00:00:01Z muse 0.4.0]"]})
    ds = xr.Dataset()

    add_history(ds, {}, "synthesis", sources=(first, second))

    assert _calls(ds) == ["load", "align()", "synthesis()"]


def test_a_source_read_without_the_library_passes_on_its_old_names_as_current_ones() -> None:
    old = {"Chianti": "10.0", "chianti_version": "11.0.2", "fiasco": "0.4", "HISTORY": ["load"], "version": "0.3"}
    source = xr.Dataset(attrs=old)
    ds = xr.Dataset()

    update_attrs(ds, source)
    add_history(ds, {}, "demo", sources=(source,))

    # the current name wins; the old provenance is inherited as history, not copied as attributes
    assert {key: ds.attrs[key] for key in ("chianti_version", "fiasco_version")} == {
        "chianti_version": "11.0.2",
        "fiasco_version": "0.4",
    }
    assert _calls(ds) == ["load", "demo()"]
    assert not {"HISTORY", "version", "Chianti", "fiasco"} & set(ds.attrs)


def test_update_attrs_does_not_mutate_source_attrs() -> None:
    source = xr.Dataset(attrs={"history": ["load"], "instrument": "MUSE"})
    ds = xr.Dataset()
    update_attrs(ds, source, instrument="other")
    assert source.attrs == {"history": ["load"], "instrument": "MUSE"}


def test_add_history_inherits_history_from_sources() -> None:
    ds = xr.Dataset(attrs={"history": ["start"]})
    raster = xr.Dataset(attrs={"history": ["make_raster()"]})
    response = xr.Dataset(attrs={"history": ["make_response()"]})
    add_history(ds, {}, "demo", sources=(raster, response))
    assert _calls(ds) == ["start", "make_raster()", "make_response()", "demo()"]


def test_add_history_does_not_duplicate_inherited_history() -> None:
    source = xr.Dataset(attrs={"history": ["load"]})
    # A result derived through xarray operations may already carry its source history.
    ds = xr.Dataset(attrs={"history": ["load"]})
    add_history(ds, {}, "demo", sources=(source,))
    assert _calls(ds) == ["load", "demo()"]


def test_add_history_nested_source_histories_merge_in_any_order() -> None:
    loaded = xr.Dataset(attrs={"history": ["load"]})
    transformed = xr.Dataset(attrs={"history": ["load", "transform"]})
    for sources in [(loaded, transformed), (transformed, loaded)]:
        ds = xr.Dataset()
        add_history(ds, {}, "demo", sources=sources)
        assert _calls(ds) == ["load", "transform", "demo()"]


@pytest.mark.parametrize("key", ["history", "date_created", "date_modified", "muse_version"])
def test_update_attrs_rejects_provenance_updates(key: str) -> None:
    ds = xr.Dataset()
    value = ["fake"] if key == "history" else "fake"
    with pytest.raises(ValueError, match="owned by add_history"):
        update_attrs(ds, **{key: value})
