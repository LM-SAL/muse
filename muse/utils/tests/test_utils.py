import re

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from attrs import evolve

import astropy.units as u

from muse.utils import utils as utils_module
from muse.utils.utils import _instrument_defaults_label, add_history, coord_as_unit, update_attrs
from muse.variables import DEFAULTS_MUSE
from muse.version import version

STAMP = re.compile(r" \[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) muse (\S+)\]$")


def _calls(ds):
    """
    The history entries of ``ds`` without their stamps.
    """
    return [STAMP.sub("", entry) for entry in ds.attrs["history"]]


def test_coord_as_unit_converts_coordinate_values() -> None:
    # a logarithmic unit, which a conversion by a factor gets wrong
    ds = xr.Dataset(coords={"axis": ("axis", [6.0, 7.0], {"units": "dex(mK)"})})

    converted = coord_as_unit(ds, "axis", u.dex(u.K), "axis")

    np.testing.assert_allclose(converted, [3.0, 4.0])
    assert converted.attrs["units"] == str(u.dex(u.K))


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

    ds = _record(xr.Dataset(), gain=quantity_grid, shift=[1.0, 2.0] * u.km / u.s, weights=grid, label=np.array([3.0]))

    assert _calls(ds) == [
        (
            "_record(ds=<Dataset>, gain=<2 values, 300000000000000.0 to 1000000000000000.0 K / cm3>, "
            "shift=<2 values, 1.0 to 2.0 km / s>, flag=True, weights=<5 values, 0.0 to 4.0 K>, label=[3.0])"
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
    ds = _record(xr.Dataset({"a": ("x", [1, 2, 3])}), weights=np.ones(20))
    assert "ds" not in ds.attrs  # Keyword inputs with defaults are stored, the required positional `ds` is not
    assert ds.attrs["gain"] == 2.0
    assert ds.attrs["label"] == "muse"
    assert "weights" not in ds.attrs  # Large array dropped, not serializable as an attr


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed:RuntimeWarning")
def test_add_history_stored_attrs_round_trip(tmp_path) -> None:
    ds = _record(xr.Dataset({"a": ("x", [1, 2, 3])}), shift=3 * u.km / u.s)
    ds.to_netcdf(tmp_path / "out.nc", engine="netcdf4")
    loaded = xr.open_dataset(tmp_path / "out.nc", engine="netcdf4")

    # The coerced keyword inputs must survive a real write/read cycle.
    assert loaded.attrs["gain"] == 2.0
    assert loaded.attrs["flag"] == 1
    assert loaded.attrs["shift"] == "3.0 km / s"
    assert loaded.attrs["label"] == "muse"


def test_add_history_keeps_an_unserializable_keyword_input_in_its_entry_alone(caplog) -> None:
    ds = _record(xr.Dataset(), weights=object())

    assert "weights" not in ds.attrs
    assert "weights=" in _calls(ds)[0]
    assert "weights" not in caplog.text  # the entry holds it, so nothing is lost and nothing is logged


def test_add_history_does_not_compute_dask_arrays() -> None:
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


def test_add_history_records_bare_name_without_inputs() -> None:
    def demo(ds, n=1): ...

    ds = xr.Dataset()
    add_history(ds, {}, demo)
    # A string label has no signature to inspect, so its inputs are skipped even when passed.
    add_history(ds, {"x": 1, "gain": 10.0}, "calibrate")

    assert _calls(ds) == ["demo()", "calibrate()"]
    assert not {"n", "x", "gain"} & set(ds.attrs)


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
    assert "level" not in source.attrs


def test_a_history_entry_reads_no_array_from_its_file(tmp_path) -> None:
    xr.Dataset({"grid": ("x", np.arange(5.0), {"units": "K"})}).to_netcdf(tmp_path / "grid.nc")

    with xr.open_dataset(tmp_path / "grid.nc") as opened:
        ds = _record(xr.Dataset(), weights=opened.grid)

    assert "weights=<array of shape (5,) K>" in ds.attrs["history"][0]


def test_the_two_outputs_of_one_call_are_one_step_even_stamped_apart() -> None:
    first = xr.Dataset(attrs={"history": ["load", "align() [2026-01-01T00:00:00Z muse 0.4.0]"]})
    second = xr.Dataset(attrs={"history": ["load", "align() [2026-01-01T00:00:01Z muse 0.4.0]"]})
    ds = xr.Dataset()

    add_history(ds, {}, "synthesis", sources=(first, second))

    assert _calls(ds) == ["load", "align()", "synthesis()"]


def test_the_label_of_instrument_defaults_follows_every_value_and_the_values_alone(monkeypatch) -> None:
    custom = evolve(DEFAULTS_MUSE, pixel_width_SG=0.8 * u.arcsec)
    late = ("gain_CI", "pair_energy_SG", "wavelength_start_SG", "channel_spectral_order_SG")
    changed = [evolve(custom, **{name: 2 * getattr(custom, name)}) for name in late]

    assert len({_instrument_defaults_label(defaults) for defaults in [custom, *changed]}) == 1 + len(late)
    # an integer and its float are one value
    as_float = evolve(custom, channel_spectral_order_SG=custom.channel_spectral_order_SG.astype(float))
    assert _instrument_defaults_label(as_float) == _instrument_defaults_label(custom)
    # the MUSE values of another muse are told from today's
    today = _instrument_defaults_label(DEFAULTS_MUSE)
    older = evolve(DEFAULTS_MUSE, number_of_slits_SG=5)
    monkeypatch.setattr(utils_module, "DEFAULTS_MUSE", older)
    assert _instrument_defaults_label(older).split()[0] == "DEFAULTS_MUSE"
    assert _instrument_defaults_label(older) != today


def test_add_history_inherits_history_from_sources() -> None:
    ds = xr.Dataset(attrs={"history": ["start"]})
    raster = xr.Dataset(attrs={"history": ["make_raster()"]})
    response = xr.Dataset(attrs={"history": ["make_response()"]})
    add_history(ds, {}, "demo", sources=(raster, response))
    assert _calls(ds) == ["start", "make_raster()", "make_response()", "demo()"]


def test_add_history_nested_source_histories_merge_in_any_order() -> None:
    loaded = xr.Dataset(attrs={"history": ["load"]})
    transformed = xr.Dataset(attrs={"history": ["load", "transform"]})
    for sources in [(loaded, transformed), (transformed, loaded)]:
        ds = xr.Dataset()
        add_history(ds, {}, "demo", sources=sources)
        assert _calls(ds) == ["load", "transform", "demo()"]


def test_update_attrs_rejects_provenance_updates() -> None:
    with pytest.raises(ValueError, match="owned by add_history"):
        update_attrs(xr.Dataset(), history=["fake"])
