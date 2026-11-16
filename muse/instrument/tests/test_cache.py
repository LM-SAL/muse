"""
Tests of the cached line lists and continua on synthetic backends: no test reads
CHIANTI.
"""

import os
import shutil
import contextlib

import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument.atomic import cache
from muse.instrument.atomic.backends import _validate_plasma_inputs
from muse.instrument.atomic.cache import load_or_create_continuum, load_or_create_line_list
from muse.instrument.response.io import _open_dataset_file
from muse.tests.helpers import fake_continuum, fake_line_list
from muse.utils.utils import _to_zarr

TEMPERATURE = xr.DataArray(10 ** np.array([5.5, 6.0, 6.5]) * u.K, dims="logT")
LINE_LIST = {"abundance": "a", "wavelength_range": [150, 192] * u.AA, "temperature": TEMPERATURE}
CONTINUUM = {"abundance": "a", "wavelength_grid": [170.0, 171.0, 172.0] * u.AA, "temperature": TEMPERATURE}


def _pressure(values, dim="pressure"):
    return xr.DataArray(values * u.K / u.cm**3, dims=dim)


def _density(values):
    return xr.DataArray(values * u.cm**-3, dims="density")


def synthetic_line_list_backend(temperature, *, density=None, pressure=None, **_kwargs):
    """
    Three lines on the requested grids, the same at every plasma point.
    """
    temperature, plasma = _validate_plasma_inputs(temperature, density, pressure)
    line_list = fake_line_list(3, wavelength=[165.0, 171.073, 180.0], logT=np.log10(temperature.data))
    axis = "logD" if density is not None else plasma.dims[0]
    values = np.log10(plasma.data) if density is not None else plasma.data
    line_list["gofnt"] = line_list.gofnt.expand_dims({axis: values})
    line_list[axis].attrs["units"] = "dex(1 / cm3)" if density is not None else "K / cm3"
    return line_list.assign_attrs(chianti_version="11.0.2")


@pytest.fixture
def line_list_calls(monkeypatch):
    calls = []

    def backend(temperature, **kwargs):
        calls.append(kwargs)
        return synthetic_line_list_backend(temperature, **kwargs)

    monkeypatch.setitem(cache._LINE_LIST_BACKENDS, "fiasco", backend)
    return calls


@pytest.fixture
def continuum_calls(monkeypatch):
    calls = []

    def backend(temperature, wavelength_grid, **kwargs):
        calls.append(kwargs)
        return fake_continuum(temperature, wavelength_grid, **kwargs)

    monkeypatch.setitem(cache._CONTINUUM_BACKENDS, "fiasco", backend)
    return calls


def test_a_line_list_is_created_once_and_then_read(tmp_path, line_list_calls):
    kwargs = {**LINE_LIST, "pressure": _pressure([3e15])}

    first = load_or_create_line_list(tmp_path, **kwargs)
    second = load_or_create_line_list(tmp_path, **kwargs)

    assert len(line_list_calls) == 1
    # the file alone: no lock and no staging file are left
    assert [path.name for path in tmp_path.iterdir()] == ["line_list_fiasco_150-192A_a.zarr.zip"]
    xr.testing.assert_identical(first, second)
    assert first.gofnt.dims == ("pressure", "logT", "trans_index")
    assert first.pressure.attrs["units"] == "K / cm3"
    assert first.attrs["minimum_abundance"] == 1e-21


@pytest.mark.usefixtures("line_list_calls", "continuum_calls")
def test_the_file_names_say_what_the_files_hold(tmp_path):
    # The ends are not rounded: 25.4-900 must not find the list of 25-900.
    load_or_create_line_list(tmp_path, **{**LINE_LIST, "wavelength_range": [25.4, 900] * u.AA}, density=_density([1e9]))
    load_or_create_continuum(tmp_path, **CONTINUUM, pressure=_pressure([3e15]))

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "continuum_fiasco_170-172A_n3_a.zarr.zip",
        "line_list_fiasco_25.4-900A_a_density.zarr.zip",
    ]


@pytest.mark.parametrize(
    ("load", "kwargs"), [(load_or_create_line_list, LINE_LIST), (load_or_create_continuum, CONTINUUM)]
)
def test_a_missing_cache_can_be_an_error(tmp_path, line_list_calls, continuum_calls, load, kwargs):
    with pytest.raises(FileNotFoundError, match="does not exist, and create_if_missing is False"):
        load(tmp_path / "cache", create_if_missing=False, pressure=_pressure([3e15]), **kwargs)
    assert not line_list_calls
    assert not continuum_calls
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    ("grid", "created", "requested"),
    [("density", [1e8, 1e9], [1e9, 1e8]), ("pressure", [5e15, 3e15, 7e15], [7e15, 5e15])],
    ids=["reordered", "unsorted-file"],
)
def test_a_cache_serves_the_plasma_points_it_holds(tmp_path, continuum_calls, grid, created, requested):
    make = _density if grid == "density" else _pressure
    load_or_create_continuum(tmp_path, **CONTINUUM, **{grid: make(created)})

    actual = load_or_create_continuum(tmp_path, **CONTINUUM, **{grid: make(requested)})

    assert len(continuum_calls) == 1
    axis = "logD" if grid == "density" else "pressure"
    np.testing.assert_array_equal(actual[axis].values, np.log10(requested) if grid == "density" else requested)
    # the two-photon part of the synthetic continuum is 1e-40 times the plasma value, over 1e-27
    two_photon = actual.spectral_response.sel(line="two-photon").transpose(axis, "logT", "wavelength_bin").values
    np.testing.assert_allclose(
        two_photon, np.broadcast_to(1e-13 * np.array(requested)[:, None, None], two_photon.shape)
    )
    assert "wavelength_bounds" in actual.coords


MISMATCHES = {
    "nearby-pressure": ({"pressure": _pressure([3.0001e15])}, "holds the pressure values"),
    "temperatures": ({"temperature": TEMPERATURE.isel(logT=[0, 1])}, "is on other temperatures than requested"),
    "minimum-abundance": ({"minimum_abundance": 1e-2}, "minimum abundance 1e-21, not the requested 0.01"),
}


@pytest.mark.parametrize(("other", "problem"), MISMATCHES.values(), ids=MISMATCHES)
@pytest.mark.parametrize("what", ["line list", "continuum"])
def test_a_cache_that_does_not_hold_the_request_is_refused_and_kept(
    tmp_path, line_list_calls, continuum_calls, what, other, problem
):
    """
    Both caches do the same: neither computes again, and neither replaces a file other
    processes may be reading.
    """
    if what == "line list":
        load, kwargs, calls = load_or_create_line_list, LINE_LIST, line_list_calls
    else:
        load, kwargs, calls = load_or_create_continuum, CONTINUUM, continuum_calls
    kwargs = {**kwargs, "pressure": _pressure([3e15, 5e15])}
    load(tmp_path, **kwargs)
    (path,) = tmp_path.iterdir()
    written = path.read_bytes()

    with pytest.raises(ValueError, match=f"{problem}.*; delete it or use another cache directory"):
        load(tmp_path, **{**kwargs, **other})

    assert len(calls) == 1
    assert [found.name for found in tmp_path.iterdir()] == [path.name]
    assert path.read_bytes() == written


def test_two_grids_with_one_file_name_do_not_share_a_continuum(tmp_path, continuum_calls):
    kwargs = {**CONTINUUM, "pressure": _pressure([3e15])}
    load_or_create_continuum(tmp_path, **kwargs)

    # the same ends and the same number of wavelengths, so the same name
    with pytest.raises(ValueError, match="is on other wavelengths than requested"):
        load_or_create_continuum(tmp_path, **{**kwargs, "wavelength_grid": [170.0, 171.001, 172.0] * u.AA})

    assert len(continuum_calls) == 1


@pytest.mark.usefixtures("line_list_calls")
def test_the_requested_values_come_back_bit_for_bit(tmp_path, continuum_calls):
    """
    A pressure given in SI is one bit away from the one in the file; a line list and a
    continuum that each kept their own number would not align.
    """
    load_or_create_continuum(tmp_path, **CONTINUUM, pressure=_pressure([1e15]))
    in_si = xr.DataArray([1e21] * u.K / u.m**3, dims="pressure")
    requested = in_si.data.to_value(u.K / u.cm**3).tolist()
    assert requested != [1e15]

    continuum = load_or_create_continuum(tmp_path, **CONTINUUM, pressure=in_si)
    line_list = load_or_create_line_list(tmp_path, **LINE_LIST, pressure=in_si)

    assert continuum.pressure.values.tolist() == requested
    assert line_list.pressure.values.tolist() == requested
    np.testing.assert_array_equal(continuum.logT.values, np.log10(TEMPERATURE.data.value))
    np.testing.assert_array_equal(line_list.logT.values, continuum.logT.values)
    assert len(continuum_calls) == 1


def test_a_pressure_axis_of_another_name_is_stored_as_pressure(tmp_path, line_list_calls):
    line_list = load_or_create_line_list(tmp_path, **LINE_LIST, pressure=_pressure([3e15], dim="p"))

    assert line_list_calls[0]["pressure"].dims == ("pressure",)
    assert line_list.gofnt.dims == ("pressure", "logT", "trans_index")
    # and a call with the usual name reads the same file
    load_or_create_line_list(tmp_path, **LINE_LIST, pressure=_pressure([3e15]))
    assert len(line_list_calls) == 1


def test_the_chianti_version_of_a_cache_written_by_muse_0_3_is_read(tmp_path, line_list_calls):
    kwargs = {**LINE_LIST, "pressure": _pressure([3e15])}
    load_or_create_line_list(tmp_path, **kwargs)
    (path,) = tmp_path.glob("line_list_*.zarr.zip")
    with _open_dataset_file(path) as current:
        old = current.load()
    old.attrs["Chianti"] = old.attrs.pop("chianti_version")
    path.unlink()
    _to_zarr(old, path)

    load_or_create_line_list(tmp_path, chianti_version="11.0.2", **kwargs)

    assert len(line_list_calls) == 1


@pytest.mark.usefixtures("line_list_calls")
def test_a_line_list_of_another_chianti_version_is_refused(tmp_path):
    kwargs = {**LINE_LIST, "pressure": _pressure([3e15])}
    cache_directory = tmp_path / "run" / "cache"  # not there yet: the first call makes it

    assert (
        load_or_create_line_list(cache_directory, chianti_version="11.0.2", **kwargs).attrs["chianti_version"]
        == "11.0.2"
    )
    # the version is checked on a line list read from a file too, as the file name does not carry it
    with pytest.raises(ValueError, match=r"CHIANTI 11\.0\.2, not 9\.0\.1"):
        load_or_create_line_list(cache_directory, chianti_version="9.0.1", **kwargs)


def test_unknown_backends_and_unsafe_names_are_refused(tmp_path):
    pressure = _pressure([3e15])
    with pytest.raises(ValueError, match="backend must be one of"):
        load_or_create_line_list(tmp_path, backend="idl", pressure=pressure, **LINE_LIST)
    with pytest.raises(ValueError, match="backend must be one of"):
        load_or_create_continuum(tmp_path, backend="idl", pressure=pressure, **CONTINUUM)
    # an abundance is part of the file name: this one would write outside the directory
    with pytest.raises(ValueError, match="abundance is part of a file name"):
        load_or_create_line_list(tmp_path, pressure=pressure, **{**LINE_LIST, "abundance": "../a"})
    assert not list(tmp_path.iterdir())


def test_a_cache_another_process_is_creating_is_not_created_twice(tmp_path, line_list_calls):
    kwargs = {**LINE_LIST, "pressure": _pressure([3e15])}
    lock = tmp_path / "line_list_fiasco_150-192A_a.zarr.zip.lock"
    lock.write_text("process 7 on node12, job 99, since 2026-01-01 00:00 UTC")

    # the error says who holds the lock, for the person who has to judge whether it is stale
    with pytest.raises(
        FileExistsError, match=r"is being created by process 7 on node12, job 99.*delete .*a\.zarr\.zip\.lock"
    ):
        load_or_create_line_list(tmp_path, **kwargs)

    assert not line_list_calls
    assert lock.exists()  # not this call's to remove
    lock.unlink()
    load_or_create_line_list(tmp_path, **kwargs)
    assert not lock.exists()


def test_the_lock_says_which_process_holds_it(tmp_path, monkeypatch):
    seen = []

    def backend(temperature, **kwargs):
        seen.append((tmp_path / "line_list_fiasco_150-192A_a.zarr.zip.lock").read_text())
        return synthetic_line_list_backend(temperature, **kwargs)

    monkeypatch.setitem(cache._LINE_LIST_BACKENDS, "fiasco", backend)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")

    load_or_create_line_list(tmp_path, **LINE_LIST, pressure=_pressure([3e15]))

    assert seen[0].startswith(f"process {os.getpid()} on ")
    assert ", job 4242, since " in seen[0]


def test_a_cache_finished_while_waiting_for_the_lock_is_read(tmp_path, line_list_calls, monkeypatch):
    """
    Between the check for the file and the lock another process may have finished it: it
    is then read, not created a second time.
    """
    kwargs = {**LINE_LIST, "pressure": _pressure([3e15])}
    finished = load_or_create_line_list(tmp_path / "other", **kwargs)

    @contextlib.contextmanager
    def lock_taken_late(path, _what):
        shutil.copy(tmp_path / "other" / path.name, path)
        yield

    monkeypatch.setattr(cache, "_creation_lock", lock_taken_late)
    (tmp_path / "cache").mkdir()

    xr.testing.assert_identical(load_or_create_line_list(tmp_path / "cache", **kwargs), finished)
    assert len(line_list_calls) == 1


def test_a_failed_creation_releases_its_lock(tmp_path, monkeypatch):
    def no_database(_temperature, **_kwargs):
        msg = "no database"
        raise RuntimeError(msg)

    monkeypatch.setitem(cache._LINE_LIST_BACKENDS, "fiasco", no_database)

    with pytest.raises(RuntimeError, match="no database"):
        load_or_create_line_list(tmp_path, **LINE_LIST, pressure=_pressure([3e15]))

    assert not list(tmp_path.iterdir())


def test_limited_threads_caps_numexpr_and_puts_it_back(monkeypatch):
    numexpr = pytest.importorskip("numexpr")
    monkeypatch.setattr(os, "process_cpu_count", lambda: 64)
    before = numexpr.set_num_threads(3)
    try:
        with cache._limited_threads(2):
            assert numexpr.set_num_threads(2) == 2
        assert numexpr.set_num_threads(3) == 3
        # asking for more threads than numexpr is set to leaves it as it is
        with cache._limited_threads(8):
            assert numexpr.set_num_threads(3) == 3
        # and a process that may use one CPU gets one thread, whatever is asked
        monkeypatch.setattr(os, "process_cpu_count", lambda: 1)
        with cache._limited_threads(2):
            assert numexpr.set_num_threads(1) == 1
        assert numexpr.set_num_threads(3) == 3
    finally:
        numexpr.set_num_threads(before)


def test_limited_threads_never_increases_a_pool(monkeypatch):
    # Stand-ins for the pools: a test must not depend on which BLAS the machine has.
    seen = []
    monkeypatch.setattr(
        cache,
        "threadpool_info",
        lambda: [
            {"user_api": "blas", "num_threads": 2},
            {"user_api": "openmp", "num_threads": 16},
            {"user_api": "openmp", "num_threads": 3},
        ],
    )
    monkeypatch.setattr(cache, "threadpool_limits", lambda limits: seen.append(limits) or contextlib.nullcontext())
    monkeypatch.setattr(os, "process_cpu_count", lambda: 64)

    with cache._limited_threads(4):
        pass
    monkeypatch.setattr(os, "process_cpu_count", lambda: 1)
    with cache._limited_threads(4):
        pass

    assert seen == [{"blas": 2, "openmp": 3}, {"blas": 1, "openmp": 1}]
    with pytest.raises(ValueError, match="threads must be a positive integer"), cache._limited_threads(0):
        pass
