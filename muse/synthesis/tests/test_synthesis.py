import gc
import os
import time
import tracemalloc

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from muse.synthesis import _backends
from muse.synthesis.synthesis import vdem_synthesis
from muse.transforms.transforms import reshape_x_to_slit_step
from muse.variables import DEFAULTS_MUSE


def _mps_available() -> bool:
    try:
        import torch  # NOQA: PLC0415
    except ImportError:
        return False
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def test_vdem_synthesis(response, raster) -> None:
    detector_response = vdem_synthesis(raster, response)
    assert isinstance(detector_response.flux.data, np.ndarray)
    # A response read with chunked=True keeps the flux lazy next to an in-memory raster.
    assert isinstance(vdem_synthesis(raster, response.chunk({"logT": 1})).flux.data, da.Array)
    assert set(detector_response.data_vars) == {"flux"}
    assert dict(detector_response.sizes) == {
        "y": 32,
        "step": DEFAULTS_MUSE.steps_per_raster_SG,
        "line": 7,
        "detector_x_pixel": 32,
    }
    assert np.isfinite(detector_response.flux).all()
    assert detector_response.flux.attrs["units"] == "ph / s"
    # The history of the raster, then this call; the format of an entry is tested with add_history.
    assert [entry.split("(")[0] for entry in detector_response.attrs["history"]] == [
        "reshape_x_to_slit_step",
        "vdem_synthesis",
    ]


def test_vdem_synthesis_rechunks_contracted_dims_to_single_chunk(response, raster) -> None:
    # Operands chunked along the contracted dims (e.g. a response read with
    # chunked=True, whose files store logT chunks of 1) must be rechunked before
    # the einsum: da.einsum otherwise pairs every chunk of one operand with
    # every chunk of the other, exploding the task graph and peak memory.
    eager = vdem_synthesis(raster, response)
    lazy = vdem_synthesis(
        raster.chunk({"logT": 1, "doppler_velocity": 2}),
        response.chunk({"line": 1, "logT": 1, "doppler_velocity": 2}),
    )

    # ~309 tasks with the rechunk in place; deleting _single_chunk_over on either
    # operand pairs chunks instead (960+ tasks), so bound the graph structurally.
    assert len(dict(lazy.flux.data.__dask_graph__())) < 600
    np.testing.assert_allclose(lazy.flux.compute().values, eager.flux.values, rtol=1e-12)


def _coordless_inputs() -> tuple[xr.Dataset, xr.Dataset]:
    raster = xr.Dataset(
        {"vdem": (("logT", "doppler_velocity", "y"), np.arange(1.0, 9.0).reshape(2, 2, 2), {"units": "cm-5"})}
    )
    response = xr.Dataset(
        {
            "detector_response": (
                ("line", "doppler_velocity", "logT"),
                [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 2.0], [3.0, 0.0]]],
                {"units": "cm5"},
            )
        },
        coords={
            "line_wavelength": ("line", [100.0, 200.0], {"units": "Angstrom"}),
            "detector_wavelength": ("line", [100.0, 200.0], {"units": "Angstrom"}),
        },
    )
    return raster, response


def test_vdem_synthesis_contracts_values() -> None:
    raster, response = _coordless_inputs()

    result = vdem_synthesis(raster, response, sum_over=("logT", "doppler_velocity"))

    assert result.flux.dims == ("y", "line")
    np.testing.assert_array_equal(result.flux, [[8.0, 19.0], [10.0, 24.0]])
    # Index coordinates even for coordinate-less inputs, so .sel(line=...) and index-aligned merges work.
    assert {"y", "line"} <= set(result.xindexes)


def test_vdem_synthesis_rejects_misaligned_contraction_coordinates(response, raster) -> None:
    response = response.assign_coords(logT=response.logT.values + 1e-9)

    with pytest.raises(ValueError, match="align_response_and_vdem"):
        vdem_synthesis(raster, response)


def test_vdem_synthesis_rejects_one_sided_index_coordinates(response, raster) -> None:
    # A shared dim indexed on one input only bypasses join="exact" and would
    # contract positionally against unordered labels; it must fail loudly.
    bare = response.drop_vars("logT")

    with pytest.raises(ValueError, match="'logT' has an index coordinate on only one input"):
        vdem_synthesis(raster, bare)


def test_vdem_synthesis_keeps_conflicting_coords_from_raster_except_on_response_only_dims(response, raster) -> None:
    # A coord both inputs carry with different values must survive with the
    # raster's value (pre-refactor behavior), not silently vanish.
    raster = raster.assign_coords(instrument="muse-raster")
    response = response.assign_coords(instrument="muse-response")

    result = vdem_synthesis(raster, response)

    assert str(result.instrument.values) == "muse-raster"
    # One slit of the raster keeps slit as a scalar coord; the kept slit axis of the response owns it.
    one_slit = vdem_synthesis(raster.isel(slit=3), response.isel(slit=[3]), sum_over=("logT", "doppler_velocity"))
    np.testing.assert_array_equal(one_slit.slit, response.slit[[3]])


def test_vdem_synthesis_torch_backend_matches_numpy(response, raster) -> None:
    pytest.importorskip("torch")
    numpy_flux = vdem_synthesis(raster, response, backend="numpy").flux
    accel_flux = vdem_synthesis(raster, response, backend="torch").flux

    assert isinstance(accel_flux.data, np.ndarray)
    np.testing.assert_allclose(accel_flux.values, numpy_flux.values, rtol=1e-4)


def _torch_budget(raster, response, flux, rows: int) -> int:
    """
    A budget whose half, minus the response, holds ``rows`` rows of the product along y.
    """
    per_row = 8 * (raster.vdem.size + flux.size) // raster.sizes["y"]
    return 2 * (rows * per_row + 8 * response.detector_response.size)


def test_vdem_synthesis_torch_in_chunks_matches_one_product(response, raster, monkeypatch) -> None:
    """
    An inverted VDEM has slit before y; slit is shared with the response, so the product is
    cut along y: in chunks of three rows, and a row at a time on the CPU below a budget of
    one row, as the host pages rather than failing.
    """
    pytest.importorskip("torch")
    # slit stays in the output, so it is a candidate chunk axis that the response shares
    sum_over = ("logT", "doppler_velocity")
    expected = vdem_synthesis(raster, response, sum_over=sum_over, backend="torch").flux
    slit_first = raster.transpose("logT", "doppler_velocity", "slit", "y", "step")
    for budget in (_torch_budget(slit_first, response, expected, 3), 1):
        monkeypatch.setattr(_backends, "_available_memory_bytes", lambda _device, budget=budget: budget)
        result = vdem_synthesis(slit_first, response, sum_over=sum_over, backend="torch").flux
        np.testing.assert_allclose(
            result.transpose(*expected.dims).values, expected.values, rtol=1e-5, atol=1e-6 * float(abs(expected).max())
        )


def test_vdem_synthesis_torch_refuses_a_budget_below_one_row_on_an_accelerator(response, raster, monkeypatch) -> None:
    """
    The accelerator branch needs no accelerator: the budget is refused before any tensor
    is placed, so this runs where the MPS tests skip.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(_backends, "_resolve_device", lambda _device: torch.device("mps"))
    monkeypatch.setattr(_backends, "_empty_device_cache", lambda _device: None)
    monkeypatch.setattr(_backends, "_available_memory_bytes", lambda _device: 1)
    with pytest.raises(MemoryError, match="does not fit"):
        vdem_synthesis(raster, response, device="mps", backend="torch")


@pytest.mark.skipif(not _mps_available(), reason="no MPS device")
def test_vdem_synthesis_torch_on_mps_refuses_a_budget_below_one_row(response, raster, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1)
    with pytest.raises(MemoryError, match="does not fit"):
        vdem_synthesis(raster, response, device="mps", backend="torch")


@pytest.mark.skipif(not _mps_available(), reason="no MPS device")
def test_vdem_synthesis_torch_on_mps_refuses_an_all_zero_product(response, raster, monkeypatch) -> None:
    """
    Metal returns zeros instead of raising when it runs out of memory; the synthesis
    must not pass them on.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "einsum", lambda *_args, **_kwargs: torch.zeros(1, device="mps"))
    with pytest.raises(RuntimeError, match="all zero"):
        vdem_synthesis(raster, response, device="mps", backend="torch")


@pytest.mark.skipif(not _mps_available(), reason="no MPS device")
def test_vdem_synthesis_torch_on_mps_empties_the_cache_before_sizing_its_chunks(response, raster, monkeypatch) -> None:
    """
    The MPS allocator's cached blocks count as allocated, so a solve run just before
    would leave no budget.
    """
    torch = pytest.importorskip("torch")
    calls = []
    measure = _backends._available_memory_bytes
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("emptied"))
    monkeypatch.setattr(_backends, "_available_memory_bytes", lambda device: calls.append("sized") or measure(device))
    vdem_synthesis(raster, response, device="mps", backend="torch")
    assert calls[:2] == ["emptied", "sized"]


@pytest.mark.cuda
def test_vdem_synthesis_torch_on_cuda_empties_the_cache_before_sizing_its_chunks(response, raster, monkeypatch) -> None:
    """
    The CUDA allocator keeps the blocks a solve reserved, which mem_get_info counts as
    used.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    calls = []
    measure = _backends._available_memory_bytes
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("emptied"))
    monkeypatch.setattr(_backends, "_available_memory_bytes", lambda device: calls.append("sized") or measure(device))
    vdem_synthesis(raster, response, device="cuda", backend="torch")
    assert calls[:2] == ["emptied", "sized"]


@pytest.mark.performance
def test_vdem_synthesis_performance_regression(response, raster) -> None:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("performance baselines need an uncontended machine; rerun with -n 0")
    response = response.isel(detector_x_pixel=np.arange(1024) % response.sizes["detector_x_pixel"]).assign_coords(
        detector_x_pixel=np.arange(1024)
    )
    # Untraced baselines (2026-08-14, threads scheduler pinned to 4 workers):
    # numpy 0.03 s/146 MiB, dask 0.19 s/514 MiB, chunked-response 0.14 s/508 MiB.
    # Limits are deliberately wide for shared CI runners; graph-shape regressions are
    # caught structurally by test_vdem_synthesis_rechunks_contracted_dims_to_single_chunk.
    cases = {
        "numpy": (raster, response, 1.0, 300),
        "dask": (raster.chunk({"logT": 5, "step": 4}), response, 2.0, 1024),
        "chunked-response": (
            raster.chunk({"step": 4}),
            response.chunk({"logT": 1, "doppler_velocity": 2}),
            2.0,
            1024,
        ),
    }

    for name, (case_raster, case_response, max_seconds, max_peak_mib) in cases.items():
        with dask.config.set(scheduler="threads", num_workers=4):
            vdem_synthesis(case_raster, case_response).flux.compute()  # warm caches and imports
            gc.collect()
            start = time.perf_counter()
            vdem_synthesis(case_raster, case_response).flux.compute()
            elapsed = time.perf_counter() - start
            gc.collect()
            tracemalloc.start()  # traced separately: tracing inflates wall-clock ~1.3x
            try:
                vdem_synthesis(case_raster, case_response).flux.compute()
                peak_mib = tracemalloc.get_traced_memory()[1] / 1024**2
            finally:
                tracemalloc.stop()

        assert elapsed < max_seconds, f"{name} synthesis took {elapsed:.2f} s (limit {max_seconds:.2f} s)"
        assert peak_mib < max_peak_mib, f"{name} synthesis peaked at {peak_mib:.0f} MiB (limit {max_peak_mib} MiB)"


def test_vdem_synthesis_accepts_generic_wavelength_space_names(response, vdem) -> None:
    # A slit-less response with the create_spectral_response vocabulary synthesizes
    # with default arguments: names are renamed and slit is dropped from sum_over.
    single_slit = response.isel(slit=0, drop=True)
    generic = single_slit.rename(
        {
            "detector_response": "spectral_response",
            "detector_wavelength": "wavelength_grid",
        }
    )

    canonical = vdem_synthesis(vdem, single_slit, sum_over=("logT", "doppler_velocity"))
    accepted = vdem_synthesis(vdem, generic)

    xr.testing.assert_identical(accepted.flux, canonical.flux)
    assert accepted.attrs["sum_over"] == ["logT", "doppler_velocity"]


def test_vdem_synthesis_rejects_unknown_sum_over_dim(response, raster) -> None:
    with pytest.raises(ValueError, match=r"'bogus' is not a response dimension"):
        vdem_synthesis(raster, response, sum_over=("bogus",))


def test_vdem_synthesis_rejects_the_legacy_vdop_axis(response, raster) -> None:
    # A stray vdop is not in sum_over, so einsum would carry it into the output and contract
    # the other input's doppler_velocity alone: no error, wrong flux. It must fail loudly.
    with pytest.raises(ValueError, match=r"vdem uses the legacy 'vdop' dimension"):
        vdem_synthesis(raster.rename(doppler_velocity="vdop"), response)
    # read_response keeps the vdop of a legacy response file
    with pytest.raises(ValueError, match=r"response uses the legacy 'vdop' dimension"):
        vdem_synthesis(raster, response.rename(doppler_velocity="vdop"))


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("raster_without_slit", "response has a slit dimension"),
        ("response_without_slit", "vdem has a slit dimension but response does not"),
        ("wavelength_response_without_slit", "vdem has a slit dimension but response does not"),
    ],
)
def test_vdem_synthesis_rejects_mismatched_slit_dimensions(response, raster, case, match) -> None:
    # A spectrograph response and its raster have the slit axis on both sides or on neither; contracting
    # a multi-slit response against a raster that has no slit axis is not a meaningful synthesis.
    if case == "raster_without_slit":
        raster = raster.isel(slit=0, drop=True)
    else:
        response = response.isel(slit=0, drop=True)
    if case == "wavelength_response_without_slit":
        # the shape create_spectral_response returns: no slit, and the spectral axis in wavelength space
        response = response.rename(detector_x_pixel="wavelength_bin")

    with pytest.raises(ValueError, match=match):
        vdem_synthesis(raster, response)


def test_vdem_synthesis_images_each_slit_position_with_an_imager_response(response, raster, vdem) -> None:
    # An imager response has no slit and no spectral axis, as map_response_to_ci_detector returns it.
    imager = response.isel(slit=0, detector_x_pixel=0, drop=True)

    out = vdem_synthesis(raster, imager)

    # The same image as synthesizing the field in x and y and putting it on the raster afterwards.
    expected = reshape_x_to_slit_step(vdem_synthesis(vdem, imager))
    assert set(out.flux.dims) == {"step", "slit", "y", "line"}
    xr.testing.assert_allclose(out.flux.transpose(*expected.flux.dims), expected.flux)


def test_vdem_synthesis_requires_detector_wavelength_only_while_slit_is_kept(response, raster) -> None:
    response = response.reset_coords("detector_wavelength")

    with pytest.raises(ValueError, match=r"response\.detector_wavelength is missing"):
        vdem_synthesis(raster, response, sum_over=("logT", "doppler_velocity"))
    assert "detector_wavelength" not in vdem_synthesis(raster, response).coords


def test_vdem_synthesis_converts_wavelength_coords_to_angstrom(response, raster) -> None:
    response_nm = response.assign_coords(
        line_wavelength=response.line_wavelength / 10.0,
        detector_wavelength=response.detector_wavelength / 10.0,
    )
    response_nm.line_wavelength.attrs["units"] = "nm"
    response_nm.detector_wavelength.attrs["units"] = "nm"

    detector_response = vdem_synthesis(raster, response_nm, sum_over=("logT", "doppler_velocity"))

    np.testing.assert_allclose(detector_response.line_wavelength.values, response.line_wavelength.values)
    np.testing.assert_allclose(detector_response.detector_wavelength.values, response.detector_wavelength.values)
    assert detector_response.line_wavelength.attrs["units"] == "Angstrom"
    assert detector_response.detector_wavelength.attrs["units"] == "Angstrom"


@pytest.mark.cuda
def test_vdem_synthesis_cuda_matches_cpu(response, raster) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cpu = vdem_synthesis(raster, response)
    gpu = vdem_synthesis(raster, response, device="cuda:0", backend="torch")
    np.testing.assert_allclose(gpu.flux.values, cpu.flux.values, rtol=1e-5, atol=1e-6)


def test_vdem_synthesis_requires_units(response, raster) -> None:
    with pytest.raises(ValueError, match=r"response\.detector_response is missing"):
        vdem_synthesis(raster, response.drop_vars("detector_response"))
    del raster.vdem.attrs["units"]

    with pytest.raises(ValueError, match=r"vdem\.vdem must define units"):
        vdem_synthesis(raster, response)
