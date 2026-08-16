import gc
import os
import time
import tracemalloc

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.synthesis.synthesis import vdem_synthesis
from muse.tests.helpers import assert_dataset_structure, fake_vdem_single_doppler_velocity
from muse.transforms.transforms import reshape_x_to_slit_step

SPEED_OF_LIGHT_KMS = 299792.458


def test_vdem_synthesis(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    detector_response = vdem_synthesis(reshaped_vdem, response)
    assert isinstance(detector_response.flux.data, np.ndarray)
    assert_dataset_structure(
        detector_response,
        data_vars=("flux",),
        coords=("y", "step", "line", "detector_x_pixel", "line_wavelength"),
        sizes={"y": 32, "step": 11, "line": 7, "detector_x_pixel": 32},
        finite_vars=("flux",),
    )
    assert detector_response.flux.attrs["units"] == "ph / s"
    assert detector_response.attrs["HISTORY"] == [
        "reshape_x_to_slit_step(ds=ds, nslits=35, nraster=11)",
        (
            "vdem_synthesis(raster=raster, response=response, sum_over=('logT', 'doppler_velocity', 'slit'), "
            "cuda_device=None, backend=numpy)"
        ),
    ]
    np.testing.assert_array_equal(
        detector_response.line_wavelength.values,
        [108.355, 108.117, 108.355, 171.073, 171.073, 284.163, 284.163],
    )


def test_vdem_synthesis_keeps_dask_inputs_lazy(response, vdem) -> None:
    # A dask-backed raster (e.g. from open_zarr) must produce a lazy flux whose
    # computed values match the eager numpy path.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    eager = vdem_synthesis(reshaped_vdem, response)
    lazy = vdem_synthesis(reshaped_vdem.chunk({"logT": 5, "step": 4}), response)

    assert isinstance(lazy.flux.data, da.Array)
    np.testing.assert_allclose(lazy.flux.compute().values, eager.flux.values, rtol=1e-12)


def test_vdem_synthesis_rechunks_contracted_dims_to_single_chunk(response, vdem) -> None:
    # Operands chunked along the contracted dims (e.g. a response read with
    # chunked=True, whose files store logT chunks of 1) must be rechunked before
    # the einsum: da.einsum otherwise pairs every chunk of one operand with
    # every chunk of the other, exploding the task graph and peak memory.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    eager = vdem_synthesis(reshaped_vdem, response)
    lazy = vdem_synthesis(
        reshaped_vdem.chunk({"logT": 1, "doppler_velocity": 2}),
        response.chunk({"line": 1, "logT": 1, "doppler_velocity": 2}),
    )

    assert isinstance(lazy.flux.data, da.Array)
    assert lazy.flux.data.npartitions <= reshaped_vdem.sizes["y"] * response.sizes["line"]
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


def test_vdem_synthesis_materializes_index_coordinates() -> None:
    # Every output dim must carry an index coordinate even for coordinate-less
    # inputs, so .sel(line=...) and index-aligned concat/merge keep working.
    raster, response = _coordless_inputs()

    result = vdem_synthesis(raster, response, sum_over=("logT", "doppler_velocity"))

    for dim in result.flux.dims:
        assert dim in result.coords, f"{dim} has no index coordinate"
    np.testing.assert_array_equal(result.y.values, [0, 1])
    np.testing.assert_array_equal(result.line.values, [0, 1])


def test_vdem_synthesis_rejects_misaligned_contraction_coordinates(response, vdem) -> None:
    raster = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    response = response.assign_coords(logT=response.logT.values + 1e-9)

    with pytest.raises(ValueError, match="align_response_and_vdem"):
        vdem_synthesis(raster, response)


def test_vdem_synthesis_rejects_one_sided_index_coordinates(response, vdem) -> None:
    # A shared dim indexed on one input only bypasses join="exact" and would
    # contract positionally against unordered labels; it must fail loudly.
    raster = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    bare = response.drop_vars("logT")

    with pytest.raises(ValueError, match="'logT' has an index coordinate on only one input"):
        vdem_synthesis(raster, bare)


def test_vdem_synthesis_preserves_coordinate_attrs(response, vdem) -> None:
    # Coordinate units must survive the contraction: downstream consumers
    # (reshape_x_to_slit_step, match_fov) require units on the spatial coords.
    raster = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)

    result = vdem_synthesis(raster, response)

    assert result.y.attrs == raster.y.attrs
    assert result.y.attrs["units"] == "arcsec"


def test_vdem_synthesis_accepts_quantity_backed_data(response, vdem) -> None:
    # Duck arrays (e.g. astropy Quantity from a units-aware pipeline) must contract
    # like plain numpy instead of hitting einsum's subscripts-only Quantity path.
    raster = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    quantity_raster = raster.assign(vdem=raster.vdem.copy(data=u.Quantity(raster.vdem.data, copy=False)))

    expected = vdem_synthesis(raster, response)
    result = vdem_synthesis(quantity_raster, response)

    np.testing.assert_array_equal(result.flux.values, expected.flux.values)


def test_vdem_synthesis_keeps_conflicting_scalar_coords_from_raster(response, vdem) -> None:
    # A coord both inputs carry with different values must survive with the
    # raster's value (pre-refactor behavior), not silently vanish.
    raster = reshape_x_to_slit_step(vdem, nslits=35, nraster=11).assign_coords(instrument="muse-raster")
    response = response.assign_coords(instrument="muse-response")

    result = vdem_synthesis(raster, response)

    assert str(result.instrument.values) == "muse-raster"


def test_vdem_synthesis_preserves_response_output_coords(response, vdem) -> None:
    raster = reshape_x_to_slit_step(vdem, nslits=35, nraster=11).assign_coords(
        channel=("line", np.full(response.sizes["line"], 999))
    )

    result = vdem_synthesis(raster, response)

    np.testing.assert_array_equal(result.channel, response.channel)


def test_vdem_synthesis_propagates_dataset_level_coords(response, vdem) -> None:
    # A raster coord living on a response-only output dim (so not attached to the
    # vdem variable) must still propagate to the output.
    line_tags = [f"l{i}" for i in range(response.sizes["line"])]
    raster = reshape_x_to_slit_step(vdem, nslits=35, nraster=11).assign_coords(line_tag=("line", line_tags))

    result = vdem_synthesis(raster, response)

    assert list(result.line_tag.values) == line_tags


def test_vdem_synthesis_preserves_internal_component_kind(response, vdem) -> None:
    component_kind = ["line", "line", "contaminants", "line", "contaminants", "line", "contaminants"]
    response = response.assign_coords(component_kind=("line", component_kind))
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)

    result = vdem_synthesis(reshaped_vdem, response)

    np.testing.assert_array_equal(result.component_kind, component_kind)


def test_vdem_synthesis_torch_backend_matches_numpy(response, vdem) -> None:
    pytest.importorskip("torch")
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    numpy_flux = vdem_synthesis(reshaped_vdem, response, backend="numpy").flux
    accel_flux = vdem_synthesis(reshaped_vdem, response, backend="torch").flux

    assert isinstance(accel_flux.data, np.ndarray)
    np.testing.assert_allclose(accel_flux.values, numpy_flux.values, rtol=1e-4)


@pytest.mark.cost
def test_vdem_synthesis_cost_regression(response, vdem) -> None:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("cost baselines need an uncontended machine; rerun with -n 0")
    raster = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
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


def test_vdem_synthesis_doppler_shifts_line_centroid(response) -> None:
    # Emission at a single doppler_velocity must place the line at lambda * (1 + v/c): synthesis
    # encodes velocity as a wavelength shift, the core spectral behaviour.
    def centroid(vdop_kms):
        reshaped = reshape_x_to_slit_step(fake_vdem_single_doppler_velocity(vdop_kms), nslits=35, nraster=11)
        flux = vdem_synthesis(reshaped, response, sum_over=("logT", "doppler_velocity")).flux
        spectrum = flux.isel(line=0, slit=17).sum(dim=["y", "step"]).values
        wavelength = flux.detector_wavelength.isel(line=0, slit=17).values
        return float((spectrum * wavelength).sum() / spectrum.sum())

    rest_wavelength = 108.355
    expected_shift = rest_wavelength * 300.0 / SPEED_OF_LIGHT_KMS
    blue, zero, red = centroid(-300.0), centroid(0.0), centroid(300.0)
    assert blue < zero < red
    # The fixture samples 32 detector pixels, so the centroid resolves the shift to
    # ~10%; assert direction and magnitude, not an exact match.
    np.testing.assert_allclose(red - zero, expected_shift, rtol=0.15)
    np.testing.assert_allclose(zero - blue, expected_shift, rtol=0.15)


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


def test_vdem_synthesis_rejects_unknown_sum_over_dim(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    with pytest.raises(ValueError, match=r"'bogus' is not a response dimension"):
        vdem_synthesis(reshaped_vdem, response, sum_over=("bogus",))


@pytest.mark.parametrize("legacy_input", ["raster", "response"])
def test_vdem_synthesis_rejects_the_legacy_vdop_axis(response, vdem, legacy_input) -> None:
    # A stray vdop is not in sum_over, so einsum would carry it into the output and contract
    # the other input's doppler_velocity alone: no error, wrong flux. It must fail loudly.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    if legacy_input == "raster":
        reshaped_vdem = reshaped_vdem.rename(doppler_velocity="vdop")
    else:
        response = response.rename(doppler_velocity="vdop")

    with pytest.raises(ValueError, match=rf"{legacy_input} uses the legacy 'vdop' dimension"):
        vdem_synthesis(reshaped_vdem, response)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("raster_without_slit", "response has a slit dimension"),
        ("response_without_slit", "raster has a slit dimension but response does not"),
    ],
)
def test_vdem_synthesis_rejects_mismatched_slit_dimensions(response, vdem, case, match) -> None:
    # The slit axis has to be present on both sides or neither; contracting a multi-slit
    # response against a raster that has no slit axis is not a meaningful synthesis.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    if case == "raster_without_slit":
        reshaped_vdem = reshaped_vdem.isel(slit=0, drop=True)
    else:
        response = response.isel(slit=0, drop=True)

    with pytest.raises(ValueError, match=match):
        vdem_synthesis(reshaped_vdem, response)


def test_vdem_synthesis_requires_present_arrays(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    bad_response = response.drop_vars("detector_response")
    with pytest.raises(ValueError, match=r"response\.detector_response is missing"):
        vdem_synthesis(reshaped_vdem, bad_response)


def test_vdem_synthesis_rejects_invalid_unit_string(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    bad_response = response.copy(deep=True)
    bad_response.detector_response.attrs["units"] = "not-a-real-unit"
    with pytest.raises(ValueError, match=r"response\.detector_response units must be a valid astropy unit"):
        vdem_synthesis(reshaped_vdem, bad_response)


def test_vdem_synthesis_rejects_non_length_wavelength_units(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    bad_response = response.copy(deep=True)
    bad_response.line_wavelength.attrs["units"] = "km/s"
    with pytest.raises(ValueError, match=r"response\.line_wavelength units must be convertible to Angstrom"):
        vdem_synthesis(reshaped_vdem, bad_response)


@pytest.mark.parametrize("name", ["line_wavelength", "detector_wavelength"])
def test_vdem_synthesis_requires_response_wavelength_coords(response, vdem, name) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    bad_response = response.reset_coords(name)
    kwargs = {"sum_over": ("logT", "doppler_velocity")} if name == "detector_wavelength" else {}

    with pytest.raises(ValueError, match=rf"response\.{name} is missing"):
        vdem_synthesis(reshaped_vdem, bad_response, **kwargs)


def test_vdem_synthesis_does_not_require_discarded_detector_wavelength(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    response = response.reset_coords("detector_wavelength")

    result = vdem_synthesis(reshaped_vdem, response)

    assert "detector_wavelength" not in result.coords


def test_vdem_synthesis_keeps_slit_and_assigns_sg_wvl(response, vdem) -> None:
    # Not summing over slit leaves it as a flux dimension, which triggers the
    # detector_wavelength coordinate assignment branch.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    detector_response = vdem_synthesis(reshaped_vdem, response, sum_over=("logT", "doppler_velocity"))
    assert "slit" in detector_response.flux.dims
    assert_dataset_structure(
        detector_response,
        data_vars=("flux",),
        coords=("y", "slit", "step", "line", "detector_x_pixel", "line_wavelength", "detector_wavelength"),
        sizes={"y": 32, "slit": 35, "step": 11, "line": 7, "detector_x_pixel": 32},
        finite_vars=("flux",),
    )
    assert detector_response.detector_wavelength.dims == ("line", "slit", "detector_x_pixel")


def test_vdem_synthesis_converts_wavelength_coords_to_angstrom(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    response_nm = response.assign_coords(
        line_wavelength=response.line_wavelength / 10.0,
        detector_wavelength=response.detector_wavelength / 10.0,
    )
    response_nm.line_wavelength.attrs["units"] = "nm"
    response_nm.detector_wavelength.attrs["units"] = "nm"

    detector_response = vdem_synthesis(reshaped_vdem, response_nm, sum_over=("logT", "doppler_velocity"))

    np.testing.assert_allclose(detector_response.line_wavelength.values, response.line_wavelength.values)
    np.testing.assert_allclose(detector_response.detector_wavelength.values, response.detector_wavelength.values)
    assert detector_response.line_wavelength.attrs["units"] == "Angstrom"
    assert detector_response.detector_wavelength.attrs["units"] == "Angstrom"


@pytest.mark.cuda
def test_vdem_synthesis_cuda_matches_cpu(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    cpu = vdem_synthesis(reshaped_vdem, response)
    gpu = vdem_synthesis(reshaped_vdem, response, cuda_device=0, backend="torch")
    assert_dataset_structure(
        gpu,
        data_vars=("flux",),
        coords=("y", "step", "line", "detector_x_pixel", "line_wavelength"),
        sizes={"y": 32, "step": 11, "line": 7, "detector_x_pixel": 32},
        finite_vars=("flux",),
    )
    np.testing.assert_allclose(gpu.flux.values, cpu.flux.values, rtol=1e-5, atol=1e-6)


def test_vdem_synthesis_requires_raster_vdem_units(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    del reshaped_vdem.vdem.attrs["units"]

    with pytest.raises(ValueError, match=r"raster\.vdem must define units"):
        vdem_synthesis(reshaped_vdem, response)


@pytest.mark.parametrize("name", ["detector_response", "line_wavelength", "detector_wavelength"])
def test_vdem_synthesis_requires_response_units(response, vdem, name) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    bad_response = response.copy(deep=True)
    del bad_response[name].attrs["units"]
    kwargs = {"sum_over": ("logT", "doppler_velocity")} if name == "detector_wavelength" else {}

    with pytest.raises(ValueError, match=rf"response\.{name} must define units"):
        vdem_synthesis(reshaped_vdem, bad_response, **kwargs)
