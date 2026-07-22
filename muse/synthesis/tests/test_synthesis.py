import dask.array as da
import numpy as np
import pytest

from muse.synthesis.synthesis import _build_einsum_indices, vdem_synthesis
from muse.tests.helpers import assert_dataset_structure, fake_vdem_single_vdop
from muse.transforms.transforms import reshape_x_to_slit_step

SPEED_OF_LIGHT_KMS = 299792.458

# Dimension names for the science contraction used across the einsum-index tests.
RASTER_DIMS = ("logT", "vdop", "y", "slit", "step")
RESPONSE_DIMS = ("line", "vdop", "logT", "slit", "detector_x_pixel")


def test_build_einsum_indices_contracts_shared_dims() -> None:
    # Shared dims (logT, vdop, slit) reuse the raster letters; the default sum_over
    # contracts them, leaving y/step from the raster and line/detector_x_pixel from the response.
    einsum_str, out_str, out_dims = _build_einsum_indices(RASTER_DIMS, RESPONSE_DIMS, ("logT", "vdop", "slit"))
    assert einsum_str == "abcde,fbadg"
    assert out_str == "cefg"
    assert out_dims == ["y", "step", "line", "detector_x_pixel"]


def test_build_einsum_indices_keeps_unsummed_slit() -> None:
    # Dropping slit from sum_over keeps it as an output dimension.
    einsum_str, out_str, out_dims = _build_einsum_indices(RASTER_DIMS, RESPONSE_DIMS, ("logT", "vdop"))
    assert einsum_str == "abcde,fbadg"
    assert out_str == "cdefg"
    assert out_dims == ["y", "slit", "step", "line", "detector_x_pixel"]


def test_build_einsum_indices_general_shapes() -> None:
    # Letter assignment and shared-dim reuse are independent of the science names,
    # and the produced spec is a well-formed einsum.
    einsum_str, out_str, out_dims = _build_einsum_indices(("a_dim", "shared"), ("shared", "b_dim"), ("shared",))
    assert einsum_str == "ab,bc"
    assert out_str == "ac"
    assert out_dims == ["a_dim", "b_dim"]
    result = np.einsum(f"{einsum_str}->{out_str}", np.ones((2, 3)), np.ones((3, 4)))
    assert result.shape == (2, 4)


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
        "vdem_synthesis(raster=raster, response=response, sum_over=('logT', 'vdop', 'slit'), "
        "cuda_device=None, backend=numpy)",
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


def test_vdem_synthesis_flux_matches_independent_einsum(response, vdem) -> None:
    # Independently recompute one flux value with a plain xarray multiply + sum over
    # the shared logT/vdop/slit dims, and compare to the einsum result. This
    # guards the einsum index bookkeeping, not just the output shape.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    result = vdem_synthesis(reshaped_vdem, response)

    it, istep, iline, ipixel = (int(i) for i in np.unravel_index(int(result.flux.values.argmax()), result.flux.shape))
    contribution = reshaped_vdem.vdem.isel(y=it, step=istep) * response.detector_response.isel(
        line=iline,
        detector_x_pixel=ipixel,
    )
    expected = float(contribution.sum().values)  # sums over logT, vdop, slit
    got = float(result.flux.isel(y=it, step=istep, line=iline, detector_x_pixel=ipixel).values)
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_vdem_synthesis_preserves_internal_component_kind(response, vdem) -> None:
    component_kind = ["line", "line", "contaminants", "line", "contaminants", "line", "contaminants"]
    response = response.assign_coords(component_kind=("line", component_kind))
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)

    result = vdem_synthesis(reshaped_vdem, response)

    np.testing.assert_array_equal(result.component_kind, component_kind)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_vdem_synthesis_accelerator_backend_matches_numpy(response, vdem, backend) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    numpy_flux = vdem_synthesis(reshaped_vdem, response, backend="numpy").flux
    accel_flux = vdem_synthesis(reshaped_vdem, response, backend=backend).flux

    assert isinstance(accel_flux.data, np.ndarray)
    np.testing.assert_allclose(accel_flux.values, numpy_flux.values, rtol=1e-4)


def test_vdem_synthesis_is_linear_in_vdem(response, vdem) -> None:
    # Synthesis is a tensor contraction, so scaling the VDEM scales the flux. An
    # identity/passthrough that ignored the response could not preserve this.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    base = vdem_synthesis(reshaped_vdem, response).flux
    scaled_raster = reshaped_vdem.copy(deep=True)
    scaled_raster["vdem"] = scaled_raster.vdem * 3.0
    scaled = vdem_synthesis(scaled_raster, response).flux
    np.testing.assert_allclose(scaled.values, 3.0 * base.values, rtol=1e-5)


def test_vdem_synthesis_zeroing_response_removes_only_that_line(response, vdem) -> None:
    # Zeroing one line's response must null exactly that line's flux and leave the
    # rest untouched: proof the output is driven by the response, not echoing VDEM.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    base = vdem_synthesis(reshaped_vdem, response).flux
    muted_response = response.copy(deep=True)
    muted_response.detector_response[3] = 0.0
    out = vdem_synthesis(reshaped_vdem, muted_response).flux
    assert float(base.isel(line=3).sum()) > 0.0
    assert float(out.isel(line=3).sum()) == 0.0
    np.testing.assert_array_equal(out.isel(line=0).values, base.isel(line=0).values)


def test_vdem_synthesis_doppler_shifts_line_centroid(response) -> None:
    # Emission at a single vdop must place the line at lambda * (1 + v/c): synthesis
    # encodes velocity as a wavelength shift, the core spectral behaviour.
    def centroid(vdop_kms):
        reshaped = reshape_x_to_slit_step(fake_vdem_single_vdop(vdop_kms), nslits=35, nraster=11)
        flux = vdem_synthesis(reshaped, response, sum_over=("logT", "vdop")).flux
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


def test_vdem_synthesis_rejects_unknown_sum_over_dim(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    with pytest.raises(ValueError, match=r"'bogus' is not a response dimension"):
        vdem_synthesis(reshaped_vdem, response, sum_over=("bogus",))


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
    kwargs = {"sum_over": ("logT", "vdop")} if name == "detector_wavelength" else {}

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
    detector_response = vdem_synthesis(reshaped_vdem, response, sum_over=("logT", "vdop"))
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

    detector_response = vdem_synthesis(reshaped_vdem, response_nm, sum_over=("logT", "vdop"))

    np.testing.assert_allclose(detector_response.line_wavelength.values, response.line_wavelength.values)
    np.testing.assert_allclose(detector_response.detector_wavelength.values, response.detector_wavelength.values)
    assert detector_response.line_wavelength.attrs["units"] == "Angstrom"
    assert detector_response.detector_wavelength.attrs["units"] == "Angstrom"


@pytest.mark.cuda
def test_vdem_synthesis_cuda_matches_cpu(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    cpu = vdem_synthesis(reshaped_vdem, response)
    gpu = vdem_synthesis(reshaped_vdem, response, cuda_device=0, backend="jax")
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
    kwargs = {"sum_over": ("logT", "vdop")} if name == "detector_wavelength" else {}

    with pytest.raises(ValueError, match=rf"response\.{name} must define units"):
        vdem_synthesis(reshaped_vdem, bad_response, **kwargs)
