import numpy as np
import pytest

from muse.synthesis.synthesis import _build_einsum_indices, vdem_synthesis
from muse.tests.helpers import assert_dataset_structure, fake_vdem_single_vdop
from muse.transforms.transforms import reshape_x_to_slit_step

SPEED_OF_LIGHT_KMS = 299792.458

# Dimension names for the science contraction used across the einsum-index tests.
RASTER_DIMS = ("logT", "vdop", "y", "slit", "step")
RESPONSE_DIMS = ("line", "vdop", "logT", "slit", "SG_xpixel")


def test_build_einsum_indices_contracts_shared_dims() -> None:
    # Shared dims (logT, vdop, slit) reuse the raster letters; the default sum_over
    # contracts them, leaving y/step from the raster and line/SG_xpixel from the response.
    einsum_str, out_str, out_dims = _build_einsum_indices(RASTER_DIMS, RESPONSE_DIMS, ("logT", "vdop", "slit"))
    assert einsum_str == "abcde,fbadg"
    assert out_str == "cefg"
    assert out_dims == ["y", "step", "line", "SG_xpixel"]


def test_build_einsum_indices_keeps_unsummed_slit() -> None:
    # Dropping slit from sum_over keeps it as an output dimension.
    einsum_str, out_str, out_dims = _build_einsum_indices(RASTER_DIMS, RESPONSE_DIMS, ("logT", "vdop"))
    assert einsum_str == "abcde,fbadg"
    assert out_str == "cdefg"
    assert out_dims == ["y", "slit", "step", "line", "SG_xpixel"]


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
        coords=("y", "step", "line", "SG_xpixel", "line_wvl"),
        sizes={"y": 32, "step": 11, "line": 7, "SG_xpixel": 32},
        finite_vars=("flux",),
    )
    assert detector_response.flux.attrs["units"] == "ph / s"
    assert detector_response.attrs["HISTORY"] == [
        "reshape_x_to_slit_step(ds=ds, nslits=35, nraster=11)",
        "vdem_synthesis(raster=raster, response=response, sum_over=('logT', 'vdop', 'slit'), "
        "cuda_device=None, backend=numpy)",
    ]
    np.testing.assert_array_equal(
        detector_response.line_wvl.values,
        [108.355, 108.117, 108.355, 171.073, 171.073, 284.163, 284.163],
    )


def test_vdem_synthesis_flux_matches_independent_einsum(response, vdem) -> None:
    # Independently recompute one flux value with a plain xarray multiply + sum over
    # the shared logT/vdop/slit dims, and compare to the einsum result. This
    # guards the einsum index bookkeeping, not just the output shape.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    result = vdem_synthesis(reshaped_vdem, response)

    it, istep, iline, ipixel = (int(i) for i in np.unravel_index(int(result.flux.values.argmax()), result.flux.shape))
    contribution = reshaped_vdem.vdem.isel(y=it, step=istep) * response.SG_resp.isel(line=iline, SG_xpixel=ipixel)
    expected = float(contribution.sum().values)  # sums over logT, vdop, slit
    got = float(result.flux.isel(y=it, step=istep, line=iline, SG_xpixel=ipixel).values)
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_vdem_synthesis_numpy_backend(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)

    result = vdem_synthesis(reshaped_vdem, response, backend="numpy")

    assert isinstance(result.flux.data, np.ndarray)
    assert_dataset_structure(
        result,
        data_vars=("flux",),
        coords=("y", "step", "line", "SG_xpixel", "line_wvl"),
        sizes={"y": 32, "step": 11, "line": 7, "SG_xpixel": 32},
        finite_vars=("flux",),
    )


def test_vdem_synthesis_jax_backend_matches_numpy(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    numpy_flux = vdem_synthesis(reshaped_vdem, response, backend="numpy").flux
    jax_flux = vdem_synthesis(reshaped_vdem, response, backend="jax").flux

    assert isinstance(jax_flux.data, np.ndarray)
    np.testing.assert_allclose(jax_flux.values, numpy_flux.values, rtol=1e-4)


def test_vdem_synthesis_torch_backend_matches_numpy(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    numpy_flux = vdem_synthesis(reshaped_vdem, response, backend="numpy").flux
    torch_flux = vdem_synthesis(reshaped_vdem, response, backend="torch").flux

    assert isinstance(torch_flux.data, np.ndarray)
    np.testing.assert_allclose(torch_flux.values, numpy_flux.values, rtol=1e-4)


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
    muted_response.SG_resp[3] = 0.0
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
        wavelength = flux.SG_wvl.isel(line=0, slit=17).values
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
    bad_response = response.drop_vars("SG_resp")
    with pytest.raises(ValueError, match=r"response\.SG_resp is missing"):
        vdem_synthesis(reshaped_vdem, bad_response)


def test_vdem_synthesis_rejects_invalid_unit_string(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    bad_response = response.copy(deep=True)
    bad_response.SG_resp.attrs["units"] = "not-a-real-unit"
    with pytest.raises(ValueError, match=r"response\.SG_resp units must be a valid astropy unit"):
        vdem_synthesis(reshaped_vdem, bad_response)


def test_vdem_synthesis_rejects_non_length_wavelength_units(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    bad_response = response.copy(deep=True)
    bad_response.line_wvl.attrs["units"] = "km/s"
    with pytest.raises(ValueError, match=r"response\.line_wvl units must be convertible to Angstrom"):
        vdem_synthesis(reshaped_vdem, bad_response)


@pytest.mark.parametrize("name", ["line_wvl", "SG_wvl"])
def test_vdem_synthesis_requires_response_wavelength_coords(response, vdem, name) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    bad_response = response.reset_coords(name)

    with pytest.raises(ValueError, match=rf"response\.{name} is missing"):
        vdem_synthesis(reshaped_vdem, bad_response)


def test_vdem_synthesis_keeps_slit_and_assigns_sg_wvl(response, vdem) -> None:
    # Not summing over slit leaves it as a flux dimension, which triggers the
    # SG_wvl coordinate assignment branch.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    detector_response = vdem_synthesis(reshaped_vdem, response, sum_over=("logT", "vdop"))
    assert "slit" in detector_response.flux.dims
    assert_dataset_structure(
        detector_response,
        data_vars=("flux",),
        coords=("y", "slit", "step", "line", "SG_xpixel", "line_wvl", "SG_wvl"),
        sizes={"y": 32, "slit": 35, "step": 11, "line": 7, "SG_xpixel": 32},
        finite_vars=("flux",),
    )
    assert detector_response.SG_wvl.dims == ("line", "slit", "SG_xpixel")


@pytest.mark.cuda
def test_vdem_synthesis_cuda_matches_cpu(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    cpu = vdem_synthesis(reshaped_vdem, response)
    gpu = vdem_synthesis(reshaped_vdem, response, cuda_device=0, backend="jax")
    assert_dataset_structure(
        gpu,
        data_vars=("flux",),
        coords=("y", "step", "line", "SG_xpixel", "line_wvl"),
        sizes={"y": 32, "step": 11, "line": 7, "SG_xpixel": 32},
        finite_vars=("flux",),
    )
    np.testing.assert_allclose(gpu.flux.values, cpu.flux.values, rtol=1e-5, atol=1e-6)


def test_vdem_synthesis_requires_raster_vdem_units(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    del reshaped_vdem.vdem.attrs["units"]

    with pytest.raises(ValueError, match=r"raster\.vdem must define units"):
        vdem_synthesis(reshaped_vdem, response)


@pytest.mark.parametrize("name", ["SG_resp", "line_wvl", "SG_wvl"])
def test_vdem_synthesis_requires_response_units(response, vdem, name) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    bad_response = response.copy(deep=True)
    del bad_response[name].attrs["units"]

    with pytest.raises(ValueError, match=rf"response\.{name} must define units"):
        vdem_synthesis(reshaped_vdem, bad_response)
