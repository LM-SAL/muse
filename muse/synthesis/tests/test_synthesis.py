import numpy as np
import pytest
import torch

from muse.synthesis.synthesis import vdem_synthesis
from muse.tests.helpers import assert_dataset_structure
from muse.transforms.transforms import reshape_x_to_slit_step


def test_vdem_synthesis(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    detector_response = vdem_synthesis(reshaped_vdem, response)
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
        "vdem_synthesis(raster=raster, response=response, sum_over=('logT', 'vdop', 'slit'), cuda_device=None)",
    ]
    np.testing.assert_array_equal(
        detector_response.line_wvl.values,
        [108.355, 108.117, 108.355, 171.073, 171.073, 284.163, 284.163],
    )


def test_vdem_synthesis_flux_matches_independent_einsum(response, vdem) -> None:
    # Independently recompute one flux value with a plain xarray multiply + sum over
    # the shared logT/vdop/slit dims, and compare to the torch-einsum result. This
    # guards the einsum index bookkeeping, not just the output shape.
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    result = vdem_synthesis(reshaped_vdem, response)

    it, istep, iline, ipixel = (int(i) for i in np.unravel_index(int(result.flux.values.argmax()), result.flux.shape))
    contribution = reshaped_vdem.vdem.isel(y=it, step=istep) * response.SG_resp.isel(line=iline, SG_xpixel=ipixel)
    expected = float(contribution.sum().values)  # sums over logT, vdop, slit
    got = float(result.flux.isel(y=it, step=istep, line=iline, SG_xpixel=ipixel).values)
    np.testing.assert_allclose(got, expected, rtol=1e-5)


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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_vdem_synthesis_cuda_matches_cpu(response, vdem) -> None:
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    cpu = vdem_synthesis(reshaped_vdem, response)
    gpu = vdem_synthesis(reshaped_vdem, response, cuda_device=0)
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
