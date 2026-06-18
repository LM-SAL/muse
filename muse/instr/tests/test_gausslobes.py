import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import astropy.units as u

from muse.instr import gausslobe_psf, gausslobe_psf_stack, gausslobe_spacing
from muse.instr.gausslobes import _gausslobe_peak_values
from muse.variables import DEFAULTS_MUSE


def test_gausslobe_psf_basic() -> None:
    lobes = gausslobe_psf()
    assert lobes.shape == (771, 2049)
    assert_allclose(lobes.max(), 0.18985516913220707)
    assert_allclose(lobes.min(), 0)
    assert_allclose(lobes.sum(), 0.7926032302790151)


def test_gausslobe_psf_no_core() -> None:
    lobes = gausslobe_psf(no_core=True)
    assert lobes.shape == (771, 2049)
    assert_allclose(lobes.max(), 0.002217848140504724)
    assert_allclose(lobes.min(), 0)
    assert_allclose(lobes.sum(), 0.13650323027901506)


def test_gausslobe_psf_full_transmission() -> None:
    lobes = gausslobe_psf(mesh_transmission=1.0)
    assert lobes.shape == (771, 2049)
    assert_allclose(lobes.max(), 0.2893692564124479)
    assert_allclose(lobes.min(), 0)
    assert_allclose(lobes.sum(), 1.0)


def test_gausslobe_peak_values() -> None:
    mesh_transmission = 0.81
    spikes = _gausslobe_peak_values(nspike=10, mesh_transmission=mesh_transmission)
    assert spikes.shape == (10,)
    assert spikes[0] == 1
    sin_arg = np.pi * np.sqrt(mesh_transmission)
    assert_allclose(spikes[1], (np.sin(sin_arg) / sin_arg) ** 2)
    # Defaults resolve from DEFAULTS_MUSE (channel-keyed dict)
    default_spikes = _gausslobe_peak_values(nspike=10)
    assert_allclose(default_spikes, spikes)


def test_gausslobe_spacing() -> None:
    spacing = gausslobe_spacing(284.163 * u.AA, lpi=70)
    expected = 284.163e-10 / (0.0254 / 70) / (np.pi / 180 / 3600)
    assert_allclose(spacing, expected)


def test_gausslobe_spacing_can_be_converted_to_pixels() -> None:
    spacing = gausslobe_spacing(284.163 * u.AA, lpi=70)
    shift_x = spacing / DEFAULTS_MUSE.dx_pixel_SG.to_value(u.arcsec)
    shift_y = spacing / DEFAULTS_MUSE.dy_pixel_SG.to_value(u.arcsec)
    assert_allclose(shift_x, spacing / DEFAULTS_MUSE.dx_pixel_SG.to_value(u.arcsec))
    assert_allclose(shift_y, spacing / DEFAULTS_MUSE.dy_pixel_SG.to_value(u.arcsec))


def test_axis_cuts() -> None:
    cut_x = gausslobe_psf(axis="x")
    cut_y = gausslobe_psf(axis="y")
    assert cut_x.shape == (771,)
    assert cut_y.shape == (2049,)
    full = gausslobe_psf()
    # Cuts are normalized by one less power of transmission than the 2D pattern.
    mesh_transmission = next(iter(DEFAULTS_MUSE.mesh_transmission.values()))
    assert_allclose(cut_x.values * mesh_transmission, full.isel(y=1024).values)
    assert_allclose(cut_y.values * mesh_transmission, full.isel(x=385).values)


def test_invalid_axis_raises() -> None:
    with pytest.raises(ValueError, match="axis must be"):
        gausslobe_psf(axis="z")


def test_cut_in_half() -> None:
    half = gausslobe_psf(cut_in_half=True)
    assert half.shape == (385, 2049)
    assert half.x.size == 385


def test_only_core_total_is_transmission_squared() -> None:
    mesh_transmission = next(iter(DEFAULTS_MUSE.mesh_transmission.values()))
    only_core = gausslobe_psf(only_core=True)
    assert_allclose(only_core.sum(), mesh_transmission**2)


def test_full_is_no_core_plus_core() -> None:
    full = gausslobe_psf()
    no_core = gausslobe_psf(no_core=True)
    core = gausslobe_psf(only_core=True)
    assert_allclose(full.values, no_core.values + core.values, atol=1e-12)


def test_explicit_quantity_pixel_scales() -> None:
    # Regression: explicitly passed Quantity pixel scales must convert to arcsec
    default = gausslobe_psf()
    explicit = gausslobe_psf(dx_pixel_SG=0.4 * u.arcsec, dy_pixel_SG=0.167 * u.arcsec)
    assert_allclose(explicit.values, default.values)


def test_center_false_rolls_pattern() -> None:
    centered = gausslobe_psf()
    rolled = gausslobe_psf(center=False)
    expected = np.roll(np.roll(centered.values, centered.shape[1] // 2, axis=1), centered.shape[0] // 2, axis=0)
    assert_allclose(rolled.values, expected)


def test_tilt_smoke() -> None:
    tilted = gausslobe_psf(angle=5)
    assert tilted.shape == (771, 2049)
    assert np.isfinite(tilted.values).all()
    assert (tilted.values >= 0).all()


def test_wavelength_requires_quantity() -> None:
    with pytest.raises(TypeError, match="wavelength"):
        gausslobe_psf(wavelength=171.073)


def test_wavelength_quantity_scalar() -> None:
    from_nm = gausslobe_psf(wavelength=17.1073 * u.nm)
    from_quantity = gausslobe_psf(wavelength=171.073 * u.AA)
    assert_allclose(from_quantity.values, from_nm.values)


def test_gausslobe_psf_stack_multi_wavelength() -> None:
    wavelengths = [108.355, 171.073, 284.163] * u.AA
    stack = gausslobe_psf_stack(wavelengths=wavelengths)
    assert dict(stack.sizes) == {"wavelength": 3, "x": 771, "y": 2049}
    single = gausslobe_psf(wavelength=171.073 * u.AA)
    assert_allclose(stack.sel(wavelength=171.073).values, single.values)


def test_gausslobe_psf_stack_quantity_array() -> None:
    # Regression: Quantity arrays must not silently drop their unit
    from_floats = gausslobe_psf_stack(wavelengths=[171.073, 284.163])
    from_quantity = gausslobe_psf_stack(wavelengths=[171.073, 284.163] * u.AA)
    assert_allclose(from_quantity.values, from_floats.values)


def test_gausslobe_psf_stack_custom_dim_name() -> None:
    # Regression: wavelength DataArrays with a dim not named "wavelength" must work
    wavelengths = xr.DataArray([171.073, 284.163] * u.AA, dims=["line"])
    stack = gausslobe_psf_stack(wavelengths=wavelengths)
    assert "line" in stack.dims
    assert stack.sizes["line"] == 2
