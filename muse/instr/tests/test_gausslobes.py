import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import astropy.units as u

from muse.instr.gausslobes import (
    core_psf_gausslobes,
    gausslobes,
    gausslobes_distance,
    gausslobes_peak,
    gausslobes_single_wavelength,
)
from muse.variables import DEFAULTS_MUSE


def test_gausslobes_basic() -> None:
    lobes = gausslobes()
    assert lobes.shape == (771, 2049)
    assert_allclose(lobes.max(), 0.1898729107229633)
    assert_allclose(lobes.min(), 0)
    assert_allclose(lobes.sum(), 0.792603230279015)


def test_gausslobes_no_core() -> None:
    lobes = gausslobes(no_core=True)
    assert lobes.shape == (771, 2049)
    assert_allclose(lobes.max(), 0.0022180481987020277)
    assert_allclose(lobes.min(), 0)
    assert_allclose(lobes.sum(), 0.13650323027901506)


def test_gausslobes_full_transmission() -> None:
    lobes = gausslobes(transmission=1.0)
    assert lobes.shape == (771, 2049)
    assert_allclose(lobes.max(), 0.2893962973982065)
    assert_allclose(lobes.min(), 0)
    assert_allclose(lobes.sum(), 1.0)


def test_gausslobes_peak() -> None:
    transmission = 0.81
    spikes = gausslobes_peak(nspike=10, transmission=transmission)
    assert spikes.shape == (10,)
    assert spikes[0] == 1
    sin_arg = np.pi * np.sqrt(transmission)
    assert_allclose(spikes[1], (np.sin(sin_arg) / sin_arg) ** 2)
    # Defaults resolve from DEFAULTS_MUSE (channel-keyed dict)
    default_spikes = gausslobes_peak(nspike=10)
    assert_allclose(default_spikes, spikes)


def test_gausslobes_distance_arcsec() -> None:
    spacing = gausslobes_distance(284.163 * u.AA, lpi=70, arcsec=True)
    expected = 284.163e-10 / (0.0254 / 70) / (np.pi / 180 / 3600)
    assert_allclose(spacing, expected)


def test_gausslobes_distance_pixels() -> None:
    spacing = gausslobes_distance(284.163 * u.AA, lpi=70, arcsec=True)
    shift_x, shift_y = gausslobes_distance(284.163 * u.AA, lpi=70)
    assert_allclose(shift_x, spacing / DEFAULTS_MUSE.dx_pixel_SG.to_value(u.arcsec))
    assert_allclose(shift_y, spacing / DEFAULTS_MUSE.dy_pixel_SG.to_value(u.arcsec))


def test_gausslobes_distance_lpi_types() -> None:
    base = gausslobes_distance(284.163 * u.AA, lpi=70, arcsec=True)
    assert_allclose(gausslobes_distance(284.163 * u.AA, lpi=np.int64(70), arcsec=True), base)
    assert_allclose(gausslobes_distance(284.163 * u.AA, lpi=[70], arcsec=True), base)
    assert_allclose(gausslobes_distance(284.163 * u.AA, lpi={284: 70}, arcsec=True), base)


def test_axis_cuts() -> None:
    cut_x = gausslobes_single_wavelength(axis="x")
    cut_y = gausslobes_single_wavelength(axis="y")
    assert cut_x.shape == (771,)
    assert cut_y.shape == (2049,)
    full = gausslobes_single_wavelength()
    # Cuts are normalized by one less power of transmission than the 2D pattern.
    transmission = next(iter(DEFAULTS_MUSE.mesh_transmission.values()))
    assert_allclose(cut_x.values * transmission, full.isel(y=1024).values)
    assert_allclose(cut_y.values * transmission, full.isel(x=385).values)


def test_cut_in_half() -> None:
    half = gausslobes_single_wavelength(cut_in_half=True)
    assert half.shape == (385, 2049)
    assert half.x.size == 385


def test_only_core_matches_core_psf_wrapper() -> None:
    only_core = gausslobes_single_wavelength(only_core=True)
    core = core_psf_gausslobes()
    assert_allclose(core.values, only_core.values)
    assert_allclose(core.x.values, only_core.x.values)
    assert_allclose(core.y.values, only_core.y.values)


def test_only_core_total_is_transmission_squared() -> None:
    transmission = next(iter(DEFAULTS_MUSE.mesh_transmission.values()))
    only_core = gausslobes_single_wavelength(only_core=True)
    assert_allclose(only_core.sum(), transmission**2)


def test_full_is_no_core_plus_core() -> None:
    full = gausslobes_single_wavelength()
    no_core = gausslobes_single_wavelength(no_core=True)
    core = gausslobes_single_wavelength(only_core=True)
    assert_allclose(full.values, no_core.values + core.values, atol=1e-12)


def test_explicit_quantity_pixel_scales() -> None:
    # Regression: explicitly passed Quantity pixel scales must convert to arcsec
    default = gausslobes_single_wavelength()
    explicit = gausslobes_single_wavelength(xpix_scale=0.4 * u.arcsec, ypix_scale=0.167 * u.arcsec)
    assert_allclose(explicit.values, default.values)


def test_center_false_rolls_pattern() -> None:
    centered = gausslobes_single_wavelength()
    rolled = gausslobes_single_wavelength(center=False)
    expected = np.roll(np.roll(centered.values, centered.shape[1] // 2, axis=1), centered.shape[0] // 2, axis=0)
    assert_allclose(rolled.values, expected)


def test_tilt_smoke() -> None:
    tilted = gausslobes_single_wavelength(angle=5)
    assert tilted.shape == (771, 2049)
    assert np.isfinite(tilted.values).all()
    assert (tilted.values >= 0).all()


def test_wavelength_requires_quantity() -> None:
    with pytest.raises(TypeError, match="wavelength"):
        gausslobes_single_wavelength(wavelength=171.073)


def test_wavelength_quantity_scalar() -> None:
    from_nm = gausslobes_single_wavelength(wavelength=17.1073 * u.nm)
    from_quantity = gausslobes_single_wavelength(wavelength=171.073 * u.AA)
    assert_allclose(from_quantity.values, from_nm.values)


def test_wrapper_multi_wavelength() -> None:
    wavelengths = [108.355, 171.073, 284.163] * u.AA
    stack = gausslobes(wavelength=wavelengths)
    assert dict(stack.sizes) == {"wavelength": 3, "x": 771, "y": 2049}
    single = gausslobes_single_wavelength(wavelength=171.073 * u.AA)
    assert_allclose(stack.sel(wavelength=171.073).values, single.values)


def test_wrapper_quantity_array() -> None:
    # Regression: Quantity arrays must not silently drop their unit
    from_floats = gausslobes(wavelength=[171.073, 284.163])
    from_quantity = gausslobes(wavelength=[171.073, 284.163] * u.AA)
    assert_allclose(from_quantity.values, from_floats.values)


def test_wrapper_custom_dim_name() -> None:
    # Regression: wavelength DataArrays with a dim not named "wavelength" must work
    wavelengths = xr.DataArray([171.073, 284.163] * u.AA, dims=["line"])
    stack = gausslobes(wavelength=wavelengths)
    assert "line" in stack.dims
    assert stack.sizes["line"] == 2


def test_instrumental_width_sg_uses_shared_constant() -> None:
    width = DEFAULTS_MUSE.instrumental_width_sg
    expected = 0.0815 * u.AA / DEFAULTS_MUSE.FWHM_TO_SIGMA / DEFAULTS_MUSE.channel_spectral_order
    assert_allclose(width.values, expected.values)
    assert width.sel(channel=284).item() == 0.0815 * u.AA / DEFAULTS_MUSE.FWHM_TO_SIGMA
