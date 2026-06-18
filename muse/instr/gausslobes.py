import numpy as np
import xarray as xr

import astropy.units as u
from astropy.stats import gaussian_sigma_to_fwhm

from muse.utils.documentation import format_docstring
from muse.utils.utils import add_history
from muse.variables import DEFAULTS_MUSE

__all__ = [
    "gausslobes",
    "gausslobes_distance",
    "gausslobes_peak",
    "gausslobes_single_wavelength",
]

_AXES = {None, "x", "y"}


def _first_value(value):
    # Defaults are channel-keyed dicts; one value per call (call per wavelength).
    if isinstance(value, dict):
        return next(iter(value.values()))
    return value


def _lines_per_inch(value):
    value = _first_value(value)
    return value.to_value(1 / u.imperial.inch) if isinstance(value, u.Quantity) else float(value)


def _validate_axis(axis):
    if axis in _AXES:
        return
    msg = "axis must be None, 'x', or 'y'"
    raise ValueError(msg)


def _gaussian(axis_values, center, sigma):
    return np.exp(-((axis_values - center) ** 2) / (2 * sigma**2))


def _axis_indices(axis, nspike):
    xind = np.array([0]) if axis == "y" else np.arange(-nspike, nspike + 1)
    yind = np.array([0]) if axis == "x" else np.arange(-nspike, nspike + 1)
    return xind, yind


def _axis_lobes(axis_values, indices, spacing, sigma, spike_values):
    lobes = np.zeros_like(axis_values)
    for index in indices:
        lobes += spike_values[abs(index)] * _gaussian(axis_values, spacing * index, sigma)
    return lobes


def _aligned_mesh_psf(x, y, xind, yind, spacing, sigma_x, sigma_y, spike_values):
    gx = _axis_lobes(x, xind, spacing, sigma_x, spike_values)
    gy = _axis_lobes(y, yind, spacing, sigma_y, spike_values)
    return gx[:, None] * gy[None, :]


def _tilted_mesh_psf(x, y, xind, yind, spacing, angle, sigma_x, sigma_y, spike_values):
    xxind, yyind = np.meshgrid(xind, yind)
    xxind_flat = xxind.ravel()
    yyind_flat = yyind.ravel()
    weights = spike_values[np.abs(xxind_flat)] * spike_values[np.abs(yyind_flat)]
    xxpos = xxind_flat * spacing
    yypos = yyind_flat * spacing
    angle_rad = np.deg2rad(angle)
    rxxpos = xxpos * np.cos(angle_rad) - yypos * np.sin(angle_rad)
    ryypos = xxpos * np.sin(angle_rad) + yypos * np.cos(angle_rad)
    gx_all = _gaussian(x[None, :], rxxpos[:, None], sigma_x)
    gy_all = _gaussian(y[None, :], ryypos[:, None], sigma_y)
    return np.matmul((gx_all * weights[:, None]).T, gy_all)


@format_docstring("DEFAULTS_MUSE", mesh_transmission="mesh_transmission")
def gausslobes_peak(*, nspike=100, mesh_transmission=DEFAULTS_MUSE.mesh_transmission):
    """
    Generate the relative peak values of the side lobes to the core.

    Parameters
    ----------
    nspike : int, optional
        Number of side lobes.
        Default is 100.
    mesh_transmission : float, optional
        Transmission of the grating.
        Default is {mesh_transmission}.

    Returns
    -------
    spike_value : numpy.array
        Relative peak values of the side lobes to the core.
    """
    mesh_transmission = _first_value(mesh_transmission)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_arg = np.arange(nspike) * np.pi * np.sqrt(mesh_transmission)
        spike_value = (np.sin(sin_arg) / sin_arg) ** 2
    spike_value[0] = 1
    return spike_value


@format_docstring("DEFAULTS_MUSE", lpi="lpi")
@u.quantity_input(wave0=u.AA)
def gausslobes_distance(wave0, *, lpi=DEFAULTS_MUSE.lpi, arcsec=False):
    """
    Generate the distance between core and 1st side lobes in both x and y directions in
    pixel unit.

    Parameters
    ----------
    wave0 : `astropy.units.Quantity`
        Wavelength.
    lpi : int, optional
        Lines per inch of the grating.
        Default is {lpi}.
    arcsec : bool
        Return the spacing in arcseconds instead of per-axis pixel shifts.
        Default is False

    Returns
    -------
    spacing : float
        Distance between core and 1st side lobes in arcseconds.
        Only returned when ``arcsec`` is True.
    shift_x, shift_y : float
        Distance between core and 1st side lobes in x and y direction in pixel
        units. Returned when ``arcsec`` is False.
    """
    lpi = _lines_per_inch(lpi)
    asec = np.pi / 180 / 3600
    wave0 = wave0.to_value(u.AA)
    spacing = wave0 * 1e-10 / (0.0254 / lpi) / asec
    if arcsec:
        return spacing
    dx_pixel_SG = DEFAULTS_MUSE.dx_pixel_SG.to_value(u.arcsec)
    dy_pixel_SG = DEFAULTS_MUSE.dy_pixel_SG.to_value(u.arcsec)
    shift_x = spacing / dx_pixel_SG  # in pixel scale
    shift_y = spacing / dy_pixel_SG  # in pixel scale
    return shift_x, shift_y


@format_docstring(
    "DEFAULTS_MUSE",
    wavelength="main_lines_SG_wavelength",
    psf_fwhm_x="psf_fwhm_x",
    psf_fwhm_y="psf_fwhm_y",
    lpi="lpi",
    number_of_slits_SG="number_of_slits_SG",
    steps_per_raster_SG="steps_per_raster_SG",
    oversample_x_SG="oversample_x_SG",
    oversample_y_SG="oversample_y_SG",
    pixels_SG="pixels_SG",
    mesh_transmission="mesh_transmission",
    dx_pixel_SG="dx_pixel_SG",
    dy_pixel_SG="dy_pixel_SG",
)
@u.quantity_input(
    wavelength=u.AA,
    psf_fwhm_x=u.arcsec,
    psf_fwhm_y=u.arcsec,
    pixels_SG=u.pix,
    dx_pixel_SG=u.arcsec,
    dy_pixel_SG=u.arcsec,
)
def gausslobes_single_wavelength(
    *,
    wavelength=DEFAULTS_MUSE.main_lines_SG_wavelength["Fe XV 284.163"],
    center=True,
    no_core=False,
    only_core=False,
    psf_fwhm_x=DEFAULTS_MUSE.psf_fwhm_x,
    psf_fwhm_y=DEFAULTS_MUSE.psf_fwhm_y,
    lpi=DEFAULTS_MUSE.lpi,
    number_of_slits_SG=DEFAULTS_MUSE.number_of_slits_SG,
    steps_per_raster_SG=DEFAULTS_MUSE.steps_per_raster_SG,
    oversample_x_SG=DEFAULTS_MUSE.oversample_x_SG,
    oversample_y_SG=DEFAULTS_MUSE.oversample_y_SG,
    pixels_SG=DEFAULTS_MUSE.pixels_SG,
    spike_values=None,
    mesh_transmission=DEFAULTS_MUSE.mesh_transmission,
    dx_pixel_SG=DEFAULTS_MUSE.dx_pixel_SG,
    dy_pixel_SG=DEFAULTS_MUSE.dy_pixel_SG,
    cut_in_half=False,
    axis=None,
    angle=None,
):
    """
    Creates the Gausslobe pattern.

    By default, all value default to the MUSE specification,
    i.e., 0.4" x 0.167" pixels.

    Parameters
    ----------
    wavelength : `u.Quantity`
        Wavelength in Angstroms (or equivalent).
        Defaults to {wavelength} Angstroms.
    center : `bool`
        Option to center the pattern, by default `True`.
    no_core : `bool`
        Option to subtract out core lobe, by default `False`.
    only_core : `bool`
        Return only the core lobe (no side lobes), by default `False`.
    psf_fwhm_x : `float`
        FWHM in arcsec, by default {psf_fwhm_x}.
    psf_fwhm_y : `float`
        FWHM in arcsec, by default {psf_fwhm_y}.
    lpi : `int`
        Lines per inch of the mesh grid, by default {lpi}.
    number_of_slits_SG : `int`
        Number of slits, by default {number_of_slits_SG}.
    steps_per_raster_SG : `int`
        Number of steps in one raster, by default {steps_per_raster_SG}.
    oversample_x_SG : `int`
        Over sample factor for x pixels, by default {oversample_x_SG}.
    oversample_y_SG : `int`
        Over sample factor for y pixels, by default {oversample_y_SG}.
    pixels_SG :  `int`
        Number of along-slit pixels, i.e., y-axis size, by default {pixels_SG}.
    spike_values : `numpy.array`
        Lobe intensities below taken from FFT of quarter circle aperture with given LPI mesh.
        Note variation of lobe-to-core intensities with wavelength could be added,
        though not a big effect within  current bandpass.
        By default `None`.
    mesh_transmission : `float`
        Mesh transmission, by default {mesh_transmission}.
    dx_pixel_SG : `float`
        Pixel size in x-axis in arcsec, by default {dx_pixel_SG}.
    dy_pixel_SG : `float`
        Pixel size in y-axis in arcsec, by default {dy_pixel_SG}.
    cut_in_half : `bool`
        Return only the first half of the pattern along the x-axis, by default
        `False`.
    axis : {{'None' | 'x' | 'y'}}, optional
        Along specific direction only, by default `None`.
        Which means both x and y directions
    angle : 'float'
        Tilt angle between slit-scan coordinate and mesh grid in degree.
        By default 'None', which means 0 degree (aligned each other)

    Returns
    -------
    `xarray.DataArray`
        The Gausslobe pattern.

    References
    ----------
    This is a Python version of the original code from `Adrian.Daw <adrian.daw@nasa.gov>`__.
    IDL code version dated - 24 Oct 2022.
    New definition of core PSF. Correct the rotation. - 25 Feb 2026. K. Cho.
    """
    _validate_axis(axis)
    psf_fwhm_x = psf_fwhm_x.to_value(u.arcsec)
    psf_fwhm_y = psf_fwhm_y.to_value(u.arcsec)
    lpi = _lines_per_inch(lpi)
    pixels_SG = pixels_SG.to_value(u.pix)
    mesh_transmission = _first_value(mesh_transmission)
    dx_pixel_SG = dx_pixel_SG.to_value(u.arcsec)
    dy_pixel_SG = dy_pixel_SG.to_value(u.arcsec)
    tilt = angle not in (0, None)

    coverage = number_of_slits_SG * steps_per_raster_SG
    ny = int(2 * pixels_SG + 1)
    noy = int(ny * oversample_y_SG)
    nx = int(2 * coverage + 1)
    nox = int(nx * oversample_x_SG)

    midy = int(noy / 2)
    midx = int(nox / 2)
    sigma_x = psf_fwhm_x / gaussian_sigma_to_fwhm
    sigma_y = psf_fwhm_y / gaussian_sigma_to_fwhm

    y = dy_pixel_SG / oversample_y_SG * (np.arange(noy) - midy)
    x = dx_pixel_SG / oversample_x_SG * (np.arange(nox) - midx)

    midy_sub = int(ny / 2)
    midx_sub = int(nx / 2)
    y_sub = dy_pixel_SG * (np.arange(ny) - midy_sub)
    x_sub = dx_pixel_SG * (np.arange(nx) - midx_sub)

    spacing = gausslobes_distance(wavelength, lpi=lpi, arcsec=True)

    if spike_values is None:
        if tilt:
            nspike = int(np.sqrt((pixels_SG * dy_pixel_SG) ** 2 + (coverage * dx_pixel_SG) ** 2) / spacing)
        else:
            nspike = int(np.max([pixels_SG * dy_pixel_SG, coverage * dx_pixel_SG]) / spacing)
        spike_values = gausslobes_peak(nspike=nspike + 1, mesh_transmission=mesh_transmission)
    else:
        nspike = len(spike_values) - 1

    spike_values = np.asarray(spike_values)
    gx = spike_values[0] * _gaussian(x, 0, sigma_x)
    gy = spike_values[0] * _gaussian(y, 0, sigma_y)
    core_psf = gx[:, None] * gy[None, :]

    core_total = np.sum(core_psf)

    if only_core:
        mesh_psf = core_psf
    else:
        xind, yind = _axis_indices(axis, nspike)
        if tilt:  # mesh is not aligned with slit and scan direction
            mesh_psf = _tilted_mesh_psf(x, y, xind, yind, spacing, angle, sigma_x, sigma_y, spike_values)
        else:  # mesh is aligned with slit and scan direction
            mesh_psf = _aligned_mesh_psf(x, y, xind, yind, spacing, sigma_x, sigma_y, spike_values)
        if no_core:
            mesh_psf -= core_psf
    mesh_psf = mesh_psf.reshape(nx, oversample_x_SG, ny, oversample_y_SG).sum(axis=(1, 3))
    mesh_psf = mesh_transmission**2 * mesh_psf / core_total
    if axis is not None:
        mesh_psf /= mesh_transmission
    if not center:
        mesh_psf = np.roll(
            np.roll(mesh_psf, mesh_psf.shape[1] // 2, axis=1),
            mesh_psf.shape[0] // 2,
            axis=0,
        )
    psf = xr.DataArray(mesh_psf, coords={"x": x_sub, "y": y_sub}, dims=["x", "y"])
    psf.x.attrs["units"] = "arcsec"
    psf.y.attrs["units"] = "arcsec"
    if cut_in_half:
        psf = psf.isel(x=slice(None, nx // 2))
    add_history(psf, locals(), gausslobes_single_wavelength)

    if axis == "x":
        return psf.isel(y=midy_sub)
    if axis == "y":
        return psf.isel(x=midx_sub)
    return psf


def gausslobes(**kwargs):
    """
    Generate Gausslobe PSFs for each wavelength in an xarray.DataArray.

    Parameters
    ----------
    wavelength : xarray.DataArray, array-like, or scalar
        Wavelengths (Angstroms or equivalent). Scalars produce a single PSF.
    kwargs : dict
        Additional keyword arguments for `gausslobes_single_wavelength`.

    Returns
    -------
    psf_stack : xarray.DataArray
        Stacked PSFs with the wavelength dimension(s) plus (x, y).
    """
    wavelength = kwargs.get("wavelength", DEFAULTS_MUSE.main_lines_SG_wavelength["Fe XV 284.163"])
    if np.size(wavelength) == 1 and not isinstance(wavelength, xr.DataArray):
        return gausslobes_single_wavelength(**kwargs)

    call_kwargs = {k: v for k, v in kwargs.items() if k != "wavelength"}
    if isinstance(wavelength, xr.DataArray):
        dim, attrs = wavelength.dims[0], wavelength.attrs
    else:
        dim, attrs = "wavelength", {}
        if isinstance(wavelength, u.Quantity):
            wavelength = wavelength.to_value(u.AA)
    values = np.asarray(wavelength)  # xarray/Quantity -> AA magnitudes
    coord = xr.DataArray(values, dims=[dim], coords={dim: values}, name=dim, attrs=attrs)

    psf_list = [gausslobes_single_wavelength(wavelength=v * u.AA, **call_kwargs) for v in values]
    psf_stack = xr.concat(psf_list, dim=coord)
    add_history(psf_stack, locals(), gausslobes)
    return psf_stack
