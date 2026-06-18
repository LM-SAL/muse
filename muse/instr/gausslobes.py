import numpy as np
import xarray as xr

import astropy.units as u

from muse.utils.documentation import format_docstring
from muse.utils.utils import add_history
from muse.variables import DEFAULTS_MUSE

__all__ = [
    "core_psf_gausslobes",
    "gausslobes",
    "gausslobes_distance",
    "gausslobes_peak",
    "gausslobes_single_wavelength",
]


def _first_value(value):
    if isinstance(value, dict):
        return next(iter(value.values()))
    if isinstance(value, list | tuple):
        return value[0]
    return value


def _lines_per_inch(value):
    value = _first_value(value)
    if isinstance(value, u.Quantity):
        return value.to_value(1 / u.imperial.inch)
    return float(np.ravel(value)[0]) if np.ndim(value) else float(value)


@format_docstring("DEFAULTS_MUSE", transmission="mesh_transmission")
def gausslobes_peak(*, nspike=100, transmission=DEFAULTS_MUSE.mesh_transmission):
    """
    Generate the relative peak values of the side lobes to the core.

    Parameters
    ----------
    nspike : int, optional
        Number of side lobes.
        Default is 100.
    transmission : float, optional
        Transmission of the grating.
        Default is {transmission}.

    Returns
    -------
    spike_value : numpy.array
        Relative peak values of the side lobes to the core.
    """
    transmission = _first_value(transmission)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_arg = np.arange(nspike) * np.pi * np.sqrt(transmission)
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
    xpix_scale = DEFAULTS_MUSE.dx_pixel_SG.to_value(u.arcsec)
    ypix_scale = DEFAULTS_MUSE.dy_pixel_SG.to_value(u.arcsec)
    shift_x = spacing / xpix_scale  # in pixel scale
    shift_y = spacing / ypix_scale  # in pixel scale
    return shift_x, shift_y


@format_docstring(
    "DEFAULTS_MUSE",
    wavelength="main_lines_SG_wavelength",
    fwhm_x="psf_fwhm_x",
    fwhm_y="psf_fwhm_y",
    lpi="lpi",
    nslits="number_of_slits_SG",
    nsteps="steps_per_raster_SG",
    oversample_x="oversample_x_SG",
    oversample_y="oversample_y_SG",
    slit_dim="pixels_SG",
    transmission="mesh_transmission",
    xpix_scale="dx_pixel_SG",
    ypix_scale="dy_pixel_SG",
)
@u.quantity_input(wavelength=u.AA, fwhm_x=u.arcsec, fwhm_y=u.arcsec, xpix_scale=u.arcsec, ypix_scale=u.arcsec)
def gausslobes_single_wavelength(
    *,
    wavelength=DEFAULTS_MUSE.main_lines_SG_wavelength["Fe XV 284.163"],
    center=True,
    no_core=False,
    only_core=False,
    fwhm_x=DEFAULTS_MUSE.psf_fwhm_x,
    fwhm_y=DEFAULTS_MUSE.psf_fwhm_y,
    lpi=DEFAULTS_MUSE.lpi,
    nslits=DEFAULTS_MUSE.number_of_slits_SG,
    nsteps=DEFAULTS_MUSE.steps_per_raster_SG,
    oversample_x=DEFAULTS_MUSE.oversample_x_SG,
    oversample_y=DEFAULTS_MUSE.oversample_y_SG,
    slit_dim=DEFAULTS_MUSE.pixels_SG,
    spike_values=None,
    transmission=DEFAULTS_MUSE.mesh_transmission,
    xpix_scale=DEFAULTS_MUSE.dx_pixel_SG,
    ypix_scale=DEFAULTS_MUSE.dy_pixel_SG,
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
    fwhm_x : `float`
        FWHM in arcsec, by default {fwhm_x}.
    fwhm_y : `float`
        FWHM in arcsec, by default {fwhm_y}.
    lpi : `int`
        Lines per inch of the mesh grid, by default {lpi}.
    nslits : `int`
        Number of slits, by default {nslits}.
    nsteps : `int`
        Number of steps in one raster, by default {nsteps}.
    oversample_x : `int`
        Over sample factor for x pixels, by default {oversample_x}.
    oversample_y : `int`
        Over sample factor for y pixels, by default {oversample_y}.
    slit_dim :  `int`
        Number of along-slit pixels, i.e., y-axis size, by default {slit_dim}.
    spike_values : `numpy.array`
        Lobe intensities below taken from FFT of quarter circle aperture with given LPI mesh.
        Note variation of lobe-to-core intensities with wavelength could be added,
        though not a big effect within  current bandpass.
        By default `None`.
    transmission : `float`
        Mesh transmission, by default {transmission}.
    xpix_scale : `float`
        Pixel size in x-axis in arcsec, by default {xpix_scale}.
    ypix_scale : `float`
        Pixel size in y-axis in arcsec, by default {ypix_scale}.
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
    if isinstance(fwhm_x, u.Quantity):
        fwhm_x = fwhm_x.to_value(u.arcsec)
    if isinstance(fwhm_y, u.Quantity):
        fwhm_y = fwhm_y.to_value(u.arcsec)
    lpi = _lines_per_inch(lpi)
    if isinstance(slit_dim, u.Quantity):
        slit_dim = slit_dim.to_value(u.pix)
    transmission = _first_value(transmission)
    if isinstance(xpix_scale, u.Quantity):
        xpix_scale = xpix_scale.to_value(u.arcsec)
    if isinstance(ypix_scale, u.Quantity):
        ypix_scale = ypix_scale.to_value(u.arcsec)
    tilt = angle not in (0, None)

    coverage = nslits * nsteps
    ny = int(2 * slit_dim + 1)
    noy = int(ny * oversample_y)
    nx = int(2 * coverage + 1)
    nox = int(nx * oversample_x)

    midy = int(noy / 2)
    midx = int(nox / 2)
    sigma_x = fwhm_x / DEFAULTS_MUSE.FWHM_TO_SIGMA
    sigma_y = fwhm_y / DEFAULTS_MUSE.FWHM_TO_SIGMA

    y = ypix_scale / oversample_y * (np.arange(noy) - midy)
    x = xpix_scale / oversample_x * (np.arange(nox) - midx)

    midy_sub = int(ny / 2)
    midx_sub = int(nx / 2)
    y_sub = ypix_scale * (np.arange(ny) - midy_sub)
    x_sub = xpix_scale * (np.arange(nx) - midx_sub)

    spacing = gausslobes_distance(wavelength, lpi=lpi, arcsec=True)

    if spike_values is None:
        if tilt:
            nspike = int(np.sqrt((slit_dim * ypix_scale) ** 2 + (coverage * xpix_scale) ** 2) / spacing)
        else:
            nspike = int(np.max([slit_dim * ypix_scale, coverage * xpix_scale]) / spacing)
        spike_values = gausslobes_peak(nspike=nspike + 1, transmission=transmission)
    else:
        nspike = len(spike_values) - 1

    gx = spike_values[0] * np.exp(-(x**2 / (2 * sigma_x**2)))
    gy = spike_values[0] * np.exp(-(y**2 / (2 * sigma_y**2)))
    core_psf = gx[:, None] * gy[None, :]

    core_total = np.sum(core_psf)

    if only_core:
        mesh_psf = core_psf
    else:
        xind = np.array([0]) if axis == "y" else np.arange(-nspike, nspike + 1)
        yind = np.array([0]) if axis == "x" else np.arange(-nspike, nspike + 1)
        if tilt:  # mesh is not aligned with slit and scan direction
            xxind, yyind = np.meshgrid(xind, yind)
            xxind_flat = xxind.ravel()
            yyind_flat = yyind.ravel()
            weights = spike_values[np.abs(xxind_flat)] * spike_values[np.abs(yyind_flat)]
            xxpos = xxind_flat * spacing
            yypos = yyind_flat * spacing
            angle_rad = np.deg2rad(angle)
            rxxpos = xxpos * np.cos(angle_rad) - yypos * np.sin(angle_rad)
            ryypos = xxpos * np.sin(angle_rad) + yypos * np.cos(angle_rad)
            gx_all = np.exp(-((x[None, :] - rxxpos[:, None]) ** 2) / (2 * sigma_x**2))
            gy_all = np.exp(-((y[None, :] - ryypos[:, None]) ** 2) / (2 * sigma_y**2))
            gx_weighted = gx_all * weights[:, None]
            mesh_psf = np.matmul(gx_weighted.T, gy_all)
        else:  # mesh is aligned with slit and scan direction
            gx = np.zeros(int(nox))
            gy = np.zeros(int(noy))
            for i in xind:
                gx += spike_values[abs(i)] * np.exp(-((x - spacing * i) ** 2 / (2 * sigma_x**2)))
            for j in yind:
                gy += spike_values[abs(j)] * np.exp(-((y - spacing * j) ** 2 / (2 * sigma_y**2)))
            mesh_psf = gx[:, None] * gy[None, :]
        if no_core:
            mesh_psf -= core_psf
    mesh_psf = mesh_psf.reshape(nx, oversample_x, ny, oversample_y).sum(axis=(1, 3))
    mesh_psf = transmission**2 * mesh_psf / core_total
    if axis is not None and np.size(axis) < 2:
        mesh_psf /= transmission
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

    if np.all(np.squeeze(axis) == "x"):
        return psf.isel(y=midy_sub)
    if np.all(np.squeeze(axis) == "y"):
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
    if not isinstance(wavelength, xr.DataArray):
        if np.size(wavelength) == 1:
            return gausslobes_single_wavelength(**kwargs)
        if isinstance(wavelength, u.Quantity):
            wavelength = wavelength.to_value(u.AA)
        wavelength = xr.DataArray(wavelength, dims=["wavelength"], coords={"wavelength": wavelength})

    psf_list = []
    call_kwargs = dict(kwargs)
    call_kwargs.pop("wavelength", None)

    # Flatten wavelength while preserving original dims structure
    flat_wvl = wavelength.stack(flat_dim=wavelength.dims)
    for wvl_val in flat_wvl.values:
        if not isinstance(wvl_val, u.Quantity):
            wvl_val = wvl_val * u.AA
        psf = gausslobes_single_wavelength(wavelength=wvl_val, **call_kwargs)
        psf_list.append(psf)

    # Stack along flattened dimension
    psf_stack = xr.concat(psf_list, dim="flat_dim")
    # Restore original multi-index coordinates
    psf_stack = psf_stack.assign_coords(flat_dim=flat_wvl.coords["flat_dim"])
    # Unstack to restore original wavelength dimensions
    psf_stack = psf_stack.unstack("flat_dim")

    if "wavelength" in psf_stack.coords:
        psf_stack.wavelength.attrs.update(wavelength.attrs)

    add_history(psf_stack, locals(), gausslobes)

    return psf_stack


def core_psf_gausslobes(**kwargs):
    """
    Creates the core (no side lobes) Gausslobe pattern.

    Thin wrapper around `gausslobes_single_wavelength` with ``only_core=True``;
    see that function for the accepted parameters.

    Returns
    -------
    `xarray.DataArray`
        The core Gausslobe pattern.
    """
    return gausslobes_single_wavelength(only_core=True, **kwargs)
