import string

import numpy as np
import numpy.typing as npt
import xarray as xr

import astropy.units as u
from astropy.constants import c as speed_of_light

from muse.log import logger
from muse.transforms.transforms import reshape_x_to_slit_step
from muse.utils.utils import add_history, require_unit

__all__ = ["calculate_moments", "create_simple_vdem", "doppler_to_wavelength", "wavelength_to_doppler"]


def create_simple_vdem(
    temperature: npt.ArrayLike,
    velocity: npt.ArrayLike,
    ne_nh: npt.ArrayLike,
    cell_length: npt.ArrayLike,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    velocity_axis: npt.ArrayLike,
    log_temperature_axis: npt.ArrayLike,
) -> xr.Dataset:
    r"""
    Calculates DEM as a function of temperature and velocity,
    x (0) and y (1) axes are horizontal z (2) vertical.
    Right hand rule.
    velocity is LOS velocity and positive is towards the observer [km/s].

    Parameters
    ----------
    temperature : numpy.ndarray
        3D array of gas temperature in K.
    velocity : numpy.ndarray
        3D array of line-of-sight velocity in km/s (positive towards the observer).
    ne_nh : numpy.ndarray
        3D array of ``n_e * n_H`` in 1/cm^6.
    cell_length : numpy.ndarray
        1D array of cell length along the line-of-sight (z) axis in cm (may be non-uniform).
    x : numpy.ndarray
        1D array of the x-axis coordinate in cm.
    y : numpy.ndarray
        1D array of the y-axis coordinate in cm.
    velocity_axis : numpy.ndarray
        1D velocity bin centers in km/s.
    log_temperature_axis : numpy.ndarray
        1D temperature bin centers in log10(K).

    Returns
    -------
    xarray.Dataset
        VDEM with dimensions of the 2D spatial axes from the simulation plus
        temperature and velocity bins.

    Raises
    ------
    ValueError
        If ``temperature`` is not 3D, if ``velocity``/``ne_nh`` do not match its
        shape, or if ``cell_length``/``x``/``y`` lengths do not match the
        corresponding axes.

    Notes
    -----
    Remove any convection zone from the box first.

    Integration is along the z axis.

    The intensity of a spectral line can be defined as :math:`I = \int n_e^2\, G(T) dl`, where
    :math:`n_e` is the electron density, :math:`G(T)` is the contribution function, and the emission
    measure is :math:`EM = \int n_e^2\, dl`. This can be defined also as
    :math:`I = \int DEM\, G(T) dT`, where DEM is the differential emission measure typically defined
    as :math:`DEM(T) = n_e^2\, dl / dT`. However, one could break the EM into finite voxels as a
    function of other variables. This is relevant when the response function depends on various
    variables. For MUSE or any spectrograph, one could build a response function which depends on the
    velocity (Doppler shift), hence it will need a VDEM which is defined as follows:

    .. math::

        VDEM = \sum_l n_e(T, v_{los})^2 \Delta l / (\Delta T\, \Delta v_{los})

    So, with a response function that includes the Doppler shift information
    (:math:`G(T,v_{los})_\lambda`), the convolution of the VDEM with this response function will be

    .. math::

        I(\lambda) = \sum_T \sum_{vlos} VDEM\, G(T,v_{los})_\lambda \Delta T\, \Delta v_{los}

    and the total intensity:

    .. math::

        I = \sum_\lambda I(\lambda) d\lambda
          = \sum_\lambda \sum_T \sum_{vlos} VDEM\, G(T,v_{los})_\lambda
            \Delta T\, \Delta v_{los}\, \Delta\lambda
          = \sum_T DEM\, G(T) \Delta T

    In principle, we could even create VDEM as a function of density

    .. math::

        VDEM = \sum_l n_e(T, v_{los})^2 / \Delta l / (\Delta T\, \Delta v_{los}\, \Delta n_e)

    or abundances.

    Working with VDEMs allows us to create a single one and synthesize any optically thin spectral
    line.
    """
    temperature = np.asarray(temperature)
    velocity = np.asarray(velocity)
    ne_nh = np.asarray(ne_nh)
    cell_length = np.asarray(cell_length)
    velocity_axis = np.asarray(velocity_axis)
    log_temperature_axis = np.asarray(log_temperature_axis)
    x = np.asarray(x)
    y = np.asarray(y)

    # Snapshot the numpy inputs for provenance before they are down-cast to float32.
    call_inputs = dict(locals())

    if temperature.ndim != 3:
        msg = f"temperature must be a 3D array, got {temperature.ndim}D"
        raise ValueError(msg)
    if velocity.shape != temperature.shape or ne_nh.shape != temperature.shape:
        msg = f"velocity {velocity.shape} and ne_nh {ne_nh.shape} must match temperature {temperature.shape}"
        raise ValueError(msg)
    if temperature.shape[2] != len(cell_length):
        msg = (
            f"cell_length ({len(cell_length)}) must match temperature along the "
            f"line-of-sight (z) axis ({temperature.shape[2]})"
        )
        raise ValueError(msg)
    if temperature.shape[0] != len(x) or temperature.shape[1] != len(y):
        msg = (
            f"x ({len(x)}) and y ({len(y)}) lengths must match the non-LOS "
            f"temperature dimensions {temperature.shape[:2]}"
        )
        raise ValueError(msg)

    cell_length = cell_length.astype(np.float32, copy=False)
    velocity, temperature, ne_nh, velocity_axis, log_temperature_axis = [
        q.astype(np.float32, copy=False) for q in [velocity, temperature, ne_nh, velocity_axis, log_temperature_axis]
    ]

    n_velocity_bins = len(velocity_axis)
    n_temperature_bins = len(log_temperature_axis)

    log_temperature_bin_width = log_temperature_axis[1] - log_temperature_axis[0]
    velocity_bin_width = velocity_axis[1] - velocity_axis[0]

    # Each line-of-sight cell spans the temperatures between it and its neighbour; its
    # emission is distributed across temperature bins by the log-T overlap (DEM = dl/dT).
    temperature_prev = np.roll(temperature, 1, axis=2)
    temperature_prev[:, :, 0] = 100.0
    max_temperature = np.maximum(temperature, temperature_prev)
    min_temperature = np.minimum(temperature, temperature_prev)

    # The VDEM array has shape [n_temperature_bins, n_velocity_bins, x, y]
    # (the line-of-sight z axis is integrated out).
    vdem = np.zeros((n_temperature_bins, n_velocity_bins, *velocity.shape[:2]), dtype=ne_nh.dtype)
    for i_temperature in range(n_temperature_bins):
        bin_lo = 10.0 ** (log_temperature_axis[i_temperature] - log_temperature_bin_width / 2.0)
        bin_hi = 10.0 ** (log_temperature_axis[i_temperature] + log_temperature_bin_width / 2.0)
        log_temperature_clipped = np.log10(np.clip(temperature, bin_lo, bin_hi))
        log_temperature_prev_clipped = np.log10(np.clip(temperature_prev, bin_lo, bin_hi))
        bin_fraction = np.abs(log_temperature_prev_clipped - log_temperature_clipped) / log_temperature_bin_width
        temperature_mask = (max_temperature >= bin_lo) & (min_temperature < bin_hi)

        for i_velocity in range(n_velocity_bins):
            voxel_mask = (
                (velocity >= velocity_axis[i_velocity] - velocity_bin_width / 2.0)
                & temperature_mask
                & (velocity < velocity_axis[i_velocity] + velocity_bin_width / 2.0)
            )
            # n_e * n_H * bin_fraction * cell_length summed along the line of sight
            los_integrand = ne_nh * bin_fraction * voxel_mask * cell_length.reshape(1, 1, -1)
            vdem[i_temperature, i_velocity, ...] = los_integrand.sum(axis=2)

    vdem_ds = xr.Dataset()
    vdem_ds["vdem"] = xr.DataArray(
        vdem[:, ::-1, :, :] / 1e27,
        dims=["logT", "vdop", "x", "y"],
        coords={
            "logT": log_temperature_axis,
            "vdop": -velocity_axis[::-1],
            "x": x,
            "y": y,
        },
        attrs={
            "description": "VDEM(logT, vdop, x, y)",
            "units": "1e27 / cm5",
        },
    )
    vdem_ds.x.attrs["long_name"] = "X"
    vdem_ds.y.attrs["long_name"] = "Y"
    vdem_ds.logT.attrs["long_name"] = r"log$_{10}$(T)"
    vdem_ds.vdop.attrs["long_name"] = r"v$_{Doppler}$"
    vdem_ds.x.attrs["units"] = "cm"
    vdem_ds.y.attrs["units"] = "cm"
    vdem_ds.logT.attrs["units"] = "dex(K)"
    vdem_ds.vdop.attrs["units"] = "km/s"
    add_history(vdem_ds, call_inputs, create_simple_vdem)
    return vdem_ds


def calculate_moments(
    spectrum: xr.Dataset,
    *,
    moment_dim: str = "SG_xpixel",
    vmax: float | None = None,
    vmask: float | None = None,
    vdop_reference: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Compute the zeroth, first, and second moments from a spectrum.

    Parameters
    ----------
    spectrum : `xarray.Dataset`
        Input spectrum. Must carry a ``dopp_vel`` coordinate (km/s); run
        `wavelength_to_doppler` first if you only have wavelengths.
    moment_dim : `str`, optional
        Spectral axis to integrate the line profile over, by default ``"SG_xpixel"``.
        The Doppler velocities used for the moments come from the ``dopp_vel``
        coordinate, which is normalized to km/s on entry.
    vmax : `float` or None, optional
        Maximum absolute velocity (km/s) to include in the integration, by default None.
    vmask : `float` or None, optional
        Half-width (in ``SG_xpixel``) of the window kept around the line peak, by default None.
        Only used together with ``vmax``.
    vdop_reference : `xarray.Dataset`, optional
        Doppler shift proxy, e.g., from the main line obtained by the
        SDC code, by default `None`.

    Returns
    -------
    `xarray.Dataset`
        Dataset containing the moments.
    """
    require_unit(spectrum, "flux", "spectrum.flux")
    dopp_unit = require_unit(spectrum, "dopp_vel", "spectrum.dopp_vel", coord_only=True, convertible_to=u.km / u.s)
    # Normalize to km/s so the raw .data used by the einsum is correct regardless of input unit.
    spectrum = spectrum.assign_coords(dopp_vel=spectrum.dopp_vel * dopp_unit.to(u.km / u.s))
    spectrum.dopp_vel.attrs["units"] = str(u.km / u.s)

    # Build the einsum spec contracting dopp_vel against flux: each dim gets one letter,
    # dopp_vel dims reuse the flux letter so they contract. moment_dim is dropped from the
    # moment output but kept in the vmax (velocity-cut) output. Masking never changes dims,
    # so this single spec serves both the vmax cut and the moment sums.
    letters = list(string.ascii_lowercase)
    dopp_letters = {dim: letters[i] for i, dim in enumerate(spectrum.dopp_vel.dims)}
    n_dopp = len(dopp_letters)
    dopp_spec = "".join(dopp_letters.values())
    flux_spec = ""
    moment_out = ""
    vmax_out = ""
    for i_flux, flux_dim in enumerate(spectrum.flux.dims):
        letter = dopp_letters.get(flux_dim, letters[n_dopp + i_flux])
        flux_spec += letter
        vmax_out += letter
        if flux_dim != moment_dim:
            moment_out += letter
    einsum_str = f"{dopp_spec},{flux_spec}"
    logger.debug(f"{einsum_str}->{moment_out}")
    if vmax is not None and vdop_reference is not None:
        velocity = spectrum["dopp_vel"]
        first_moment_proxy = reshape_x_to_slit_step(
            vdop_reference["SDC main, 1st mom"].sel(line=["Fe XIX", "Fe IX", "Fe XV"])
        )
        velocity_mask = xr.where(
            np.abs(velocity - first_moment_proxy) > velocity.differentiate("SG_xpixel") * vmask, 0.0, 1.0
        )
        velocity_mask = velocity_mask.where(np.abs(velocity) < vmax, 0.0 * velocity)
        masked_spectrum = spectrum.assign(flux=velocity_mask * spectrum.flux)
    elif vmax is not None:
        velocity = spectrum["dopp_vel"]
        velocity_mask = xr.where(np.abs(velocity) > vmax, 0.0 * velocity, 1.0 + 0.0 * velocity)
        masked_spectrum = spectrum.assign(
            flux=xr.DataArray(
                np.einsum(f"{einsum_str}->{vmax_out}", velocity_mask, spectrum.flux),
                dims=spectrum.flux.dims,
            )
        )
        if vmask is not None:
            peak_xpixel = masked_spectrum.flux.argmax(dim=["SG_xpixel"])["SG_xpixel"].expand_dims(
                {"SG_xpixel": np.size(masked_spectrum["SG_xpixel"].to_numpy())}
            )
            masked_spectrum = masked_spectrum.assign(
                flux=masked_spectrum.flux.where(np.abs(peak_xpixel - masked_spectrum.coords["SG_xpixel"]) < vmask, 0)
            )
    else:
        masked_spectrum = spectrum
    masked_spectrum = masked_spectrum.assign(flux=masked_spectrum.flux.where(masked_spectrum.flux > 0, 0))
    zeroth = masked_spectrum.flux.sum(dim=moment_dim)
    velocity_data = masked_spectrum.dopp_vel.data
    first = np.einsum(f"{einsum_str}->{moment_out}", velocity_data, masked_spectrum.flux) / zeroth
    # Note that int(I (u-I1)^2 du)/I0 = (int(I u^2 du))/I0 - I1^2
    second = np.sqrt(
        np.einsum(f"{einsum_str}->{moment_out}", velocity_data**2, masked_spectrum.flux) / zeroth - first**2,
    )
    # zeroth/first/second are already DataArrays carrying the non-moment dims and coords.
    moments = xr.Dataset()
    moments["0th"] = zeroth
    moments["1st"] = first
    moments["2nd"] = second
    moments.attrs = dict(spectrum.attrs)
    moments["0th"].attrs = dict(masked_spectrum.flux.attrs)
    moments["1st"].attrs["units"] = str(u.km / u.s)
    moments["2nd"].attrs["units"] = str(u.km / u.s)
    add_history(moments, locals(), calculate_moments)
    return moments


def wavelength_to_doppler(response: xr.Dataset) -> xr.Dataset:
    """
    Add a Doppler-shift coordinate in km/s derived from wavelengths.

    Parameters
    ----------
    response : `xarray.Dataset`
        Must include ``SG_wvl`` and ``line_wvl`` coordinates.

    Returns
    -------
    `xarray.Dataset`
        A new dataset with an added ``dopp_vel`` coordinate in km/s.
    """
    sg_unit = require_unit(response, "SG_wvl", "response.SG_wvl", coord_only=True, convertible_to=u.AA)
    line_unit = require_unit(response, "line_wvl", "response.line_wvl", coord_only=True, convertible_to=u.AA)
    c_kms = speed_of_light.to_value(u.km / u.s)
    sg_wvl = response.coords["SG_wvl"] * sg_unit.to(u.AA)
    line_wvl = response.coords["line_wvl"] * line_unit.to(u.AA)
    dopp_vel = (sg_wvl / line_wvl - 1) * c_kms
    dopp_vel.attrs["units"] = str(u.km / u.s)
    response = response.assign_coords(dopp_vel=dopp_vel)
    add_history(response, locals(), wavelength_to_doppler)
    return response


def doppler_to_wavelength(response: xr.Dataset) -> xr.Dataset:
    """
    Add a wavelength coordinate in Angstrom derived from a Doppler shift.

    Parameters
    ----------
    response : `xarray.Dataset`
        Must include ``dopp_vel`` and ``line_wvl`` coordinates.

    Returns
    -------
    `xarray.Dataset`
        A new dataset with an added ``SG_wvl`` coordinate in Angstrom.
    """
    dopp_unit = require_unit(response, "dopp_vel", "response.dopp_vel", coord_only=True, convertible_to=u.km / u.s)
    line_unit = require_unit(response, "line_wvl", "response.line_wvl", coord_only=True, convertible_to=u.AA)
    c_kms = speed_of_light.to_value(u.km / u.s)
    line_wvl = response.coords["line_wvl"] * line_unit.to(u.AA)
    dopp_vel = response.coords["dopp_vel"] * dopp_unit.to(u.km / u.s)
    sg_wvl = line_wvl * (1 + dopp_vel / c_kms)
    sg_wvl.attrs["units"] = str(u.AA)
    response = response.assign_coords(SG_wvl=sg_wvl)
    add_history(response, locals(), doppler_to_wavelength)
    return response
