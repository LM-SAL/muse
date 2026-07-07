from venv import logger

import numpy as np
import numpy.typing as npt
import xarray as xr

import astropy.units as u
from astropy.constants import c as speed_of_light

from muse.utils.utils import add_history, coord_as_unit, require_unit

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
    n_x_chunks: int = 1,
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
    n_x_chunks : int, optional
        Split the x axis into this many contiguous blocks and process them one at a time,
        by default 1. The line-of-sight integration is independent per (x, y) column, so this
        is exact (identical result to ``n_x_chunks=1``), just trading a few extra cheap passes
        for a smaller peak memory footprint. Clamped down to ``len(x)`` if larger.

    Returns
    -------
    xarray.Dataset
        VDEM with dimensions of the 2D spatial axes from the simulation plus
        temperature and velocity bins.

    Raises
    ------
    ValueError
        If ``temperature`` is not 3D, if ``velocity``/``ne_nh`` do not match its
        shape, if ``cell_length``/``x``/``y`` lengths do not match the
        corresponding axes, or if ``temperature``/``velocity``/``ne_nh`` contain
        non-finite values.

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
    for name, cube in (("temperature", temperature), ("velocity", velocity), ("ne_nh", ne_nh)):
        if not np.isfinite(cube).all():
            msg = f"{name} contains non-finite values (NaN or inf), this will cause issues during synthesis"
            logger.warning(msg)
    n_x_chunks = min(n_x_chunks, len(x))

    n_velocity_bins = len(velocity_axis)
    n_temperature_bins = len(log_temperature_axis)
    n_y = len(y)
    log_temperature_bin_width = log_temperature_axis[1] - log_temperature_axis[0]
    velocity_bin_width = velocity_axis[1] - velocity_axis[0]
    # Contiguous half-open velocity bins [center - dv/2, center + dv/2): a voxel sitting exactly
    # on an edge lands in the upper bin, matching the (>= bin_lo) & (< bin_hi) temperature convention below.
    velocity_edges = np.append(velocity_axis - velocity_bin_width / 2.0, velocity_axis[-1] + velocity_bin_width / 2.0)
    cell_length_los = cell_length.reshape(1, 1, -1)

    # The line-of-sight integration is independent per (x, y) column, so processing contiguous
    # x-blocks one at a time and concatenating is exact; n_x_chunks only lowers peak memory.
    blocks = []
    for x_block in np.array_split(np.arange(len(x)), n_x_chunks):
        n_x_block = len(x_block)
        block = slice(x_block[0], x_block[-1] + 1)
        # Input dtypes are kept as-is; the vdem output dtype follows ne_nh, and the LOS
        # accumulation below happens in float64 regardless (np.bincount always sums in float64).
        ne_nh_block = ne_nh[block] / 1e27  # normalize to the 1e27 / cm^5 output units
        temperature_block = temperature[block]
        # Each line-of-sight cell spans the temperatures between it and its neighbour; its
        # emission is distributed across temperature bins by the log-T overlap (DEM = dl/dT).
        temperature_prev = np.roll(temperature_block, 1, axis=2)
        temperature_prev[:, :, 0] = 100.0
        max_temperature = np.maximum(temperature_block, temperature_prev)
        min_temperature = np.minimum(temperature_block, temperature_prev)
        # Every voxel falls in exactly one velocity bin, so scatter each voxel onto a flat
        # (velocity_bin, x, y) index; that index has no z axis, so np.bincount's accumulation
        # is the line-of-sight sum. Out-of-range voxels get invalid indices but are masked out below.
        velocity_bin = np.searchsorted(velocity_edges, velocity[block], side="right") - 1
        in_velocity_range = (velocity_bin >= 0) & (velocity_bin < n_velocity_bins)
        scatter_index = velocity_bin * (n_x_block * n_y) + np.arange(n_x_block * n_y).reshape(n_x_block, n_y, 1)

        vdem_block = np.zeros((n_temperature_bins, n_velocity_bins, n_x_block, n_y), dtype=ne_nh_block.dtype)
        for i_temperature in range(n_temperature_bins):
            bin_lo = 10.0 ** (log_temperature_axis[i_temperature] - log_temperature_bin_width / 2.0)
            bin_hi = 10.0 ** (log_temperature_axis[i_temperature] + log_temperature_bin_width / 2.0)
            log_temperature_clipped = np.log10(np.clip(temperature_block, bin_lo, bin_hi))
            log_temperature_prev_clipped = np.log10(np.clip(temperature_prev, bin_lo, bin_hi))
            bin_fraction = np.abs(log_temperature_prev_clipped - log_temperature_clipped) / log_temperature_bin_width
            voxel_mask = in_velocity_range & (max_temperature >= bin_lo) & (min_temperature < bin_hi)
            # n_e * n_H * bin_fraction * cell_length, scattered into its velocity bin and summed along z.
            los_integrand = ne_nh_block * bin_fraction * cell_length_los
            vdem_block[i_temperature] = np.bincount(
                scatter_index[voxel_mask],
                weights=los_integrand[voxel_mask],
                minlength=n_velocity_bins * n_x_block * n_y,
            ).reshape(n_velocity_bins, n_x_block, n_y)
        blocks.append(vdem_block)

    vdem_ds = xr.Dataset()
    vdem_ds["vdem"] = xr.DataArray(
        np.concatenate(blocks, axis=2)[:, ::-1],
        dims=["logT", "vdop", "x", "y"],
        coords={
            "logT": log_temperature_axis,
            "vdop": -velocity_axis[::-1],
            "x": x,
            "y": y,
        },
    )
    vdem_ds["vdem"].attrs = {
        "description": "VDEM(logT, vdop, x, y)",
        "units": "1e27 / cm5",
    }
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
    integration_name: str = "flux",
    doppler_name: str = "dopp_vel",
    vmax: float | None = None,
    vmask: float | None = None,
) -> xr.Dataset:
    """
    Compute the zeroth, first, and second moments from a spectrum.

    Parameters
    ----------
    spectrum : `xarray.Dataset`
        Input spectrum. Must carry a Doppler-velocity coordinate (km/s); run
        `wavelength_to_doppler` first if you only have wavelengths.
    moment_dim : `str`, optional
        Spectral axis to integrate the line profile over, by default ``"SG_xpixel"``.
        The Doppler velocities used for the moments come from the ``doppler_name``
        coordinate, which is normalized to km/s on entry.
    integration_name : `str`, optional
        Name of the variable to integrate over ``spectrum``, by default ``"flux"``.
    doppler_name : `str`, optional
        Name of the Doppler-velocity coordinate in ``spectrum``, by default ``"dopp_vel"``.
    vmax : `float` or None, optional
        Maximum absolute velocity (km/s) to include in the integration, by default None.
    vmask : `float` or None, optional
        Half-width (in ``SG_xpixel``) of the window kept around the line peak, by default None.
        Only used together with ``vmax``.

    Returns
    -------
    `xarray.Dataset`
        Dataset containing the moments.
    """
    require_unit(spectrum, integration_name, f"spectrum.{integration_name}")
    if doppler_name not in spectrum.coords:
        msg = f"spectrum is missing the {doppler_name!r} coordinate; run wavelength_to_doppler first to add it."
        raise ValueError(msg)
    dopp_unit = require_unit(
        spectrum, doppler_name, f"spectrum.{doppler_name}", coord_only=True, convertible_to=u.km / u.s
    )
    # Normalize to km/s so the raw .data used by the einsum is correct regardless of input unit.
    spectrum = spectrum.assign_coords({doppler_name: spectrum[doppler_name] * dopp_unit.to(u.km / u.s)})
    spectrum[doppler_name].attrs["units"] = str(u.km / u.s)

    if vmax is not None:
        velocity = spectrum[doppler_name]
        velocity_mask = xr.where(np.abs(velocity) > vmax, 0.0, 1.0)
        masked_flux = (spectrum[integration_name] * velocity_mask).transpose(*spectrum[integration_name].dims)
        masked_spectrum = spectrum.assign({integration_name: masked_flux})
        if vmask is not None:
            peak_index = masked_spectrum[integration_name].argmax(dim=moment_dim)
            peak_coord = masked_spectrum[moment_dim].isel({moment_dim: peak_index})
            distance = np.abs(masked_spectrum[moment_dim] - peak_coord)
            masked_spectrum = masked_spectrum.assign(
                {integration_name: masked_spectrum[integration_name].where(distance < vmask, 0)}
            )
    else:
        masked_spectrum = spectrum
    masked_spectrum = masked_spectrum.assign(
        {
            integration_name: masked_spectrum[integration_name]
            .where(masked_spectrum[integration_name] > 0, 0)
            .assign_attrs(spectrum[integration_name].attrs)
        }
    )
    zeroth = masked_spectrum[integration_name].sum(dim=moment_dim)
    velocity = masked_spectrum[doppler_name]
    # Pixels with no flux (e.g. fully masked) would divide by zero; leave them NaN, not inf.
    safe_zeroth = zeroth.where(zeroth > 0)
    first = (masked_spectrum[integration_name] * velocity).sum(dim=moment_dim) / safe_zeroth
    # Note that int(I (u-I1)^2 du)/I0 = (int(I u^2 du))/I0 - I1^2
    variance = (masked_spectrum[integration_name] * velocity**2).sum(dim=moment_dim) / safe_zeroth - first**2
    second = np.sqrt(variance.clip(min=0))
    # zeroth/first/second are already DataArrays carrying the non-moment dims and coords.
    moments = xr.Dataset({"0th": zeroth, "1st": first, "2nd": second}, attrs=dict(spectrum.attrs))
    moments["0th"].attrs = dict(masked_spectrum[integration_name].attrs)
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
    c_kms = speed_of_light.to_value(u.km / u.s)
    sg_wvl = coord_as_unit(response, "SG_wvl", u.AA, "response.SG_wvl")
    line_wvl = coord_as_unit(response, "line_wvl", u.AA, "response.line_wvl")
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
    c_kms = speed_of_light.to_value(u.km / u.s)
    line_wvl = coord_as_unit(response, "line_wvl", u.AA, "response.line_wvl")
    dopp_vel = coord_as_unit(response, "dopp_vel", u.km / u.s, "response.dopp_vel")
    sg_wvl = line_wvl * (1 + dopp_vel / c_kms)
    sg_wvl.attrs["units"] = str(u.AA)
    response = response.assign_coords(SG_wvl=sg_wvl)
    add_history(response, locals(), doppler_to_wavelength)
    return response
