from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
import xarray as xr

import astropy.units as u
from astropy.constants import c as speed_of_light

from muse.log import logger
from muse.utils.utils import add_history, coord_as_unit, require_unit, update_attrs

__all__ = ["calculate_moments", "create_simple_vdem", "doppler_to_wavelength", "wavelength_to_doppler"]

_VDEM_X_BLOCK_SIZE = 32
# Deliberate fixed cap, not os.cpu_count(): it bounds peak memory (block temporaries
# scale with worker count) and this runs nested under gallery/CI parallelism on
# 2-4 vCPU builders, where a cpu-derived pool would oversubscribe.
_VDEM_MAX_WORKERS = 4


def create_simple_vdem(
    temperature: npt.ArrayLike,
    velocity: npt.ArrayLike,
    ne_nh: npt.ArrayLike,
    cell_length: npt.ArrayLike,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    velocity_axis: npt.ArrayLike,
    log_temperature_axis: npt.ArrayLike,
    integration_axis: int = 2,
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
        3D array of the velocity component along ``integration_axis`` in km/s
        (positive towards the observer).
    ne_nh : numpy.ndarray
        3D array of ``n_e * n_H`` in 1/cm^6.
    cell_length : numpy.ndarray
        1D array of cell length along the line-of-sight (``integration_axis``) axis in cm
        (may be non-uniform).
    x : numpy.ndarray
        1D array of the coordinate of the first non-LOS axis in cm.
    y : numpy.ndarray
        1D array of the coordinate of the second non-LOS axis in cm.
    velocity_axis : numpy.ndarray
        1D velocity bin centers in km/s.
    log_temperature_axis : numpy.ndarray
        1D temperature bin centers in log10(K).
    integration_axis : int, optional
        Axis of ``temperature``/``velocity``/``ne_nh`` to integrate along, by default 2.
        ``x`` and ``y`` label the two remaining axes in their original order, and the
        integration enters the box at index 0 of this axis.

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
        corresponding axes. Non-finite values in ``temperature``/``velocity``/``ne_nh``
        only log a warning; the affected voxels can propagate NaN into the output.

    Notes
    -----
    Remove any convection zone from the box first.

    Integration is along ``integration_axis`` (the last axis by default).

    Intermediate arrays are processed in x blocks (a few worker threads at a time) to
    bound peak memory. The returned VDEM is still allocated eagerly in full.

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

        VDEM = \sum_l n_e(T, v_{los}, n_e)^2\, \Delta l / (\Delta T\, \Delta v_{los}\, \Delta n_e)

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
    # Views only; the rest of the function integrates over the last axis.
    temperature, velocity, ne_nh = (np.moveaxis(q, integration_axis, -1) for q in (temperature, velocity, ne_nh))
    if velocity.shape != temperature.shape or ne_nh.shape != temperature.shape:
        msg = f"velocity {velocity.shape} and ne_nh {ne_nh.shape} must match temperature {temperature.shape}"
        raise ValueError(msg)
    if temperature.shape[2] != len(cell_length):
        msg = (
            f"cell_length ({len(cell_length)}) must match temperature along the "
            f"line-of-sight axis {integration_axis} ({temperature.shape[2]})"
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

    n_velocity_bins = len(velocity_axis)
    n_temperature_bins = len(log_temperature_axis)
    n_x = len(x)
    n_y = len(y)
    log_temperature_bin_width = log_temperature_axis[1] - log_temperature_axis[0]
    velocity_bin_width = velocity_axis[1] - velocity_axis[0]
    # Contiguous half-open velocity bins [center - dv/2, center + dv/2): a voxel sitting exactly
    # on an edge lands in the upper bin, matching the half-open temperature-bin convention below.
    velocity_edges = np.append(velocity_axis - velocity_bin_width / 2.0, velocity_axis[-1] + velocity_bin_width / 2.0)
    temperature_lower_edge = log_temperature_axis[0] - log_temperature_bin_width / 2.0
    temperature_upper_edge = log_temperature_axis[-1] + log_temperature_bin_width / 2.0
    cell_length_los = cell_length.reshape(1, 1, -1)

    # Only the output scales with n_x; full-cube intermediates stay bounded by the block
    # size times the number of worker threads.
    vdem_dtype = (np.empty(0, dtype=ne_nh.dtype) / 1e27).dtype
    vdem = np.zeros((n_temperature_bins, n_velocity_bins, n_x, n_y), dtype=vdem_dtype)

    def process_block(block_start: int) -> None:
        block = slice(block_start, min(block_start + _VDEM_X_BLOCK_SIZE, n_x))
        n_x_block = block.stop - block.start
        ne_nh_block = ne_nh[block] / 1e27  # normalize to the 1e27 / cm^5 output units
        # n_e * n_H * cell_length is the emission each voxel spreads over temperature bins.
        emission_block = ne_nh_block * cell_length_los
        temperature_block = temperature[block]
        # Each line-of-sight cell spans the temperatures between it and its neighbour; its
        # emission is distributed across temperature bins by the log-T overlap (DEM = dl/dT).
        log_temperature_block = np.log10(temperature_block)
        log_temperature_prev = np.roll(log_temperature_block, 1, axis=2)
        log_temperature_prev[:, :, 0] = 2.0  # boundary cell is entered from 100 K
        max_log_temperature = np.maximum(log_temperature_block, log_temperature_prev)
        min_log_temperature = np.minimum(log_temperature_block, log_temperature_prev)
        # Every voxel falls in exactly one velocity bin, so scatter each voxel onto a flat
        # (velocity_bin, x, y) index; that index has no z axis, so np.bincount's accumulation
        # is the line-of-sight sum.
        velocity_bin = np.searchsorted(velocity_edges, velocity[block], side="right") - 1
        voxel_mask = (
            (velocity_bin >= 0)
            & (velocity_bin < n_velocity_bins)
            & (max_log_temperature >= temperature_lower_edge)
            & (min_log_temperature < temperature_upper_edge)
        )
        scatter_index = velocity_bin * (n_x_block * n_y) + np.arange(n_x_block * n_y).reshape(n_x_block, n_y, 1)

        # Flatten the contributing voxels; each spans [segment_lo, segment_hi] in log T,
        # clamped to the temperature axis.
        n_spatial = n_velocity_bins * n_x_block * n_y
        spatial_index = scatter_index[voxel_mask]
        emission = emission_block[voxel_mask]
        segment_lo = np.clip(min_log_temperature[voxel_mask], temperature_lower_edge, temperature_upper_edge)
        segment_hi = np.clip(max_log_temperature[voxel_mask], temperature_lower_edge, temperature_upper_edge)
        bin_lo = np.clip(
            ((segment_lo - temperature_lower_edge) // log_temperature_bin_width).astype(np.intp),
            0,
            n_temperature_bins - 1,
        )
        bin_hi = np.clip(
            ((segment_hi - temperature_lower_edge) // log_temperature_bin_width).astype(np.intp),
            0,
            n_temperature_bins - 1,
        )
        # A voxel contributes its bin overlap / bin width to every bin it spans. Instead of
        # looping over temperature bins, scatter the partial overlaps of the two end bins as
        # point weights; interior bins (weight = full emission, rare because adjacent cells
        # usually differ by less than a bin) are expanded per spanned bin with np.repeat.
        single = bin_lo == bin_hi
        multi = ~single
        upper_edge_of_bin_lo = temperature_lower_edge + (bin_lo[multi] + 1) * log_temperature_bin_width
        lower_edge_of_bin_hi = temperature_lower_edge + bin_hi[multi] * log_temperature_bin_width
        interior_counts = bin_hi[multi] - bin_lo[multi] - 1
        first_interior_bin = np.repeat(bin_lo[multi] + 1, interior_counts)
        total_interior = first_interior_bin.size
        # Within-group offsets 0..count-1 turn the repeated first bin into every interior bin.
        group_starts = np.cumsum(interior_counts) - interior_counts
        interior_bins = first_interior_bin + np.arange(total_interior) - np.repeat(group_starts, interior_counts)
        point_bins = np.concatenate((bin_lo[single], bin_lo[multi], bin_hi[multi], interior_bins))
        point_index = np.concatenate(
            (
                spatial_index[single],
                spatial_index[multi],
                spatial_index[multi],
                np.repeat(spatial_index[multi], interior_counts),
            )
        )
        point_weights = np.concatenate(
            (
                emission[single] * (segment_hi[single] - segment_lo[single]) / log_temperature_bin_width,
                emission[multi] * (upper_edge_of_bin_lo - segment_lo[multi]) / log_temperature_bin_width,
                emission[multi] * (segment_hi[multi] - lower_edge_of_bin_hi) / log_temperature_bin_width,
                np.repeat(emission[multi], interior_counts),
            )
        )
        binned = np.bincount(
            point_bins * n_spatial + point_index,
            weights=point_weights,
            minlength=n_temperature_bins * n_spatial,
        )
        vdem[:, :, block, :] = binned.reshape(n_temperature_bins, n_velocity_bins, n_x_block, n_y)

    # Blocks are independent and write disjoint x slices; numpy releases the GIL in the
    # heavy ops (log10, searchsorted, bincount), so threads scale without extra copies.
    with ThreadPoolExecutor(max_workers=_VDEM_MAX_WORKERS) as executor:
        for done in [executor.submit(process_block, start) for start in range(0, n_x, _VDEM_X_BLOCK_SIZE)]:
            done.result()

    vdem_ds = xr.Dataset()
    vdem_ds["vdem"] = xr.DataArray(
        vdem[:, ::-1],
        dims=["logT", "doppler_velocity", "x", "y"],
        coords={
            "logT": log_temperature_axis,
            "doppler_velocity": -velocity_axis[::-1],
            "x": x,
            "y": y,
        },
    )
    vdem_ds["vdem"].attrs = {
        "description": "VDEM(logT, doppler_velocity, x, y)",
        "units": "1e27 / cm5",
    }
    vdem_ds.x.attrs["long_name"] = "X"
    vdem_ds.y.attrs["long_name"] = "Y"
    vdem_ds.logT.attrs["long_name"] = r"log$_{10}$(T)"
    vdem_ds.doppler_velocity.attrs["long_name"] = r"v$_{Doppler}$"
    vdem_ds.x.attrs["units"] = "cm"
    vdem_ds.y.attrs["units"] = "cm"
    vdem_ds.logT.attrs["units"] = "dex(K)"
    vdem_ds.doppler_velocity.attrs["units"] = "km/s"
    add_history(vdem_ds, call_inputs, create_simple_vdem)
    return vdem_ds


def calculate_moments(
    spectrum: xr.Dataset,
    *,
    moment_dim: str = "detector_x_pixel",
    integration_name: str = "flux",
    doppler_name: str = "doppler_velocity",
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
        Spectral axis to integrate the line profile over, by default
        ``"detector_x_pixel"``.
        The Doppler velocities used for the moments come from the ``doppler_name``
        coordinate, which is normalized to km/s on entry.
    integration_name : `str`, optional
        Name of the variable to integrate over ``spectrum``, by default ``"flux"``.
    doppler_name : `str`, optional
        Name of the Doppler-velocity coordinate in ``spectrum``, by default
        ``"doppler_velocity"``.
    vmax : `float` or None, optional
        Maximum absolute velocity (km/s) to include in the integration, by default None.
    vmask : `float` or None, optional
        Half-width (in ``detector_x_pixel``) of the window kept around the line
        peak, by default None.
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
    moments = xr.Dataset({"0th": zeroth, "1st": first, "2nd": second})
    update_attrs(moments, spectrum)
    moments["0th"].attrs = dict(masked_spectrum[integration_name].attrs)
    moments["1st"].attrs["units"] = str(u.km / u.s)
    moments["2nd"].attrs["units"] = str(u.km / u.s)
    add_history(moments, locals(), calculate_moments, sources=(spectrum,))
    return moments


def wavelength_to_doppler(response: xr.Dataset) -> xr.Dataset:
    """
    Add a Doppler-shift coordinate in km/s derived from wavelengths.

    Parameters
    ----------
    response : `xarray.Dataset`
        Must include ``detector_wavelength`` and ``line_wavelength``
        coordinates.

    Returns
    -------
    `xarray.Dataset`
        A new dataset with an added ``doppler_velocity`` coordinate in km/s.
    """
    c_kms = speed_of_light.to_value(u.km / u.s)
    detector_wavelength = coord_as_unit(
        response,
        "detector_wavelength",
        u.AA,
        "response.detector_wavelength",
    )
    line_wavelength = coord_as_unit(response, "line_wavelength", u.AA, "response.line_wavelength")
    doppler_velocity = (detector_wavelength / line_wavelength - 1) * c_kms
    doppler_velocity.attrs["units"] = str(u.km / u.s)
    response = response.assign_coords(doppler_velocity=doppler_velocity)
    add_history(response, locals(), wavelength_to_doppler)
    return response


def doppler_to_wavelength(response: xr.Dataset) -> xr.Dataset:
    """
    Add a wavelength coordinate in Angstrom derived from a Doppler shift.

    Parameters
    ----------
    response : `xarray.Dataset`
        Must include ``doppler_velocity`` and ``line_wavelength`` coordinates.

    Returns
    -------
    `xarray.Dataset`
        A new dataset with an added ``detector_wavelength`` coordinate in
        Angstrom.
    """
    c_kms = speed_of_light.to_value(u.km / u.s)
    line_wavelength = coord_as_unit(response, "line_wavelength", u.AA, "response.line_wavelength")
    doppler_velocity = coord_as_unit(
        response,
        "doppler_velocity",
        u.km / u.s,
        "response.doppler_velocity",
    )
    detector_wavelength = line_wavelength * (1 + doppler_velocity / c_kms)
    detector_wavelength.attrs["units"] = str(u.AA)
    response = response.assign_coords(detector_wavelength=detector_wavelength)
    add_history(response, locals(), doppler_to_wavelength)
    return response
