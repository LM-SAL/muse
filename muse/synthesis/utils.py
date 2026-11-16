from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr

import astropy.units as u
from astropy.constants import c as speed_of_light

from muse.log import logger
from muse.utils.utils import add_history, coord_as_unit, require_unit, update_attrs

__all__ = [
    "apply_exposure_time",
    "calculate_moments",
    "create_simple_vdem",
    "doppler_to_wavelength",
    "remove_exposure_time",
    "sum_lines_per_channel",
    "wavelength_to_doppler",
]

# Deliberate fixed cap, not os.cpu_count(): it bounds peak memory (per-slice temporaries
# scale with worker count) and this runs nested under gallery/CI parallelism on
# 2-4 vCPU builders, where a cpu-derived pool would oversubscribe.
_VDEM_MAX_WORKERS = 4


@u.quantity_input(
    temperature=u.K,
    velocity=u.km / u.s,
    ne_nh=u.cm**-6,
    cell_length=u.cm,
    x=u.cm,
    y=u.cm,
    velocity_grid=u.km / u.s,
    logT_grid=u.dex(u.K),
)
def create_simple_vdem(
    *,
    temperature: u.Quantity,
    velocity: u.Quantity,
    ne_nh: u.Quantity,
    cell_length: u.Quantity,
    x: u.Quantity,
    y: u.Quantity,
    velocity_grid: u.Quantity,
    logT_grid: u.Quantity,
    integration_axis: int = 2,
) -> xr.Dataset:
    r"""
    Calculates DEM as a function of temperature and velocity,
    x (0) and y (1) axes are horizontal z (2) vertical.
    Right hand rule.
    velocity is LOS velocity and positive is towards the observer.

    Every argument is keyword-only, and every physical one is an
    `astropy.units.Quantity` in any unit convertible to the one given below.

    Parameters
    ----------
    temperature : `astropy.units.Quantity`
        3D array of gas temperature, in K.
    velocity : `astropy.units.Quantity`
        3D array of the velocity component along ``integration_axis``, in km/s
        (positive towards the observer).
    ne_nh : `astropy.units.Quantity`
        3D array of ``n_e * n_H``, in 1/cm^6.
    cell_length : `astropy.units.Quantity`
        1D array of cell length along the line-of-sight (``integration_axis``) axis,
        in cm (may be non-uniform).
    x : `astropy.units.Quantity`
        1D array of the coordinate of the first non-LOS axis, in cm.
    y : `astropy.units.Quantity`
        1D array of the coordinate of the second non-LOS axis, in cm.
    velocity_grid : `astropy.units.Quantity`
        1D velocity bin centers, in km/s.
    logT_grid : `astropy.units.Quantity`
        1D temperature bin centers, in dex(K), as the ``logT`` coordinate of a VDEM
        and of a response, e.g. ``np.arange(5.5, 7.6, 0.1) * u.dex(u.K)``.
    integration_axis : int, optional
        Axis of ``temperature``/``velocity``/``ne_nh`` to integrate along, by default 2.
        ``x`` and ``y`` label the two remaining axes in their original order, and the
        integration enters the box at index 0 of this axis.

    Returns
    -------
    xarray.Dataset
        VDEM with dimensions of the 2D spatial axes from the simulation plus
        temperature and velocity bins. Its ``x`` and ``y`` coordinates are in cm,
        ``logT`` in dex(K) and ``doppler_velocity`` in km/s.

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

    Intermediate arrays are processed one x slice per worker thread; temporaries stay
    bounded by the slice size (y times z) and the fixed worker count.

    The returned VDEM is still allocated eagerly in full.

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
    # Plain arrays in the units the binning works in. Taken before call_inputs, so that the
    # history names the arrays instead of listing their values.
    temperature = temperature.to_value(u.K)
    velocity = velocity.to_value(u.km / u.s)
    ne_nh = ne_nh.to_value(u.cm**-6)
    cell_length = cell_length.to_value(u.cm)
    x = x.to_value(u.cm)
    y = y.to_value(u.cm)
    velocity_grid = velocity_grid.to_value(u.km / u.s)
    logT_grid = logT_grid.to_value(u.dex(u.K))
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

    n_velocity_bins = len(velocity_grid)
    n_temperature_bins = len(logT_grid)
    n_x = len(x)
    n_y = len(y)
    log_temperature_bin_width = logT_grid[1] - logT_grid[0]
    velocity_bin_width = velocity_grid[1] - velocity_grid[0]
    # Contiguous half-open velocity bins [center - dv/2, center + dv/2): a voxel sitting exactly
    # on an edge lands in the upper bin, matching the half-open temperature-bin convention below.
    velocity_edges = np.append(velocity_grid - velocity_bin_width / 2.0, velocity_grid[-1] + velocity_bin_width / 2.0)
    log_temperature_lower_edge = logT_grid[0] - log_temperature_bin_width / 2.0
    log_temperature_upper_edge = logT_grid[-1] + log_temperature_bin_width / 2.0
    cell_length_los = cell_length.reshape(1, -1)

    # Only the output scales with n_x; full-cube intermediates stay bounded by the
    # (y, z) slice size times the number of worker threads.
    vdem_dtype = (np.empty(0, dtype=ne_nh.dtype) / 1e27).dtype
    vdem = np.zeros((n_temperature_bins, n_velocity_bins, n_x, n_y), dtype=vdem_dtype)

    def process_slice(ix: int) -> None:
        ne_nh_slice = ne_nh[ix] / 1e27  # normalize to the 1e27 / cm^5 output units
        # n_e * n_H * cell_length is the emission each voxel spreads over temperature bins.
        emission_slice = ne_nh_slice * cell_length_los
        # Each line-of-sight cell spans the temperatures between it and its neighbour; its
        # emission is distributed across temperature bins by the log-T overlap (DEM = dl/dT).
        log_temperature_slice = np.log10(temperature[ix])
        log_temperature_previous = np.roll(log_temperature_slice, 1, axis=1)
        log_temperature_previous[:, 0] = 2.0  # boundary cell is entered from 100 K
        max_log_temperature = np.maximum(log_temperature_slice, log_temperature_previous)
        min_log_temperature = np.minimum(log_temperature_slice, log_temperature_previous)
        # Every voxel falls in exactly one velocity bin, so scatter each voxel onto a flat
        # (velocity_bin, y) index; that index has no z axis, so np.bincount's accumulation
        # is the line-of-sight sum.
        velocity_bin = np.searchsorted(velocity_edges, velocity[ix], side="right") - 1
        voxel_mask = (
            (velocity_bin >= 0)
            & (velocity_bin < n_velocity_bins)
            & (max_log_temperature >= log_temperature_lower_edge)
            & (min_log_temperature < log_temperature_upper_edge)
        )
        scatter_index = velocity_bin * n_y + np.arange(n_y).reshape(n_y, 1)

        # Flatten the contributing voxels; each spans [segment_lo, segment_hi] in log T,
        # clamped to the temperature axis.
        n_velocity_y = n_velocity_bins * n_y
        velocity_y_index = scatter_index[voxel_mask]
        emission = emission_slice[voxel_mask]
        segment_lo = np.clip(min_log_temperature[voxel_mask], log_temperature_lower_edge, log_temperature_upper_edge)
        segment_hi = np.clip(max_log_temperature[voxel_mask], log_temperature_lower_edge, log_temperature_upper_edge)
        bin_lo = np.clip(
            ((segment_lo - log_temperature_lower_edge) // log_temperature_bin_width).astype(np.intp),
            0,
            n_temperature_bins - 1,
        )
        bin_hi = np.clip(
            ((segment_hi - log_temperature_lower_edge) // log_temperature_bin_width).astype(np.intp),
            0,
            n_temperature_bins - 1,
        )
        # A voxel contributes its bin overlap / bin width to every bin it spans. Instead of
        # looping over temperature bins, scatter the partial overlaps of the two end bins as
        # point weights; interior bins (weight = full emission, rare because adjacent cells
        # usually differ by less than a bin) are expanded per spanned bin with np.repeat.
        single = bin_lo == bin_hi
        multi = ~single
        upper_edge_of_bin_lo = log_temperature_lower_edge + (bin_lo[multi] + 1) * log_temperature_bin_width
        lower_edge_of_bin_hi = log_temperature_lower_edge + bin_hi[multi] * log_temperature_bin_width
        interior_counts = bin_hi[multi] - bin_lo[multi] - 1
        first_interior_bin = np.repeat(bin_lo[multi] + 1, interior_counts)
        total_interior = first_interior_bin.size
        # Within-group offsets 0..count-1 turn the repeated first bin into every interior bin.
        group_starts = np.cumsum(interior_counts) - interior_counts
        interior_bins = first_interior_bin + np.arange(total_interior) - np.repeat(group_starts, interior_counts)
        point_bins = np.concatenate((bin_lo[single], bin_lo[multi], bin_hi[multi], interior_bins))
        point_index = np.concatenate(
            (
                velocity_y_index[single],
                velocity_y_index[multi],
                velocity_y_index[multi],
                np.repeat(velocity_y_index[multi], interior_counts),
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
            point_bins * n_velocity_y + point_index,
            weights=point_weights,
            minlength=n_temperature_bins * n_velocity_y,
        )
        vdem[:, :, ix, :] = binned.reshape(n_temperature_bins, n_velocity_bins, n_y)

    # Slices are independent and write disjoint x indices; numpy releases the GIL in the
    # heavy ops (log10, searchsorted, bincount), so threads scale without extra copies.
    with ThreadPoolExecutor(max_workers=_VDEM_MAX_WORKERS) as executor:
        for done in [executor.submit(process_slice, ix) for ix in range(n_x)]:
            done.result()

    vdem_ds = xr.Dataset()
    vdem_ds["vdem"] = xr.DataArray(
        vdem[:, ::-1],
        dims=["logT", "doppler_velocity", "x", "y"],
        coords={
            "logT": logT_grid,
            "doppler_velocity": -velocity_grid[::-1],
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


@u.quantity_input(maximum_doppler_velocity=u.km / u.s)
def calculate_moments(
    spectrum: xr.Dataset,
    *,
    moment_dim: str = "detector_x_pixel",
    variable_name: str = "flux",
    doppler_name: str = "detector_doppler_velocity",
    maximum_doppler_velocity: u.Quantity | None = None,
    peak_half_width: float | None = None,
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
    variable_name : `str`, optional
        Name of the variable to integrate over ``spectrum``, by default ``"flux"``.
    doppler_name : `str`, optional
        Name of the Doppler-velocity coordinate in ``spectrum``, by default
        ``"detector_doppler_velocity"``, which `wavelength_to_doppler` adds. Its
        dimensions must be a subset of the dimensions of ``variable_name``.
    maximum_doppler_velocity : `astropy.units.Quantity` or None, optional
        Maximum absolute Doppler velocity to include in the integration, convertible
        to km/s, by default None.
    peak_half_width : `float` or None, optional
        Half-width of the window kept around the line peak, in the units of the
        ``moment_dim`` coordinate (detector pixels by default), by default None.
        With ``maximum_doppler_velocity`` the peak is found within that velocity.

    Returns
    -------
    `xarray.Dataset`
        Dataset containing the moments.
    """
    require_unit(spectrum, variable_name, f"spectrum.{variable_name}")
    if doppler_name not in spectrum.coords:
        msg = f"spectrum is missing the {doppler_name!r} coordinate; run wavelength_to_doppler first to add it."
        raise ValueError(msg)
    doppler_unit = require_unit(
        spectrum, doppler_name, f"spectrum.{doppler_name}", coord_only=True, convertible_to=u.km / u.s
    )
    # Normalize to km/s so the raw .data used by the einsum is correct regardless of input unit.
    spectrum = spectrum.assign_coords({doppler_name: spectrum[doppler_name] * doppler_unit.to(u.km / u.s)})
    spectrum[doppler_name].attrs["units"] = str(u.km / u.s)
    if not set(spectrum[doppler_name].dims).issubset(spectrum[variable_name].dims):
        msg = f"spectrum.{doppler_name} dimensions must be a subset of spectrum.{variable_name} dimensions"
        raise ValueError(msg)

    masked_spectrum = spectrum
    if maximum_doppler_velocity is not None:
        # NaN Doppler coordinates stay included, matching the old 0/1 mask.
        maximum_speed = maximum_doppler_velocity.to_value(u.km / u.s)  # the coordinate is in km/s by now
        masked_flux = spectrum[variable_name].where(~(np.abs(spectrum[doppler_name]) > maximum_speed), 0)
        masked_spectrum = spectrum.assign({variable_name: masked_flux})
    if peak_half_width is not None:
        peak_index = masked_spectrum[variable_name].argmax(dim=moment_dim)
        peak_coord = masked_spectrum[moment_dim].isel({moment_dim: peak_index})
        distance = np.abs(masked_spectrum[moment_dim] - peak_coord)
        masked_spectrum = masked_spectrum.assign(
            {variable_name: masked_spectrum[variable_name].where(distance < peak_half_width, 0)}
        )
    masked_spectrum = masked_spectrum.assign(
        {
            variable_name: masked_spectrum[variable_name]
            .where(masked_spectrum[variable_name] > 0, 0)
            .assign_attrs(spectrum[variable_name].attrs)
        }
    )
    zeroth = masked_spectrum[variable_name].sum(dim=moment_dim)
    velocity = masked_spectrum[doppler_name]
    # Pixels with no flux (e.g. fully masked) would divide by zero; leave them NaN, not inf.
    safe_zeroth = zeroth.where(zeroth > 0)
    first = (masked_spectrum[variable_name] * velocity).sum(dim=moment_dim) / safe_zeroth
    # Note that int(I (u-I1)^2 du)/I0 = (int(I u^2 du))/I0 - I1^2
    variance = (masked_spectrum[variable_name] * velocity**2).sum(dim=moment_dim) / safe_zeroth - first**2
    second = np.sqrt(variance.clip(min=0))
    # zeroth/first/second are already DataArrays carrying the non-moment dims and coords.
    moments = xr.Dataset({"0th": zeroth, "1st": first, "2nd": second})
    update_attrs(moments, spectrum)
    moments["0th"].attrs = dict(masked_spectrum[variable_name].attrs)
    moments["1st"].attrs["units"] = str(u.km / u.s)
    moments["2nd"].attrs["units"] = str(u.km / u.s)
    add_history(moments, locals(), calculate_moments, sources=(spectrum,))
    return moments


def wavelength_to_doppler(data: xr.Dataset) -> xr.Dataset:
    """
    Add a Doppler-shift coordinate in km/s derived from wavelengths.

    Parameters
    ----------
    data : `xarray.Dataset`
        A spectrum or a response. It must include the ``detector_wavelength``
        and ``line_wavelength`` coordinates.

    Returns
    -------
    `xarray.Dataset`
        A new dataset with an added ``detector_doppler_velocity`` coordinate in
        km/s: the Doppler shift of each detector pixel from the line. It is not
        the ``doppler_velocity`` axis of a VDEM or a response, which stays as
        it is.
    """
    speed_of_light_kms = speed_of_light.to_value(u.km / u.s)
    detector_wavelength = coord_as_unit(data, "detector_wavelength", u.AA, "data.detector_wavelength")
    line_wavelength = coord_as_unit(data, "line_wavelength", u.AA, "data.line_wavelength")
    detector_doppler_velocity = (detector_wavelength / line_wavelength - 1) * speed_of_light_kms
    detector_doppler_velocity.attrs["units"] = str(u.km / u.s)
    result = data.assign_coords(detector_doppler_velocity=detector_doppler_velocity)
    add_history(result, locals(), wavelength_to_doppler)
    return result


def doppler_to_wavelength(data: xr.Dataset, *, doppler_name: str = "detector_doppler_velocity") -> xr.Dataset:
    """
    Add a wavelength coordinate in Angstrom derived from a Doppler shift.

    Parameters
    ----------
    data : `xarray.Dataset`
        A spectrum or a response. It must include the coordinate that
        ``doppler_name`` names and the ``line_wavelength`` coordinate.
    doppler_name : `str`, optional
        Name of the Doppler-velocity coordinate to convert, by default
        ``"detector_doppler_velocity"``, which `wavelength_to_doppler` adds.
        Pass ``"doppler_velocity"`` for the velocity axis of a response.

    Returns
    -------
    `xarray.Dataset`
        A new dataset with an added ``detector_wavelength`` coordinate in
        Angstrom.

    Raises
    ------
    ValueError
        If ``data`` already has a ``detector_wavelength`` coordinate.
    """
    if "detector_wavelength" in data.coords:
        msg = "data already has a detector_wavelength coordinate; drop it first to replace it"
        raise ValueError(msg)
    speed_of_light_kms = speed_of_light.to_value(u.km / u.s)
    line_wavelength = coord_as_unit(data, "line_wavelength", u.AA, "data.line_wavelength")
    doppler_velocity = coord_as_unit(data, doppler_name, u.km / u.s, f"data.{doppler_name}")
    detector_wavelength = line_wavelength * (1 + doppler_velocity / speed_of_light_kms)
    detector_wavelength.attrs["units"] = str(u.AA)
    result = data.assign_coords(detector_wavelength=detector_wavelength)
    add_history(result, locals(), doppler_to_wavelength)
    return result


def _time_base(unit: u.UnitBase) -> u.UnitBase | None:
    """
    The time unit in the denominator of ``unit``, e.g. ``s`` for ``ph / s``, or `None`
    when ``unit`` is not a rate.
    """
    for base, power in zip(unit.bases, unit.powers, strict=True):
        if base.physical_type == "time" and power == -1:
            return base
    return None


@u.quantity_input(exposure_time=u.s)
def apply_exposure_time(spectrum: xr.Dataset, exposure_time: u.Quantity) -> xr.Dataset:
    """
    Turn the rate of a spectrum into the counts of one exposure.

    Parameters
    ----------
    spectrum : `xarray.Dataset`
        Spectrum whose ``flux`` is a rate, such as ``ph / s``. ``flux`` must
        define units in the attrs.
    exposure_time : `astropy.units.Quantity`
        Length of the exposure, a scalar convertible to seconds.

    Returns
    -------
    `xarray.Dataset`
        A new dataset whose ``flux`` is multiplied by the exposure time and no
        longer per unit time, e.g. ``ph`` for ``ph / s``. The time is recorded
        as the scalar coordinate ``exposure_time`` in seconds, which
        `remove_exposure_time` reads.

    Raises
    ------
    ValueError
        If ``flux`` has no units or is not per unit time, if ``exposure_time``
        is not a positive scalar, or if the spectrum already carries an
        exposure time.
    """
    flux_unit = require_unit(spectrum, "flux", "spectrum.flux")
    # A data variable of that name counts too: the coordinate would replace it without a word.
    if "exposure_time" in spectrum.variables:
        msg = "spectrum already has an exposure time applied; call remove_exposure_time first"
        raise ValueError(msg)
    time_base = _time_base(flux_unit)
    if time_base is None:
        msg = f"spectrum.flux is in {flux_unit}, which is not a rate: its unit must be per unit time, such as 'ph / s'"
        raise ValueError(msg)
    if not exposure_time.isscalar:
        msg = "exposure_time must be a scalar"
        raise ValueError(msg)
    if not np.isfinite(exposure_time.value) or exposure_time.value <= 0:
        msg = f"exposure_time must be finite and positive, got {exposure_time}"
        raise ValueError(msg)
    exposure_time = exposure_time.to(u.s)  # a statement so add_history records the time in seconds
    flux = (spectrum.flux * exposure_time.to_value(time_base)).assign_attrs(
        {**spectrum.flux.attrs, "units": str(flux_unit * time_base)}
    )
    result = spectrum.assign(flux=flux).assign_coords(
        exposure_time=((), exposure_time.to_value(u.s), {"units": str(u.s)})
    )
    add_history(result, locals(), apply_exposure_time)
    return result


def remove_exposure_time(spectrum: xr.Dataset) -> xr.Dataset:
    """
    Divide the exposure time back out of a spectrum, turning counts into a rate.

    Parameters
    ----------
    spectrum : `xarray.Dataset`
        Spectrum returned by `apply_exposure_time`: its ``flux`` must define
        units and it must carry the ``exposure_time`` coordinate.

    Returns
    -------
    `xarray.Dataset`
        A new dataset without the ``exposure_time`` coordinate, whose ``flux``
        is divided by the exposure time and so is per second, e.g. ``ph / s``
        for ``ph``.

    Raises
    ------
    ValueError
        If ``flux`` has no units or the spectrum carries no exposure time.
    """
    flux_unit = require_unit(spectrum, "flux", "spectrum.flux")
    if "exposure_time" not in spectrum.coords:
        msg = "spectrum has no exposure_time coordinate; apply_exposure_time records it"
        raise ValueError(msg)
    seconds = float(coord_as_unit(spectrum, "exposure_time", u.s, "spectrum.exposure_time"))
    flux = (spectrum.flux / seconds).assign_attrs({**spectrum.flux.attrs, "units": str(flux_unit / u.s)})
    result = spectrum.assign(flux=flux).drop_vars("exposure_time")
    add_history(result, locals(), remove_exposure_time)
    return result


def sum_lines_per_channel(data: xr.Dataset, *, sum_slits: bool = False) -> xr.Dataset:
    """
    Sum the lines of each channel, which the detector records on the same pixels.

    Parameters
    ----------
    data : `xarray.Dataset`
        A spectrum or a response with a ``line`` dimension and a ``channel``
        coordinate along it.
    sum_slits : `bool`, optional
        Also sum over ``slit`` when the data has that dimension, by default
        `False`.

    Returns
    -------
    `xarray.Dataset`
        A new dataset in which ``channel`` replaces ``line``. Every data
        variable along ``line`` is the sum of the lines of the channel, and a
        NaN in a line stays a NaN in the sum. Every coordinate along ``line``,
        such as ``gain``, takes the value of the first line of the channel;
        ``line_wavelength`` does too where every line of the channel has the
        same wavelength, and is NaN otherwise. The channels keep the order in
        which they first appear. With ``sum_slits``, the coordinates along
        ``slit`` are dropped.

    Raises
    ------
    ValueError
        If ``data`` has no ``line`` dimension or no ``channel`` coordinate along
        it.

    Notes
    -----
    Sum after converting photons to electrons or DN, not before. A channel
    whose lines differ in wavelength has no single photon energy, so its
    ``line_wavelength`` is NaN in the sum and
    `muse.instrument.transform_flux_units` refuses to take it across the photon
    step.
    """
    if "line" not in data.dims or "channel" not in data.coords or data.channel.dims != ("line",):
        msg = "data must have a line dimension and a channel coordinate along it"
        raise ValueError(msg)
    channels, first_line = np.unique(data.channel.values, return_index=True)
    channels, first_line = channels[np.argsort(first_line)], np.sort(first_line)
    # Start from the first line of each channel, which gives the coordinates; the sums then replace the
    # data variables along line.
    # errors="ignore": a line dimension need not have a coordinate of its own.
    result = data.isel(line=first_line).swap_dims(line="channel").drop_vars("line", errors="ignore")
    for name, variable in data.data_vars.items():
        if "line" not in variable.dims:
            continue
        summed = variable.groupby("channel").sum(skipna=False, keep_attrs=True).sel(channel=channels)
        result[name] = summed.transpose(*("channel" if dim == "line" else dim for dim in variable.dims))
    if "line_wavelength" in data.coords and data.line_wavelength.dims == ("line",):
        # The sum of lines at different wavelengths has no single photon energy: without a wavelength,
        # transform_flux_units refuses the photon step instead of taking the first line's for all.
        wavelengths = data.line_wavelength.groupby("channel")
        mixed = (wavelengths.max(skipna=False) != wavelengths.min(skipna=False)).sel(channel=channels)
        result = result.assign_coords(line_wavelength=result.line_wavelength.where(~mixed.values))
    if sum_slits and "slit" in result.dims:
        result = result.sum("slit", skipna=False, keep_attrs=True)
    add_history(result, locals(), sum_lines_per_channel)
    return result
