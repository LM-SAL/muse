from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr

import astropy.units as u
from astropy.constants import c as speed_of_light

from muse.log import logger
from muse.utils.utils import _require_increasing_axis, add_history, coord_as_unit, require_unit, update_attrs

__all__ = [
    "apply_exposure_time",
    "calculate_moments",
    "create_simple_vdem",
    "doppler_to_wavelength",
    "remove_exposure_time",
    "sum_lines_per_channel",
    "wavelength_to_doppler",
]

# Fixed, not os.cpu_count(): bounds peak memory and does not oversubscribe nested CI runs.
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
    Bin a simulation box into a VDEM in temperature and line-of-sight velocity.

    Axes 0 (x) and 1 (y) are horizontal and axis 2 (z) vertical, right-handed. Physical
    arguments take any unit convertible to the one given below.

    Parameters
    ----------
    temperature : `astropy.units.Quantity`
        3D gas temperature, in K.
    velocity : `astropy.units.Quantity`
        3D velocity along ``integration_axis``, in km/s, positive towards the observer.
    ne_nh : `astropy.units.Quantity`
        3D ``n_e * n_H``, in 1/cm^6.
    cell_length : `astropy.units.Quantity`
        1D cell length along ``integration_axis``, in cm (may be non-uniform).
    x : `astropy.units.Quantity`
        1D coordinate of the first non-LOS axis, in cm.
    y : `astropy.units.Quantity`
        1D coordinate of the second non-LOS axis, in cm.
    velocity_grid : `astropy.units.Quantity`
        1D velocity bin centers, in km/s, increasing and evenly spaced, in the frame of ``velocity``
        (positive towards the observer); the output ``doppler_velocity`` axis is ``-velocity_grid[::-1]``.
    logT_grid : `astropy.units.Quantity`
        1D temperature bin centers, increasing and evenly spaced, in dex(K) like the ``logT`` of a response, e.g.
        ``np.arange(5.5, 7.6, 0.1) * u.dex(u.K)``.
    integration_axis : int, optional
        Axis of the 3D arrays to integrate along, by default 2, entered at index 0.
        ``x`` and ``y`` label the other two axes in their order.

    Returns
    -------
    xarray.Dataset
        VDEM over ``logT`` (dex(K)), ``doppler_velocity`` (km/s), ``x`` and ``y`` (cm), in
        1e27 cm^-5 per bin. Plasma moving towards the observer gets a negative ``doppler_velocity``.

    Raises
    ------
    ValueError
        If ``temperature`` is not 3D, a grid is not increasing and evenly spaced, or another input
        does not match its shape or axes. Non-finite
        values only log a warning, and can propagate NaN into the output.

    Notes
    -----
    Remove any convection zone from the box first. Each x slice is binned on a worker
    thread, so temporaries scale with one (y, z) slice; the VDEM is allocated in full.

    A line intensity is :math:`I = \int n_e n_H\, G(T)\, dl`, with :math:`G(T)` the
    contribution function. Each bin holds the column emission measure of its log T and
    velocity range, in 1e27 cm^-5 and not divided by the bin widths:

    .. math::

        VDEM = \sum_l n_e n_H\, \Delta l\, f_l

    with :math:`f_l` the fraction of cell :math:`l`'s log T span, from the previous cell's
    log T to its own, that lies in the bin. A cell with no span, such as the first one,
    puts all its emission in its own bin, and plasma outside the bins contributes nothing.
    With a response :math:`G(T, v_{los})_\lambda` per unit emission measure that includes
    the Doppler shift,

    .. math::

        I(\lambda) = \sum_T \sum_{v_{los}} VDEM\, G(T, v_{los})_\lambda

    so one VDEM synthesizes any optically thin line.
    """
    # Before call_inputs, so the history names the arrays instead of listing their values.
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
    # Views; the binning below integrates over the last axis.
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

    for name, grid in (("velocity_grid", velocity_grid), ("logT_grid", logT_grid)):
        _require_increasing_axis(grid, name)
        # The bins take the first spacing for the whole grid. Allow the rounding of arange or float32, as in _resample_axis.
        eps = np.finfo(grid.dtype if np.issubdtype(grid.dtype, np.floating) else float).eps
        tolerance = 16 * eps * max(np.abs(grid).max(), 1.0)
        if grid.size < 2 or np.abs(grid - np.linspace(grid[0], grid[-1], grid.size)).max() > tolerance:
            msg = f"{name} must hold at least two evenly spaced values"
            raise ValueError(msg)
    n_velocity_bins = len(velocity_grid)
    n_temperature_bins = len(logT_grid)
    n_x = len(x)
    n_y = len(y)
    log_temperature_bin_width = logT_grid[1] - logT_grid[0]
    velocity_bin_width = velocity_grid[1] - velocity_grid[0]
    # Half-open bins, as for temperature below: a voxel on an edge lands in the upper bin.
    velocity_edges = np.append(velocity_grid - velocity_bin_width / 2.0, velocity_grid[-1] + velocity_bin_width / 2.0)
    log_temperature_lower_edge = logT_grid[0] - log_temperature_bin_width / 2.0
    log_temperature_upper_edge = logT_grid[-1] + log_temperature_bin_width / 2.0
    cell_length_los = cell_length.reshape(1, -1)

    vdem_dtype = (np.empty(0, dtype=ne_nh.dtype) / 1e27).dtype
    vdem = np.zeros((n_temperature_bins, n_velocity_bins, n_x, n_y), dtype=vdem_dtype)
    # Per x slice: the emission measure from the lowest logT edge up, its part hotter than the grid, its part faster.
    left_out = np.zeros((n_x, 3))

    def process_slice(ix: int) -> None:
        ne_nh_slice = ne_nh[ix] / 1e27  # normalize to the 1e27 / cm^5 output units
        emission_slice = ne_nh_slice * cell_length_los
        # Cell i spans the log T from cell i-1 to its own, and the first cell is isothermal. Cell i's
        # density and velocity stand for that whole interval, a second-order simplification.
        log_temperature_slice = np.log10(temperature[ix])
        log_temperature_previous = np.roll(log_temperature_slice, 1, axis=1)
        log_temperature_previous[:, 0] = log_temperature_slice[:, 0]
        max_log_temperature = np.maximum(log_temperature_slice, log_temperature_previous)
        min_log_temperature = np.minimum(log_temperature_slice, log_temperature_previous)
        # The flat (velocity_bin, y) index has no z axis, so np.bincount sums the line of sight.
        velocity_bin = np.searchsorted(velocity_edges, velocity[ix], side="right") - 1
        warm = log_temperature_slice >= log_temperature_lower_edge
        hot = log_temperature_slice >= log_temperature_upper_edge
        fast = warm & ~hot & ((velocity_bin < 0) | (velocity_bin >= n_velocity_bins))
        left_out[ix] = [np.nansum(emission_slice[mask], dtype=np.float64) for mask in (warm, hot, fast)]
        voxel_mask = (
            (velocity_bin >= 0)
            & (velocity_bin < n_velocity_bins)
            & (max_log_temperature >= log_temperature_lower_edge)
            & (min_log_temperature < log_temperature_upper_edge)
        )
        scatter_index = velocity_bin * n_y + np.arange(n_y).reshape(n_y, 1)

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
        # A bin gets the cell's emission times overlap / cell span: partial at the two end bins,
        # a bin width on the (rare) interior ones, so the shares sum to the emission inside the grid.
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
        span = max_log_temperature[voxel_mask] - min_log_temperature[voxel_mask]
        emission_per_dex = np.divide(emission, span, out=np.zeros_like(emission), where=span > 0)
        point_weights = np.concatenate(
            (
                # An isothermal cell (no span) puts all its emission in its own bin.
                np.where(
                    span[single] > 0,
                    emission_per_dex[single] * (segment_hi[single] - segment_lo[single]),
                    emission[single],
                ),
                emission_per_dex[multi] * (upper_edge_of_bin_lo - segment_lo[multi]),
                emission_per_dex[multi] * (segment_hi[multi] - lower_edge_of_bin_hi),
                np.repeat(emission_per_dex[multi] * log_temperature_bin_width, interior_counts),
            )
        )
        binned = np.bincount(
            point_bins * n_velocity_y + point_index,
            weights=point_weights,
            minlength=n_temperature_bins * n_velocity_y,
        )
        vdem[:, :, ix, :] = binned.reshape(n_temperature_bins, n_velocity_bins, n_y)

    # Slices write disjoint x; numpy releases the GIL in log10, searchsorted and bincount.
    with ThreadPoolExecutor(max_workers=_VDEM_MAX_WORKERS) as executor:
        for done in [executor.submit(process_slice, ix) for ix in range(n_x)]:
            done.result()

    warm, hot, fast = left_out.sum(axis=0)
    # Plasma cooler than the grid is expected, such as a chromosphere; hotter or faster plasma means the grid is too narrow.
    if hot + fast > 1e-6 * warm:
        logger.info(
            f"Of the emission measure from logT {log_temperature_lower_edge:.2f} up, {100 * hot / warm:.3g} % is "
            f"hotter than logT {log_temperature_upper_edge:.2f} and {100 * fast / warm:.3g} % moves faster than "
            f"velocity_grid ({velocity_edges[0]:.1f} to {velocity_edges[-1]:.1f} km/s); both are left out of the VDEM"
        )

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


@u.quantity_input(maximum_doppler_velocity=u.km / u.s, peak_half_width=u.pix)
def calculate_moments(
    spectrum: xr.Dataset,
    *,
    moment_dim: str = "detector_x_pixel",
    variable_name: str = "flux",
    doppler_name: str = "detector_doppler_velocity",
    maximum_doppler_velocity: u.Quantity | None = None,
    peak_half_width: u.Quantity | None = None,
) -> xr.Dataset:
    """
    Compute the zeroth, first, and second moments from a spectrum.

    Parameters
    ----------
    spectrum : `xarray.Dataset`
        Spectrum with a Doppler-velocity coordinate in any velocity unit, which
        `wavelength_to_doppler` adds.
    moment_dim : `str`, optional
        Spectral axis to integrate over, by default ``"detector_x_pixel"``.
    variable_name : `str`, optional
        Variable to integrate, by default ``"flux"``.
    doppler_name : `str`, optional
        Doppler-velocity coordinate, by default ``"detector_doppler_velocity"``. Its
        dimensions must be a subset of those of ``variable_name``.
    maximum_doppler_velocity : `astropy.units.Quantity` or None, optional
        Largest absolute Doppler velocity to include, by default None.
    peak_half_width : `astropy.units.Quantity` or None, optional
        Half-width in pixels of the window kept around the line peak, by default None.
        With ``maximum_doppler_velocity`` the peak is found within that velocity.

    Returns
    -------
    `xarray.Dataset`
        The moments ``0th``, ``1st`` and ``2nd``, the last two in km/s. Velocity and
        width use only pixels with a finite Doppler velocity and are NaN when there are
        none.
    """
    require_unit(spectrum, variable_name, f"spectrum.{variable_name}")
    if doppler_name not in spectrum.coords:
        msg = f"spectrum is missing the {doppler_name!r} coordinate; run wavelength_to_doppler first to add it."
        raise ValueError(msg)
    doppler_unit = require_unit(
        spectrum, doppler_name, f"spectrum.{doppler_name}", coord_only=True, convertible_to=u.km / u.s
    )
    # The sums below take raw values, which must be in km/s.
    spectrum = spectrum.assign_coords({doppler_name: spectrum[doppler_name] * doppler_unit.to(u.km / u.s)})
    spectrum[doppler_name].attrs["units"] = str(u.km / u.s)
    if not set(spectrum[doppler_name].dims).issubset(spectrum[variable_name].dims):
        msg = f"spectrum.{doppler_name} dimensions must be a subset of spectrum.{variable_name} dimensions"
        raise ValueError(msg)

    masked_spectrum = spectrum
    if maximum_doppler_velocity is not None:
        # ~(>) keeps NaN Doppler coordinates in, as the old 0/1 mask did.
        maximum_speed = maximum_doppler_velocity.to_value(u.km / u.s)
        masked_flux = spectrum[variable_name].where(~(np.abs(spectrum[doppler_name]) > maximum_speed), 0)
        masked_spectrum = spectrum.assign({variable_name: masked_flux})
    if peak_half_width is not None:
        peak_index = masked_spectrum[variable_name].argmax(dim=moment_dim)
        distance = np.abs(xr.DataArray(np.arange(masked_spectrum.sizes[moment_dim]), dims=moment_dim) - peak_index)
        masked_spectrum = masked_spectrum.assign(
            {variable_name: masked_spectrum[variable_name].where(distance < peak_half_width.to_value(u.pix), 0)}
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
    finite = np.isfinite(velocity)
    # Flux without a velocity counts in the 0th moment only; none left (or no flux) gives NaN, not inf.
    weight = masked_spectrum[variable_name].where(finite, 0)
    # A zero weight times an infinite velocity is NaN and warns on dask arrays, so zero the velocity there too.
    velocity = velocity.where(finite, 0)
    weight_sum = weight.sum(dim=moment_dim)
    safe_weight_sum = weight_sum.where(weight_sum > 0)
    first = (weight * velocity).sum(dim=moment_dim) / safe_weight_sum
    # int(I (u-I1)^2 du)/I0 = (int(I u^2 du))/I0 - I1^2
    variance = (weight * velocity**2).sum(dim=moment_dim) / safe_weight_sum - first**2
    second = np.sqrt(variance.clip(min=0))
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
        A spectrum or a response with ``detector_wavelength`` and ``line_wavelength``.

    Returns
    -------
    `xarray.Dataset`
        ``data`` with a ``detector_doppler_velocity`` coordinate, the shift of each detector
        pixel from the line, positive to the red. The ``doppler_velocity`` axis of a VDEM or
        a response is left as it is.
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
        A spectrum or a response with ``line_wavelength`` and the ``doppler_name``
        coordinate.
    doppler_name : `str`, optional
        Doppler-velocity coordinate to convert, by default ``"detector_doppler_velocity"``;
        ``"doppler_velocity"`` for the velocity axis of a response.

    Returns
    -------
    `xarray.Dataset`
        ``data`` with a ``detector_wavelength`` coordinate in Angstrom.

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
        Spectrum whose ``flux`` has a rate unit in its attrs, such as ``ph / s``.
    exposure_time : `astropy.units.Quantity`
        Scalar length of the exposure, in any time unit.

    Returns
    -------
    `xarray.Dataset`
        ``spectrum`` with ``flux`` times the exposure time, e.g. ``ph`` for ``ph / s``, and
        the scalar coordinate ``exposure_time`` in seconds, which `remove_exposure_time`
        reads.

    Raises
    ------
    ValueError
        If ``flux`` has no units or is not per unit time, ``exposure_time`` is not a
        positive scalar, or the spectrum already carries an exposure time.
    """
    flux_unit = require_unit(spectrum, "flux", "spectrum.flux")
    # A data variable counts too: the coordinate would silently replace it.
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
        Spectrum from `apply_exposure_time`, with ``flux`` units and the ``exposure_time``
        coordinate.

    Returns
    -------
    `xarray.Dataset`
        ``spectrum`` without ``exposure_time``, its ``flux`` divided by it and so per
        second, e.g. ``ph / s`` for ``ph``.

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
        A spectrum or a response with a ``line`` dimension and a ``channel`` coordinate
        along it.
    sum_slits : `bool`, optional
        Also sum over ``slit``, if present, by default `False`.

    Returns
    -------
    `xarray.Dataset`
        ``data`` with ``channel`` in place of ``line``. Data variables along ``line`` are
        summed per channel, a NaN staying NaN. Coordinates along ``line``, such as
        ``gain``, take the value of the first line of the channel; ``line_wavelength``
        too where all lines of the channel share it, and NaN otherwise. Channels keep the
        order they first appear in. With ``sum_slits``, coordinates along ``slit`` are
        dropped.

    Raises
    ------
    ValueError
        If ``data`` has no ``line`` dimension or no ``channel`` coordinate along it.

    Notes
    -----
    Sum after converting photons to electrons or DN: a channel whose lines differ in
    wavelength has no single photon energy, and `muse.instrument.transform_flux_units`
    refuses to take it across the photon step.
    """
    if "line" not in data.dims or "channel" not in data.coords or data.channel.dims != ("line",):
        msg = "data must have a line dimension and a channel coordinate along it"
        raise ValueError(msg)
    channels, first_line = np.unique(data.channel.values, return_index=True)
    channels, first_line = channels[np.argsort(first_line)], np.sort(first_line)
    # The first line of each channel gives the coordinates; the sums replace the variables.
    result = data.isel(line=first_line).swap_dims(line="channel").drop_vars("line")
    for name, variable in data.data_vars.items():
        if "line" not in variable.dims:
            continue
        summed = variable.groupby("channel").sum(skipna=False, keep_attrs=True).sel(channel=channels)
        result[name] = summed.transpose(*("channel" if dim == "line" else dim for dim in variable.dims))
    if "line_wavelength" in data.coords and data.line_wavelength.dims == ("line",):
        wavelengths = data.line_wavelength.groupby("channel")
        mixed = (wavelengths.max(skipna=False) != wavelengths.min(skipna=False)).sel(channel=channels)
        result = result.assign_coords(line_wavelength=result.line_wavelength.where(~mixed.values))
    if sum_slits and "slit" in result.dims:
        result = result.sum("slit", skipna=False, keep_attrs=True)
    add_history(result, locals(), sum_lines_per_channel)
    return result
