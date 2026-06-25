from functools import cache

import numpy as np
import xarray as xr

from muse.utils.utils import _use_jax, add_history, jax_to_numpy, numpy_to_jax

__all__ = ["create_simple_vdem"]


def _create_vdem_array_numpy(temperature, velocity, ne_nh, cell_length, velocity_axis, log_temperature_axis):
    n_velocity_bins = len(velocity_axis)
    n_temperature_bins = len(log_temperature_axis)

    log_temperature_bin_width = log_temperature_axis[1] - log_temperature_axis[0]
    velocity_bin_width = velocity_axis[1] - velocity_axis[0]

    temperature_prev = np.roll(temperature, 1, axis=2)
    temperature_prev[:, :, 0] = 100.0
    max_temperature = np.maximum(temperature, temperature_prev)
    min_temperature = np.minimum(temperature, temperature_prev)

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
            los_integrand = ne_nh * bin_fraction * voxel_mask * cell_length.reshape(1, 1, -1)
            vdem[i_temperature, i_velocity, ...] = los_integrand.sum(axis=2)
    return vdem


@cache
def _create_vdem_array_jax_kernel():
    import jax  # NOQA: PLC0415 - optional backend
    import jax.numpy as jnp  # NOQA: PLC0415 - optional backend

    @jax.jit
    def kernel(temperature, velocity, ne_nh, cell_length, velocity_axis, log_temperature_axis):
        # Rolled lax loops, not Python for-loops: under jit a Python loop unrolls into
        # n_temperature_bins * n_velocity_bins graph copies (slow compile, big memory).
        # The temperature loop must stay a loop anyway -- each voxel spreads across
        # several T bins, so vectorising T would materialise (n_T, x, y, z).
        n_velocity_bins = velocity_axis.shape[0]
        n_temperature_bins = log_temperature_axis.shape[0]

        log_temperature_bin_width = log_temperature_axis[1] - log_temperature_axis[0]
        velocity_bin_width = velocity_axis[1] - velocity_axis[0]
        cell_length = cell_length.reshape(1, 1, -1)

        temperature_prev = jnp.roll(temperature, 1, axis=2)
        temperature_prev = temperature_prev.at[:, :, 0].set(100.0)
        max_temperature = jnp.maximum(temperature, temperature_prev)
        min_temperature = jnp.minimum(temperature, temperature_prev)

        def temperature_step(i_temperature, vdem):
            bin_lo = 10.0 ** (log_temperature_axis[i_temperature] - log_temperature_bin_width / 2.0)
            bin_hi = 10.0 ** (log_temperature_axis[i_temperature] + log_temperature_bin_width / 2.0)
            log_temperature_clipped = jnp.log10(jnp.clip(temperature, bin_lo, bin_hi))
            log_temperature_prev_clipped = jnp.log10(jnp.clip(temperature_prev, bin_lo, bin_hi))
            bin_fraction = jnp.abs(log_temperature_prev_clipped - log_temperature_clipped) / log_temperature_bin_width
            temperature_mask = (max_temperature >= bin_lo) & (min_temperature < bin_hi)
            weight = ne_nh * bin_fraction * temperature_mask * cell_length

            def velocity_step(i_velocity, vdem):
                voxel_mask = (velocity >= velocity_axis[i_velocity] - velocity_bin_width / 2.0) & (
                    velocity < velocity_axis[i_velocity] + velocity_bin_width / 2.0
                )
                contribution = (weight * voxel_mask).sum(axis=2)
                return vdem.at[i_temperature, i_velocity].set(contribution)

            return jax.lax.fori_loop(0, n_velocity_bins, velocity_step, vdem)

        vdem = jnp.zeros((n_temperature_bins, n_velocity_bins, *velocity.shape[:2]), dtype=ne_nh.dtype)
        return jax.lax.fori_loop(0, n_temperature_bins, temperature_step, vdem)

    return kernel


def create_simple_vdem(
    temperature,
    velocity,
    ne_nh,
    cell_length,
    x,
    y,
    velocity_axis,
    log_temperature_axis,
    *,
    cuda_device: int | None = None,
    backend: str | None = None,
):
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
    cuda_device : int or None, optional
        CUDA device index, or None for CPU. Default is None.
    backend : str or None, optional
        Force ``"jax"`` or ``"numpy"``. If None (default), use JAX when it is
        installed and fall back to NumPy otherwise.

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

    # Snapshot the numpy inputs for provenance before they are converted to tensors.
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

    if _use_jax(cuda_device, backend):
        cell_length = numpy_to_jax(cell_length.astype(np.float32, copy=False), cuda_device=cuda_device)
        velocity, temperature, ne_nh, velocity_axis, log_temperature_axis = [
            numpy_to_jax(q.astype(np.float32, copy=False), cuda_device=cuda_device)
            for q in [velocity, temperature, ne_nh, velocity_axis, log_temperature_axis]
        ]
        vdem = _create_vdem_array_jax_kernel()(
            temperature, velocity, ne_nh, cell_length, velocity_axis, log_temperature_axis
        )
        log_temperature_axis, velocity_axis, vdem = [
            jax_to_numpy(q) for q in [log_temperature_axis, velocity_axis, vdem]
        ]
    else:
        cell_length = cell_length.astype(np.float32, copy=False)
        velocity, temperature, ne_nh, velocity_axis, log_temperature_axis = [
            q.astype(np.float32, copy=False)
            for q in [velocity, temperature, ne_nh, velocity_axis, log_temperature_axis]
        ]
        vdem = _create_vdem_array_numpy(temperature, velocity, ne_nh, cell_length, velocity_axis, log_temperature_axis)

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
