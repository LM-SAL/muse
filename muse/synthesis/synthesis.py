import string
from collections.abc import Hashable, Sequence

import numpy as np
import xarray as xr

import astropy.units as u

from muse.log import logger
from muse.utils.documentation import format_docstring
from muse.utils.utils import (
    _resolve_backend,
    add_history,
    coord_as_unit,
    jax_to_numpy,
    numpy_to_jax,
    numpy_to_torch,
    require_unit,
    torch_to_numpy,
    update_attrs,
)
from muse.variables import DEFAULTS_MUSE

__all__ = ["vdem_synthesis"]


def _calc_einsum(
    *,
    raster: xr.Dataset,
    response: xr.Dataset,
    einsum_str: str,
    out_str: str,
    cuda_device: int | None = None,
    backend: str = "numpy",
):
    """
    Compute the tensor product using the selected array backend.

    Parameters
    ----------
    raster : `xarray.Dataset`
        VDEM raster dataset.
    response : `xarray.Dataset`
        Response function dataset.
    einsum_str : `str`
        Einsum input string.
    out_str : `str`
        Einsum output string.
    cuda_device : `int` or `None`, optional
        CUDA device index for GPU use (requires ``backend="jax"`` or ``"torch"``), or None for CPU.
    backend : `str`, optional
        ``"numpy"`` (default), ``"jax"``, or ``"torch"``. JAX and Torch are opt-in.

    Returns
    -------
    array-like
        Result of the einsum operation.
    """
    backend = _resolve_backend(cuda_device, backend)
    logger.debug(f"Using {backend} for synthesis")
    if backend == "torch":
        import torch  # NOQA: PLC0415

        return torch_to_numpy(
            torch.einsum(
                f"{einsum_str}->{out_str}",
                numpy_to_torch(raster.vdem.data, cuda_device=cuda_device),
                numpy_to_torch(response.detector_response.data, cuda_device=cuda_device),
            )
        )
    if backend == "jax":
        import jax  # NOQA: PLC0415
        import jax.numpy as jnp  # NOQA: PLC0415

        return jax_to_numpy(
            jnp.einsum(
                f"{einsum_str}->{out_str}",
                numpy_to_jax(raster.vdem.data, cuda_device=cuda_device),
                numpy_to_jax(response.detector_response.data, cuda_device=cuda_device),
                precision=jax.lax.Precision.HIGHEST,
            )
        )
    return np.einsum(
        f"{einsum_str}->{out_str}",
        np.asarray(raster.vdem.data),
        np.asarray(response.detector_response.data),
    )


def _build_einsum_indices(
    raster_dims: tuple[Hashable, ...],
    response_dims: tuple[Hashable, ...],
    sum_over: Sequence[str],
) -> tuple[str, str, list[Hashable]]:
    """
    Build the einsum spec for contracting the VDEM raster with the response.

    Each unique dimension name gets one index letter; dimensions shared by both
    operands reuse the same letter (so einsum contracts over them). The output
    keeps every dimension not in ``sum_over``, in raster-then-response order.

    Parameters
    ----------
    raster_dims, response_dims : `tuple` of `str`
        Dimension names of ``raster.vdem`` and ``response.detector_response``.
    sum_over : `tuple` of `str`
        Dimension names to contract over.

    Returns
    -------
    einsum_str : `str`
        Input spec, e.g. ``"abcde,fbcdg"``.
    out_str : `str`
        Output spec for the non-summed dimensions.
    out_dims : `list` of `str`
        Output dimension names, aligned with ``out_str``.
    """
    letters = iter(string.ascii_lowercase)
    dim_to_letter = {}
    for dim in (*raster_dims, *response_dims):
        if dim not in dim_to_letter:
            dim_to_letter[dim] = next(letters)

    out_dims = []
    for dim in (*raster_dims, *response_dims):
        if dim not in sum_over and dim not in out_dims:
            out_dims.append(dim)

    raster_spec = "".join(dim_to_letter[dim] for dim in raster_dims)
    response_spec = "".join(dim_to_letter[dim] for dim in response_dims)
    out_str = "".join(dim_to_letter[dim] for dim in out_dims)
    return f"{raster_spec},{response_spec}", out_str, out_dims


def _validate_inputs(
    raster: xr.Dataset, response: xr.Dataset, sum_over: Sequence[str]
) -> tuple[u.UnitBase, u.UnitBase]:
    """
    Validate ``raster``/``response`` structure and units for synthesis.

    Checks slit-dimension consistency, that every ``sum_over`` dimension exists on
    the response, and that ``vdem``/``detector_response`` define valid units
    while ``line_wavelength``/``detector_wavelength`` are coordinates with
    wavelength units.

    Returns
    -------
    `tuple` of `astropy.units.Unit`
        ``(raster.vdem unit, response.detector_response unit)``, reused for the
        flux unit.
    """
    if "slit" in response.dims and "slit" not in raster.dims and response.slit.size > 1:
        msg = "response has a slit dimension and number of slits are greater than one, but raster does not"
        raise ValueError(msg)
    if "slit" not in response.dims and "slit" in raster.dims:
        msg = "raster has a slit dimension but response does not"
        raise ValueError(msg)
    for dim in sum_over:
        if dim not in response.dims:
            msg = f"{dim!r} is not a response dimension"
            raise ValueError(msg)
    raster_vdem_unit = require_unit(raster, "vdem", "raster.vdem")
    response_unit = require_unit(response, "detector_response", "response.detector_response")
    require_unit(response, "line_wavelength", "response.line_wavelength", coord_only=True, convertible_to=u.AA)
    require_unit(
        response,
        "detector_wavelength",
        "response.detector_wavelength",
        coord_only=True,
        convertible_to=u.AA,
    )
    return raster_vdem_unit, response_unit


@format_docstring("DEFAULTS_MUSE", sum_over="sum_over_dims_synthesis")
def vdem_synthesis(
    raster: xr.Dataset,
    response: xr.Dataset,
    *,
    sum_over: Sequence[str] = DEFAULTS_MUSE.sum_over_dims_synthesis,
    cuda_device: int | None = None,
    backend: str = "numpy",
) -> xr.Dataset:
    """
    Given a VDEM raster, and response function(s) synthesize observables by computing
    the tensor product.

    Parameters
    ----------
    raster : `xarray.Dataset`
        VDEM raster. ``vdem`` must define units in the attrs.
    response : `xarray.Dataset`
        Response functions. ``detector_response``, ``line_wavelength``, and
        ``detector_wavelength`` must define units in the attrs.
    sum_over : `tuple` of `str`
        Dimensions to sum over, by default {sum_over}.
    cuda_device : `int`, optional
        CUDA device index for GPU use (requires ``backend="jax"`` or ``"torch"``), defaults to None (CPU).
    backend : `str`, optional
        ``"numpy"`` (default), ``"jax"``, or ``"torch"``. JAX and Torch are
        opt-in: neither is selected implicitly, so results do not change with
        what is installed. The JAX and Torch paths downcast float64 inputs to
        float32; the NumPy path keeps the input dtype.

    Returns
    -------
    `xarray.Dataset`
        Dataset of the spectrum on the detector.
    """
    raster_vdem_unit, response_unit = _validate_inputs(raster, response, sum_over)
    einsum_str, out_str, dims = _build_einsum_indices(raster.vdem.dims, response.detector_response.dims, sum_over)
    logger.debug(
        f"einsum {einsum_str}->{out_str}: "
        f"vdem{np.shape(raster.vdem.data)} x detector_response{np.shape(response.detector_response.data)}"
    )
    einsum_result = _calc_einsum(
        raster=raster,
        response=response,
        einsum_str=einsum_str,
        out_str=out_str,
        cuda_device=cuda_device,
        backend=backend,
    )
    ds = xr.Dataset()
    update_attrs(ds, raster)
    update_attrs(ds, response)
    for key in dims:
        ds[key] = raster[key] if key in raster.vdem.dims else response[key]
    out_dims = set(dims)
    coords = {name: coord for name, coord in raster.coords.items() if set(coord.dims) <= out_dims}
    coords.update(
        {name: coord for name, coord in response.detector_response.coords.items() if set(coord.dims) <= out_dims}
    )

    logger.debug(f"flux {tuple(dims)} shape {np.shape(einsum_result)}")
    da = xr.DataArray(data=einsum_result, dims=dims, coords=coords)
    ds["flux"] = da
    ds = ds.assign_coords(line_wavelength=coord_as_unit(response, "line_wavelength", u.AA, "response.line_wavelength"))
    ds.flux.attrs.update({"units": str(raster_vdem_unit * response_unit)})
    # detector_wavelength carries a slit dimension, so only attach it when slit
    # survives in the output (or the response never had one); otherwise it would
    # re-introduce slit.
    slit_preserved = "slit" not in response.detector_response.dims or "slit" in ds.flux.dims
    if slit_preserved and "detector_wavelength" in response.coords:
        ds = ds.assign_coords(
            detector_wavelength=coord_as_unit(
                response,
                "detector_wavelength",
                u.AA,
                "response.detector_wavelength",
            )
        )
    add_history(ds, locals(), vdem_synthesis)
    return ds
