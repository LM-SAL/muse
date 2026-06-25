import string

import numpy as np
import xarray as xr

import astropy.units as u

from muse.log import logger
from muse.utils.documentation import format_docstring
from muse.utils.utils import (
    _resolve_backend,
    add_history,
    jax_to_numpy,
    numpy_to_jax,
    numpy_to_torch,
    torch_to_numpy,
    update_attrs,
)
from muse.variables import DEFAULTS_MUSE

__all__ = ["vdem_synthesis"]


def _array_unit(ds: xr.Dataset, name: str, label: str, *, convertible_to=None):
    try:
        array = ds[name]
    except KeyError as exc:
        msg = f"{label} is missing"
        raise ValueError(msg) from exc
    if "units" not in array.attrs:
        msg = f"{label} must define units"
        raise ValueError(msg)
    try:
        unit = u.Unit(array.attrs["units"])
    except (TypeError, ValueError) as exc:
        msg = f"{label} units must be a valid astropy unit"
        raise ValueError(msg) from exc
    if convertible_to is not None:
        try:
            unit.to(convertible_to)
        except u.UnitConversionError as exc:
            msg = f"{label} units must be convertible to {convertible_to}"
            raise ValueError(msg) from exc
    return unit


def _calc_einsum(
    *,
    raster: xr.Dataset,
    response: xr.Dataset,
    einsum_str: str,
    out_str: str,
    cuda_device: int | None = None,
    backend: str | None = "numpy",
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
    backend : `str` or `None`, optional
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
                numpy_to_torch(response.SG_resp.data, cuda_device=cuda_device),
            )
        )
    if backend == "jax":
        import jax  # NOQA: PLC0415
        import jax.numpy as jnp  # NOQA: PLC0415

        return jax_to_numpy(
            jnp.einsum(
                f"{einsum_str}->{out_str}",
                numpy_to_jax(raster.vdem.data, cuda_device=cuda_device),
                numpy_to_jax(response.SG_resp.data, cuda_device=cuda_device),
                precision=jax.lax.Precision.HIGHEST,
            )
        )
    return np.einsum(
        f"{einsum_str}->{out_str}",
        np.asarray(raster.vdem.data),
        np.asarray(response.SG_resp.data),
    )


def _build_einsum_indices(raster_dims, response_dims, sum_over):
    """
    Build the einsum spec for contracting the VDEM raster with the response.

    Each unique dimension name gets one index letter; dimensions shared by both
    operands reuse the same letter (so einsum contracts over them). The output
    keeps every dimension not in ``sum_over``, in raster-then-response order.

    Parameters
    ----------
    raster_dims, response_dims : `tuple` of `str`
        Dimension names of ``raster.vdem`` and ``response.SG_resp``.
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


@format_docstring("DEFAULTS_MUSE", sum_over="sum_over_dims_synthesis")
def vdem_synthesis(
    raster: xr.Dataset,
    response: xr.Dataset,
    *,
    sum_over=DEFAULTS_MUSE.sum_over_dims_synthesis,
    cuda_device: int | None = None,
    backend: str | None = "numpy",
) -> xr.Dataset:
    """
    Given a VDEM raster, and response function(s) synthesize observables by
    computing the tensor product.

    Parameters
    ----------
    raster : `xarray.Dataset`
        VDEM raster. ``vdem`` must define units in the attrs.
    response : `xarray.Dataset`
        Response functions. ``SG_resp``, ``line_wvl``,
        and ``SG_wvl`` must define units in the attrs.
    sum_over : `tuple` of `str`
        Dimensions to sum over, by default {sum_over}.
    cuda_device : `int`, optional
        CUDA device index for GPU use (requires ``backend="jax"`` or ``"torch"``), defaults to None (CPU).
    backend : `str` or `None`, optional
        ``"numpy"`` (default), ``"jax"``, or ``"torch"``. JAX and Torch are
        opt-in: neither is selected implicitly, so results do not change with
        what is installed. The JAX and Torch paths downcast float64 inputs to
        float32; the NumPy path keeps the input dtype.

    Returns
    -------
    `xarray.Dataset`
        Dataset of the spectrum on the detector.
    """
    for dim in sum_over:
        if dim not in response.dims:
            msg = f"{dim!r} is not a response dimension"
            raise ValueError(msg)

    raster_vdem_unit = _array_unit(raster, "vdem", "raster.vdem")
    response_sg_resp_unit = _array_unit(response, "SG_resp", "response.SG_resp")
    _array_unit(response, "line_wvl", "response.line_wvl", convertible_to=u.AA)
    _array_unit(response, "SG_wvl", "response.SG_wvl", convertible_to=u.AA)

    einsum_str, out_str, dims = _build_einsum_indices(raster.vdem.dims, response.SG_resp.dims, sum_over)

    logger.debug(
        f"einsum {einsum_str}->{out_str}: vdem{np.shape(raster.vdem.data)} x SG_resp{np.shape(response.SG_resp.data)}"
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

    coords_depending_on_summed_dim = []
    for dim in sum_over:
        coords_depending_on_summed_dim.append(
            {coord_name for coord_name, coord in response.SG_resp.coords.items() if dim in coord.dims}
        )
        coords_depending_on_summed_dim.append(
            {coord_name for coord_name, _ in raster.vdem.coords.items() if dim in raster.vdem.dims}
        )
    coords_depending_on_summed_dim = set.union(*coords_depending_on_summed_dim)

    response_coords = set(response.SG_resp.coords) - coords_depending_on_summed_dim
    raster_coords = set(raster.coords) - coords_depending_on_summed_dim

    coords = {}
    for key in response_coords:
        coords[key] = response.coords[key]
    for key in raster_coords - response_coords:
        coords[key] = raster.coords[key]

    logger.debug(f"flux {tuple(dims)} shape {np.shape(einsum_result)}")
    da = xr.DataArray(data=einsum_result, dims=dims, coords=coords)
    ds["flux"] = da

    ds.flux.attrs.update({"units": str(raster_vdem_unit * response_sg_resp_unit)})

    # SG_wvl carries a slit dimension, so only attach it when slit survives in the
    # output (or the response never had one); otherwise it would re-introduce slit.
    slit_preserved = "slit" not in response.SG_resp.dims or "slit" in ds.flux.dims
    if slit_preserved and "SG_wvl" in response:
        ds = ds.assign_coords(SG_wvl=response.SG_wvl)

    add_history(ds, locals(), vdem_synthesis)
    return ds
