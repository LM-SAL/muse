import string
import contextlib

import numpy as np
import torch
import xarray as xr

import astropy.units as u

from muse.log import logger
from muse.utils.documentation import format_docstring
from muse.utils.utils import add_history, numpy_to_torch, torch_to_numpy
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
):
    """
    Compute the tensor product using torch.einsum, optionally on a CUDA device.

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
        CUDA device index for GPU use, or None for CPU.

    Returns
    -------
    `numpy.ndarray`
        Result of the einsum operation.
    """
    device_context = torch.cuda.device(f"cuda:{cuda_device}") if cuda_device is not None else contextlib.nullcontext()
    with device_context:
        logger.debug(f"Using torch on cuda:{cuda_device}" if cuda_device is not None else "Using CPU with torch")
        result = torch.einsum(
            f"{einsum_str}->{out_str}",
            numpy_to_torch(raster.vdem.data, cuda_device=cuda_device).float(),
            numpy_to_torch(response.SG_resp.data, cuda_device=cuda_device).float(),
        )
    return torch_to_numpy(result)


@format_docstring("DEFAULTS_MUSE", sum_over="sum_over_dims_synthesis")
def vdem_synthesis(
    raster: xr.Dataset,
    response: xr.Dataset,
    *,
    sum_over=DEFAULTS_MUSE.sum_over_dims_synthesis,
    cuda_device: int | None = None,
) -> xr.Dataset:
    """
    Given a VDEM raster, and response function(s) synthesize observables by
    computing the tensor product.

    Parameters
    ----------
    raster : `xarray.Dataset`
        VDEM raster (e.g., from raster_simulation_vdem). ``vdem`` must define units.
    response : `xarray.Dataset`
        Response functions (e.g., from `read_response`). ``SG_resp``, ``line_wvl``,
        and ``SG_wvl`` must define units.
    sum_over : `tuple(str)`
        Dimensions to sum over, by default {sum_over}.
    cuda_device : `int`, optional
        CUDA device index for GPU use, defaults to None (CPU).

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

    index_list = list(string.ascii_lowercase)
    index_dim_dict = {}
    einsum_str = ""
    out_str = ""
    for ik, k in enumerate(raster.vdem.dims):
        einsum_str += index_list[ik]
        index_dim_dict[k] = index_list[ik]
        if k not in sum_over and index_list[ik] not in out_str:
            out_str += index_list[ik]
    einsum_str += ","
    for ij, j in enumerate(response.SG_resp.dims):
        if j in raster.vdem.dims:
            einsum_str += index_dim_dict[j]
            index = index_dim_dict[j]
        else:
            index_dim_dict[j] = index_list[ik + ij + 1]
            einsum_str += index_list[ik + ij + 1]
            index = index_list[ik + ij + 1]
        if j not in sum_over and out_str.find(index) == -1:
            out_str += index_list[ik + ij + 1]

    index_coord_dict = {}
    for ik, k in enumerate(raster.vdem.dims):
        index_coord_dict[k] = index_list[ik]

    for ij, j in enumerate(response.SG_resp.dims):
        if j not in raster.vdem.dims:
            index_coord_dict[j] = index_list[ik + ij + 1]

    logger.debug(f"raster dims {raster.dims}")
    logger.debug(f"response.SG_resp dims {response.SG_resp.dims}")
    logger.debug(f"einsum in {einsum_str}, einsum out {out_str}")
    logger.debug(
        f"shape of: raster.vdem {np.shape(raster.vdem.data)} of response.SG_resp{np.shape(response.SG_resp.data)}",
    )

    einsum_result = _calc_einsum(
        raster=raster,
        response=response,
        einsum_str=einsum_str,
        out_str=out_str,
        cuda_device=cuda_device,
    )
    ds = xr.Dataset()
    ds.attrs.update(raster.attrs)
    ds.attrs.update(response.attrs)
    dims = []
    for key, value in index_dim_dict.items():
        if value in out_str:
            ds[key] = raster[key] if key in raster.vdem.dims else response[key]
            dims.append(key)

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

    logger.debug(f"Shape of result: {np.shape(einsum_result)}")
    logger.debug(f"dims: {dims}")
    logger.debug(f"index_dim_dict: {index_dim_dict.keys()}")
    logger.debug(f"coords: {coords}")
    da = xr.DataArray(data=einsum_result, dims=dims, coords=coords)
    ds["flux"] = da

    ds.flux.attrs.update({"units": str(raster_vdem_unit * response_sg_resp_unit)})

    if "slit" not in response.SG_resp.dims or ("slit" in ds.flux.dims):
        if ("SG_wvl" in response.coords) and ("SG_wvl" not in ds.coords):
            ds = ds.assign_coords(SG_wvl=response.coords["SG_wvl"])

        try:
            ds["SG_wvl"] = response.SG_wvl
            ds = ds.set_coords("SG_wvl")
        except Exception:  # NOQA: BLE001
            logger.debug("No SG_wvl found.")

    add_history(ds, locals(), vdem_synthesis)
    return ds
