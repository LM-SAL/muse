from typing import TYPE_CHECKING
from collections.abc import Sequence

import dask.array as da
import numpy as np
import xarray as xr

import astropy.units as u

from muse.log import logger
from muse.synthesis._backends import _resolve_backend, torch_einsum
from muse.utils.utils import add_history, coord_as_unit, require_unit, update_attrs

if TYPE_CHECKING:
    import torch

__all__ = ["vdem_synthesis"]

# The names of create_spectral_response, accepted for the detector names.
_WAVELENGTH_SPACE_ALIASES = {
    "spectral_response": "detector_response",
    "wavelength_grid": "detector_wavelength",
}
# A response with neither of these as a dimension has no spectral axis: it is an imager's.
_SPECTRAL_DIMS = {"detector_x_pixel", "wavelength_bin"}


def _single_chunk_over(array: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
    """
    Rechunk ``dims`` of a dask-backed array to one chunk; return numpy arrays unchanged.

    Otherwise ``da.einsum`` pairs every chunk of one operand with every chunk of the
    other, which explodes the task graph, runtime and peak memory.
    """
    if not isinstance(array.data, da.Array):
        return array
    return array.chunk({dim: -1 for dim in dims if dim in array.dims})


def _check_shared_indexes(vdem: xr.DataArray, other: xr.DataArray, name: str) -> None:
    # join="exact" checks only indexed dims; one indexed on one side would contract by position.
    for dim in sorted(set(vdem.dims) & set(other.dims)):
        if (dim in vdem.xindexes) != (dim in other.xindexes):
            unindexed = name if dim in vdem.xindexes else "vdem"
            regrid = " or regrid with muse.instrument.align_response_and_vdem" if name == "response" else ""
            msg = (
                f"{dim!r} has an index coordinate on only one input ({unindexed} lacks it), so labels "
                f"cannot be aligned; add the coordinate with assign_coords{regrid}"
            )
            raise ValueError(msg)


def _density_index(vdem: xr.DataArray, density: xr.DataArray, response: xr.Dataset) -> xr.DataArray:
    """
    Each ``vdem`` bin's fractional position on the ``logD`` grid of ``response``, which
    is sorted, at its ``density``, held at the ends beyond the grid.
    """
    if "logD" in vdem.dims:
        msg = (
            "vdem.vdem already has a logD axis, which carries its density; density is only for a VDEM without one, "
            "so synthesize this one without density and with 'logD' in sum_over"
        )
        raise ValueError(msg)
    unit = require_unit(density.to_dataset(name="density"), "density", "density", convertible_to=u.cm**-3)
    if not set(density.dims) <= set(vdem.dims):
        msg = f"density dimensions {density.dims} must be among those of vdem.vdem {vdem.dims}"
        raise ValueError(msg)
    _check_shared_indexes(vdem, density, "density")
    grid = coord_as_unit(response, "logD", u.dex(u.cm**-3), "response.logD").values
    if np.unique(grid).size < grid.size:
        msg = "response.logD repeats a density, so the planes to interpolate between are ambiguous"
        raise ValueError(msg)
    density = unit.to(u.cm**-3, density)
    # A bin without emission adds nothing, not NaN: its weighted mean density is 0/0, stored as 0 or NaN.
    # xr.where joins exactly, so labels that differ raise.
    log_density = xr.where(vdem != 0, np.log10(density.where(density > 0)), grid[0])
    index = xr.apply_ufunc(np.interp, log_density, kwargs={"xp": grid, "fp": np.arange(grid.size)}, dask="parallelized")
    if grid.size == 1:
        # np.interp returns the only plane even for a NaN query.
        index = index.where(log_density.notnull())
    # np.interp returns float64; cast so a float32 VDEM keeps float32 planes and flux.
    return index.astype(np.result_type(vdem.dtype, 0.0))


def _validate_inputs(vdem: xr.Dataset, response: xr.Dataset, sum_over: Sequence[str]) -> tuple[u.UnitBase, u.UnitBase]:
    """
    Check the slits, the legacy ``vdop`` axis, ``sum_over``, indexes and units.

    Returns
    -------
    `tuple` of `astropy.units.Unit`
        The units of ``vdem.vdem`` and ``response.detector_response``.
    """
    if "slit" in response.dims and "slit" not in vdem.dims and response.slit.size > 1:
        msg = "response has a slit dimension and number of slits are greater than one, but vdem does not"
        raise ValueError(msg)
    # An imager response images each slit position on its own, so only the VDEM may have slit.
    if "slit" not in response.dims and "slit" in vdem.dims and _SPECTRAL_DIMS & set(response.dims):
        msg = "vdem has a slit dimension but response does not"
        raise ValueError(msg)
    # 'vdop' is not in sum_over: einsum would keep it and return a wrong flux without an error.
    for name, dataset in (("vdem", vdem), ("response", response)):
        if "vdop" in dataset.dims:
            msg = f"{name} uses the legacy 'vdop' dimension; rename it with {name}.rename(vdop='doppler_velocity')"
            raise ValueError(msg)
    for dim in sum_over:
        if dim not in response.dims:
            msg = f"{dim!r} is not a response dimension"
            raise ValueError(msg)
    vdem_unit = require_unit(vdem, "vdem", "vdem.vdem")
    response_unit = require_unit(response, "detector_response", "response.detector_response")
    _check_shared_indexes(vdem.vdem, response.detector_response, "response")
    require_unit(response, "line_wavelength", "response.line_wavelength", coord_only=True, convertible_to=u.AA)
    if "slit" not in response.detector_response.dims or "slit" not in sum_over:
        require_unit(
            response,
            "detector_wavelength",
            "response.detector_wavelength",
            coord_only=True,
            convertible_to=u.AA,
        )
    return vdem_unit, response_unit


def vdem_synthesis(
    vdem: xr.Dataset,
    response: xr.Dataset,
    *,
    density: xr.DataArray | None = None,
    sum_over: Sequence[str] = ("logT", "doppler_velocity", "slit"),
    device: "str | torch.device | None" = None,
    backend: str = "numpy",
) -> xr.Dataset:
    """
    Synthesize observables as the tensor product of a VDEM and a response.

    Parameters
    ----------
    vdem : `xarray.Dataset`
        VDEM, as it is or reshaped onto slits and raster steps, laid out as
        `muse.synthesis.create_simple_vdem` returns it: a ``vdem`` variable over ``logT``
        (dex(K)) and ``doppler_velocity`` (km/s, negative towards the observer), with its
        units in the attrs, each bin holding the column emission measure of n_e n_H, not
        divided by the bin widths (see that function's Notes).
    response : `xarray.Dataset`
        Response functions. ``detector_response`` and ``line_wavelength`` must define
        units, and ``detector_wavelength`` too when its slit dimension stays in the output.
        Indexed coordinates shared with ``vdem`` must match exactly; a dimension indexed
        on only one input is rejected. The names of
        `muse.instrument.create_spectral_response` (``spectral_response``,
        ``wavelength_grid``) are renamed to the detector names; such a response is still
        per Å per steradian, with no detector mapping or photon conversion. An imager
        response, from `muse.instrument.map_response_to_ci_detector`, has no slit
        dimension and no spectral axis; it images each slit of the VDEM on its own, and
        ``slit`` stays in the output.
    density : `xarray.DataArray`, optional
        Emission-measure-weighted mean electron density of each ``vdem`` bin, the
        integral of n_e^2 n_H dl over that of n_e n_H dl for the plasma in the bin.
        :cite:t:`mckevitt_2026` evaluate the contribution function at such a density for
        each temperature bin. It needs units convertible to cm^-3 in its attrs, and lies
        over any of the dimensions of ``vdem.vdem``, with its labels. The response, which
        then needs the ``logD`` axis of a line list made with a ``density``, is
        interpolated linearly in ``logD`` at each bin's density and held at the end planes
        beyond the grid; ``logD`` joins ``sum_over``. Every plane enters the sum, so a NaN
        on any of them makes the flux of its line NaN in every pixel. A bin without
        emission adds nothing whatever its density; one with emission and a NaN or
        nonpositive density makes the flux it adds to NaN. To resample the VDEM with
        `muse.transforms.match_fov`, which resamples each variable on its own, resample
        ``density * vdem.vdem`` with it, with ``density`` in cm^-3 and set to 0 where
        there is no emission (NaN would spread), and divide by the resampled
        ``vdem.vdem``; this does not hold with a ``pad_mode`` of ``"maximum"``,
        ``"minimum"`` or ``"median"``, which are not linear.
    sum_over : `tuple` of `str`
        Dimensions to sum over, by default ``("logT", "doppler_velocity", "slit")``.
        ``"slit"`` is dropped when the response has no slit dimension.
    device : `str`, `torch.device` or `None`, optional
        Torch device of the product (needs ``backend="torch"``): ``"cpu"``, ``"cuda:0"``,
        ``"mps"``, or ``"auto"`` for the first available of CUDA, MPS and the CPU. By
        default `None`, the CPU.
    backend : `str`, optional
        ``"numpy"`` (default) or ``"torch"``, never chosen implicitly. Torch computes in
        float32, in chunks sized to half the free device and host memory; NumPy keeps the
        input dtype.

    Returns
    -------
    `xarray.Dataset`
        The spectrum on the detector, or the image for an imager response, as ``flux``,
        over the dimensions of ``vdem`` and ``response`` not in ``sum_over``, in the product
        of their units (ph/s for a VDEM from `muse.synthesis.create_simple_vdem` and a
        detector response). With NumPy, dask-backed inputs give a lazy ``flux``; Torch
        computes eagerly.

    Raises
    ------
    ValueError
        If an input fails validation (missing units, mismatched slit dimensions, the
        legacy ``vdop`` axis, a ``sum_over`` dimension missing from the response, or a
        shared dimension indexed on only one input), or shared indexed coordinates differ;
        regrid with `muse.instrument.align_response_and_vdem`. Also if ``density`` lacks
        units convertible to cm^-3, has a dimension ``vdem.vdem`` lacks or labels that
        differ from the VDEM's, or the response has no ``logD`` axis or repeats a value on
        it, or ``vdem.vdem`` already has one.
    MemoryError
        With ``backend="torch"`` on an accelerator, if one row of the product does not fit
        next to the response.
    """
    renames = {
        old: new
        for old, new in _WAVELENGTH_SPACE_ALIASES.items()
        if (old in response.variables or old in response.dims) and new not in response.variables
    }
    if renames:
        logger.info(f"Accepting wavelength-space response names, renaming: {renames}")
        response = response.rename(renames)
    if "slit" in sum_over and "slit" not in response.dims:
        logger.info("The response has no slit dimension, dropping 'slit' from sum_over")
        sum_over = tuple(dim for dim in sum_over if dim != "slit")
    if density is not None:
        sum_over = (*sum_over, "logD")
    vdem_unit, response_unit = _validate_inputs(vdem, response, sum_over)
    vdem = vdem.assign(vdem=_single_chunk_over(vdem.vdem, sum_over))
    response = response.assign(detector_response=_single_chunk_over(response.detector_response, sum_over))
    if density is None:
        pairs = [(vdem.vdem, response.detector_response)]
    else:
        response = response.sortby("logD")
        # The density's chunks carry over to each plane's share of the VDEM.
        index = _density_index(vdem.vdem, _single_chunk_over(density, sum_over), response)
        # One plane at a time: all at once would hold the VDEM once per plane.
        pairs = (
            (vdem.vdem * (1 - abs(index - plane)).clip(min=0), response.detector_response.isel(logD=plane, drop=True))
            for plane in range(response.sizes["logD"])
        )
    dim_indices = {
        dim: index for index, dim in enumerate(dict.fromkeys((*vdem.vdem.dims, *response.detector_response.dims)))
    }
    # Shared dims in the VDEM's order, else np.einsum copies the VDEM to line them up.
    response_dims = sorted(
        (dim for dim in response.detector_response.dims if density is None or dim != "logD"),
        key=dim_indices.__getitem__,
    )
    vdem_indices = [dim_indices[dim] for dim in vdem.vdem.dims]
    response_indices = [dim_indices[dim] for dim in response_dims]
    output_dims = [dim for dim in dim_indices if dim not in sum_over]
    output_indices = [dim_indices[dim] for dim in output_dims]
    backend = _resolve_backend(device, backend)
    logger.debug(
        f"Using {backend} to contract vdem{np.shape(vdem.vdem.data)} and "
        f"detector_response{np.shape(response.detector_response.data)} over {tuple(sum_over)} "
        f"-> flux{tuple(output_dims)}"
    )

    def contract(vdem_values, detector_response):
        if backend == "torch":
            return torch_einsum(vdem_values, detector_response, vdem_indices, response_indices, output_indices, device)
        return np.einsum(vdem_values, vdem_indices, detector_response, response_indices, output_indices, optimize=True)

    try:
        flux = sum(
            xr.apply_ufunc(
                contract,
                vdem_values,
                detector_response,
                input_core_dims=[list(vdem.vdem.dims), response_dims],
                output_core_dims=[output_dims],
                join="exact",
                dask="allowed",
                keep_attrs=True,
            )
            for vdem_values, detector_response in pairs
        )
    except xr.AlignmentError as err:
        msg = (
            "vdem and response coordinates disagree on a shared dimension; "
            f"regrid the response with muse.instrument.align_response_and_vdem ({err})"
        )
        raise ValueError(msg) from err
    flux.attrs = {"units": str(vdem_unit * response_unit)}
    ds = flux.to_dataset(name="flux")
    # apply_ufunc drops disagreeing coords: the VDEM's win, except along response-only dims.
    response_only_dims = set(response.detector_response.dims) - set(vdem.vdem.dims)
    restored = {name: coord for name, coord in vdem.coords.items() if set(coord.dims) <= set(output_dims)}
    restored.update(
        {
            name: coord
            for name, coord in response.coords.items()
            if set(coord.dims) <= set(output_dims)
            and (name not in restored or response_only_dims.intersection(coord.dims))
        }
    )
    restored |= {
        dim: (vdem[dim] if dim in vdem.vdem.dims else response[dim])
        for dim in ds.flux.dims
        if dim not in restored and dim not in ds.coords
    }
    ds = ds.assign_coords(restored)
    ds = ds.assign_coords(line_wavelength=coord_as_unit(response, "line_wavelength", u.AA, "response.line_wavelength"))
    # detector_wavelength runs along slit, which it would bring back once summed away.
    slit_summed = "slit" in response.detector_response.dims and "slit" not in ds.flux.dims
    if not slit_summed and "detector_wavelength" in response.coords:
        ds = ds.assign_coords(
            detector_wavelength=coord_as_unit(
                response,
                "detector_wavelength",
                u.AA,
                "response.detector_wavelength",
            )
        )
    update_attrs(ds, vdem)
    update_attrs(ds, response)
    add_history(
        ds, locals(), vdem_synthesis, sources=(vdem, response) if density is None else (vdem, response, density)
    )
    return ds
