"""
Spectral response functions built from CHIANTI line lists.
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr
from scipy.special import erf
from tqdm import tqdm

import astropy.constants as const
import astropy.units as u

from muse.log import logger
from muse.utils.utils import _require_increasing_axis, add_history, coord_as_unit, require_unit, update_attrs
from muse.variables import DEFAULTS_MUSE

__all__ = ["create_continuum_response", "create_log_wavelength_edges", "create_spectral_response"]

# The Gaussian is exactly zero beyond this point in float64.
_GAUSSIAN_WINDOW_SIGMA = 40
_GAUSSIAN_EXPRESSION = "scaled * exp(-0.5 * (shift / width) ** 2) / gaussian_norm / width"
_BIN_METHODS = ("center", "average")


@u.quantity_input(wavelength_range=u.AA)
def create_log_wavelength_edges(wavelength_range: u.Quantity, fractional_width: float) -> u.Quantity:
    """
    Edges of logarithmically spaced wavelength bins covering a range.

    Each bin is ``fractional_width`` wide relative to its lower edge, so every bin spans the
    same velocity width; pass the result to `create_spectral_response` as ``wavelength_edges``
    for a coarse imager grid.

    Parameters
    ----------
    wavelength_range : `astropy.units.Quantity`
        Two-element range to cover; the last edge is at or beyond its upper end.
    fractional_width : `float`
        Bin width relative to its lower edge, e.g. ``0.01``.

    Returns
    -------
    `astropy.units.Quantity`
        The bin edges in Angstrom, one more than the number of bins.
    """
    lower, upper = np.asarray(wavelength_range.to_value(u.AA), dtype=float).reshape(-1)[[0, -1]]
    if not (0 < lower < upper) or not fractional_width > 0:
        msg = (
            "create_log_wavelength_edges needs an increasing positive wavelength_range and a positive fractional_width"
        )
        raise ValueError(msg)
    number_of_bins = int(np.ceil(np.log(upper / lower) / np.log1p(fractional_width)))
    return lower * (1 + fractional_width) ** np.arange(number_of_bins + 1) * u.AA


def create_spectral_response(
    line_list: xr.Dataset,
    wavelength_grid: u.Quantity | None = None,
    *,
    main_lines: Sequence[str] | None,
    instrumental_width: u.Quantity = 0 * u.AA,
    doppler_velocity: u.Quantity | None = None,
    nonthermal_velocity: u.Quantity | None = None,
    effective_area: xr.DataArray | None = None,
    include_contaminants: bool = False,
    wavelength_edges: u.Quantity | None = None,
    bin_method: str | None = None,
) -> xr.Dataset:
    """
    Create an instrument-neutral wavelength-space spectral response.

    Parameters
    ----------
    line_list : `xarray.Dataset`
        CHIANTI line list with its ``logT`` coordinate in ``dex(K)``, e.g. from
        `muse.instrument.create_chianti_line_list`.
    wavelength_grid : `astropy.units.Quantity`, optional
        One-dimensional wavelength samples for one channel and spectral order; the bins
        extend halfway to the neighbouring samples. Either this or ``wavelength_edges``.
    main_lines : sequence of `str` or `None`
        Stable ``full_name`` values to retain, in output order. Repeated
        transitions with the same name are summed. If `None` or empty, no
        named lines are retained and ``include_contaminants`` must be `True`.
    instrumental_width : `astropy.units.Quantity`, optional
        Scalar instrumental-width sigma, by default 0 Angstrom.
    doppler_velocity : `astropy.units.Quantity`, optional
        Scalar or one-dimensional Doppler-velocity axis. If `None`, line
        centers remain at their rest wavelengths and no ``doppler_velocity``
        dimension is added.
    nonthermal_velocity : `astropy.units.Quantity`, optional
        Scalar or one-dimensional nonthermal-velocity axis. If `None`, no
        nonthermal broadening or ``nonthermal_velocity`` dimension is added.
    effective_area : `xarray.DataArray`, optional
        Either a one-dimensional effective-area curve with a unit-bearing
        ``wavelength`` coordinate, or a zero-dimensional scalar area convertible
        to cm**2 (e.g. ``DEFAULTS_MUSE.main_line_effective_area_SG.sel(channel=171)``)
        applied uniformly across the wavelength grid. If `None`, the response
        is not scaled by effective area.
    include_contaminants : `bool`, optional
        If `True`, include all lines not in ``main_lines`` in a single
        ``contaminants`` line. If `False`, only the main lines are returned.
    wavelength_edges : `astropy.units.Quantity`, optional
        Strictly increasing bin edges, one more than the number of bins, instead of
        ``wavelength_grid``; the bin centers become ``wavelength_grid``. Meant for coarse or
        non-uniform bins, such as `create_log_wavelength_edges` for an imager.
    bin_method : {"center", "average"}, optional
        ``"center"`` samples each Gaussian at the bin center and under-counts lines narrower
        than a bin; ``"average"`` stores the bin average, which conserves the line area on
        any grid and weights each line by a curve ``effective_area`` at its own
        Doppler-shifted center. By default ``"center"`` with ``wavelength_grid`` and
        ``"average"`` with ``wavelength_edges``.

    Returns
    -------
    `xarray.Dataset`
        Dataset containing ``spectral_response`` with temperature, velocity,
        line, and ``wavelength_bin`` dimensions. ``wavelength_grid`` contains
        the wavelength of each bin and ``wavelength_bounds`` its two edges.
        Detector geometry is not added.
    """
    call_inputs = dict(locals())
    main_lines = () if main_lines is None else main_lines
    response = _create_wavelength_response(
        line_list,
        wavelength_grid,
        instrumental_width=instrumental_width,
        doppler_velocity=doppler_velocity,
        nonthermal_velocity=nonthermal_velocity,
        effective_area=effective_area,
        main_lines=main_lines,
        include_contaminants=include_contaminants,
        wavelength_edges=wavelength_edges,
        bin_method=bin_method,
    )
    response = response.drop_vars("component_kind")
    update_attrs(response, line_list, **response.attrs)
    add_history(response, call_inputs, create_spectral_response, sources=(line_list,))
    return response


@u.quantity_input(
    wavelength_grid=u.AA,
    instrumental_width=u.AA,
    doppler_velocity=u.km / u.s,
    nonthermal_velocity=u.km / u.s,
    wavelength_edges=u.AA,
)
def _create_wavelength_response(
    line_list: xr.Dataset,
    wavelength_grid: u.Quantity | None = None,
    *,
    instrumental_width: u.Quantity = 0 * u.AA,
    doppler_velocity: u.Quantity | None = None,
    nonthermal_velocity: u.Quantity | None = None,
    effective_area: xr.DataArray | None = None,
    main_lines: Sequence[str] | None = None,
    include_contaminants: bool = False,
    wavelength_edges: u.Quantity | None = None,
    bin_method: str | None = None,
    progress: bool = True,
) -> xr.Dataset:
    """
    Build the wavelength-space response, optionally including contaminants.
    """
    if bin_method is None:
        # Sampling at the centers of coarse bins misses the lines between them altogether.
        bin_method = "center" if wavelength_edges is None else "average"
    if bin_method not in _BIN_METHODS:
        msg = f"bin_method must be one of {_BIN_METHODS}, got {bin_method!r}"
        raise ValueError(msg)
    wavelength_grid, edges = _wavelength_bins(wavelength_grid, wavelength_edges)
    instrumental_width = _instrumental_width_in_angstrom(instrumental_width)
    if doppler_velocity is not None:
        doppler_velocity = _velocity_axis(doppler_velocity, "doppler_velocity")
    if nonthermal_velocity is not None:
        nonthermal_velocity = _velocity_axis(nonthermal_velocity, "nonthermal_velocity")
    effective_area = _effective_area_in_canonical_units(effective_area)
    if include_contaminants and effective_area is not None and np.unique(effective_area.data).size == 1:
        logger.warning(
            "A flat (wavelength-independent) effective area is only representative near the main lines; "
            "applying it to contaminant lines across the band will produce wrong contaminant counts. "
            "Provide a wavelength-dependent effective_area curve for quantitative work with contaminants."
        )
    line_list = _validate_line_list(line_list)
    line_names = tuple(str(name) for name in line_list.full_name.values)
    all_lines_requested = main_lines is None
    main_lines = _validate_main_lines(line_names, main_lines)
    if len(main_lines) > 10:
        logger.warning(
            f"Building a spectral response for {len(main_lines)} main lines may take a while and use a lot of memory. "
            "If you are only interested in a few lines, pass them in `main_lines`."
        )
    if not main_lines and not include_contaminants:
        msg = "main_lines cannot be empty unless include_contaminants=True"
        raise ValueError(msg)
    if not include_contaminants and not all_lines_requested:
        # Only the selected lines are evaluated, but the per-transition centers and
        # widths below are computed for the whole list; a broad-band list can carry
        # thousands of transitions (dominating peak memory), so drop the rest first.
        selected = set(main_lines)
        keep = [i for i, name in enumerate(line_names) if name in selected]
        line_list = line_list.isel(trans_index=keep)
        line_names = tuple(str(name) for name in line_list.full_name.values)

    try:
        import periodictable as pt  # noqa: PLC0415
    except ImportError:
        msg = "periodictable is required for this function, install it with `pip install muse[chianti]`"
        raise ImportError(msg) from None

    speed_of_light_kms = const.c.to_value(u.km / u.s)
    speed_of_light_ms = const.c.to_value(u.m / u.s)
    boltzmann = const.k_B.to_value(u.J / u.K)
    proton_mass = const.m_p.to_value(u.kg)

    if doppler_velocity is not None:
        line_centers = line_list["wavelength"] * (1 + doppler_velocity / speed_of_light_kms)
    else:
        line_centers = line_list["wavelength"]

    atomic_mass = _atomic_mass_from_atomic_number(line_list.atomic_number, pt.elements, proton_mass)
    thermal_velocity = np.sqrt(boltzmann * 10**line_list.logT / atomic_mass)
    thermal_line_width = line_centers * thermal_velocity / speed_of_light_ms
    doppler_widths_squared = thermal_line_width**2 + instrumental_width**2
    if nonthermal_velocity is not None:
        doppler_widths_squared = (
            doppler_widths_squared + (line_list["wavelength"] * (nonthermal_velocity / speed_of_light_kms)) ** 2
        )
    doppler_widths = np.sqrt(doppler_widths_squared)
    gaussian_norm = np.sqrt(2 * np.pi)

    # Bin averages take a curve at each line's own center; a scalar or sampled area scales the grid below.
    area_at_center = None
    if bin_method == "average" and effective_area is not None and "wavelength" in effective_area.dims:
        area_at_center = (
            effective_area.interp(wavelength=line_centers).fillna(0).drop_vars("wavelength", errors="ignore")
        )

    main_response_parts = {name: [] for name in main_lines}
    for i, line_name in enumerate(line_names):
        if line_name not in main_response_parts:
            continue
        line_response, gofnt_scaled = _evaluate_gaussian_response(
            wavelength_grid,
            line_centers.isel(trans_index=i),
            doppler_widths.isel(trans_index=i),
            line_list.gofnt.isel(trans_index=i),
            gaussian_norm,
            edges=edges,
            bin_method=bin_method,
            area_at_center=None if area_at_center is None else area_at_center.isel(trans_index=i),
        )
        line_response = xr.DataArray(line_response, dims=gofnt_scaled.dims, coords=gofnt_scaled.coords)
        line_response = line_response.expand_dims(line=[line_name])
        line_response = line_response.assign_coords(
            line_wavelength=("line", [line_list.wavelength.isel(trans_index=i).item()], {"units": str(u.AA)}),
            component_kind=("line", ["line"]),
        )
        main_response_parts[line_name].append(line_response)

    # Most main lines map to a single transition; skip the concat + reduce copies there.
    responses = [
        parts[0]
        if len(parts) == 1
        else xr.concat(parts, dim="_transition", join="exact").sum("_transition", keep_attrs=True)
        for parts in main_response_parts.values()
    ]

    if include_contaminants:
        contaminant_indices = [i for i, name in enumerate(line_names) if name not in main_response_parts]
        if contaminant_indices:
            # A transition whose +-window (see _evaluate_gaussian_response) misses the grid
            # contributes exactly zero, and a broad-band list is mostly such lines.
            width_dims = [dim for dim in doppler_widths.dims if dim != "trans_index"]
            center_dims = [dim for dim in line_centers.dims if dim != "trans_index"]
            half_window = _GAUSSIAN_WINDOW_SIGMA * doppler_widths.max(dim=width_dims)
            lower = line_centers.min(dim=center_dims) - half_window
            upper = line_centers.max(dim=center_dims) + half_window
            on_grid = ((upper >= edges[0]) & (lower <= edges[-1])).values
            # Keep one off-grid line so an all-off-grid list still yields a zero component.
            contaminant_indices = [i for i in contaminant_indices if on_grid[i]] or contaminant_indices[:1]
        contaminant_response = _create_contaminant_response(
            line_list,
            contaminant_indices,
            wavelength_grid,
            line_centers,
            doppler_widths,
            gaussian_norm,
            edges=edges,
            bin_method=bin_method,
            area_at_center=area_at_center,
            progress=progress,
        )
    else:
        contaminant_response = None

    if contaminant_response is not None:
        contaminant_response = contaminant_response.expand_dims(line=["contaminants"])
        contaminant_response = contaminant_response.assign_coords(
            line_wavelength=("line", [np.nan], {"units": str(u.AA)}),
            component_kind=("line", ["contaminants"]),
        )
        responses.append(contaminant_response)

    responses = xr.concat(responses, dim="line", coords="different", compat="equals", join="exact")
    ds = xr.Dataset({"spectral_response": responses})

    if effective_area is not None:
        if area_at_center is None:
            # A curve is resampled onto the grid; a zero-dimensional area needs no resampling.
            scaling = effective_area
            if "wavelength" in effective_area.dims:
                scaling = (
                    effective_area.interp(wavelength=wavelength_grid).fillna(0).rename(wavelength="wavelength_grid")
                )
            ds["spectral_response"] = ds.spectral_response * scaling
        ds.spectral_response.attrs["units"] = str(DEFAULTS_MUSE.normalization * u.erg * u.cm**5 / u.s / u.sr / u.AA)
        ds.spectral_response.attrs["effective_area_at"] = "bin center" if area_at_center is None else "line center"
    else:
        ds.spectral_response.attrs["units"] = str(DEFAULTS_MUSE.normalization * u.erg * u.cm**3 / u.s / u.sr / u.AA)
    ds.spectral_response.attrs["bin_method"] = bin_method

    ds = ds.assign_coords(_wavelength_bin_coords(wavelength_grid, edges))

    ds.attrs["normalization"] = DEFAULTS_MUSE.normalization
    return ds


def _wavelength_bins(
    wavelength_grid: u.Quantity | None, wavelength_edges: u.Quantity | None
) -> tuple[xr.DataArray, np.ndarray]:
    """
    The bin centers as a ``wavelength_bin`` array and the bin edges, from either input.
    """
    if (wavelength_grid is None) == (wavelength_edges is None):
        msg = "pass exactly one of wavelength_grid and wavelength_edges"
        raise ValueError(msg)
    if wavelength_edges is not None:
        edges = np.asarray(wavelength_edges.to_value(u.AA), dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            msg = "wavelength_edges must hold at least two bin edges"
            raise ValueError(msg)
        _require_increasing_axis(edges, "wavelength_edges", positive=True)
        return _wavelength_grid_in_angstrom(0.5 * (edges[1:] + edges[:-1]) * u.AA), edges
    grid = _wavelength_grid_in_angstrom(wavelength_grid)
    return grid, _centers_to_edges(grid.data)


def _wavelength_bin_coords(centers, edges: np.ndarray) -> dict:
    """
    The ``wavelength_grid`` and ``wavelength_bounds`` coordinates of a binned response.
    """
    angstrom = {"units": str(u.AA)}
    bounds = xr.DataArray(np.stack([edges[:-1], edges[1:]], axis=-1), dims=("wavelength_bin", "bound"), attrs=angstrom)
    return {
        "wavelength_grid": (
            "wavelength_bin",
            np.asarray(centers, dtype=float),
            {**angstrom, "bounds": "wavelength_bounds"},
        ),
        "wavelength_bounds": bounds,
    }


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """
    Edges halfway between neighbouring centers, extrapolated by half a bin at the ends.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        msg = "at least two bin centers are needed to infer bin edges"
        raise ValueError(msg)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def _instrumental_width_in_angstrom(instrumental_width: u.Quantity) -> float:
    if not instrumental_width.isscalar:
        msg = "instrumental_width must be scalar"
        raise ValueError(msg)
    converted = instrumental_width.to_value(u.AA)
    if not np.all(np.isfinite(converted)) or np.any(converted < 0):
        msg = "instrumental_width must contain finite, non-negative values"
        raise ValueError(msg)
    return converted


def _effective_area_in_canonical_units(effective_area: xr.DataArray | None) -> xr.DataArray | None:
    if effective_area is None:
        return None
    if not isinstance(effective_area, xr.DataArray):
        msg = (
            "effective_area must be an xarray.DataArray, "
            "e.g. DEFAULTS_MUSE.main_line_effective_area_SG.sel(channel=...)"
        )
        raise TypeError(msg)
    if effective_area.ndim == 0:
        # A scalar area applies uniformly, so it stays zero-dimensional
        # and simply multiplies the response.
        data = effective_area.data
        units = effective_area.attrs.get("units")
        try:
            units = u.Unit(units) if units is not None else None
        except (TypeError, ValueError) as exc:
            msg = "effective_area units must be a valid astropy unit"
            raise ValueError(msg) from exc
        if not isinstance(data, u.Quantity):
            # attrs["units"] names the unit the raw value is already in.
            data = u.Quantity(data, units)
        elif units is not None and not units.is_equivalent(data.unit):
            # The payload carries its own unit, so contradicting attrs are a bug, not a request
            # to convert. Trusting either one silently is how a wrong effective area gets in.
            msg = f"effective_area units {units} contradict the data unit {data.unit}"
            raise ValueError(msg)
        try:
            value = data.to_value(u.cm**2)
        except u.UnitConversionError as exc:
            msg = "effective_area must be convertible to cm2"
            raise ValueError(msg) from exc
        if not np.isfinite(value) or value < 0:
            msg = "effective_area must contain finite, non-negative values"
            raise ValueError(msg)
        return xr.DataArray(value, attrs={"units": str(u.cm**2)})
    if effective_area.dims != ("wavelength",):
        msg = "effective_area must be one-dimensional with wavelength as its only dimension"
        raise ValueError(msg)
    dataset = xr.Dataset({"effective_area": effective_area})
    area_unit = require_unit(dataset, "effective_area", "effective_area", convertible_to=u.cm**2)
    wavelength = coord_as_unit(dataset, "wavelength", u.AA, "effective_area wavelength coordinate")
    converted = (effective_area * area_unit.to(u.cm**2)).assign_attrs({**effective_area.attrs, "units": str(u.cm**2)})
    if not np.all(np.isfinite(converted)) or np.any(converted < 0):
        msg = "effective_area must contain finite, non-negative values"
        raise ValueError(msg)
    _require_increasing_axis(wavelength, "effective_area wavelength coordinate")
    return converted.assign_coords(wavelength=wavelength)


def _velocity_axis(values: u.Quantity, dim: str) -> xr.DataArray:
    values = np.atleast_1d(values.to_value(u.km / u.s))
    if values.ndim != 1:
        msg = f"{dim} must be scalar or one-dimensional"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = f"{dim} must contain only finite values"
        raise ValueError(msg)
    if dim == "nonthermal_velocity" and np.any(values < 0):
        msg = "nonthermal_velocity must contain non-negative values"
        raise ValueError(msg)
    coordinate = xr.DataArray(values, dims=dim, attrs={"units": str(u.km / u.s)})
    return xr.DataArray(values, dims=dim, coords={dim: coordinate})


def _wavelength_grid_in_angstrom(wavelength_grid: u.Quantity) -> xr.DataArray:
    values = wavelength_grid.to_value(u.AA)
    _require_increasing_axis(values, "wavelength_grid")
    return xr.DataArray(values, dims="wavelength_bin", attrs={"units": str(u.AA)})


def _create_contaminant_response(
    line_list: xr.Dataset,
    contaminant_indices: list[int],
    wavelength_grid: xr.DataArray,
    line_centers: xr.DataArray,
    doppler_widths: xr.DataArray,
    gaussian_norm: float,
    *,
    edges: np.ndarray,
    bin_method: str,
    area_at_center: xr.DataArray | None = None,
    progress: bool,
) -> xr.DataArray | None:
    if not contaminant_indices:
        return None
    accumulator = None
    iterator = tqdm(contaminant_indices, desc="Spectral contaminants", unit="line") if progress else contaminant_indices
    for i in iterator:
        accumulator, gofnt_scaled = _evaluate_gaussian_response(
            wavelength_grid,
            line_centers.isel(trans_index=i),
            doppler_widths.isel(trans_index=i),
            line_list.gofnt.isel(trans_index=i),
            gaussian_norm,
            accumulator=accumulator,
            edges=edges,
            bin_method=bin_method,
            area_at_center=None if area_at_center is None else area_at_center.isel(trans_index=i),
        )
    return xr.DataArray(accumulator, dims=gofnt_scaled.dims, coords=gofnt_scaled.coords)


def _validate_line_list(line_list: xr.Dataset) -> xr.Dataset:
    if not isinstance(line_list, xr.Dataset):
        msg = "line_list must be an xarray.Dataset"
        raise TypeError(msg)
    if "trans_index" not in line_list.dims:
        line_list = line_list.expand_dims("trans_index")
    if line_list.sizes["trans_index"] == 0:
        msg = "line_list must not be empty"
        raise ValueError(msg)
    missing = [name for name in ("wavelength", "atomic_number", "gofnt", "full_name") if name not in line_list]
    if missing:
        msg = f"line_list is missing required variables: {', '.join(missing)}"
        raise ValueError(msg)
    if "logT" not in line_list.coords:
        msg = "line_list must include a logT coordinate"
        raise ValueError(msg)
    if "logT" not in line_list.gofnt.dims:
        msg = "line_list.gofnt must include a logT dimension"
        raise ValueError(msg)
    missing_trans_index = [
        name
        for name in ("wavelength", "atomic_number", "gofnt", "full_name")
        if "trans_index" not in line_list[name].dims
    ]
    if missing_trans_index:
        msg = f"line_list variables must include a trans_index dimension: {', '.join(missing_trans_index)}"
        raise ValueError(msg)
    for name in ("wavelength", "atomic_number", "gofnt", "logT"):
        try:
            values = np.asarray(line_list[name], dtype=float)
        except (TypeError, ValueError) as exc:
            msg = f"line_list.{name} must contain numeric values"
            raise ValueError(msg) from exc
        if values.size == 0 or not np.all(np.isfinite(values)):
            msg = f"line_list.{name} must contain finite values"
            raise ValueError(msg)
    if np.any(line_list.wavelength <= 0):
        msg = "line_list.wavelength must contain positive values"
        raise ValueError(msg)
    if np.any(line_list.gofnt < 0):
        msg = "line_list.gofnt must contain non-negative values"
        raise ValueError(msg)
    atomic_number = np.asarray(line_list.atomic_number)
    if np.any(atomic_number <= 0) or np.any(atomic_number != np.floor(atomic_number)):
        msg = "line_list.atomic_number must contain positive integers"
        raise ValueError(msg)
    if any(not isinstance(name, (str, np.str_)) or not name.strip() for name in line_list.full_name.values):
        msg = "line_list.full_name must contain non-empty strings"
        raise ValueError(msg)
    wavelength_unit = require_unit(
        line_list,
        "wavelength",
        "line_list.wavelength",
        convertible_to=u.AA,
    )
    gofnt_unit = require_unit(
        line_list,
        "gofnt",
        "line_list.gofnt",
        convertible_to=u.erg * u.cm**3 / (u.s * u.sr),
    )
    require_unit(line_list, "logT", "line_list.logT", coord_only=True, convertible_to=u.dex(u.K))
    wavelength = (line_list.wavelength * wavelength_unit.to(u.AA)).assign_attrs(
        {**line_list.wavelength.attrs, "units": str(u.AA)}
    )
    gofnt_target_unit = u.erg * u.cm**3 / (u.s * u.sr)
    gofnt = (line_list.gofnt * gofnt_unit.to(gofnt_target_unit)).assign_attrs(
        {**line_list.gofnt.attrs, "units": str(gofnt_target_unit)}
    )
    return line_list.assign(wavelength=wavelength, gofnt=gofnt)


def _validate_main_lines(line_names: tuple[str, ...], main_lines: Sequence[str] | None) -> tuple[str, ...]:
    if main_lines is None:
        return tuple(dict.fromkeys(line_names))
    if isinstance(main_lines, str) or not isinstance(main_lines, Sequence):
        msg = "main_lines must be a sequence of full_name strings or None"
        raise TypeError(msg)
    main_lines = tuple(main_lines)
    if any(not isinstance(name, str) for name in main_lines):
        msg = "main_lines must contain only full_name strings"
        raise TypeError(msg)
    if len(main_lines) != len(set(main_lines)):
        msg = "main_lines must contain unique full_name values"
        raise ValueError(msg)
    missing = [name for name in main_lines if name not in line_names]
    if missing:
        msg = f"main_lines not found in line_list.full_name: {', '.join(missing)}"
        raise ValueError(msg)
    return main_lines


def _atomic_mass_from_atomic_number(atomic_number: xr.DataArray, elements, proton_mass: float) -> xr.DataArray:
    max_atomic_number = max(element.number for element in elements)
    if np.any(np.asarray(atomic_number) > max_atomic_number):
        msg = f"line_list.atomic_number must be between 1 and {max_atomic_number}"
        raise ValueError(msg)
    atomic_mass = atomic_number.copy()
    masses = np.array([elements[int(value)].mass for value in atomic_mass.data.reshape(-1)])
    atomic_mass.data = masses.reshape(atomic_mass.shape) * proton_mass
    return atomic_mass


def _evaluate_gaussian_response(
    wavelength_grid: xr.DataArray,
    line_center: xr.DataArray,
    doppler_width: xr.DataArray,
    gofnt: xr.DataArray,
    gaussian_norm: float,
    *,
    accumulator: np.ndarray | None = None,
    edges: np.ndarray | None = None,
    bin_method: str = "center",
    area_at_center: xr.DataArray | None = None,
) -> tuple[np.ndarray, xr.DataArray]:
    """
    Add one line's Gaussian to ``accumulator`` (a new full-grid array when `None`).

    With ``bin_method="center"`` the Gaussian is sampled at the bin centers; with
    ``"average"`` each bin, bounded by ``edges``, holds its average of the Gaussian, and
    ``area_at_center`` (the effective area at the line center, which may vary with the
    Doppler velocity) scales the line.
    """
    try:
        import numexpr  # noqa: PLC0415
    except ImportError:
        msg = "numexpr is required for this function, install it with `pip install muse[chianti]`"
        raise ImportError(msg) from None

    average = bin_method == "average"
    if average and edges is None:
        msg = "bin averages need the bin edges"
        raise ValueError(msg)
    center, width_for_bounds = xr.broadcast(line_center, doppler_width)
    half_window = _GAUSSIAN_WINDOW_SIGMA * width_for_bounds
    window_lower, window_upper = np.min((center - half_window).data), np.max((center + half_window).data)
    if average:
        # Every bin overlapping the window, so a line narrower than a bin lands in the bin holding it.
        start = max(0, int(np.searchsorted(edges, window_lower, side="right")) - 1)
        stop = min(wavelength_grid.size, int(np.searchsorted(edges, window_upper, side="left")))
    else:
        start = np.searchsorted(wavelength_grid.data, window_lower, side="left")
        stop = np.searchsorted(wavelength_grid.data, window_upper, side="right")

    # Broadcast only the +-window slice of the grid: the Gaussian is exactly zero
    # outside it, and broadcasting the full grid first materializes several
    # (logT x doppler x wavelength) float64 temporaries per line (~5 GB for the
    # 108 band) where the window needs a small fraction of that.
    window_points = (
        xr.DataArray(edges[start : stop + 1], dims="wavelength_bin")
        if average
        else wavelength_grid.isel(wavelength_bin=slice(start, stop))
    )
    shift = (window_points - line_center).broadcast_like(gofnt)
    width, shift = xr.broadcast(doppler_width, shift)
    gofnt_scaled = (gofnt / DEFAULTS_MUSE.normalization).broadcast_like(width)
    gofnt_scaled, width, shift = xr.broadcast(gofnt_scaled, width, shift)
    wavelength_axis = gofnt_scaled.get_axis_num("wavelength_bin")
    if average:
        gofnt_scaled = gofnt_scaled.isel(wavelength_bin=slice(None, -1))
        block = _bin_averaged_gaussian(
            gofnt_scaled.data, shift.data, width.data, np.diff(edges[start : stop + 1]), wavelength_axis
        )
    else:
        block = numexpr.evaluate(
            _GAUSSIAN_EXPRESSION,
            local_dict={
                "scaled": gofnt_scaled.data,
                "shift": shift.data,
                "width": width.data,
                "gaussian_norm": gaussian_norm,
            },
        )
    if area_at_center is not None:
        block = block * area_at_center.broadcast_like(gofnt_scaled).transpose(*gofnt_scaled.dims).data
    if accumulator is None:
        full_shape = list(gofnt_scaled.shape)
        full_shape[wavelength_axis] = wavelength_grid.size
        accumulator = np.zeros(tuple(full_shape), dtype=gofnt_scaled.dtype)
    index = [slice(None)] * accumulator.ndim
    index[wavelength_axis] = slice(start, stop)
    accumulator[tuple(index)] += block
    return accumulator, gofnt_scaled


def _bin_averaged_gaussian(
    scaled: np.ndarray, shift_at_edges: np.ndarray, width: np.ndarray, bin_widths: np.ndarray, wavelength_axis: int
) -> np.ndarray:
    """
    Each bin's average of the Gaussian: erf differences at the bin edges over the bin
    width.
    """
    import numexpr  # noqa: PLC0415

    erf_at_edges = erf(shift_at_edges / (width * np.sqrt(2)))
    lower_index = [slice(None)] * erf_at_edges.ndim
    upper_index = [slice(None)] * erf_at_edges.ndim
    lower_index[wavelength_axis], upper_index[wavelength_axis] = slice(None, -1), slice(1, None)
    shape = [-1 if axis == wavelength_axis else 1 for axis in range(erf_at_edges.ndim)]
    return numexpr.evaluate(
        "scaled * 0.5 * (upper - lower) / bin_width",
        local_dict={
            "scaled": scaled,
            "upper": erf_at_edges[tuple(upper_index)],
            "lower": erf_at_edges[tuple(lower_index)],
            "bin_width": bin_widths.reshape(shape),
        },
    )


@u.quantity_input(wavelength_edges=u.AA, doppler_velocity=u.km / u.s)
def create_continuum_response(
    continuum: xr.Dataset,
    wavelength_edges: u.Quantity,
    *,
    effective_area: xr.DataArray,
    doppler_velocity: u.Quantity | None = None,
) -> xr.Dataset:
    """
    Fold a rest-frame continuum through an effective area onto wavelength bins.

    For every Doppler velocity the continuum is shifted (light observed at a wavelength was
    emitted at that wavelength over ``1 + v / c``), multiplied by the area at the observed
    wavelength and averaged over each bin, so a bin keeps the structure of the area inside it.
    The result has the layout of `create_spectral_response` with ``bin_method="average"`` and
    concatenates with a line response along ``line``.

    Parameters
    ----------
    continuum : `xarray.Dataset`
        From `muse.instrument.create_chianti_continuum` or
        `muse.instrument.create_fiasco_continuum`, on a grid fine enough to resolve the
        effective area.
    wavelength_edges : `astropy.units.Quantity`
        Strictly increasing bin edges, one more than the number of bins.
    effective_area : `xarray.DataArray`
        A curve with a ``wavelength`` coordinate, or one value that holds at every
        wavelength, in units convertible to cm2, as for `create_spectral_response`.
    doppler_velocity : `astropy.units.Quantity`, optional
        Scalar or one-dimensional bulk velocities; adds a ``doppler_velocity`` dimension.

    Returns
    -------
    `xarray.Dataset`
        ``spectral_response`` on the bins in the continuum's units times cm2, with
        ``wavelength_grid`` at the bin centers and ``wavelength_bounds``.
    """
    call_inputs = dict(locals())
    response_unit = require_unit(continuum, "spectral_response", "continuum.spectral_response")
    grid = np.asarray(coord_as_unit(continuum, "wavelength_grid", u.AA, "continuum.wavelength_grid"), dtype=float)
    centers, edges = _wavelength_bins(None, wavelength_edges)
    if effective_area is None:
        msg = "effective_area is required: a continuum response is the continuum seen through an area"
        raise ValueError(msg)
    area_on_grid = _effective_area_in_canonical_units(effective_area)
    if area_on_grid.ndim:
        area_on_grid = (
            area_on_grid.interp(wavelength=xr.DataArray(grid, dims="wavelength_bin")).fillna(0).drop_vars("wavelength")
        )
    rest = continuum.spectral_response.assign_coords(wavelength_bin=grid)

    if doppler_velocity is None:
        binned = _bin_average(rest * area_on_grid, grid, edges)
    else:
        velocities = np.atleast_1d(doppler_velocity.to_value(u.km / u.s)).astype(float)
        speed_of_light = const.c.to_value(u.km / u.s)
        pieces = []
        for velocity in velocities:
            shifted = rest.interp(wavelength_bin=grid / (1 + velocity / speed_of_light), kwargs={"fill_value": 0.0})
            pieces.append(_bin_average(shifted.assign_coords(wavelength_bin=grid) * area_on_grid, grid, edges))
        axis = xr.DataArray(velocities, dims="doppler_velocity", attrs={"units": str(u.km / u.s)})
        binned = xr.concat(pieces, dim=axis).transpose("line", "doppler_velocity", ...)
    binned.attrs = {
        **continuum.spectral_response.attrs,
        "units": str(response_unit * u.cm**2),
        "effective_area_at": "bin average",
        "bin_method": "average",
    }
    result = xr.Dataset({"spectral_response": binned}, attrs=dict(continuum.attrs)).assign_coords(
        _wavelength_bin_coords(centers, edges)
    )
    add_history(result, call_inputs, create_continuum_response, sources=(continuum,))
    return result


def _bin_average(values: xr.DataArray, x: np.ndarray, edges: np.ndarray) -> xr.DataArray:
    """
    Average a density sampled at ``x`` (the ``wavelength_bin`` axis) over the bins
    between ``edges``.

    The samples are joined by straight lines and that piecewise-linear density is
    integrated exactly over every bin; the density is zero outside the sampled range.
    """
    if x.size < 2:
        msg = "the continuum needs at least two wavelength samples to average over bins"
        raise ValueError(msg)
    y = np.moveaxis(np.asarray(values.data, dtype=float), values.get_axis_num("wavelength_bin"), -1)
    dx = np.diff(x)
    cumulative = np.zeros(y.shape)
    cumulative[..., 1:] = np.cumsum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)
    # The integral up to every edge: the cumulative value at the sample before it plus the
    # exact integral of the linear segment up to the edge.
    index = np.clip(np.searchsorted(x, edges, side="right") - 1, 0, x.size - 2)
    t = np.clip(edges - x[index], 0.0, dx[index])
    slope = (y[..., index + 1] - y[..., index]) / dx[index]
    at_edges = cumulative[..., index] + y[..., index] * t + 0.5 * slope * t**2
    averaged = np.diff(at_edges, axis=-1) / np.diff(edges)
    averaged = np.moveaxis(averaged, -1, values.get_axis_num("wavelength_bin"))
    coords = {name: coord for name, coord in values.coords.items() if "wavelength_bin" not in coord.dims}
    return xr.DataArray(averaged, dims=values.dims, coords=coords, attrs=values.attrs)
