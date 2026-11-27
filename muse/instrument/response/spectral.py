"""
Wavelength-space responses of spectral lines and the continuum.
"""

from collections.abc import Sequence

import numexpr
import numpy as np
import periodictable as pt
import xarray as xr
from scipy.special import erf
from tqdm import tqdm

import astropy.constants as const
import astropy.units as u
from astropy.stats import gaussian_sigma_to_fwhm

from muse.log import logger
from muse.utils.utils import _require_increasing_axis, add_history, coord_as_unit, require_unit, update_attrs

__all__ = ["create_continuum_response", "create_log_wavelength_edges", "create_spectral_response"]

# The Gaussian is exactly zero beyond this point in float64.
_GAUSSIAN_WINDOW_SIGMA = 40
_GAUSSIAN_EXPRESSION = "scaled * exp(-0.5 * (shift / width) ** 2) / gaussian_norm / width"
_BIN_METHODS = ("center", "average")
_NORMALIZATION = 1e-27


@u.quantity_input(wavelength_range=u.AA)
def create_log_wavelength_edges(wavelength_range: u.Quantity, fractional_width: float) -> u.Quantity:
    """
    Create the edges of logarithmically spaced wavelength bins that cover a range.

    Every bin spans the same velocity width; pass the result to `create_spectral_response` as
    ``wavelength_edges`` for a coarse imager grid.

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

    Raises
    ------
    ValueError
        If ``wavelength_range`` is not positive and increasing, or ``fractional_width`` is
        not positive.
    """
    lower, upper = np.asarray(wavelength_range.to_value(u.AA), dtype=float).reshape(-1)[[0, -1]]
    if not (0 < lower < upper) or not fractional_width > 0:
        msg = "wavelength_range must be positive and increasing, and fractional_width must be positive"
        raise ValueError(msg)
    number_of_bins = int(np.ceil(np.log(upper / lower) / np.log1p(fractional_width)))
    return lower * (1 + fractional_width) ** np.arange(number_of_bins + 1) * u.AA


def create_spectral_response(
    line_list: xr.Dataset,
    wavelength_grid: u.Quantity | None = None,
    *,
    main_lines: Sequence[str] | None,
    instrumental_fwhm: u.Quantity = 0 * u.AA,
    doppler_velocity: u.Quantity | None = None,
    nonthermal_velocity: u.Quantity | None = None,
    effective_area: xr.DataArray | xr.Dataset | None = None,
    include_contaminants: bool = False,
    wavelength_edges: u.Quantity | None = None,
    bin_method: str = "average",
) -> xr.Dataset:
    """
    Create an instrument-neutral wavelength-space spectral response.

    Parameters
    ----------
    line_list : `xarray.Dataset`
        Line list with its ``logT`` coordinate in ``dex(K)``, e.g. from
        `muse.instrument.create_chianti_line_list` or
        `muse.instrument.create_fiasco_line_list`.
    wavelength_grid : `astropy.units.Quantity`, optional
        One-dimensional bin centers for one channel and spectral order; each bin extends
        halfway to its neighbors. Give this or ``wavelength_edges``.
    main_lines : sequence of `str` or `None`
        ``full_name`` values to keep, in output order; transitions that share a name are
        summed. If `None` or empty, ``include_contaminants`` must be `True`.
    instrumental_fwhm : `astropy.units.Quantity`, optional
        Scalar instrumental FWHM, by default 0 Angstrom; its sigma adds in quadrature to
        the thermal and nonthermal widths. For a MUSE SG channel,
        ``DEFAULTS_MUSE.instrumental_fwhm_SG.sel(channel=171).data``.
    doppler_velocity : `astropy.units.Quantity`, optional
        Scalar or one-dimensional; adds a ``doppler_velocity`` dimension. If `None`, lines
        stay at their rest wavelengths.
    nonthermal_velocity : `astropy.units.Quantity`, optional
        Scalar or one-dimensional 1/e (most probable) velocity xi of the EIS and IRIS
        literature, which adds a sigma of lambda xi / (c sqrt 2) and a
        ``nonthermal_velocity`` dimension. If `None`, no nonthermal broadening.
    effective_area : `xarray.DataArray` or `xarray.Dataset`, optional
        A curve along a ``wavelength`` coordinate with units, or a zero-dimensional area
        applied at every wavelength (e.g.
        ``DEFAULTS_MUSE.main_line_effective_area_SG.sel(channel=171)``), convertible to
        cm2; or the Dataset of `muse.instrument.create_effective_area`. If `None`, the
        response is not scaled by an area.
    include_contaminants : `bool`, optional
        If `True`, sum every line not in ``main_lines`` into one ``contaminants`` line.
    wavelength_edges : `astropy.units.Quantity`, optional
        Strictly increasing bin edges, one more than the bins, instead of
        ``wavelength_grid``, e.g. from `create_log_wavelength_edges`; the bin centers
        become ``wavelength_grid``.
    bin_method : {"center", "average"}, optional
        ``"center"`` samples each Gaussian at the bin center, which miscounts a line
        narrower than a bin (too high on a bin center, too low between two).
        ``"average"``, the default, averages it over each bin, which conserves a line's
        area on any grid that covers it. An ``effective_area`` curve is applied at each
        line's Doppler-shifted center with ``"average"``, at each bin center with
        ``"center"``.

    Returns
    -------
    `xarray.Dataset`
        ``spectral_response`` along temperature, velocity, ``line`` and
        ``wavelength_bin``, with the bin centers in ``wavelength_grid`` and their edges
        in ``wavelength_bounds``.
    """
    call_inputs = dict(locals())
    main_lines = () if main_lines is None else main_lines
    response = _build_spectral_response(
        line_list,
        wavelength_grid,
        instrumental_fwhm=instrumental_fwhm,
        doppler_velocity=doppler_velocity,
        nonthermal_velocity=nonthermal_velocity,
        effective_area=effective_area,
        main_lines=main_lines,
        include_contaminants=include_contaminants,
        wavelength_edges=wavelength_edges,
        bin_method=bin_method,
    )
    update_attrs(response, line_list, **response.attrs)
    # The area keeps its own lineage, such as the calibration an AIA area was made with.
    area = call_inputs["effective_area"]
    add_history(
        response, call_inputs, create_spectral_response, sources=(line_list,) if area is None else (line_list, area)
    )
    return response


@u.quantity_input(
    wavelength_grid=u.AA,
    instrumental_fwhm=u.AA,
    doppler_velocity=u.km / u.s,
    nonthermal_velocity=u.km / u.s,
    wavelength_edges=u.AA,
)
def _build_spectral_response(
    line_list: xr.Dataset,
    wavelength_grid: u.Quantity | None = None,
    *,
    instrumental_fwhm: u.Quantity = 0 * u.AA,
    doppler_velocity: u.Quantity | None = None,
    nonthermal_velocity: u.Quantity | None = None,
    effective_area: xr.DataArray | None = None,
    main_lines: Sequence[str],
    include_contaminants: bool = False,
    wavelength_edges: u.Quantity | None = None,
    bin_method: str = "average",
) -> xr.Dataset:
    """
    Build the wavelength-space response, optionally including contaminants.
    """
    if bin_method not in _BIN_METHODS:
        msg = f"bin_method must be one of {_BIN_METHODS}, got {bin_method!r}"
        raise ValueError(msg)
    wavelength_grid, edges = _wavelength_bins(wavelength_grid, wavelength_edges)
    instrumental_sigma = _instrumental_sigma_in_angstrom(instrumental_fwhm)
    if doppler_velocity is not None:
        doppler_velocity = _velocity_axis(doppler_velocity, "doppler_velocity")
    if nonthermal_velocity is not None:
        nonthermal_velocity = _velocity_axis(nonthermal_velocity, "nonthermal_velocity")
    effective_area = _effective_area_in_cm2(effective_area)
    if include_contaminants and effective_area is not None and np.unique(effective_area.data).size == 1:
        logger.warning(
            "A flat (wavelength-independent) effective area is only representative near the main lines; "
            "applying it to contaminant lines across the band will produce wrong contaminant counts. "
            "Provide a wavelength-dependent effective_area curve for quantitative work with contaminants."
        )
    line_list = _validate_line_list(line_list)
    line_names = tuple(str(name) for name in line_list.full_name.values)
    main_lines = _validate_main_lines(line_names, main_lines)
    if len(main_lines) > 10:
        logger.warning(
            f"Building a spectral response for {len(main_lines)} main lines may take a while and use a lot of memory. "
            "If you are only interested in a few lines, pass them in `main_lines`."
        )
    if not main_lines and not include_contaminants:
        msg = "main_lines cannot be empty unless include_contaminants=True"
        raise ValueError(msg)
    if not include_contaminants:
        # Centers and widths below are per transition; thousands of unselected ones dominate peak memory.
        selected = set(main_lines)
        keep = [i for i, name in enumerate(line_names) if name in selected]
        line_list = line_list.isel(transition=keep)
        line_names = tuple(str(name) for name in line_list.full_name.values)

    speed_of_light_kms = const.c.to_value(u.km / u.s)
    speed_of_light_ms = const.c.to_value(u.m / u.s)
    boltzmann = const.k_B.to_value(u.J / u.K)
    # periodictable masses are in atomic mass units, not proton masses.
    atomic_mass_unit = const.u.to_value(u.kg)

    if doppler_velocity is not None:
        line_centers = line_list["wavelength"] * (1 + doppler_velocity / speed_of_light_kms)
    else:
        line_centers = line_list["wavelength"]

    atomic_mass = _atomic_mass_from_atomic_number(line_list.atomic_number, pt.elements, atomic_mass_unit)
    thermal_velocity = np.sqrt(boltzmann * 10**line_list.logT / atomic_mass)
    thermal_line_width = line_centers * thermal_velocity / speed_of_light_ms
    line_widths_squared = thermal_line_width**2 + instrumental_sigma**2
    if nonthermal_velocity is not None:
        # the 1/e velocity xi of a Maxwellian is sqrt(2) times its sigma
        line_widths_squared = line_widths_squared + (line_centers * (nonthermal_velocity / speed_of_light_kms)) ** 2 / 2
    line_widths = np.sqrt(line_widths_squared)
    gaussian_norm = np.sqrt(2 * np.pi)

    area_at_center = None
    if bin_method == "average" and effective_area is not None and "wavelength" in effective_area.dims:
        area_at_center = (
            effective_area.interp(wavelength=line_centers).fillna(0).drop_vars("wavelength", errors="ignore")
        )

    main_response_parts = {name: [] for name in main_lines}
    for i, line_name in enumerate(line_names):
        if line_name not in main_response_parts:
            continue
        line_response, template = _evaluate_gaussian_response(
            wavelength_grid,
            line_centers.isel(transition=i),
            line_widths.isel(transition=i),
            line_list.gofnt.isel(transition=i),
            gaussian_norm,
            edges=edges,
            bin_method=bin_method,
            area_at_center=None if area_at_center is None else area_at_center.isel(transition=i),
        )
        line_response = xr.DataArray(line_response, dims=template.dims, coords=template.coords)
        line_response = line_response.expand_dims(line=[line_name])
        line_response = line_response.assign_coords(
            line_wavelength=("line", [line_list.wavelength.isel(transition=i).item()], {"units": str(u.AA)}),
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
            # A transition whose window misses the grid adds exactly zero; a broad-band list is mostly such.
            width_dims = [dim for dim in line_widths.dims if dim != "transition"]
            center_dims = [dim for dim in line_centers.dims if dim != "transition"]
            half_window = _GAUSSIAN_WINDOW_SIGMA * line_widths.max(dim=width_dims)
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
            line_widths,
            gaussian_norm,
            edges=edges,
            bin_method=bin_method,
            area_at_center=area_at_center,
        )
    else:
        contaminant_response = None

    if contaminant_response is not None:
        contaminant_response = contaminant_response.expand_dims(line=["contaminants"])
        contaminant_response = contaminant_response.assign_coords(
            line_wavelength=("line", [np.nan], {"units": str(u.AA)}),
        )
        responses.append(contaminant_response)

    responses = xr.concat(responses, dim="line", coords="different", compat="equals", join="exact")
    ds = xr.Dataset({"spectral_response": responses})

    if effective_area is not None:
        if area_at_center is None:
            scaling = effective_area
            if "wavelength" in effective_area.dims:
                scaling = (
                    effective_area.interp(wavelength=wavelength_grid).fillna(0).rename(wavelength="wavelength_grid")
                )
            ds["spectral_response"] = ds.spectral_response * scaling
        ds.spectral_response.attrs["units"] = str(_NORMALIZATION * u.erg * u.cm**5 / u.s / u.sr / u.AA)
        ds.spectral_response.attrs["effective_area_at"] = "bin center" if area_at_center is None else "line center"
    else:
        ds.spectral_response.attrs["units"] = str(_NORMALIZATION * u.erg * u.cm**3 / u.s / u.sr / u.AA)
    ds.spectral_response.attrs["bin_method"] = bin_method

    ds = ds.assign_coords(_wavelength_bin_coords(wavelength_grid, edges))

    ds.attrs["normalization"] = _NORMALIZATION
    return ds


def _wavelength_bins(
    wavelength_grid: u.Quantity | None, wavelength_edges: u.Quantity | None
) -> tuple[xr.DataArray, np.ndarray]:
    """
    The bin centers as a ``wavelength_bin`` array and the bin edges, from either input.
    """
    if (wavelength_grid is None) == (wavelength_edges is None):
        msg = "exactly one of wavelength_grid and wavelength_edges must be given"
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
    Edges halfway between neighboring centers, extrapolated by half a bin at the ends.
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


def _instrumental_sigma_in_angstrom(instrumental_fwhm: u.Quantity) -> float:
    """
    The Gaussian sigma of ``instrumental_fwhm``, in Angstrom.
    """
    if not instrumental_fwhm.isscalar:
        msg = "instrumental_fwhm must be scalar"
        raise ValueError(msg)
    converted = instrumental_fwhm.to_value(u.AA)
    if not np.all(np.isfinite(converted)) or np.any(converted < 0):
        msg = "instrumental_fwhm must contain finite, non-negative values"
        raise ValueError(msg)
    return converted / gaussian_sigma_to_fwhm


def _effective_area_in_cm2(effective_area: xr.DataArray | xr.Dataset | None) -> xr.DataArray | None:
    if effective_area is None:
        return None
    if isinstance(effective_area, xr.Dataset) and "effective_area" in effective_area:
        # The Dataset of create_effective_area or read_effective_area; its scalar channel stays behind.
        effective_area = effective_area["effective_area"].reset_coords(drop=True)
    if not isinstance(effective_area, xr.DataArray):
        msg = (
            "effective_area must be an xarray.DataArray, "
            "e.g. DEFAULTS_MUSE.main_line_effective_area_SG.sel(channel=...)"
        )
        raise TypeError(msg)
    if effective_area.ndim == 0:
        data = effective_area.data
        if not isinstance(data, u.Quantity):
            # attrs["units"] names the unit the raw value is already in.
            data = u.Quantity(data, effective_area.attrs.get("units"))
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
    area_unit = require_unit(dataset, "effective_area", "effective_area")
    if not area_unit.is_equivalent(u.cm**2):
        msg = f"effective_area units {area_unit} must be convertible to cm2"
        raise ValueError(msg)
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
    line_widths: xr.DataArray,
    gaussian_norm: float,
    *,
    edges: np.ndarray,
    bin_method: str,
    area_at_center: xr.DataArray | None = None,
) -> xr.DataArray | None:
    if not contaminant_indices:
        return None
    accumulator = None
    for i in tqdm(contaminant_indices, desc="Spectral contaminants", unit="line"):
        accumulator, template = _evaluate_gaussian_response(
            wavelength_grid,
            line_centers.isel(transition=i),
            line_widths.isel(transition=i),
            line_list.gofnt.isel(transition=i),
            gaussian_norm,
            accumulator=accumulator,
            edges=edges,
            bin_method=bin_method,
            area_at_center=None if area_at_center is None else area_at_center.isel(transition=i),
        )
    return xr.DataArray(accumulator, dims=template.dims, coords=template.coords)


def _validate_line_list(line_list: xr.Dataset) -> xr.Dataset:
    if "trans_index" in line_list.dims:
        msg = "line_list lies along trans_index, as muse 0.3 wrote line lists: rename that dimension to transition"
        raise ValueError(msg)
    if "transition" not in line_list.dims:
        msg = "line_list must lie along a transition dimension, as the line-list builders write it"
        raise ValueError(msg)
    if line_list.sizes["transition"] == 0:
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
    missing_transition = [
        name
        for name in ("wavelength", "atomic_number", "gofnt", "full_name")
        if "transition" not in line_list[name].dims
    ]
    if missing_transition:
        msg = f"line_list variables must include a transition dimension: {', '.join(missing_transition)}"
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


def _validate_main_lines(line_names: tuple[str, ...], main_lines: Sequence[str]) -> tuple[str, ...]:
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


def _atomic_mass_from_atomic_number(atomic_number: xr.DataArray, elements, atomic_mass_unit: float) -> xr.DataArray:
    atomic_mass = atomic_number.copy()
    masses = np.array([elements[int(value)].mass for value in atomic_mass.data.reshape(-1)])
    atomic_mass.data = masses.reshape(atomic_mass.shape) * atomic_mass_unit
    return atomic_mass


def _evaluate_gaussian_response(
    wavelength_grid: xr.DataArray,
    line_center: xr.DataArray,
    line_width: xr.DataArray,
    gofnt: xr.DataArray,
    gaussian_norm: float,
    *,
    accumulator: np.ndarray | None = None,
    edges: np.ndarray,
    bin_method: str,
    area_at_center: xr.DataArray | None = None,
) -> tuple[np.ndarray, xr.DataArray]:
    """
    Add one line's Gaussian to ``accumulator`` (a new full-grid array when `None`).

    ``area_at_center``, the effective area at the line center for each Doppler velocity,
    scales the line.
    """
    average = bin_method == "average"
    center, width_for_bounds = xr.broadcast(line_center, line_width)
    half_window = _GAUSSIAN_WINDOW_SIGMA * width_for_bounds
    window_lower, window_upper = np.min((center - half_window).data), np.max((center + half_window).data)
    if average:
        # Every bin overlapping the window, so a line narrower than a bin lands in the bin holding it.
        start = max(0, int(np.searchsorted(edges, window_lower, side="right")) - 1)
        stop = min(wavelength_grid.size, int(np.searchsorted(edges, window_upper, side="left")))
    else:
        start = np.searchsorted(wavelength_grid.data, window_lower, side="left")
        stop = np.searchsorted(wavelength_grid.data, window_upper, side="right")

    # Broadcast only the window: the full grid makes ~5 GB of temporaries per line in the 108 band.
    window_points = (
        xr.DataArray(edges[start : stop + 1], dims="wavelength_bin")
        if average
        else wavelength_grid.isel(wavelength_bin=slice(start, stop))
    )
    shift = (window_points - line_center).broadcast_like(gofnt)
    width, shift = xr.broadcast(line_width, shift)
    gofnt_scaled = (gofnt / _NORMALIZATION).broadcast_like(width)
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
    effective_area: xr.DataArray | xr.Dataset,
    doppler_velocity: u.Quantity | None = None,
) -> xr.Dataset:
    """
    Fold a rest-frame continuum through an effective area onto wavelength bins.

    For each Doppler velocity the continuum is shifted to ``lambda (1 + v / c)`` and divided
    by ``1 + v / c``, which keeps its integral; the product with the area at the observed
    wavelength is averaged over each bin, so a bin keeps the structure of the area inside it.
    The continuum is linear between its samples and constant beyond them, which fills the gap
    a shift opens at one end.

    The result has the dimensions and coordinates of a `create_spectral_response` result with
    ``bin_method="average"``, not always in the same order; `xarray.concat` joins the two
    along ``line``.

    Parameters
    ----------
    continuum : `xarray.Dataset`
        From `muse.instrument.create_chianti_continuum` or
        `muse.instrument.create_fiasco_continuum`, on a grid fine enough to resolve the
        effective area.
    wavelength_edges : `astropy.units.Quantity`
        Strictly increasing bin edges, one more than the number of bins.
    effective_area : `xarray.DataArray` or `xarray.Dataset`
        As for `create_spectral_response`, but required.
    doppler_velocity : `astropy.units.Quantity`, optional
        Scalar or one-dimensional bulk velocities; adds a ``doppler_velocity`` dimension.

    Returns
    -------
    `xarray.Dataset`
        ``spectral_response`` on the bins in the continuum's units times cm2, with
        ``wavelength_grid`` at the bin centers and ``wavelength_bounds``.

    Raises
    ------
    ValueError
        If ``effective_area`` is `None` or not in units convertible to cm2, ``wavelength_edges``
        is not increasing, a bin lies wholly beyond the samples of the continuum, or the
        continuum lacks units or has fewer than two wavelengths.
    """
    call_inputs = dict(locals())
    response_unit = require_unit(continuum, "spectral_response", "continuum.spectral_response")
    continuum_grid = np.asarray(
        coord_as_unit(continuum, "wavelength_grid", u.AA, "continuum.wavelength_grid"), dtype=float
    )
    centers, edges = _wavelength_bins(None, wavelength_edges)
    if effective_area is None:
        msg = "effective_area is required: a continuum response is the continuum seen through an area"
        raise ValueError(msg)
    if continuum_grid.size < 2:
        msg = "the continuum needs at least two wavelength samples to average over bins"
        raise ValueError(msg)
    if edges[1] <= continuum_grid[0] or edges[-2] >= continuum_grid[-1]:
        msg = (
            f"wavelength_edges reach beyond the continuum, which is given from {continuum_grid[0]:g} to "
            f"{continuum_grid[-1]:g} Angstrom: a bin lies wholly outside it"
        )
        raise ValueError(msg)
    area = _effective_area_in_cm2(effective_area)
    # Samples for every velocity: the outer edges and a curve's ends and just beyond, where it drops to zero.
    fixed = [edges[[0, -1]]]
    if area.ndim:
        ends = area.wavelength.values[[0, -1]]
        fixed += [ends, np.nextafter(ends, [-np.inf, np.inf])]
    rest_continuum = continuum.spectral_response.assign_coords(wavelength_bin=continuum_grid)
    velocities = [0.0] if doppler_velocity is None else np.atleast_1d(doppler_velocity.to_value(u.km / u.s))
    pieces = []
    for velocity in velocities:
        stretch = 1 + float(velocity) / const.c.to_value(u.km / u.s)
        # Observed samples keep the shifted continuum's shape, rest ones an area on the continuum grid.
        positions = np.unique(np.concatenate([*fixed, continuum_grid, continuum_grid * stretch]))
        positions = positions[(positions >= edges[0]) & (positions <= edges[-1])]
        area_there = area
        if area.ndim:
            area_there = (
                area.interp(wavelength=xr.DataArray(positions, dims="wavelength_bin")).fillna(0).drop_vars("wavelength")
            )
        # Clipping holds the continuum constant beyond its ends.
        emitted = rest_continuum.interp(
            wavelength_bin=np.clip(positions / stretch, continuum_grid[0], continuum_grid[-1])
        )
        product = emitted.assign_coords(wavelength_bin=positions) / stretch * area_there
        pieces.append(_bin_average(product, positions, edges))
    if doppler_velocity is None:
        binned = pieces[0]
    else:
        axis = xr.DataArray(
            np.asarray(velocities, dtype=float), dims="doppler_velocity", attrs={"units": str(u.km / u.s)}
        )
        binned = xr.concat(pieces, dim=axis).transpose("line", "doppler_velocity", ...)
    binned.attrs = {
        **continuum.spectral_response.attrs,
        "units": str(response_unit * u.cm**2),
        "effective_area_at": "bin average",
        "bin_method": "average",
    }
    result = xr.Dataset({"spectral_response": binned}).assign_coords(_wavelength_bin_coords(centers, edges))
    # New data: the attributes of the continuum, but not its dates or the muse that made it.
    update_attrs(result, continuum)
    add_history(result, call_inputs, create_continuum_response, sources=(continuum, effective_area))
    return result


def _bin_average(values: xr.DataArray, x: np.ndarray, edges: np.ndarray) -> xr.DataArray:
    """
    Average ``values``, sampled at ``x`` along ``wavelength_bin``, over the bins between
    ``edges``.

    The piecewise-linear curve through the samples is integrated exactly; ``x`` spans
    ``edges``.
    """
    y = np.moveaxis(np.asarray(values.data, dtype=float), values.get_axis_num("wavelength_bin"), -1)
    dx = np.diff(x)
    cumulative = np.zeros(y.shape)
    cumulative[..., 1:] = np.cumsum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)
    # The integral to each edge: the cumulative one at the sample before it plus the partial segment.
    index = np.clip(np.searchsorted(x, edges, side="right") - 1, 0, x.size - 2)
    t = edges - x[index]
    slope = (y[..., index + 1] - y[..., index]) / dx[index]
    at_edges = cumulative[..., index] + y[..., index] * t + 0.5 * slope * t**2
    averaged = np.diff(at_edges, axis=-1) / np.diff(edges)
    averaged = np.moveaxis(averaged, -1, values.get_axis_num("wavelength_bin"))
    coords = {name: coord for name, coord in values.coords.items() if "wavelength_bin" not in coord.dims}
    return xr.DataArray(averaged, dims=values.dims, coords=coords, attrs=values.attrs)
