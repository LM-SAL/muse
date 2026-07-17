"""
Spectral response functions built from CHIANTI line lists.
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.utils.utils import add_history, require_unit

__all__ = ["create_band_response"]

_EFFECTIVE_AREA_METHODS = ("linear", "nearest", "quadratic", "cubic")
_RESPONSE_NORMALIZATION = 1e-27


def create_band_response(
    line_list: xr.Dataset,
    wavelength_grid: u.Quantity | xr.DataArray,
    *,
    main_lines: Sequence[str],
    instrumental_width: u.Quantity | xr.DataArray = 0 * u.AA,
    doppler_velocity: u.Quantity | xr.DataArray | None = None,
    nonthermal_velocity: u.Quantity | xr.DataArray | None = None,
    effective_area: xr.DataArray | None = None,
    effective_area_method: str = "linear",
) -> xr.Dataset:
    """
    Create an instrument-neutral wavelength-space band response.

    Parameters
    ----------
    line_list : `xarray.Dataset`
        CHIANTI line list, e.g. from
        `muse.instrument.linelist.create_chianti_line_list`.
    wavelength_grid : `astropy.units.Quantity` or `xarray.DataArray`
        Wavelength samples. Quantities must be one-dimensional. A unit-bearing
        `xarray.DataArray` may carry extra physical dimensions but must use
        ``wavelength_bin`` as its sampling dimension.
    main_lines : sequence of `str`
        Stable ``full_name`` values to retain, in output order. Repeated
        transitions with the same name are summed. Unselected lines are not
        returned.
    instrumental_width : `astropy.units.Quantity` or `xarray.DataArray`, optional
        Instrumental-width sigma with wavelength units, by default 0 Angstrom.
        Quantities must be scalar. Use a unit-bearing `xarray.DataArray` to
        carry named dimensions.
    doppler_velocity : `astropy.units.Quantity` or `xarray.DataArray`, optional
        Doppler-velocity axis with velocity units.
    nonthermal_velocity : `astropy.units.Quantity` or `xarray.DataArray`, optional
        Nonthermal-velocity axis with velocity units.
    effective_area : `xarray.DataArray`, optional
        Effective area interpolated onto the wavelength grid.
    effective_area_method : `str`, optional
        Effective-area interpolation method, by default ``"linear"``.

    Returns
    -------
    `xarray.Dataset`
        Dataset containing ``spectral_response`` with temperature, velocity,
        line, and ``wavelength_bin`` dimensions. ``wavelength_grid`` contains
        the wavelength of each bin. Detector geometry is not added.
    """
    call_inputs = dict(locals())
    response = _create_wavelength_response(
        line_list,
        wavelength_grid,
        instrumental_width=instrumental_width,
        effective_area_method=effective_area_method,
        doppler_velocity=doppler_velocity,
        nonthermal_velocity=nonthermal_velocity,
        effective_area=effective_area,
        main_lines=main_lines,
    )
    add_history(response, call_inputs, create_band_response)
    return response


def _create_wavelength_response(
    line_list: xr.Dataset,
    wavelength_grid: u.Quantity | xr.DataArray,
    instrumental_width: u.Quantity | xr.DataArray = 0 * u.AA,
    effective_area_method: str = "linear",
    doppler_velocity: u.Quantity | xr.DataArray | None = None,
    nonthermal_velocity: u.Quantity | xr.DataArray | None = None,
    effective_area: xr.DataArray | None = None,
    main_lines: Sequence[str] | None = None,
    *,
    include_contaminants: bool = False,
) -> xr.Dataset:
    """
    Compute a spectral response as a function of velocity and temperature.

    Parameters
    ----------
    line_list : `xarray.Dataset`
        CHIANTI line list, e.g. from
        `muse.instrument.linelist.create_chianti_line_list`.
    wavelength_grid : `astropy.units.Quantity` or `xarray.DataArray`
        Wavelength samples. Quantities must be one-dimensional. A unit-bearing
        `xarray.DataArray` may carry extra physical dimensions but must use
        ``wavelength_bin`` as its sampling dimension.
    instrumental_width : `astropy.units.Quantity` or `xarray.DataArray`, optional
        Instrumental-width sigma with wavelength units, by default 0 Angstrom.
        Quantities must be scalar. Use a unit-bearing `xarray.DataArray` to
        carry named dimensions.
    effective_area_method : `str`, optional
        Effective-area interpolation method, by default ``"linear"``.
    doppler_velocity : `astropy.units.Quantity` or `xarray.DataArray`, optional
        Doppler-velocity axis with velocity units.
    nonthermal_velocity : `astropy.units.Quantity` or `xarray.DataArray`, optional
        Nonthermal-velocity axis with velocity units.
    effective_area : `xarray.DataArray`, optional
        Effective area interpolated onto the output wavelength grid.
    main_lines : sequence of `str`, optional
        Stable ``full_name`` values retained individually, in output order.
        Repeated transitions with the same name are summed. By default, retain
        every named line.
    include_contaminants : `bool`, optional
        Append the summed response of lines not selected by ``main_lines``, by
        default `False`. Pass an empty ``main_lines`` sequence to sum the full
        band when this is enabled.

    Returns
    -------
    `xarray.Dataset`
        Dataset containing ``spectral_response`` with temperature, velocity,
        line, and ``wavelength_bin`` dimensions. ``wavelength_grid`` contains
        the wavelength of each bin.
    """
    wavelength_grid = _wavelength_grid_in_angstrom(wavelength_grid)
    instrumental_width = _instrumental_width_in_angstrom(instrumental_width)
    if doppler_velocity is not None:
        doppler_velocity = _velocity_axis(doppler_velocity, "doppler_velocity")
    if nonthermal_velocity is not None:
        nonthermal_velocity = _velocity_axis(nonthermal_velocity, "nonthermal_velocity")
    effective_area = _effective_area_in_canonical_units(effective_area, effective_area_method)
    line_list = _validate_line_list(line_list)
    line_names = tuple(str(name) for name in line_list.full_name.values)
    main_lines = _validate_main_lines(line_names, main_lines)
    if not main_lines and not include_contaminants:
        msg = "main_lines cannot be empty unless include_contaminants=True"
        raise ValueError(msg)

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

    main_response_parts = {name: [] for name in main_lines}
    gaussian_norm = np.sqrt(2 * np.pi)
    for i, line_name in enumerate(line_names):
        if line_name not in main_response_parts:
            continue
        line_response, gofnt_scaled = _evaluate_gaussian_response(
            wavelength_grid,
            line_centers.isel(trans_index=i),
            doppler_widths.isel(trans_index=i),
            line_list.gofnt.isel(trans_index=i),
            gaussian_norm,
        )
        line_response = xr.DataArray(line_response, dims=gofnt_scaled.dims, coords=gofnt_scaled.coords)
        line_response = line_response.expand_dims(line=[line_name])
        line_response = line_response.assign_coords(
            line_wavelength=("line", [line_list.wavelength.isel(trans_index=i).item()], {"units": str(u.AA)}),
            component_kind=("line", ["line"]),
        )
        main_response_parts[line_name].append(line_response)

    responses = [xr.concat(parts, dim="_transition").sum("_transition") for parts in main_response_parts.values()]

    if include_contaminants:
        contaminant_indices = [i for i, name in enumerate(line_names) if name not in main_response_parts]
        contaminant_response = _create_contaminant_response(
            line_list,
            contaminant_indices,
            wavelength_grid,
            line_centers,
            doppler_widths,
            gaussian_norm,
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

    responses = xr.concat(responses, dim="line", coords="different", compat="equals")
    ds = xr.Dataset({"spectral_response": responses})
    spectral_response_attrs = ds.spectral_response.attrs

    if effective_area is not None:
        interp = effective_area.interp(wavelength=wavelength_grid, method=effective_area_method).fillna(0)
        interp = interp.rename(wavelength="wavelength_grid")
        ds["spectral_response"] = ds.spectral_response * interp
        ds.spectral_response.attrs.update(spectral_response_attrs)
        ds.spectral_response.attrs["units"] = str(_RESPONSE_NORMALIZATION * u.erg * u.cm**5 / u.s / u.sr / u.AA)
    else:
        ds.spectral_response.attrs["units"] = str(_RESPONSE_NORMALIZATION * u.erg * u.cm**3 / u.s / u.sr / u.AA)

    ds = ds.assign_coords(wavelength_grid=wavelength_grid.assign_attrs(units=str(u.AA)))

    ds.attrs["normalization"] = _RESPONSE_NORMALIZATION
    return ds


def _instrumental_width_in_angstrom(instrumental_width):
    if isinstance(instrumental_width, u.Quantity):
        if not instrumental_width.isscalar:
            msg = "instrumental_width Quantities must be scalar; use an xarray.DataArray for named dimensions"
            raise ValueError(msg)
        try:
            converted = instrumental_width.to_value(u.AA)
        except u.UnitConversionError as exc:
            msg = "instrumental_width units must be convertible to Angstrom"
            raise ValueError(msg) from exc
    elif isinstance(instrumental_width, xr.DataArray):
        unit = require_unit(
            xr.Dataset({"instrumental_width": instrumental_width}),
            "instrumental_width",
            "instrumental_width",
            convertible_to=u.AA,
        )
        converted = instrumental_width * unit.to(u.AA)
        converted = converted.assign_attrs({**instrumental_width.attrs, "units": str(u.AA)})
    else:
        msg = "instrumental_width must be an astropy Quantity or unit-bearing xarray.DataArray"
        raise TypeError(msg)
    if not np.all(np.isfinite(converted)) or np.any(converted < 0):
        msg = "instrumental_width must contain finite, non-negative values"
        raise ValueError(msg)
    return converted


def _effective_area_in_canonical_units(effective_area, effective_area_method):
    if effective_area is None:
        return None
    if not isinstance(effective_area, xr.DataArray):
        msg = "effective_area must be an xarray.DataArray"
        raise TypeError(msg)
    if effective_area_method not in _EFFECTIVE_AREA_METHODS:
        msg = f"effective_area_method must be one of {_EFFECTIVE_AREA_METHODS} when effective_area is supplied"
        raise ValueError(msg)
    dataset = xr.Dataset({"effective_area": effective_area})
    area_unit = require_unit(dataset, "effective_area", "effective_area", convertible_to=u.cm**2)
    wavelength_unit = require_unit(
        dataset,
        "wavelength",
        "effective_area wavelength coordinate",
        coord_only=True,
        convertible_to=u.AA,
    )
    converted = (effective_area * area_unit.to(u.cm**2)).assign_attrs({**effective_area.attrs, "units": str(u.cm**2)})
    if not np.all(np.isfinite(converted)) or np.any(converted < 0):
        msg = "effective_area must contain finite, non-negative values"
        raise ValueError(msg)
    wavelength = xr.DataArray(
        effective_area.wavelength.data * wavelength_unit.to(u.AA),
        dims=effective_area.wavelength.dims,
        attrs={**effective_area.wavelength.attrs, "units": str(u.AA)},
    )
    wavelength_values = np.asarray(wavelength)
    if (
        wavelength_values.ndim != 1
        or wavelength_values.size == 0
        or not np.all(np.isfinite(wavelength_values))
        or np.any(np.diff(wavelength_values) <= 0)
    ):
        msg = "effective_area wavelength coordinate must be one-dimensional, finite, and strictly increasing"
        raise ValueError(msg)
    return converted.assign_coords(wavelength=wavelength)


def _velocity_axis(values, dim):
    if isinstance(values, u.Quantity):
        try:
            values = values.to_value(u.km / u.s)
        except u.UnitConversionError as exc:
            msg = f"{dim} units must be convertible to km / s"
            raise ValueError(msg) from exc
    elif isinstance(values, xr.DataArray):
        unit = require_unit(xr.Dataset({dim: values}), dim, dim, convertible_to=u.km / u.s)
        values = values.data * unit.to(u.km / u.s)
    else:
        msg = f"{dim} must be an astropy Quantity or unit-bearing xarray.DataArray"
        raise TypeError(msg)
    values = np.atleast_1d(values)
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


def _wavelength_grid_in_angstrom(wavelength_grid):
    if isinstance(wavelength_grid, u.Quantity):
        try:
            values = wavelength_grid.to_value(u.AA)
        except u.UnitConversionError as exc:
            msg = "wavelength_grid units must be convertible to Angstrom"
            raise ValueError(msg) from exc
        if values.ndim != 1:
            msg = "wavelength_grid quantities must be one-dimensional"
            raise ValueError(msg)
        wavelength_grid = xr.DataArray(values, dims="wavelength_bin")
    elif isinstance(wavelength_grid, xr.DataArray):
        unit = require_unit(
            xr.Dataset({"wavelength_grid": wavelength_grid}),
            "wavelength_grid",
            "wavelength_grid",
            convertible_to=u.AA,
        )
        if "wavelength_bin" not in wavelength_grid.dims:
            msg = "wavelength_grid DataArrays must include a wavelength_bin dimension"
            raise ValueError(msg)
        wavelength_grid = wavelength_grid * unit.to(u.AA)
    else:
        msg = "wavelength_grid must be an astropy Quantity or unit-bearing xarray.DataArray"
        raise TypeError(msg)

    values = np.asarray(wavelength_grid)
    if values.size == 0 or not np.all(np.isfinite(values)):
        msg = "wavelength_grid must contain finite values"
        raise ValueError(msg)
    axis = wavelength_grid.get_axis_num("wavelength_bin")
    if np.any(np.diff(values, axis=axis) <= 0):
        msg = "wavelength_grid must be strictly increasing along wavelength_bin"
        raise ValueError(msg)
    return wavelength_grid.assign_attrs(units=str(u.AA))


def _create_contaminant_response(
    line_list,
    contaminant_indices,
    wavelength_grid,
    line_centers,
    doppler_widths,
    gaussian_norm,
):
    if not contaminant_indices:
        return None
    accumulator = None
    for i in contaminant_indices:
        accumulator, gofnt_scaled = _evaluate_gaussian_response(
            wavelength_grid,
            line_centers.isel(trans_index=i),
            doppler_widths.isel(trans_index=i),
            line_list.gofnt.isel(trans_index=i),
            gaussian_norm,
            accumulator=accumulator,
        )
    return xr.DataArray(accumulator, dims=gofnt_scaled.dims, coords=gofnt_scaled.coords)


def _validate_line_list(line_list):
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
    wavelength = (line_list.wavelength * wavelength_unit.to(u.AA)).assign_attrs(
        {**line_list.wavelength.attrs, "units": str(u.AA)}
    )
    gofnt_target_unit = u.erg * u.cm**3 / (u.s * u.sr)
    gofnt = (line_list.gofnt * gofnt_unit.to(gofnt_target_unit)).assign_attrs(
        {**line_list.gofnt.attrs, "units": str(gofnt_target_unit)}
    )
    return line_list.assign(wavelength=wavelength, gofnt=gofnt)


def _validate_main_lines(line_names, main_lines):
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


def _atomic_mass_from_atomic_number(atomic_number, elements, proton_mass):
    max_atomic_number = max(element.number for element in elements)
    if np.any(np.asarray(atomic_number) > max_atomic_number):
        msg = f"line_list.atomic_number must be between 1 and {max_atomic_number}"
        raise ValueError(msg)
    atomic_mass = atomic_number.copy()
    masses = np.array([elements[int(value)].mass for value in atomic_mass.data.reshape(-1)])
    atomic_mass.data = masses.reshape(atomic_mass.shape) * proton_mass
    return atomic_mass


def _broadcast_response_inputs(wavelength_grid, line_center, doppler_width, gofnt):
    shift = (wavelength_grid - line_center).broadcast_like(gofnt)
    width, shift = xr.broadcast(doppler_width, shift)
    gofnt_scaled = gofnt.broadcast_like(width) / _RESPONSE_NORMALIZATION
    return xr.broadcast(gofnt_scaled, width, shift)


def _evaluate_gaussian_response(
    wavelength_grid,
    line_center,
    doppler_width,
    gofnt,
    gaussian_norm,
    *,
    accumulator=None,
):
    gofnt_scaled, width, shift = _broadcast_response_inputs(wavelength_grid, line_center, doppler_width, gofnt)
    response = gofnt_scaled.data * np.exp(-0.5 * (shift.data / width.data) ** 2) / gaussian_norm / width.data
    if accumulator is not None:
        response = response + accumulator
    return response, gofnt_scaled
