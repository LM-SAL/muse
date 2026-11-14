"""
Continuum emission as wavelength-space responses: free-free, free-bound and two-photon.
"""

import numpy as np
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.instrument.atomic import (
    _broadcast_plasma_grid,
    _fiasco_ion_kwargs,
    _import_fiasco,
    _initialize_chianti,
    _over_ion_chunks,
    _plasma_axis_units,
    _validate_plasma_inputs,
)
from muse.instrument.spectral import (
    _centers_to_edges,
    _effective_area_in_canonical_units,
    _wavelength_bin_coords,
    _wavelength_bins,
)
from muse.log import logger
from muse.utils.utils import _require_increasing_axis, add_history, coord_as_unit, require_unit
from muse.variables import DEFAULTS_MUSE

__all__ = ["bin_continuum", "create_chianti_continuum", "create_fiasco_continuum"]

_COMPONENTS = ("free-free", "free-bound", "two-photon")
_CONTINUUM_UNIT = u.erg * u.cm**3 / (u.s * u.AA)


@u.quantity_input(wavelength_grid=u.AA)
def create_chianti_continuum(
    temperature: xr.DataArray,
    wavelength_grid: u.Quantity,
    *,
    density: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    abundance: str | None = None,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
) -> xr.Dataset:
    """
    Compute the continuum with ChiantiPy, summed over every ion of every abundant
    element.

    Free-free and free-bound emission do not depend on the density and are computed once;
    the two-photon emission of the hydrogen- and helium-like ions is evaluated on the plasma
    grid. The result is a wavelength-space response like `create_spectral_response` gives,
    with one ``line`` per component, at rest and without an effective area: fold it onto
    bins with `bin_continuum`.

    Parameters
    ----------
    temperature, density, pressure
        As for `create_chianti_line_list`; a density grid comes out on ``logD``.
    wavelength_grid : `astropy.units.Quantity`
        Increasing wavelengths the continuum is evaluated at.
    abundance : `str`, optional
        CHIANTI abundance name; ChiantiPy's default applies when omitted.
    minimum_abundance : `float`, optional
        Elements with an abundance relative to hydrogen below this are skipped, by default
        ``DEFAULTS_MUSE.minimum_abundance``.

    Returns
    -------
    `xarray.Dataset`
        ``spectral_response(line, ..., logT, wavelength_bin)`` in
        ``1e-27 erg cm3 / (Angstrom s sr)`` with ``line`` equal to ``free-free``, ``free-bound``
        and ``two-photon``, the plasma axis, ``wavelength_grid`` with its ``wavelength_bounds``
        and a ``line_wavelength`` of NaN.

    Raises
    ------
    RuntimeError
        If no ion contributed free-free emission, which happens when the database is not where
        ``XUVTOP`` points.
    """
    call_inputs = dict(locals())
    temperature, plasma_grid, grid = _validate_continuum_inputs(temperature, density, pressure, wavelength_grid)
    chiantipy_version, ch = _initialize_chianti()
    import ChiantiPy.tools.constants as chconst  # noqa: PLC0415
    import ChiantiPy.tools.data as chdata  # noqa: PLC0415
    import ChiantiPy.tools.io as chio  # noqa: PLC0415

    temperature_bc, density_bc, extra_coords = _broadcast_plasma_grid(
        temperature, plasma_grid, by_density=density is not None
    )
    temperatures = np.asarray(temperature.data, dtype=float).reshape(-1)
    # ChiantiPy mishandles a one-element temperature grid: evaluate it twice and keep one row.
    chianti_temperatures = np.repeat(temperatures, 2) if temperatures.size == 1 else temperatures
    densities = _densities_by_plasma_point(density_bc, temperature_bc)

    abundances = chio.abundanceRead(abundance)["abundance"] if abundance else chdata.AbundanceDefault["abundance"]
    elements = [z for z in range(1, min(30, len(abundances)) + 1) if abundances[z - 1] >= minimum_abundance]
    totals = {component: np.zeros((temperatures.size, grid.size)) for component in ("free-free", "free-bound")}
    two_photon = np.zeros((len(densities), temperatures.size, grid.size))
    contributing = dict.fromkeys(_COMPONENTS, 0)
    for z in elements:
        for stage in range(1, z + 2):
            ion_name = f"{chconst.El[z - 1]}_{stage}"
            # Not every ion has continuum data; ChiantiPy reports that by raising or by an errorMessage.
            try:
                continuum = ch.continuum(ion_name, chianti_temperatures, abundance=abundance)
            except Exception:  # noqa: BLE001, S112 - ions the database lacks
                continue
            for component, method, result_name in (
                ("free-free", continuum.freeFree, "FreeFree"),
                ("free-bound", continuum.freeBound, "FreeBound"),
            ):
                part = _chianti_component(
                    method, result_name, grid, temperatures.size, includeAbund=True, includeIoneq=True
                )
                if part is not None:
                    totals[component] += part
                    contributing[component] += 1
            if stage not in (z, z - 1):
                continue
            counted = False
            for plasma_index, electron_density in enumerate(densities):
                try:
                    ion = ch.ion(
                        ion_name,
                        temperature=chianti_temperatures,
                        eDensity=np.repeat(electron_density, 2) if temperatures.size == 1 else electron_density,
                        abundance=abundance,
                    )
                except Exception:  # noqa: BLE001, S112 - ions the database lacks
                    continue
                two_photon_part = _chianti_component(ion.twoPhoton, "TwoPhoton", grid, temperatures.size)
                if two_photon_part is not None:
                    two_photon[plasma_index] += two_photon_part
                    counted = True
            contributing["two-photon"] += counted
    logger.info(f"ChiantiPy continuum: ions contributing {contributing}")
    if contributing["free-free"] == 0:
        msg = "ChiantiPy computed free-free emission for no ion, so the continuum would be all zero; check XUVTOP"
        raise RuntimeError(msg)
    if contributing["two-photon"] == 0:
        logger.warning("The two-photon continuum is zero: no hydrogen- or helium-like ion contributed")

    continuum = _continuum_dataset(
        totals["free-free"],
        totals["free-bound"],
        two_photon,
        temperature=temperature,
        temperature_bc=temperature_bc,
        extra_coords=extra_coords,
        grid=grid,
        attrs={"Chiantipy": chiantipy_version, "Chianti": chio.versionRead(), "minimum_abundance": minimum_abundance},
    )
    add_history(continuum, call_inputs, create_chianti_continuum)
    return continuum


@u.quantity_input(wavelength_grid=u.AA)
def create_fiasco_continuum(
    temperature: xr.DataArray,
    wavelength_grid: u.Quantity,
    *,
    density: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    abundance: str | None = None,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    workers: int = 1,
    hdf5_dbase_root=None,
) -> xr.Dataset:
    """
    Compute the continuum with fiasco, summed over every ion of every abundant element.

    The counterpart of `create_chianti_continuum` with the same output, built on
    ``fiasco.IonCollection``, whose methods include the abundance and the ionization fraction
    like ChiantiPy's ``includeAbund`` and ``includeIoneq``. fiasco's continuum emission is
    not per steradian (its free-free emission since fiasco 0.9), so every component is
    divided by 4 pi to match ChiantiPy.

    Parameters
    ----------
    temperature, wavelength_grid, density, pressure, abundance, minimum_abundance
        As for `create_chianti_continuum`; fiasco's default abundance applies when omitted.
    workers : `int`, optional
        Processes the ions are split over, by default 1; ``-1`` uses every CPU this
        process may run on.
    hdf5_dbase_root : path-like, optional
        fiasco's HDF5 database; by default its configured one.

    Returns
    -------
    `xarray.Dataset`
        As for `create_chianti_continuum`.

    Raises
    ------
    RuntimeError
        If fiasco loaded no ion, which happens without its HDF5 database.

    Notes
    -----
    Needs fiasco 0.9.0 or later, and joblib when ``workers`` is not 1.
    """
    call_inputs = dict(locals())
    temperature, plasma_grid, grid = _validate_continuum_inputs(temperature, density, pressure, wavelength_grid)
    fiasco = _import_fiasco()
    temperature_bc, density_bc, extra_coords = _broadcast_plasma_grid(
        temperature, plasma_grid, by_density=density is not None
    )
    temperatures = np.asarray(temperature.data, dtype=float).reshape(-1) * u.K
    wavelengths = grid * u.AA
    densities = _densities_by_plasma_point(density_bc, temperature_bc)
    ion_kwargs = _fiasco_ion_kwargs(abundance, hdf5_dbase_root)

    ions, two_photon_ions = [], []
    for z in range(1, 31):
        try:
            element = fiasco.Element(z, temperatures, **ion_kwargs)
        except Exception:  # noqa: BLE001, S112 - elements the database lacks
            continue
        if element.abundance is None or element.abundance < minimum_abundance:
            continue
        for ion in element:
            ions.append(ion)
            if ion.charge_state + 1 in (z, z - 1):
                two_photon_ions.append(ion)
    if not ions:
        msg = "fiasco loaded no ion, so the continuum would be all zero; check its HDF5 database (hdf5_dbase_root)"
        raise RuntimeError(msg)
    free_free = sum(_over_ion_chunks(_fiasco_component, ions, workers, "free_free", wavelengths))
    free_bound = sum(_over_ion_chunks(_fiasco_component, ions, workers, "free_bound", wavelengths))
    two_photon = np.zeros((len(densities), temperatures.size, grid.size))
    for plasma_index, electron_density in enumerate(densities):
        # One density per temperature on a pressure grid: the collection evaluates the full
        # temperature x density grid, of which the diagonal is wanted.
        unique_densities = np.unique(electron_density)
        parts = _over_ion_chunks(
            _fiasco_component, two_photon_ions, workers, "two_photon", wavelengths, unique_densities * u.cm**-3
        )
        two_photon_grid = sum(parts)
        columns = np.searchsorted(unique_densities, electron_density)
        two_photon[plasma_index] = two_photon_grid[np.arange(temperatures.size), columns, :]
    logger.info(f"fiasco continuum: {len(ions)} ions, {len(two_photon_ions)} hydrogen- or helium-like")

    continuum = _continuum_dataset(
        free_free,
        free_bound,
        two_photon,
        temperature=temperature,
        temperature_bc=temperature_bc,
        extra_coords=extra_coords,
        grid=grid,
        attrs={"fiasco": fiasco.__version__, "minimum_abundance": minimum_abundance},
    )
    add_history(continuum, call_inputs, create_fiasco_continuum)
    return continuum


@u.quantity_input(wavelength_edges=u.AA, doppler_velocity=u.km / u.s)
def bin_continuum(
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
        From `create_chianti_continuum` or `create_fiasco_continuum`, on a grid fine enough
        to resolve the effective area.
    wavelength_edges : `astropy.units.Quantity`
        Strictly increasing bin edges, one more than the number of bins.
    effective_area : `xarray.DataArray`
        A curve with a ``wavelength`` coordinate, in units convertible to cm2.
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
    area_curve = _effective_area_in_canonical_units(effective_area)
    if "wavelength" not in area_curve.dims:
        msg = "effective_area must be a curve with a wavelength coordinate"
        raise ValueError(msg)
    area_on_grid = (
        area_curve.interp(wavelength=xr.DataArray(grid, dims="wavelength_bin")).fillna(0).drop_vars("wavelength")
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
    add_history(result, call_inputs, bin_continuum, sources=(continuum,))
    return result


def _validate_continuum_inputs(temperature, density, pressure, wavelength_grid):
    temperature, plasma_grid = _validate_plasma_inputs(temperature, density, pressure)
    grid = np.asarray(wavelength_grid.to_value(u.AA), dtype=float).reshape(-1)
    if grid.size < 2:
        msg = "wavelength_grid must hold at least two wavelengths"
        raise ValueError(msg)
    _require_increasing_axis(grid, "wavelength_grid", positive=True)
    return temperature, plasma_grid, grid


def _densities_by_plasma_point(density_bc: xr.DataArray, temperature_bc: xr.DataArray) -> list[np.ndarray]:
    """
    One density array per plasma point, each along the temperature axis.
    """
    plasma_dim = next(dim for dim in temperature_bc.dims if dim != "logT")
    return [np.asarray(row.data, dtype=float) for row in density_bc.transpose(plasma_dim, "logT")]


def _chianti_component(
    method, result_name: str, grid: np.ndarray, number_of_temperatures: int, **kwargs
) -> np.ndarray | None:
    """
    One ChiantiPy continuum component as ``(temperature, wavelength)``, or `None` if the
    ion lacks it; ``method`` stores its result on its object under ``result_name``.
    """
    try:
        method(grid, **kwargs)
    except Exception:  # noqa: BLE001 - ChiantiPy raises assorted errors for ions without data
        return None
    result = getattr(method.__self__, result_name)
    if "errorMessage" in result:
        return None
    intensity = np.atleast_2d(result["intensity"])
    if intensity.shape[0] != max(number_of_temperatures, 2):
        intensity = intensity.T
    return intensity[:number_of_temperatures]


def _fiasco_component(ions: list, method: str, wavelengths: u.Quantity, *args) -> np.ndarray:
    from fiasco import IonCollection  # noqa: PLC0415

    # Per steradian, as ChiantiPy reports its continua.
    return getattr(IonCollection(*ions), method)(wavelengths, *args).to_value(_CONTINUUM_UNIT) / (4 * np.pi)


def _continuum_dataset(
    free_free, free_bound, two_photon, *, temperature, temperature_bc, extra_coords, grid, attrs
) -> xr.Dataset:
    plasma_dim = next(dim for dim in temperature_bc.dims if dim != "logT")
    normalization = DEFAULTS_MUSE.normalization
    coords = {"logT": ("logT", np.log10(temperature.data), {"units": str(u.dex(u.K))}), **extra_coords}
    density_independent = [
        xr.DataArray(values / normalization, dims=("logT", "wavelength_bin")).expand_dims(
            {plasma_dim: extra_coords[plasma_dim]}
        )
        for values in (free_free, free_bound)
    ]
    components = [
        *density_independent,
        xr.DataArray(two_photon / normalization, dims=(plasma_dim, "logT", "wavelength_bin")),
    ]
    spectral_response = xr.concat(
        [component.transpose(*temperature_bc.dims, "wavelength_bin") for component in components],
        dim=xr.DataArray(list(_COMPONENTS), dims="line"),
    ).assign_coords(coords)
    spectral_response.attrs["units"] = str(normalization * u.erg * u.cm**3 / u.s / u.sr / u.AA)
    spectral_response[plasma_dim].attrs["units"] = _plasma_axis_units(plasma_dim)
    continuum = xr.Dataset({"spectral_response": spectral_response}, attrs={"normalization": normalization, **attrs})
    return continuum.assign_coords(
        **_wavelength_bin_coords(grid, _centers_to_edges(grid)),
        line_wavelength=("line", np.full(len(_COMPONENTS), np.nan), {"units": str(u.AA)}),
    )


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
