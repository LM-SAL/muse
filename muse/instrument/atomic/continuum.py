"""
Continuum emission as wavelength-space responses: free-free, free-bound and two-photon.
"""

import numpy as np
import xarray as xr

import astropy.units as u

from muse.instrument.atomic.backends import (
    _broadcast_plasma_grid,
    _fiasco_ion_kwargs,
    _import_fiasco,
    _initialize_chianti,
    _over_ion_chunks,
    _plasma_axis_units,
    _validate_plasma_inputs,
)
from muse.instrument.response.spectral import _centers_to_edges, _wavelength_bin_coords
from muse.log import logger
from muse.utils.utils import _require_increasing_axis, add_history
from muse.variables import DEFAULTS_MUSE

__all__ = ["create_chianti_continuum", "create_fiasco_continuum"]

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
    bins with `muse.instrument.create_continuum_response`.

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
        If no ion contributed free-free or free-bound emission, which happens when the
        database is not where ``XUVTOP`` points.
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
            # A neutral has no charge to radiate free-free emission and nothing recombines onto
            # it: ChiantiPy returns zeros for it, after dividing by zero. The charged ions without
            # the data of a component report an errorMessage when it is asked for.
            if stage > 1:
                continuum = ch.continuum(ion_name, chianti_temperatures, abundance=abundance)
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
                ion = ch.ion(
                    ion_name,
                    temperature=chianti_temperatures,
                    eDensity=np.repeat(electron_density, 2) if temperatures.size == 1 else electron_density,
                    abundance=abundance,
                )
                two_photon_part = _chianti_component(ion.twoPhoton, "TwoPhoton", grid, temperatures.size)
                if two_photon_part is not None:
                    two_photon[plasma_index] += two_photon_part
                    counted = True
            contributing["two-photon"] += counted
    logger.info(f"ChiantiPy continuum: ions contributing {contributing}")
    for component in ("free-free", "free-bound"):
        if contributing[component] == 0:
            msg = f"ChiantiPy computed {component} emission for no ion, so the continuum would be wrong; check XUVTOP"
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

    from fiasco.util.exceptions import MissingDatasetException, MissingIonError  # noqa: PLC0415

    # Once for all elements: each Element would otherwise compute it again, 0.4 s a time.
    proton_electron_ratio = fiasco.proton_electron_ratio(temperatures, **ion_kwargs)
    ions, two_photon_ions = [], []
    for z in range(1, 31):
        try:
            element = fiasco.Element(z, temperatures, proton_electron_ratio=proton_electron_ratio, **ion_kwargs)
        except (MissingDatasetException, MissingIonError):  # an element the database lacks
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
        # One density per temperature: fiasco pairs them itself and repeats the result along
        # its density axis, so any column of it is the answer.
        parts = _over_ion_chunks(
            _fiasco_component,
            two_photon_ions,
            workers,
            "two_photon",
            wavelengths,
            electron_density * u.cm**-3,
            couple_density_to_temperature=True,
        )
        two_photon[plasma_index] = sum(parts)[:, 0, :]
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
    except AttributeError:
        # How ChiantiPy 0.16 fails for an ion without the levels of this component: the
        # two-photon emission of 21 hydrogen- and helium-like ions of CHIANTI 11.0.2, and no
        # other error among its 495 ions. Anything else is a real error.
        return None
    result = getattr(method.__self__, result_name)
    if "errorMessage" in result:
        return None
    intensity = np.atleast_2d(result["intensity"])
    if intensity.shape[0] != max(number_of_temperatures, 2):
        intensity = intensity.T
    return intensity[:number_of_temperatures]


def _fiasco_component(ions: list, method: str, wavelengths: u.Quantity, *args, **kwargs) -> np.ndarray:
    from fiasco import IonCollection  # noqa: PLC0415

    # Per steradian, as ChiantiPy reports its continua.
    emission = getattr(IonCollection(*ions), method)(wavelengths, *args, **kwargs)
    return emission.to_value(_CONTINUUM_UNIT) / (4 * np.pi)


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
