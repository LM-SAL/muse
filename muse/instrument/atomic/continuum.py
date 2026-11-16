"""
Continuum emission as wavelength-space responses: free-free, free-bound and two-photon.
"""

from pathlib import Path

import numpy as np
import xarray as xr

import astropy.units as u

from muse.instrument.atomic.backends import (
    _MINIMUM_ABUNDANCE,
    _broadcast_plasma_grid,
    _fiasco_ion_kwargs,
    _import_fiasco,
    _initialize_chianti,
    _map_ion_groups,
    _plasma_axis_units,
    _validate_plasma_inputs,
)
from muse.instrument.response.spectral import _NORMALIZATION, _centers_to_edges, _wavelength_bin_coords
from muse.log import logger
from muse.utils.utils import _require_increasing_axis, add_history

__all__ = ["create_chianti_continuum", "create_fiasco_continuum"]

_COMPONENTS = ("free-free", "free-bound", "two-photon")
_CONTINUUM_UNIT = u.erg * u.cm**3 / (u.s * u.AA)


@u.quantity_input(wavelength_grid=u.AA)
def create_chianti_continuum(
    temperature: u.Quantity,
    wavelength_grid: u.Quantity,
    *,
    density: u.Quantity | None = None,
    pressure: u.Quantity | None = None,
    abundance: str | None = None,
    minimum_abundance: float = _MINIMUM_ABUNDANCE,
) -> xr.Dataset:
    """
    Compute the free-free, free-bound and two-photon continuum with ChiantiPy.

    Free-free and free-bound emission do not depend on density and are computed once; the
    two-photon emission of hydrogen- and helium-like ions at every density or pressure.
    The result is a rest-frame response like that of
    `muse.instrument.create_spectral_response`, one ``line`` per component and without an
    effective area; `muse.instrument.create_continuum_response` folds it onto bins.

    Parameters
    ----------
    temperature, density, pressure
        As for `create_chianti_line_list`.
    wavelength_grid : `astropy.units.Quantity`
        Increasing wavelengths the continuum is evaluated at.
    abundance : `str`, optional
        CHIANTI abundance name; by default ChiantiPy's. The name used is recorded in the
        ``abundance`` attribute.
    minimum_abundance : `float`, optional
        Elements with an abundance relative to hydrogen below this are skipped, by default
        1e-21.

    Returns
    -------
    `xarray.Dataset`
        ``spectral_response`` in 1e-27 erg cm3 / (Angstrom s sr), along ``(line, logT,
        logD, wavelength_bin)`` for a density grid and ``(line, pressure, logT,
        wavelength_bin)`` for a pressure grid. ``line`` holds ``free-free``,
        ``free-bound`` and ``two-photon``. The coordinates are ``wavelength_grid`` (the
        input) with its ``wavelength_bounds``, and ``line_wavelength``, NaN throughout.

    Raises
    ------
    RuntimeError
        If no ion contributed free-free or free-bound emission: ``minimum_abundance``
        excludes every element, or the database is not where ``XUVTOP`` points.

    Notes
    -----
    The ``XUVTOP`` environment variable must point to a local CHIANTI database.
    """
    call_inputs = dict(locals())
    temperature, plasma_grid, wavelength_grid = _validate_continuum_inputs(
        temperature, density, pressure, wavelength_grid
    )
    chiantipy_version, ch = _initialize_chianti()
    import ChiantiPy.tools.constants as chconst  # noqa: PLC0415
    import ChiantiPy.tools.data as chdata  # noqa: PLC0415
    import ChiantiPy.tools.io as chio  # noqa: PLC0415

    temperature_broadcast, density_broadcast, plasma_coords = _broadcast_plasma_grid(
        temperature, plasma_grid, by_density=density is not None
    )
    temperatures = np.asarray(temperature.data, dtype=float).reshape(-1)
    # ChiantiPy mishandles a one-element temperature grid: evaluate it twice and keep one row.
    chianti_temperatures = np.repeat(temperatures, 2) if temperatures.size == 1 else temperatures
    densities = _densities_by_plasma_point(density_broadcast, temperature_broadcast)

    abundances = chio.abundanceRead(abundance)["abundance"] if abundance else chdata.AbundanceDefault["abundance"]
    elements = [z for z in range(1, min(30, len(abundances)) + 1) if abundances[z - 1] >= minimum_abundance]
    totals = {
        component: np.zeros((temperatures.size, wavelength_grid.size)) for component in ("free-free", "free-bound")
    }
    two_photon = np.zeros((len(densities), temperatures.size, wavelength_grid.size))
    contributing = dict.fromkeys(_COMPONENTS, 0)
    for z in elements:
        for stage in range(1, z + 2):
            ion_name = f"{chconst.El[z - 1]}_{stage}"
            # A neutral has no free-free or free-bound emission, and ChiantiPy divides by zero for one.
            if stage > 1:
                continuum = ch.continuum(ion_name, chianti_temperatures, abundance=abundance)
                for component, method, result_name in (
                    ("free-free", continuum.freeFree, "FreeFree"),
                    ("free-bound", continuum.freeBound, "FreeBound"),
                ):
                    part = _chianti_component(
                        method, result_name, wavelength_grid, temperatures.size, includeAbund=True, includeIoneq=True
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
                two_photon_part = _chianti_component(ion.twoPhoton, "TwoPhoton", wavelength_grid, temperatures.size)
                if two_photon_part is not None:
                    two_photon[plasma_index] += two_photon_part
                    counted = True
            contributing["two-photon"] += counted
    logger.info(f"ChiantiPy continuum: ions contributing {contributing}")
    for component in ("free-free", "free-bound"):
        if contributing[component] == 0:
            msg = (
                f"ChiantiPy computed {component} emission for no ion, so the continuum would be wrong; "
                "check XUVTOP, and that minimum_abundance does not exclude every element"
            )
            raise RuntimeError(msg)
    if contributing["two-photon"] == 0:
        logger.warning("The two-photon continuum is zero: no hydrogen- or helium-like ion contributed")

    continuum = _continuum_dataset(
        totals["free-free"],
        totals["free-bound"],
        two_photon,
        temperature=temperature,
        temperature_broadcast=temperature_broadcast,
        plasma_coords=plasma_coords,
        wavelength_grid=wavelength_grid,
        attrs={
            "chiantipy_version": chiantipy_version,
            "chianti_version": chio.versionRead(),
            "minimum_abundance": minimum_abundance,
        },
    )
    abundance = abundance or chdata.Defaults["abundfile"]
    add_history(continuum, {**call_inputs, "abundance": abundance}, create_chianti_continuum)
    return continuum


@u.quantity_input(wavelength_grid=u.AA)
def create_fiasco_continuum(
    temperature: u.Quantity,
    wavelength_grid: u.Quantity,
    *,
    density: u.Quantity | None = None,
    pressure: u.Quantity | None = None,
    abundance: str | None = None,
    minimum_abundance: float = _MINIMUM_ABUNDANCE,
    workers: int = 1,
    hdf5_dbase_root: str | Path | None = None,
) -> xr.Dataset:
    """
    Compute the free-free, free-bound and two-photon continuum with fiasco.

    Built on ``fiasco.IonCollection``, with the output of `create_chianti_continuum`.
    fiasco's continuum is not per steradian, so every component is divided by 4 pi.

    Parameters
    ----------
    temperature, wavelength_grid, density, pressure, abundance, minimum_abundance
        As for `create_chianti_continuum`, but ``abundance`` defaults to fiasco's, which
        need not be ChiantiPy's.
    workers : `int`, optional
        Processes that share the ions, by default 1; ``-1`` uses every CPU this process
        may run on.
    hdf5_dbase_root : `str` or `pathlib.Path`, optional
        fiasco's HDF5 database; by default its configured one.

    Returns
    -------
    `xarray.Dataset`
        As for `create_chianti_continuum`.

    Raises
    ------
    RuntimeError
        If fiasco loaded no ion: ``minimum_abundance`` excludes every element, or the HDF5
        database lacks them.

    Notes
    -----
    Needs fiasco, and joblib when ``workers`` is not 1; ``pip install muse[fiasco]`` brings
    both.
    """
    call_inputs = dict(locals())
    temperature, plasma_grid, wavelength_grid = _validate_continuum_inputs(
        temperature, density, pressure, wavelength_grid
    )
    fiasco = _import_fiasco()
    temperature_broadcast, density_broadcast, plasma_coords = _broadcast_plasma_grid(
        temperature, plasma_grid, by_density=density is not None
    )
    temperatures = np.asarray(temperature.data, dtype=float).reshape(-1) * u.K
    wavelengths = wavelength_grid * u.AA
    densities = _densities_by_plasma_point(density_broadcast, temperature_broadcast)
    ion_kwargs = _fiasco_ion_kwargs(abundance, hdf5_dbase_root)

    from fiasco.util.exceptions import MissingDatasetException, MissingIonError  # noqa: PLC0415

    # Once for all elements; each Element would otherwise compute it again.
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
        msg = (
            "fiasco loaded no ion, so the continuum would be all zero; check its HDF5 database "
            "(hdf5_dbase_root), and that minimum_abundance does not exclude every element"
        )
        raise RuntimeError(msg)
    free_free = sum(_map_ion_groups(_fiasco_component, ions, workers, "free_free", wavelengths))
    free_bound = sum(_map_ion_groups(_fiasco_component, ions, workers, "free_bound", wavelengths))
    two_photon = np.zeros((len(densities), temperatures.size, wavelength_grid.size))
    for plasma_index, electron_density in enumerate(densities):
        # couple_density_to_temperature repeats the result along fiasco's density axis; any column is it.
        parts = _map_ion_groups(
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
        temperature_broadcast=temperature_broadcast,
        plasma_coords=plasma_coords,
        wavelength_grid=wavelength_grid,
        attrs={
            "fiasco_version": fiasco.__version__,
            # Where fiasco's own line list reads the database version.
            **({"chianti_version": str(ions[0]._elvlc.version)} if hasattr(ions[0], "_elvlc") else {}),
            "minimum_abundance": minimum_abundance,
        },
    )
    abundance = getattr(ions[0], "_dset_names", {}).get("abundance", abundance)
    add_history(continuum, {**call_inputs, "abundance": abundance}, create_fiasco_continuum)
    return continuum


def _validate_continuum_inputs(temperature, density, pressure, wavelength_grid):
    temperature, plasma_grid = _validate_plasma_inputs(temperature, density, pressure)
    wavelength_grid = np.asarray(wavelength_grid.to_value(u.AA), dtype=float).reshape(-1)
    if wavelength_grid.size < 2:
        msg = "wavelength_grid must hold at least two wavelengths"
        raise ValueError(msg)
    _require_increasing_axis(wavelength_grid, "wavelength_grid", positive=True)
    return temperature, plasma_grid, wavelength_grid


def _densities_by_plasma_point(
    density_broadcast: xr.DataArray, temperature_broadcast: xr.DataArray
) -> list[np.ndarray]:
    """
    One density array along ``logT`` per plasma point.
    """
    plasma_dim = next(dim for dim in temperature_broadcast.dims if dim != "logT")
    return [np.asarray(row.data, dtype=float) for row in density_broadcast.transpose(plasma_dim, "logT")]


def _chianti_component(
    method, result_name: str, wavelength_grid: np.ndarray, number_of_temperatures: int, **kwargs
) -> np.ndarray | None:
    """
    One ChiantiPy continuum component as ``(temperature, wavelength)``, or `None` if the
    ion lacks it; ``method`` stores its result on its object under ``result_name``.
    """
    try:
        method(wavelength_grid, **kwargs)
    except AttributeError:
        # ChiantiPy's failure for an ion without this component's levels, as some H- and He-like ions for two-photon.
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
    free_free, free_bound, two_photon, *, temperature, temperature_broadcast, plasma_coords, wavelength_grid, attrs
) -> xr.Dataset:
    plasma_dim = next(dim for dim in temperature_broadcast.dims if dim != "logT")
    normalization = _NORMALIZATION
    coords = {"logT": ("logT", np.log10(temperature.data), {"units": str(u.dex(u.K))}), **plasma_coords}
    density_independent = [
        xr.DataArray(values / normalization, dims=("logT", "wavelength_bin")).expand_dims(
            {plasma_dim: plasma_coords[plasma_dim]}
        )
        for values in (free_free, free_bound)
    ]
    components = [
        *density_independent,
        xr.DataArray(two_photon / normalization, dims=(plasma_dim, "logT", "wavelength_bin")),
    ]
    spectral_response = xr.concat(
        [component.transpose(*temperature_broadcast.dims, "wavelength_bin") for component in components],
        dim=xr.DataArray(list(_COMPONENTS), dims="line"),
    ).assign_coords(coords)
    spectral_response.attrs["units"] = str(normalization * u.erg * u.cm**3 / u.s / u.sr / u.AA)
    spectral_response[plasma_dim].attrs["units"] = _plasma_axis_units(plasma_dim)
    continuum = xr.Dataset({"spectral_response": spectral_response}, attrs={"normalization": normalization, **attrs})
    return continuum.assign_coords(
        **_wavelength_bin_coords(wavelength_grid, _centers_to_edges(wavelength_grid)),
        line_wavelength=("line", np.full(len(_COMPONENTS), np.nan), {"units": str(u.AA)}),
    )
