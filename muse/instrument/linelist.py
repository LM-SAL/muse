"""
CHIANTI line lists with contribution functions (GOFNT).
"""

import os
import re
import json
import warnings
from types import ModuleType
from numbers import Real
from pathlib import Path
from importlib import reload

import numpy as np
import xarray as xr

import astropy.units as u

from muse.utils.utils import add_history

__all__ = ["apply_line_boosts", "create_chianti_line_list", "create_fiasco_line_list", "rank_line_list", "select_lines"]


def create_chianti_line_list(
    temperature: xr.DataArray,
    density: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    abundance: str | None = None,
    wavelength_range: u.Quantity | None = None,
    minimum_abundance: float | None = None,
    element_list: list[str] | None = None,
    ion_list: list[str] | None = None,
) -> xr.Dataset:
    """
    Generate a line list with contribution functions using ChiantiPy.

    Parameters
    ----------
    temperature : `xarray.DataArray`
        Temperature array with an `astropy.units.Quantity` payload convertible
        to K and a ``logT`` dimension.
    density : `xarray.DataArray`, optional
        Electron density array with an `astropy.units.Quantity` payload
        convertible to cm^-3. Mutually exclusive with ``pressure``. The output
        carries this grid on a ``logD`` dimension whose coordinate is
        ``log10(density)``.
    pressure : `xarray.DataArray`, optional
        Electron pressure array with an `astropy.units.Quantity` payload
        convertible to K cm^-3. Mutually exclusive with ``density``.
    abundance : `str`, optional
        CHIANTI abundance name, e.g. ``"sun_coronal_2021_chianti"``. If not
        given, ChiantiPy's configured default applies
        (``sun_photospheric_2021_asplund`` unless a ``chiantirc`` file overrides
        it); the resolved name is recorded in the ``abundance`` attribute.
    wavelength_range : `astropy.units.Quantity`
        Two-element wavelength range convertible to Angstroms.
    minimum_abundance : `float`, optional
        Finite positive minimum elemental abundance to keep. Mutually exclusive
        with ``element_list`` and ``ion_list``.
    element_list : `list` of `str`, optional
        CHIANTI element symbols to include, such as ``"fe"`` and ``"o"``.
        Mutually exclusive with ``ion_list`` and ``minimum_abundance``.
    ion_list : `list` of `str`, optional
        CHIANTI ion names to include, such as ``"fe_9"``. Mutually exclusive
        with ``element_list`` and ``minimum_abundance``.

    Returns
    -------
    `xarray.Dataset`
        Line list with contribution functions and per-transition metadata.

    Notes
    -----
    The ``XUVTOP`` environment variable must point to a local CHIANTI database.
    ChiantiPy's ``gui`` default is forced off for headless batch jobs.
    """
    temperature, plasma_grid, wavelength_range = _validate_line_list_inputs(
        temperature, density, pressure, wavelength_range
    )
    minimum_abundance, element_list, ion_list = _validate_species_selection(minimum_abundance, element_list, ion_list)

    chiantipy_version, ch = _initialize_chianti()

    temperature_bc, density_bc, extra_coords = _broadcast_plasma_grid(
        temperature, plasma_grid, by_density=density is not None
    )
    temperature_flat = temperature_bc.data.reshape(-1)
    density_flat = density_bc.data.reshape(-1)

    chianti_kwargs = {
        "em": 1.0,
        "abundance": abundance,
        "allLines": True,
        "keepIons": True,
        "minAbund": minimum_abundance,
        "ionList": ion_list,
        "elementList": element_list,
    }
    # ChiantiPy cannot descale collision strengths for 1-element arrays; hand it scalars.
    chianti_temperature = temperature_flat.item() if temperature_flat.size == 1 else temperature_flat
    chianti_density = density_flat.item() if density_flat.size == 1 else density_flat
    bunch = ch.bunch(chianti_temperature, chianti_density, wavelength_range, **chianti_kwargs)
    abundance = getattr(bunch, "AbundanceName", abundance)
    if abundance is not None:
        abundance = Path(abundance).stem

    line_list = _chianti_bunch_to_dataset(
        bunch,
        temperature=temperature,
        temperature_bc=temperature_bc,
        extra_coords=extra_coords,
        wavelength_range=wavelength_range,
        chiantipy_version=chiantipy_version,
    )
    if line_list.sizes["trans_index"] == 0:
        msg = "CHIANTI returned no lines; check wavelength_range and the species selection"
        raise ValueError(msg)
    add_history(line_list, locals(), create_chianti_line_list)
    return line_list


def _initialize_chianti() -> tuple[str, ModuleType]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            import ChiantiPy  # noqa: PLC0415
        except ImportError:
            msg = "ChiantiPy is required for this function, install it with `pip install muse[chianti]`"
            raise ImportError(msg) from None

    xuvtop = os.environ.get("XUVTOP")
    if xuvtop is None:
        msg = (
            "The XUVTOP environment variable is not set; ChiantiPy cannot locate the CHIANTI database. "
            "Point it at a local copy of the database, e.g. `export XUVTOP=/path/to/chianti/dbase` "
            "(available from https://www.chiantidatabase.org)."
        )
        raise OSError(msg)

    with warnings.catch_warnings():
        # Without a chiantirc file, ChiantiPy evaluates os.path.isfile(False)
        # at import, which raises a RuntimeWarning on Python >= 3.14.
        warnings.simplefilter("ignore", RuntimeWarning)
        # ChiantiPy imports its optional ipyparallel implementation from core.
        warnings.filterwarnings(
            "ignore",
            message=r"ipyparallel not found\.",
            category=UserWarning,
            module=r"ChiantiPy\.core\.IpyMspectrum",
        )
        import ChiantiPy.tools.data as chdata  # noqa: PLC0415

        if not hasattr(chdata, "Defaults") or getattr(chdata, "Xuvtop", None) != xuvtop:
            chdata = reload(chdata)
        import ChiantiPy.core as ch  # noqa: PLC0415

    if not hasattr(chdata, "Defaults"):
        msg = f"ChiantiPy could not initialize the CHIANTI database at {xuvtop}"
        raise OSError(msg)

    chdata.Defaults["gui"] = False
    return ChiantiPy.__version__, ch


def _chianti_bunch_to_dataset(
    bunch,
    *,
    temperature: xr.DataArray,
    temperature_bc: xr.DataArray,
    extra_coords: dict[str, np.ndarray],
    wavelength_range: tuple[float, float],
    chiantipy_version: str,
) -> xr.Dataset:
    if getattr(bunch, "Intensity", None) is None or len(bunch.Intensity["wvl"]) == 0:
        msg = "CHIANTI returned no lines; check wavelength_range and the species selection"
        raise ValueError(msg)

    import ChiantiPy.tools.io as chio  # noqa: PLC0415

    ion_names = bunch.Intensity["ionS"]
    name_info = {
        ion: {"spectroscopic": bunch.IonInstances[ion].Spectroscopic, "Z": bunch.IonInstances[ion].Z}
        for ion in np.unique(ion_names)
    }
    per_transition = {
        "ion_name": ion_names,
        "wavelength": bunch.Intensity["wvl"],
        "lower_level_label": bunch.Intensity["pretty1"],
        "upper_level_label": bunch.Intensity["pretty2"],
        "lower_level_index": bunch.Intensity["lvl1"],
        "upper_level_index": bunch.Intensity["lvl2"],
        "spectroscopic_name": np.array([name_info[ion]["spectroscopic"] for ion in ion_names]),
        "atomic_number": np.array([name_info[ion]["Z"] for ion in ion_names]),
        "observed": bunch.Intensity["obs"] == "Y",
    }
    gofnt_values = bunch.Intensity["intensity"].reshape((*temperature_bc.data.shape, -1))
    return _line_list_dataset(
        per_transition,
        gofnt_values,
        temperature=temperature,
        temperature_bc=temperature_bc,
        extra_coords=extra_coords,
        wavelength_range=wavelength_range,
        attrs={"Chiantipy": chiantipy_version, "Chianti": chio.versionRead()},
    )


def _broadcast_plasma_grid(
    temperature: xr.DataArray, plasma_grid: xr.DataArray, *, by_density: bool
) -> tuple[xr.DataArray, xr.DataArray, dict[str, np.ndarray]]:
    """
    Temperature and density on the full grid, and the coordinate of the plasma axis.

    A density grid becomes a ``logD`` axis holding ``log10(density)``; a pressure grid
    keeps its dimension and the density at each temperature is ``pressure /
    temperature``.
    """
    if by_density:
        plasma_grid = plasma_grid.rename({plasma_grid.dims[0]: "logD"})
        temperature_bc, density_bc = xr.broadcast(temperature, plasma_grid)
        return temperature_bc, density_bc, {"logD": np.log10(plasma_grid.data)}
    density_bc = plasma_grid / temperature
    return temperature.broadcast_like(density_bc), density_bc, {plasma_grid.dims[0]: plasma_grid.data}


def _line_list_dataset(
    per_transition: dict[str, np.ndarray],
    gofnt_values: np.ndarray,
    *,
    temperature: xr.DataArray,
    temperature_bc: xr.DataArray,
    extra_coords: dict[str, np.ndarray],
    wavelength_range: tuple[float, float],
    attrs: dict,
) -> xr.Dataset:
    """
    The line-list layout both backends return, restricted to ``wavelength_range``.
    """
    line_list = xr.Dataset({name: ("trans_index", values) for name, values in per_transition.items()})
    line_list["gofnt"] = xr.DataArray(
        gofnt_values,
        dims=(*temperature_bc.dims, "trans_index"),
        coords={"logT": np.log10(temperature), **extra_coords},
    )
    line_list["logT_peak"] = np.log10(temperature[{"logT": line_list.gofnt.argmax(dim="logT")}])
    line_list["full_name"] = (
        line_list.spectroscopic_name.astype(object) + " " + line_list.wavelength.astype(str).astype(object)
    )

    line_list.attrs.update(attrs)
    line_list.wavelength.attrs["units"] = str(u.AA)
    line_list.gofnt.attrs["units"] = "erg cm3 / (s sr)"
    line_list.logT.attrs["units"] = str(u.dex(u.K))
    if "logD" in line_list.coords:
        line_list.logD.attrs["units"] = str(u.dex(u.cm**-3))

    in_range = (line_list.wavelength >= wavelength_range[0]) & (line_list.wavelength <= wavelength_range[1])
    return line_list.isel(trans_index=in_range)


def _validate_line_list_inputs(
    temperature: xr.DataArray,
    density: xr.DataArray | None,
    pressure: xr.DataArray | None,
    wavelength_range: u.Quantity | None,
) -> tuple[xr.DataArray, xr.DataArray, tuple[float, float]]:
    if density is None and pressure is None:
        msg = "Specify density or pressure"
        raise ValueError(msg)
    if density is not None and pressure is not None:
        msg = "density and pressure are mutually exclusive"
        raise ValueError(msg)

    temperature = _validate_positive_data_array(temperature, "temperature", u.K, dimension="logT")
    if density is not None:
        name = "density"
        plasma_grid = _validate_positive_data_array(density, name, u.cm**-3)
    else:
        name = "pressure"
        plasma_grid = _validate_positive_data_array(pressure, name, u.K / u.cm**3)
    if plasma_grid.dims[0] in ("logT", "trans_index"):
        msg = f"{name} dimension must not be named {plasma_grid.dims[0]!r}"
        raise ValueError(msg)

    return temperature, plasma_grid, _validate_wavelength_range(wavelength_range)


def _validate_wavelength_range(wavelength_range: u.Quantity | None) -> tuple[float, float]:
    if not isinstance(wavelength_range, u.Quantity):
        msg = "wavelength_range must be an astropy.units.Quantity convertible to Angstrom"
        raise TypeError(msg)
    try:
        values = np.asarray(wavelength_range.to_value(u.AA), dtype=float)
    except u.UnitConversionError as exc:
        msg = "wavelength_range units must be convertible to Angstrom"
        raise ValueError(msg) from exc
    if values.shape != (2,):
        msg = "wavelength_range must contain exactly two values"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "wavelength_range must contain only finite values"
        raise ValueError(msg)
    lower, upper = values
    if lower >= upper:
        msg = "wavelength_range must be in increasing order"
        raise ValueError(msg)
    return float(lower), float(upper)


def _validate_positive_data_array(
    values: xr.DataArray, name: str, unit: u.UnitBase, *, dimension: str | None = None
) -> xr.DataArray:
    if not isinstance(values, xr.DataArray):
        msg = f"{name} must be an xarray.DataArray"
        raise TypeError(msg)
    expected_dims = (dimension,) if dimension is not None else None
    if (expected_dims is not None and values.dims != expected_dims) or (expected_dims is None and values.ndim != 1):
        qualifier = f"one-dimensional {dimension}" if dimension is not None else "one-dimensional"
        msg = f"{name} must be a {qualifier} array"
        raise ValueError(msg)
    if values.size == 0:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if not isinstance(values.data, u.Quantity):
        msg = f"{name} data must be an astropy.units.Quantity convertible to {unit}"
        raise TypeError(msg)
    try:
        data = values.data.to_value(unit)
    except u.UnitConversionError as exc:
        msg = f"{name} units must be convertible to {unit}"
        raise ValueError(msg) from exc
    if not np.all(np.isfinite(data)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if np.any(data <= 0):
        msg = f"{name} must contain only positive values"
        raise ValueError(msg)
    return values.copy(data=data)


def _normalize_species_names(values: list[str] | tuple[str, ...], name: str, pattern: str) -> list[str]:
    if isinstance(values, str) or not isinstance(values, list | tuple):
        msg = f"{name} must be a list of strings"
        raise TypeError(msg)
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        msg = f"{name} must contain unique, non-empty strings"
        raise ValueError(msg)
    normalized = [value.strip().lower() for value in values]
    if len(set(normalized)) != len(normalized):
        msg = f"{name} must contain unique, non-empty strings"
        raise ValueError(msg)
    normalized.sort()
    invalid = [value for value in normalized if re.fullmatch(pattern, value) is None]
    if invalid:
        msg = f"invalid {name}: {', '.join(invalid)}"
        raise ValueError(msg)
    return normalized


def _validate_species_selection(
    minimum_abundance: float | None, element_list: list[str] | None, ion_list: list[str] | None
) -> tuple[float | None, list[str] | None, list[str] | None]:
    """
    Require exactly one species selection and normalize it.

    Returns the ``(minimum_abundance, element_list, ion_list)`` triple with the two
    unselected entries set to `None`.
    """
    given = [
        name
        for name, value in (
            ("minimum_abundance", minimum_abundance),
            ("element_list", element_list),
            ("ion_list", ion_list),
        )
        if value is not None
    ]
    if not given:
        msg = "Specify minimum_abundance, element_list, or ion_list"
        raise ValueError(msg)
    if len(given) > 1:
        msg = f"{', '.join(given)} are mutually exclusive; give only one"
        raise ValueError(msg)
    if element_list is not None:
        return None, _normalize_species_names(element_list, "element_list", r"[a-z]{1,2}"), None
    if ion_list is not None:
        return None, None, _normalize_species_names(ion_list, "ion_list", r"[a-z]{1,2}_[1-9][0-9]*d?")
    if isinstance(minimum_abundance, bool) or not isinstance(minimum_abundance, Real):
        msg = "minimum_abundance must be a real number"
        raise TypeError(msg)
    minimum_abundance = float(minimum_abundance)
    if not np.isfinite(minimum_abundance) or minimum_abundance <= 0:
        msg = "minimum_abundance must be finite and positive"
        raise ValueError(msg)
    return minimum_abundance, None, None


def create_fiasco_line_list(
    temperature: xr.DataArray,
    density: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    abundance: str | None = None,
    wavelength_range: u.Quantity | None = None,
    minimum_abundance: float | None = None,
    ion_list: list[str] | None = None,
    *,
    workers: int = 1,
    hdf5_dbase_root: str | Path | None = None,
) -> xr.Dataset:
    """
    Generate a line list with contribution functions using fiasco.

    The counterpart of `create_chianti_line_list` built on
    ``fiasco.IonCollection.build_line_list``, returning the same layout, so the two are
    interchangeable downstream.

    Parameters
    ----------
    temperature : `xarray.DataArray`
        Temperature array with an `astropy.units.Quantity` payload convertible
        to K and a ``logT`` dimension.
    density : `xarray.DataArray`, optional
        Electron density array with an `astropy.units.Quantity` payload
        convertible to cm^-3. Mutually exclusive with ``pressure``. The output
        carries this grid on a ``logD`` dimension whose coordinate is
        ``log10(density)``.
    pressure : `xarray.DataArray`, optional
        Electron pressure array with an `astropy.units.Quantity` payload
        convertible to K cm^-3. Mutually exclusive with ``density``.
    abundance : `str`, optional
        CHIANTI abundance name, e.g. ``"sun_coronal_2021_chianti"``; fiasco's
        default applies when omitted. The resolved name is recorded in the
        ``abundance`` attribute.
    wavelength_range : `astropy.units.Quantity`
        Two-element wavelength range convertible to Angstroms.
    minimum_abundance : `float`, optional
        Ions of elements with an abundance below this are skipped; by default
        every element is kept. Unlike `create_chianti_line_list` this combines
        with ``ion_list``.
    ion_list : `list` of `str`, optional
        fiasco ion names to include, such as ``"fe_9"``; by default every ion in
        the database.
    workers : `int`, optional
        Processes the ions are split over, by default 1; ``-1`` uses every core.
        Each worker holds the level populations of its ions on the whole grid.
    hdf5_dbase_root : `str` or `pathlib.Path`, optional
        fiasco's HDF5 database; by default its configured one.

    Returns
    -------
    `xarray.Dataset`
        Line list with contribution functions and per-transition metadata.

    Notes
    -----
    Needs a fiasco newer than 0.8.3 (``IonCollection.build_line_list``, merged
    2026-10-23) and joblib when ``workers`` is not 1.
    """
    call_inputs = dict(locals())
    temperature, plasma_grid, wavelength_range = _validate_line_list_inputs(
        temperature, density, pressure, wavelength_range
    )
    if ion_list is not None:
        ion_list = _normalize_species_names(ion_list, "ion_list", r"[a-z]{1,2}_[1-9][0-9]*d?")
    minimum_abundance = 0.0 if minimum_abundance is None else float(minimum_abundance)
    if not np.isfinite(minimum_abundance) or minimum_abundance < 0:
        msg = "minimum_abundance must be finite and non-negative"
        raise ValueError(msg)
    fiasco = _import_fiasco()

    temperature_bc, density_bc, extra_coords = _broadcast_plasma_grid(
        temperature, plasma_grid, by_density=density is not None
    )
    temperature_flat = temperature_bc.data.reshape(-1) * u.K
    density_flat = density_bc.data.reshape(-1) * u.cm**-3
    ion_kwargs = {
        key: value
        for key, value in (("abundance", abundance), ("hdf5_dbase_root", hdf5_dbase_root))
        if value is not None
    }
    # The proton-to-electron ratio depends only on the temperature grid and the datasets.
    proton_electron_ratio = fiasco.proton_electron_ratio(temperature_flat, **ion_kwargs)
    ions = [
        fiasco.Ion(name, temperature_flat, proton_electron_ratio=proton_electron_ratio, **ion_kwargs)
        for name in (ion_list or fiasco.list_ions(hdf5_dbase_root=hdf5_dbase_root))
    ]
    tables = [
        table
        for table in _over_ion_chunks(
            _fiasco_line_table,
            ions,
            workers,
            density_flat,
            minimum_abundance=minimum_abundance,
            wavelength_range=u.Quantity(wavelength_range, u.AA),
        )
        if table is not None
    ]
    if not tables:
        msg = "fiasco found no lines; check wavelength_range and the ion selection"
        raise ValueError(msg)
    from astropy.table import vstack  # noqa: PLC0415

    table = vstack(tables)
    roman_to_name = {ion.ion_name_roman: ion._ion_name for ion in ions}
    spectroscopic_name = np.asarray(table["ion name"]).astype(str)
    per_transition = {
        "ion_name": np.array([roman_to_name[name] for name in spectroscopic_name]),
        "wavelength": table["wavelength"].to_value(u.AA),
        "lower_level_label": np.asarray(table["lower label"]).astype(str),
        "upper_level_label": np.asarray(table["upper label"]).astype(str),
        "lower_level_index": np.asarray(table["lower level"]),
        "upper_level_index": np.asarray(table["upper level"]),
        "spectroscopic_name": spectroscopic_name,
        "atomic_number": np.asarray(table["atomic number"]),
        "observed": np.asarray(table["is observed"], dtype=bool),
    }
    # (trans_index, grid point, 1) -> (*grid dims, trans_index), per steradian
    gofnt = (table["contribution function"][:, :, 0] / (4 * np.pi * u.sr)).to_value(u.erg * u.cm**3 / (u.s * u.sr))
    gofnt_values = np.moveaxis(gofnt, 0, -1).reshape((*temperature_bc.data.shape, -1))
    line_list = _line_list_dataset(
        per_transition,
        gofnt_values,
        temperature=temperature,
        temperature_bc=temperature_bc,
        extra_coords=extra_coords,
        wavelength_range=wavelength_range,
        attrs={"fiasco": table.meta["fiasco_version"], "Chianti": table.meta["chianti_version"]},
    )
    if line_list.sizes["trans_index"] == 0:
        msg = "fiasco found no lines; check wavelength_range and the ion selection"
        raise ValueError(msg)
    abundance = getattr(ions[0], "_dset_names", {}).get("abundance", abundance)
    add_history(line_list, {**call_inputs, "abundance": abundance}, create_fiasco_line_list)
    return line_list


def _import_fiasco(*, line_lists: bool = True):
    try:
        import fiasco  # noqa: PLC0415
    except ImportError:
        msg = "fiasco is required for this function, install it with `pip install muse[fiasco]`"
        raise ImportError(msg) from None
    if line_lists and not hasattr(fiasco.IonCollection, "build_line_list"):
        msg = "fiasco.IonCollection.build_line_list is required; install a fiasco newer than 0.8.3"
        raise ImportError(msg)
    return fiasco


def _fiasco_line_table(ions: list, density: u.Quantity, **kwargs):
    """
    The line table of one chunk of ions, or `None` when fiasco finds no lines in it.
    """
    from fiasco import IonCollection  # noqa: PLC0415

    try:
        return IonCollection(*ions).build_line_list(density, couple_density_to_temperature=True, **kwargs)
    except ValueError as exc:
        if not str(exc).startswith("No lines found"):
            raise
        return None


def _over_ion_chunks(function, ions: list, workers: int, *args, **kwargs) -> list:
    """
    ``function(chunk, *args, **kwargs)`` for every chunk of ``ions``, in parallel when
    ``workers`` is not 1.
    """
    if workers == -1:
        workers = os.cpu_count() or 1
    if workers == 1 or len(ions) <= 1:
        return [function(list(ions), *args, **kwargs)]
    try:
        from joblib import Parallel, delayed  # noqa: PLC0415
    except ImportError:
        msg = "joblib is required for workers != 1, install it with `pip install muse[fiasco]`"
        raise ImportError(msg) from None
    chunks = [
        list(chunk) for chunk in np.array_split(np.array(ions, dtype=object), min(workers, len(ions))) if len(chunk)
    ]
    return Parallel(n_jobs=workers)(delayed(function)(chunk, *args, **kwargs) for chunk in chunks)


def apply_line_boosts(line_list: xr.Dataset, boosts: dict[tuple[str, int, int], float]) -> xr.Dataset:
    """
    Multiply the contribution functions of chosen transitions by a factor.

    SolarSoft's ``aia_bp_make_emiss`` multiplies the He II 303.786 A line by 20 to stand
    in for its optical depth, and every SolarSoft AIA temperature response carries that
    factor; a response meant to be compared with them needs the same.

    Parameters
    ----------
    line_list : `xarray.Dataset`
        A line list from `create_chianti_line_list` or `create_fiasco_line_list`.
    boosts : `dict`
        ``{(ion_name, lower_level_index, upper_level_index): factor}``, the levels numbered
        as in the line list (CHIANTI's ``elvlc`` order), e.g. ``{("he_2", 1, 3): 20}``.

    Returns
    -------
    `xarray.Dataset`
        A copy with ``gofnt`` scaled; the input is untouched.

    Raises
    ------
    ValueError
        If a key matches no transition or more than one.
    """
    factor = xr.ones_like(line_list.wavelength, dtype=float)
    for (ion_name, lower, upper), value in boosts.items():
        matches = (
            (line_list.ion_name == ion_name)
            & (line_list.lower_level_index == lower)
            & (line_list.upper_level_index == upper)
        )
        if int(matches.sum()) != 1:
            msg = f"{int(matches.sum())} transitions match {ion_name} {lower}-{upper}, expected exactly one"
            raise ValueError(msg)
        factor = factor.where(~matches, value)
    boosted = line_list.copy()
    boosted["gofnt"] = (line_list.gofnt * factor).assign_attrs(line_list.gofnt.attrs)
    boosted.attrs["line_boosts"] = json.dumps(
        {f"{ion} {lower}-{upper}": value for (ion, lower, upper), value in boosts.items()}
    )
    return boosted


def rank_line_list(
    line_list: xr.Dataset, effective_area: xr.DataArray | None = None, *, statistic: str = "max"
) -> xr.Dataset:
    """
    Sort the transitions by brightness, brightest first.

    Parameters
    ----------
    line_list : `xarray.Dataset`
        A line list from `create_chianti_line_list` or `create_fiasco_line_list`.
    effective_area : `xarray.DataArray`, optional
        A curve with a ``wavelength`` coordinate in Angstrom; each transition is weighted
        by the area at its wavelength, zero outside the curve.
    statistic : {"max", "sum"}, optional
        How ``gofnt`` is collapsed over temperature and the plasma axis before weighting:
        its peak (default, as the response baseline ranks contaminants) or its sum.

    Returns
    -------
    `xarray.Dataset`
        The line list reordered along ``trans_index``, with a ``brightness`` variable
        holding the ranking proxy.
    """
    if statistic not in ("max", "sum"):
        msg = f"statistic must be 'max' or 'sum', got {statistic!r}"
        raise ValueError(msg)
    brightness = getattr(line_list.gofnt, statistic)(dim=[dim for dim in line_list.gofnt.dims if dim != "trans_index"])
    if effective_area is not None:
        area = effective_area.interp(wavelength=line_list.wavelength).fillna(0).drop_vars("wavelength", errors="ignore")
        brightness = brightness * area
    order = np.argsort(-np.asarray(brightness), kind="stable")
    ranked = line_list.isel(trans_index=order)
    ranked["brightness"] = ("trans_index", np.asarray(brightness)[order])
    return ranked


def select_lines(line_list: xr.Dataset, lines) -> xr.Dataset:
    """
    Keep only the chosen transitions of a line list.

    Parameters
    ----------
    line_list : `xarray.Dataset`
        A line list from `create_chianti_line_list` or `create_fiasco_line_list`.
    lines : iterable of `tuple`
        ``(ion_name, lower_level_index, upper_level_index)`` per transition, the levels
        numbered as in the line list, e.g. ``[("he_2", 1, 3), ("he_2", 1, 4)]`` for the
        He II doublet at 303.78 A.

    Returns
    -------
    `xarray.Dataset`
        The line list reduced to those transitions, in the list's order.

    Raises
    ------
    ValueError
        If a key matches no transition or more than one.
    """
    keep = xr.zeros_like(line_list.wavelength, dtype=bool)
    for ion_name, lower, upper in lines:
        matches = (
            (line_list.ion_name == ion_name)
            & (line_list.lower_level_index == lower)
            & (line_list.upper_level_index == upper)
        )
        if int(matches.sum()) != 1:
            msg = f"{int(matches.sum())} transitions match {ion_name} {lower}-{upper}, expected exactly one"
            raise ValueError(msg)
        keep = keep | matches
    selected = line_list.isel(trans_index=keep.values)
    selected.attrs["line_selection"] = json.dumps([f"{ion} {lower}-{upper}" for ion, lower, upper in lines])
    return selected
