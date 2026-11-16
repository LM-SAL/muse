"""
The plasma grid and the ChiantiPy and fiasco backends of the line lists and continua.
"""

import os
import warnings
from types import ModuleType
from pathlib import Path
from importlib import reload

import numpy as np
import xarray as xr

import astropy.units as u
from astropy.utils import minversion

__all__ = []

# Low enough to keep every element of a CHIANTI abundance file.
_MINIMUM_ABUNDANCE = 1e-21


def _validate_plasma_inputs(
    temperature: u.Quantity, density: u.Quantity | None, pressure: u.Quantity | None
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Unitless temperature in K along ``logT``, and density in cm^-3 along ``density`` or
    pressure in K cm^-3 along ``pressure``.
    """
    if density is None and pressure is None:
        msg = "Specify density or pressure"
        raise ValueError(msg)
    if density is not None and pressure is not None:
        msg = "density and pressure are mutually exclusive"
        raise ValueError(msg)

    temperature = _validate_grid(temperature, "temperature", u.K, "logT")
    if density is not None:
        return temperature, _validate_grid(density, "density", u.cm**-3, "density")
    return temperature, _validate_grid(pressure, "pressure", u.K / u.cm**3, "pressure")


def _validate_grid(values: u.Quantity, name: str, unit: u.UnitBase, dimension: str) -> xr.DataArray:
    if not isinstance(values, u.Quantity) or values.ndim != 1:
        msg = f"{name} must be a one-dimensional astropy.units.Quantity convertible to {unit}"
        raise TypeError(msg)
    if values.size == 0:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    try:
        data = values.to_value(unit)
    except u.UnitConversionError as exc:
        msg = f"{name} units must be convertible to {unit}"
        raise ValueError(msg) from exc
    if not np.all(np.isfinite(data)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if np.any(data <= 0):
        msg = f"{name} must contain only positive values"
        raise ValueError(msg)
    return xr.DataArray(data, dims=dimension)


def _plasma_axis_units(plasma_dim: str) -> str:
    return str(u.dex(u.cm**-3) if plasma_dim == "logD" else u.K / u.cm**3)


def _broadcast_plasma_grid(
    temperature: xr.DataArray, plasma_grid: xr.DataArray, *, by_density: bool
) -> tuple[xr.DataArray, xr.DataArray, dict[str, np.ndarray]]:
    """
    Temperature and density on the full grid, and the plasma coordinate: ``logD``
    (``log10(density)``) for a density grid, else the pressure, with density ``pressure
    / temperature``.
    """
    if by_density:
        plasma_grid = plasma_grid.rename({plasma_grid.dims[0]: "logD"})
        temperature_broadcast, density_broadcast = xr.broadcast(temperature, plasma_grid)
        return temperature_broadcast, density_broadcast, {"logD": np.log10(plasma_grid.data)}
    density_broadcast = plasma_grid / temperature
    return temperature.broadcast_like(density_broadcast), density_broadcast, {plasma_grid.dims[0]: plasma_grid.data}


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
        # Without a chiantirc, ChiantiPy calls os.path.isfile(False) at import: a RuntimeWarning on Python >= 3.14.
        warnings.simplefilter("ignore", RuntimeWarning)
        # ChiantiPy.core imports its optional ipyparallel implementation.
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


def _import_fiasco():
    try:
        import fiasco  # noqa: PLC0415
    except ImportError:
        msg = "fiasco is required for this function, install it with `pip install muse[fiasco]`"
        raise ImportError(msg) from None
    # 0.9.0 added IonCollection.build_line_list; with an older one the free-free would silently be 4 pi too low.
    if not minversion(fiasco, "0.9.0"):
        msg = f"fiasco 0.9.0 or later is required, found {fiasco.__version__}"
        raise ImportError(msg)
    return fiasco


def _fiasco_ion_kwargs(abundance: str | None, hdf5_dbase_root: str | Path | None) -> dict:
    options = {"abundance": abundance, "hdf5_dbase_root": hdf5_dbase_root}
    return {key: value for key, value in options.items() if value is not None}


def _map_ion_groups(function, ions: list, workers: int, *args, single_ions: bool = False, **kwargs) -> list:
    """
    Call ``function(group, *args, **kwargs)`` on groups of ``ions`` in ``workers``
    processes; return one result per group, not in the order of ``ions``.

    One worker or one ion makes one group of all ``ions`` and starts no process.
    Otherwise each worker gets every n-th ion, as the ions of one element are neighbors
    and differ widely in cost. With ``single_ions`` each ion is its own group, those
    with the most levels first, for work whose cost grows with the levels, as a line
    list's does.
    """
    if workers < 1 and workers != -1:
        msg = f"workers must be at least 1, or -1 for every CPU this process may use, got {workers}"
        raise ValueError(msg)
    if workers == -1:
        # Not os.cpu_count: one worker per core of the machine would exhaust a small batch job's memory.
        workers = os.process_cpu_count() or 1
    if workers == 1 or len(ions) <= 1:
        return [function(list(ions), *args, **kwargs)]
    try:
        from joblib import Parallel, delayed  # noqa: PLC0415
    except ImportError:
        msg = "joblib is required for workers != 1, install it with `pip install muse[fiasco]`"
        raise ImportError(msg) from None
    if single_ions:
        # An ion without data has no n_levels.
        groups = [[ion] for ion in sorted(ions, key=lambda ion: getattr(ion, "n_levels", 0), reverse=True)]
    else:
        number_of_groups = min(workers, len(ions))
        groups = [list(ions[first::number_of_groups]) for first in range(number_of_groups)]
    with warnings.catch_warnings():
        # loky warns when it restarts a worker whose memory grew; as an error, in loky's thread, it hangs the run.
        warnings.filterwarnings("ignore", "A worker stopped while some jobs were given to the executor", UserWarning)
        # batch_size=1: joblib would otherwise bundle short tasks, and a bundle is a block of neighbors again.
        return Parallel(n_jobs=workers, batch_size=1)(delayed(function)(group, *args, **kwargs) for group in groups)
