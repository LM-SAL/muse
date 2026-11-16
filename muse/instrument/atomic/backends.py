"""
Shared by the line lists and the continua: the plasma grid they are computed on, and the
ChiantiPy and fiasco backends that compute them.
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


def _validate_plasma_inputs(
    temperature: u.Quantity, density: xr.DataArray | None, pressure: xr.DataArray | None
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    The temperature in K along ``logT`` and the plasma grid, a density in cm^-3 or a
    pressure in K cm^-3, both without units.
    """
    if density is None and pressure is None:
        msg = "Specify density or pressure"
        raise ValueError(msg)
    if density is not None and pressure is not None:
        msg = "density and pressure are mutually exclusive"
        raise ValueError(msg)

    if not isinstance(temperature, u.Quantity) or temperature.ndim != 1:
        msg = "temperature must be a one-dimensional astropy.units.Quantity convertible to K"
        raise TypeError(msg)
    temperature = _validate_positive_data_array(xr.DataArray(temperature, dims="logT"), "temperature", u.K)
    if density is not None:
        name = "density"
        plasma_grid = _validate_positive_data_array(density, name, u.cm**-3)
    else:
        name = "pressure"
        plasma_grid = _validate_positive_data_array(pressure, name, u.K / u.cm**3)
    if plasma_grid.dims[0] in ("logT", "transition"):
        msg = f"{name} dimension must not be named {plasma_grid.dims[0]!r}"
        raise ValueError(msg)
    return temperature, plasma_grid


def _validate_positive_data_array(values: xr.DataArray, name: str, unit: u.UnitBase) -> xr.DataArray:
    if not isinstance(values, xr.DataArray):
        msg = f"{name} must be an xarray.DataArray"
        raise TypeError(msg)
    if values.ndim != 1:
        msg = f"{name} must be a one-dimensional array"
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


def _plasma_axis_units(plasma_dim: str) -> str:
    """
    The units of the plasma axis: the log of a density on ``logD``, else a pressure.
    """
    return str(u.dex(u.cm**-3) if plasma_dim == "logD" else u.K / u.cm**3)


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


def _import_fiasco():
    try:
        import fiasco  # noqa: PLC0415
    except ImportError:
        msg = "fiasco is required for this function, install it with `pip install muse[fiasco]`"
        raise ImportError(msg) from None
    # 0.9.0 is the first release with IonCollection.build_line_list and with a free-free
    # emission that is not per steradian: an older fiasco would silently give a continuum that
    # is too low by a factor of 4 pi.
    if not minversion(fiasco, "0.9.0"):
        msg = f"fiasco 0.9.0 or later is required, found {fiasco.__version__}"
        raise ImportError(msg)
    return fiasco


def _fiasco_ion_kwargs(abundance: str | None, hdf5_dbase_root: str | Path | None) -> dict:
    """
    The ``abundance`` and ``hdf5_dbase_root`` keywords for fiasco's ions, without those
    left at `None`.
    """
    options = {"abundance": abundance, "hdf5_dbase_root": hdf5_dbase_root}
    return {key: value for key, value in options.items() if value is not None}


def _map_ion_groups(function, ions: list, workers: int, *args, single_ions: bool = False, **kwargs) -> list:
    """
    Call ``function(group, *args, **kwargs)`` on groups of ``ions`` and return the
    results, one per group.

    With ``workers`` equal to 1, or a single ion, there is one group, all of ``ions``,
    and no process is started. Otherwise the groups run in ``workers`` processes.

    By default there is one group per worker, and a group is every n-th ion. The ions of
    one element are neighbors in the list and differ widely in cost, so a block of
    neighbors would leave one worker with most of the work.

    With ``single_ions`` every ion is its own group. The ions with the most levels go
    first, and a free worker takes the next ion. This suits work whose cost grows with
    the number of levels, as that of a line list does: a few ions then hold most of the
    work, and groups of every n-th ion are still unequal.

    The results are not in the order of ``ions``. A caller that needs that order has to
    restore it.
    """
    if workers < 1 and workers != -1:
        msg = f"workers must be at least 1, or -1 for every CPU this process may use, got {workers}"
        raise ValueError(msg)
    if workers == -1:
        # os.process_cpu_count gives the CPUs this process may run on. os.cpu_count gives every
        # core of the machine, also where a batch job holds only a few of them, and one worker
        # per core would then exhaust the job's memory.
        workers = os.process_cpu_count() or 1
    if workers == 1 or len(ions) <= 1:
        return [function(list(ions), *args, **kwargs)]
    try:
        from joblib import Parallel, delayed  # noqa: PLC0415
    except ImportError:
        msg = "joblib is required for workers != 1, install it with `pip install muse[fiasco]`"
        raise ImportError(msg) from None
    if single_ions:
        # The number of levels of an ion, zero for one without data, ranks its cost well enough
        # to order the queue.
        groups = [[ion] for ion in sorted(ions, key=lambda ion: getattr(ion, "n_levels", 0), reverse=True)]
    else:
        number_of_groups = min(workers, len(ions))
        groups = [list(ions[first::number_of_groups]) for first in range(number_of_groups)]
    with warnings.catch_warnings():
        # loky restarts a worker whose memory grew, as it does when a worker first loads the
        # database, and says so in a warning. Where warnings are errors, that error is raised in
        # loky's own thread and the run waits forever.
        warnings.filterwarnings("ignore", "A worker stopped while some jobs were given to the executor", UserWarning)
        # batch_size=1: joblib would otherwise bundle short tasks, and a bundle is a block of neighbors again.
        return Parallel(n_jobs=workers, batch_size=1)(delayed(function)(group, *args, **kwargs) for group in groups)
