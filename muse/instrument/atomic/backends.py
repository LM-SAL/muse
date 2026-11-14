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
    temperature: xr.DataArray, density: xr.DataArray | None, pressure: xr.DataArray | None
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    The temperature in K and the density in cm^-3 or the pressure in K cm^-3, as plain
    values.
    """
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
    return temperature, plasma_grid


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
        temperature_bc, density_bc = xr.broadcast(temperature, plasma_grid)
        return temperature_bc, density_bc, {"logD": np.log10(plasma_grid.data)}
    density_bc = plasma_grid / temperature
    return temperature.broadcast_like(density_bc), density_bc, {plasma_grid.dims[0]: plasma_grid.data}


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
    # 0.9.0 brought IonCollection.build_line_list and a free-free emission that is no longer
    # per steradian: on an older fiasco the continuum would come out 4 pi too low, silently.
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


def _over_ion_chunks(function, ions: list, workers: int, *args, **kwargs) -> list:
    """
    ``function(chunk, *args, **kwargs)`` for every chunk of ``ions``, in parallel when
    ``workers`` is not 1.

    A chunk holds every n-th ion, so the results are not in the order of ``ions``: a caller
    that keeps an order has to restore it.
    """
    if workers < 1 and workers != -1:
        msg = f"workers must be at least 1, or -1 for every CPU this process may use, got {workers}"
        raise ValueError(msg)
    if workers == -1:
        # The CPUs this process may run on, not the machine's: on a Slurm node every core
        # is reported, and one worker per core runs a 16-CPU job out of memory.
        workers = os.process_cpu_count() or 1
    if workers == 1 or len(ions) <= 1:
        return [function(list(ions), *args, **kwargs)]
    try:
        from joblib import Parallel, delayed  # noqa: PLC0415
    except ImportError:
        msg = "joblib is required for workers != 1, install it with `pip install muse[fiasco]`"
        raise ImportError(msg) from None
    # Every n-th ion to a worker: the ions of one element sit together and differ widely in
    # cost, so blocks of neighbours leave one worker with a large share of the work.
    number_of_chunks = min(workers, len(ions))
    chunks = [list(ions[first::number_of_chunks]) for first in range(number_of_chunks)]
    return Parallel(n_jobs=workers)(delayed(function)(chunk, *args, **kwargs) for chunk in chunks)
