"""
Line lists and continua stored in files: created once, then read by every later call.

A file is a zarr store packed into one zip.
"""

import os
import re
import uuid
import shutil
import platform
import contextlib
from pathlib import Path
from datetime import UTC, datetime
from collections.abc import Callable

import numpy as np
import xarray as xr
from threadpoolctl import threadpool_info, threadpool_limits
from zarr.storage import ZipStore

import astropy.units as u

from muse.instrument.atomic.backends import _validate_plasma_inputs
from muse.instrument.atomic.continuum import create_chianti_continuum, create_fiasco_continuum
from muse.instrument.atomic.linelist import create_chianti_line_list, create_fiasco_line_list
from muse.log import logger
from muse.utils.utils import _to_zarr
from muse.variables import DEFAULTS_MUSE

__all__ = ["load_or_create_continuum", "load_or_create_line_list"]

_LINE_LIST_BACKENDS = {"chianti": create_chianti_line_list, "fiasco": create_fiasco_line_list}
_CONTINUUM_BACKENDS = {"chianti": create_chianti_continuum, "fiasco": create_fiasco_continuum}


@u.quantity_input(wavelength_range=u.AA)
def load_or_create_line_list(
    cache_directory: str | Path,
    *,
    abundance: str,
    wavelength_range: u.Quantity,
    temperature: xr.DataArray,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    backend: str = "fiasco",
    create_if_missing: bool = True,
    chianti_version: str | None = None,
    workers: int = 1,
    threads: int = 4,
) -> xr.Dataset:
    """
    Read a line list from a cache directory, creating it first if it is not there.

    The file is ``line_list_<backend>_<lower>-<upper>A_<abundance>.zarr.zip``, with
    ``_density`` before the suffix for a density grid. A file that holds more plasma points than requested
    serves the requested ones. One created on other temperatures, without one of the requested
    plasma points or with another minimum abundance is refused and left as it is, as a line
    list takes hours and other processes may be reading it.

    While a file is being created, ``<file>.lock`` stands beside it and says which process
    is creating it; another call for the same file raises instead of waiting. A process that
    is killed, for example at the time limit of a batch job, leaves the lock behind: delete it
    by hand before the next call.

    Parameters
    ----------
    cache_directory : `str` or `pathlib.Path`
        Directory of the cache files; created if needed.
    abundance : `str`
        CHIANTI abundance name, e.g. ``"sun_coronal_2021_chianti"``.
    wavelength_range : `astropy.units.Quantity`
        The lowest and the highest wavelength of the line list.
    temperature : `xarray.DataArray`
        Temperatures along ``logT``, with units convertible to K.
    pressure, density : `xarray.DataArray`, optional
        The plasma points: pressures with units convertible to K / cm3, or densities with
        units convertible to cm-3. Give exactly one of the two. A pressure axis is stored as
        ``pressure`` whatever the array calls it, and densities as their log10 along ``logD``.
    minimum_abundance : `float`, optional
        Elements less abundant than this are left out, by default
        ``DEFAULTS_MUSE.minimum_abundance``.
    backend : {"chianti", "fiasco"}, optional
        What creates a missing line list, by default ``"fiasco"``; ``"chianti"`` needs
        ChiantiPy and ``XUVTOP``. The file name carries the backend, so the line lists of the
        two never mix.
    create_if_missing : `bool`, optional
        If `False`, a missing file raises instead of being created, so that one of many
        processes that expect the file does not start hours of work.
    chianti_version : `str`, optional
        Refuse a line list made from another CHIANTI version than this, e.g. ``"11.0.2"``, as
        the file name does not carry the version; by default any version is accepted.
    workers : `int`, optional
        Processes the fiasco backend splits the ions over, by default 1; ``-1`` uses every CPU
        this process may run on. The ChiantiPy backend does not use it.
    threads : `int`, optional
        Upper limit on the BLAS, OpenMP and numexpr threads during the call, by default 4. A
        pool that is set lower stays as it is, and the limit is never more than the CPUs this
        process may use.

    Returns
    -------
    `xarray.Dataset`
        The line list as `muse.instrument.create_fiasco_line_list` gives it, with ``gofnt`` in
        erg cm3 / (s sr) on the requested ``logT`` and the requested ``pressure`` (K / cm3) or
        ``logD`` (dex(1 / cm3)) values.

    Raises
    ------
    FileNotFoundError
        If the file is missing and ``create_if_missing`` is `False`.
    FileExistsError
        If ``<file>.lock`` exists: another process is creating the file, or one that was
        killed left its lock behind.
    ValueError
        If ``backend`` is unknown, ``abundance`` holds characters that cannot go into a file
        name, both or neither of ``pressure`` and ``density`` are given, the file does not
        hold the request, or it is of another CHIANTI version.
    """
    request = _PlasmaRequest(temperature, pressure, density)
    lower, upper = wavelength_range.to_value(u.AA)
    path = _cache_path(
        cache_directory, "line_list", backend, _LINE_LIST_BACKENDS, f"{lower:g}-{upper:g}A", abundance, request
    )

    def create() -> xr.Dataset:
        logger.info(f"Creating the {backend} line list of {abundance} over {wavelength_range}")
        return _LINE_LIST_BACKENDS[backend](
            request.temperature,
            density=request.density,
            pressure=request.pressure,
            abundance=abundance,
            wavelength_range=wavelength_range,
            minimum_abundance=minimum_abundance,
            **({"workers": workers} if backend == "fiasco" else {}),
        )

    with _limited_threads(threads):
        line_list = _load_or_create(
            path, create, request, minimum_abundance, create_if_missing=create_if_missing, what="line list"
        )
    found = str(line_list.attrs.get("Chianti"))
    if chianti_version is not None and found != chianti_version:
        msg = (
            f"the line list cache {path} is of CHIANTI {found}, not {chianti_version}; point the backend at the "
            f"CHIANTI {chianti_version} database and use another cache directory"
        )
        raise ValueError(msg)
    return line_list


@u.quantity_input(wavelength_grid=u.AA)
def load_or_create_continuum(
    cache_directory: str | Path,
    *,
    abundance: str,
    wavelength_grid: u.Quantity,
    temperature: xr.DataArray,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    backend: str = "fiasco",
    create_if_missing: bool = True,
    workers: int = 1,
    threads: int = 4,
) -> xr.Dataset:
    """
    Read a continuum from a cache directory, creating it first if it is not there.

    It is the rest-frame continuum of one abundance on one wavelength grid. The file is
    ``continuum_<backend>_<lower>-<upper>A_n<wavelengths>_<abundance>.zarr.zip``, with
    ``_density`` before the suffix for a density grid. A file that holds more plasma points than requested
    serves the requested ones. One created on other temperatures or wavelengths, without one
    of the requested plasma points or with another minimum abundance is refused and left as it
    is. The lock file is as for `load_or_create_line_list`.

    Parameters
    ----------
    cache_directory : `str` or `pathlib.Path`
        Directory of the cache files; created if needed.
    abundance : `str`
        CHIANTI abundance name.
    wavelength_grid : `astropy.units.Quantity`
        Increasing wavelengths the continuum is computed at.
    temperature, pressure, density, minimum_abundance
        As for `load_or_create_line_list`.
    backend : {"chianti", "fiasco"}, optional
        What creates a missing continuum, by default ``"fiasco"``.
    create_if_missing, workers, threads
        As for `load_or_create_line_list`.

    Returns
    -------
    `xarray.Dataset`
        The continuum as `muse.instrument.create_fiasco_continuum` gives it:
        ``spectral_response`` in 1e-27 erg cm3 / (Angstrom s sr) for the three components along
        ``line``, on the requested ``logT`` and ``pressure`` or ``logD`` values and on
        ``wavelength_grid`` in Angstrom.

    Raises
    ------
    FileNotFoundError, FileExistsError, ValueError
        As for `load_or_create_line_list`, without the CHIANTI version.
    """
    request = _PlasmaRequest(temperature, pressure, density)
    grid = np.asarray(wavelength_grid.to_value(u.AA), dtype=float)
    span = f"{grid.min():g}-{grid.max():g}A_n{grid.size}"
    path = _cache_path(cache_directory, "continuum", backend, _CONTINUUM_BACKENDS, span, abundance, request)

    def create() -> xr.Dataset:
        logger.info(f"Creating the {backend} continuum of {abundance} on {grid.size} wavelengths")
        return _CONTINUUM_BACKENDS[backend](
            request.temperature,
            wavelength_grid,
            density=request.density,
            pressure=request.pressure,
            abundance=abundance,
            minimum_abundance=minimum_abundance,
            **({"workers": workers} if backend == "fiasco" else {}),
        )

    with _limited_threads(threads):
        return _load_or_create(
            path,
            create,
            request,
            minimum_abundance,
            create_if_missing=create_if_missing,
            what="continuum",
            wavelength_grid=grid,
        )


class _PlasmaRequest:
    """
    The temperatures and the plasma points of a call, as a line list or a continuum
    holds them.
    """

    def __init__(self, temperature: xr.DataArray, pressure: xr.DataArray | None, density: xr.DataArray | None) -> None:
        validated_temperature, plasma = _validate_plasma_inputs(temperature, density, pressure)
        self.temperature = temperature
        self.density = density
        # Whatever the caller's array calls its axis, a file and a product hold it as "pressure".
        self.pressure = pressure
        if pressure is not None and pressure.dims != ("pressure",):
            self.pressure = pressure.rename({pressure.dims[0]: "pressure"})
        self.logT = np.log10(validated_temperature.data)
        self.axis = "pressure" if density is None else "logD"
        self.values = plasma.data if density is None else np.log10(plasma.data)

    def mismatch(self, dataset: xr.Dataset) -> str | None:
        """
        What ``dataset`` lacks of the request, as the end of a sentence, or `None`.
        """
        if dataset.sizes.get("logT") != self.logT.size or not np.allclose(
            dataset.logT.values, self.logT, rtol=0, atol=1e-6
        ):
            return "is on other temperatures than requested"
        if self.axis not in dataset.dims:
            return f"has no {self.axis} axis"
        stored = dataset[self.axis].values
        if not all(np.isclose(stored, value, rtol=1e-9, atol=0).any() for value in self.values):
            return (
                f"holds the {self.axis} values {stored.tolist()}, not all of the requested {self.values.tolist()} "
                "(a file serves the plasma points it was created on, so create it on all of them at once)"
            )
        return None

    def select(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        ``dataset`` at the requested plasma points, in their order.
        """
        stored = dataset[self.axis].values
        selected = dataset.isel({self.axis: [int(np.argmin(np.abs(stored - value))) for value in self.values]})
        # The coordinates are set to the requested numbers: a value read from a file can differ
        # from the request in the last bit after a unit conversion, and a line list and a
        # continuum that each kept their own number would not align.
        return selected.assign_coords(
            logT=("logT", self.logT, dataset.logT.attrs),
            **{self.axis: (self.axis, self.values, dataset[self.axis].attrs)},
        )


def _cache_path(
    cache_directory, kind: str, backend: str, backends: dict, span: str, abundance: str, request: _PlasmaRequest
) -> Path:
    if backend not in backends:
        msg = f"backend must be one of {sorted(backends)}, got {backend!r}"
        raise ValueError(msg)
    _require_name_part("abundance", abundance, "._+-")
    suffix = "_density" if request.density is not None else ""
    # The ends are not rounded to whole Angstrom: 25.4-900 must not find the file of 25-900.
    return Path(cache_directory) / f"{kind}_{backend}_{span}_{abundance}{suffix}.zarr.zip"


def _require_name_part(name: str, value: str, punctuation: str = ".+-") -> None:
    """
    Refuse a string that goes into a file name unless it is made of plain characters: a
    slash would make a directory, and a bracket or a brace a pattern.
    """
    if not isinstance(value, str) or not re.fullmatch(f"[A-Za-z0-9{re.escape(punctuation)}]+", value):
        msg = (
            f"{name} is part of a file name and may hold only letters, digits and the characters {punctuation}; "
            f"got {value!r}"
        )
        raise ValueError(msg)


def _load_or_create(
    path: Path,
    create: Callable[[], xr.Dataset],
    request: _PlasmaRequest,
    minimum_abundance: float,
    *,
    create_if_missing: bool,
    what: str,
    wavelength_grid: np.ndarray | None = None,
) -> xr.Dataset:
    """
    The one flow of both caches: create the file if it is missing, read it, check it
    against the request and return the requested part of it.
    """
    if not path.exists():
        if not create_if_missing:
            msg = f"the {what} cache {path} does not exist, and create_if_missing is False"
            raise FileNotFoundError(msg)
        with _creation_lock(path, what):
            # Another process may have finished the file between the check above and the lock.
            if not path.exists():
                # The minimum abundance is recorded here, so that the check below does not rest
                # on what a backend stores.
                created = create().assign_attrs(minimum_abundance=minimum_abundance)
                # One chunk per plasma point and temperature, so that a call for one pressure reads
                # that pressure alone and no chunk outgrows what the compressor takes.
                chunks = {
                    name: {
                        "chunks": tuple(
                            1 if dim in (request.axis, "logT") else size
                            for dim, size in zip(variable.dims, variable.shape, strict=True)
                        )
                    }
                    for name, variable in created.data_vars.items()
                    if request.axis in variable.dims
                }
                _replace_atomically(path, lambda staging: _to_zarr(created, staging, encoding=chunks))
    logger.info(f"Reading the {what} from {path}")
    # A file created just now is read back too: the first call then returns what every later
    # one will. Opened without loading, so that only the requested part is read.
    store = ZipStore(path, mode="r")
    try:
        cached = xr.open_zarr(store, consolidated=False, chunks=None, decode_coords="all")
        if cached.attrs.get("minimum_abundance") != minimum_abundance:
            problem = (
                f"was created with the minimum abundance {cached.attrs.get('minimum_abundance')}, "
                f"not the requested {minimum_abundance}"
            )
        elif wavelength_grid is not None and (
            cached.sizes["wavelength_bin"] != wavelength_grid.size
            or not np.allclose(cached.wavelength_grid.values, wavelength_grid, rtol=1e-9, atol=0)
        ):
            problem = "is on other wavelengths than requested"
        else:
            problem = request.mismatch(cached)
        if problem:
            msg = f"the {what} cache {path} {problem}; delete it or use another cache directory"
            raise ValueError(msg)
        return request.select(cached).load()
    finally:
        store.close()


@contextlib.contextmanager
def _creation_lock(path: Path, what: str):
    """
    Hold ``<path>.lock`` for the block: of several processes that miss a cache, one
    creates it and the others stop instead of each computing it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f"{path.name}.lock")
    # Who holds it, for the person who finds it left behind: a killed process cannot remove it.
    job = os.environ.get("SLURM_JOB_ID")
    owner = (
        f"process {os.getpid()} on {platform.node()}{f', job {job}' if job else ''}, "
        f"since {datetime.now(UTC):%Y-%m-%d %H:%M} UTC"
    )
    try:
        with lock.open("x") as file:
            file.write(owner)
    except FileExistsError:
        msg = (
            f"the {what} cache {path} is being created by {lock.read_text() or 'another process'}; run again when "
            f"it is done, or delete {lock} if that process has stopped"
        )
        raise FileExistsError(msg) from None
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _replace_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """
    Call ``write`` on a staging path beside ``path`` and rename the result into place: a
    reader never finds a half-written file, and a failed write leaves nothing behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        write(staging)
        staging.replace(path)
    finally:
        # Still there only if the write or the rename failed; a zarr store that is not packed is a directory.
        if staging.is_dir():
            shutil.rmtree(staging)
        else:
            staging.unlink(missing_ok=True)


@contextlib.contextmanager
def _limited_threads(threads: int):
    """
    Cap the BLAS, OpenMP and numexpr pools for the block at the lowest of ``threads``,
    the CPUs this process may use, and what each pool is set to now: a pool is lowered,
    never increased.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        msg = f"threads must be a positive integer, got {threads!r}"
        raise ValueError(msg)
    # The CPUs of this process, not of the machine: a one-CPU task of a job array gets one thread.
    threads = min(threads, os.process_cpu_count() or 1)
    limits = {}
    for pool in threadpool_info():
        limits[pool["user_api"]] = min(pool["num_threads"], limits.get(pool["user_api"], threads))
    try:
        import numexpr  # noqa: PLC0415
    except ImportError:
        numexpr = None
    numexpr_threads = None if numexpr is None else numexpr.set_num_threads(threads)
    if numexpr_threads is not None and numexpr_threads < threads:
        numexpr.set_num_threads(numexpr_threads)
    try:
        with threadpool_limits(limits=limits):
            yield
    finally:
        if numexpr_threads is not None:
            numexpr.set_num_threads(numexpr_threads)
