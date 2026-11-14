"""
Response-function generation: cached line lists and continua, and the SG and CI products
of every band, abundance and plasma point.
"""

import os
import functools
import contextlib
from pathlib import Path
from datetime import UTC, datetime
from collections.abc import Mapping, Iterable, Sequence

import numpy as np
import xarray as xr
from threadpoolctl import threadpool_limits

import astropy.units as u

from muse.instrument.atomic import _validate_plasma_inputs, _validate_positive_data_array
from muse.instrument.continuum import bin_continuum, create_chianti_continuum, create_fiasco_continuum
from muse.instrument.effective_area import (
    _DETECTOR_ATTRS,
    _DN_PER_PHOTON,
    effective_area_wavelength_range,
    read_effective_area,
    select_effective_area_curve,
)
from muse.instrument.linelist import (
    apply_line_boosts,
    create_chianti_line_list,
    create_fiasco_line_list,
    rank_line_list,
    select_transitions,
)
from muse.instrument.radiometry import transform_response_units
from muse.instrument.response import map_response_to_ci_detector, map_response_to_sg_detector
from muse.instrument.response_io import read_spectral_response, save_response, save_spectral_response
from muse.instrument.spectral import _centers_to_edges, create_log_wavelength_edges, create_spectral_response
from muse.log import logger
from muse.variables import DEFAULTS_MUSE

__all__ = [
    "CONTINUUM_BACKENDS",
    "DEFAULT_ABUNDANCES",
    "DEFAULT_CONTINUUM_DOPPLER_VELOCITY",
    "DEFAULT_DOPPLER_VELOCITY",
    "DEFAULT_PRESSURE",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_THREADS",
    "LINE_LIST_BACKENDS",
    "generate_continuum_responses",
    "generate_responses",
    "load_or_create_continuum",
    "load_or_create_line_list",
]

#: Line-list builders by backend name.
LINE_LIST_BACKENDS = {"chianti": create_chianti_line_list, "fiasco": create_fiasco_line_list}
#: Continuum calculators by backend name.
CONTINUUM_BACKENDS = {"chianti": create_chianti_continuum, "fiasco": create_fiasco_continuum}
#: The default grids of a run.
DEFAULT_ABUNDANCES = ("sun_coronal_2021_chianti", "sun_photospheric_2021_asplund")
DEFAULT_PRESSURE = xr.DataArray([3e14, 3e15, 3e16] * u.K / u.cm**3, dims="pressure")
DEFAULT_TEMPERATURE = xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT")
DEFAULT_DOPPLER_VELOCITY = np.arange(-1000, 1010, 10) * u.km / u.s
DEFAULT_CONTINUUM_DOPPLER_VELOCITY = np.arange(-1000, 1000, 200) * u.km / u.s
#: BLAS and numexpr threads during a run; every core per process oversubscribes the machine.
DEFAULT_THREADS = 4
_PHOTON_UNITS = "1e-27 cm5 ph / (Angstrom s)"
_DN_UNITS = "1e-27 cm5 DN / (Angstrom s)"
_SG_CHANNELS = [int(channel) for channel in DEFAULTS_MUSE.channel_spectral_order_SG.channel.values]
_SG_DEFAULTS = {
    "main_lines_by_band": dict(zip(_SG_CHANNELS, DEFAULTS_MUSE.main_lines_SG, strict=True)),
    "spectral_order_by_band": dict(zip(_SG_CHANNELS, DEFAULTS_MUSE.channel_spectral_order_SG.data, strict=True)),
    "instrumental_width_by_band": dict(zip(_SG_CHANNELS, DEFAULTS_MUSE.instrumental_width_SG.data, strict=True)),
}


@contextlib.contextmanager
def _limited_threads(threads: int):
    """
    Cap the BLAS, OpenMP and numexpr pools at ``threads`` for the block, then restore
    them.
    """
    if not isinstance(threads, int) or threads < 1:
        msg = f"threads must be a positive integer, got {threads!r}"
        raise ValueError(msg)
    previous = None
    try:
        import numexpr  # noqa: PLC0415

        previous = numexpr.set_num_threads(threads)
    except ImportError:
        pass
    try:
        with threadpool_limits(limits=threads):
            yield
    finally:
        if previous is not None:
            numexpr.set_num_threads(previous)


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
    compute_if_missing: bool = True,
    chianti_version: str | None = None,
    workers: int = 1,
    threads: int = DEFAULT_THREADS,
) -> xr.Dataset:
    """
    The line list of one abundance over a wavelength range, cached in
    ``cache_directory``.

    A cached list holding more plasma points than requested serves the requested ones. One
    built on other temperatures or another abundance cut is refused and not recomputed, as a
    line list takes hours and other tasks may be reading it.

    Parameters
    ----------
    cache_directory : `str` or `pathlib.Path`
        Directory of the cache files.
    abundance, wavelength_range, temperature, pressure, density, minimum_abundance
        As for `create_chianti_line_list` and `create_fiasco_line_list`.
    backend : {"chianti", "fiasco"}, optional
        Which builder computes a missing list, by default fiasco; ``"chianti"`` needs
        ChiantiPy and ``XUVTOP``. The cache name carries the backend, so the two never mix.
    compute_if_missing : `bool`, optional
        If `False`, a missing cache raises instead of being computed, so a cluster task
        fails fast when the preparation step was skipped.
    chianti_version : `str`, optional
        Refuse a list built from another CHIANTI version, e.g. ``"11.0.2"``, cached or just
        computed, as the cache name does not carry the version; by default any is accepted.
    workers : `int`, optional
        Processes for the fiasco backend.
    threads : `int`, optional
        BLAS and numexpr threads during the run, by default 4.

    Returns
    -------
    `xarray.Dataset`
        The line list, on the requested temperatures and plasma points.
    """
    with _limited_threads(threads):
        if backend not in LINE_LIST_BACKENDS:
            msg = f"backend must be one of {sorted(LINE_LIST_BACKENDS)}, got {backend!r}"
            raise ValueError(msg)
        logT, plasma_dim, requested_values = _requested_grid(temperature, density, pressure)
        path = _line_list_cache_path(
            cache_directory, backend, abundance, wavelength_range, density_dependent=density is not None
        )
        if path.exists():
            logger.info(f"Loading the line list from {path}")
            line_list = xr.load_dataset(path)
            if not _cache_matches(line_list, logT, plasma_dim, requested_values, minimum_abundance):
                msg = (
                    f"the line-list cache {path} was built on other temperatures, plasma points or abundance "
                    "cut; delete it or use another cache directory"
                )
                raise ValueError(msg)
        elif not compute_if_missing:
            msg = f"line-list cache {path} does not exist; run the preparation step first"
            raise FileNotFoundError(msg)
        else:
            logger.info(f"Computing the {backend} line list for {abundance} over {wavelength_range}")
            kwargs = {"workers": workers} if backend == "fiasco" else {}
            # The abundance cut is recorded here, so that the check of a cached list does not
            # rest on what a backend stores.
            line_list = LINE_LIST_BACKENDS[backend](
                temperature,
                density=density,
                pressure=pressure,
                abundance=abundance,
                wavelength_range=wavelength_range,
                minimum_abundance=minimum_abundance,
                **kwargs,
            ).assign_attrs(minimum_abundance=minimum_abundance)
            encoding = {
                name: {"zlib": True, "complevel": 5}
                for name, variable in line_list.data_vars.items()
                if variable.dtype.kind in "fiu"
            }
            _replace_atomically(path, lambda staging: line_list.to_netcdf(staging, encoding=encoding))
        found = str(line_list.attrs.get("Chianti"))
        if chianti_version is not None and found != chianti_version:
            msg = (
                f"the line list {path} is CHIANTI {found}, not {chianti_version}; "
                "point the backend at that database and use a clean cache directory"
            )
            raise ValueError(msg)
        # On the caller's own grid, so the list aligns with a continuum, and at the requested
        # plasma points alone, so a list cached for every pressure serves a run of one.
        return line_list.assign_coords(logT=line_list.logT.copy(data=logT)).sel(
            {plasma_dim: requested_values}, method="nearest"
        )


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
    compute_if_missing: bool = True,
    workers: int = 1,
    threads: int = DEFAULT_THREADS,
) -> xr.Dataset:
    """
    The rest-frame continuum of one abundance on one wavelength grid, cached in
    ``cache_directory``.

    A cached cube on other temperatures, wavelengths or abundance cut is recomputed; one
    holding more plasma points than requested serves the requested ones.

    Parameters
    ----------
    cache_directory : `str` or `pathlib.Path`
        Directory of the cache files.
    abundance, wavelength_grid, temperature, pressure, density, minimum_abundance
        As for `create_chianti_continuum` and `create_fiasco_continuum`.
    backend : {"chianti", "fiasco"}, optional
        Which calculator computes a missing cube, by default fiasco.
    compute_if_missing, workers, threads
        As for `load_or_create_line_list`.

    Returns
    -------
    `xarray.Dataset`
        The continuum on exactly the requested grids.
    """
    with _limited_threads(threads):
        if backend not in CONTINUUM_BACKENDS:
            msg = f"backend must be one of {sorted(CONTINUUM_BACKENDS)}, got {backend!r}"
            raise ValueError(msg)
        logT, plasma_dim, requested_values = _requested_grid(temperature, density, pressure)
        grid = np.asarray(wavelength_grid.to_value(u.AA), dtype=float)
        path = _continuum_cache_path(
            cache_directory, backend, abundance, wavelength_grid, density_dependent=density is not None
        )
        if path.exists():
            # Closed after loading, so a recompute can replace the file on Windows too.
            with read_spectral_response(path) as opened:
                cached = opened.load()
            if _cache_matches(cached, logT, plasma_dim, requested_values, minimum_abundance, grid):
                logger.info(f"Loading the continuum from {path}")
                return cached.assign_coords(logT=cached.logT.copy(data=logT)).sel(
                    {plasma_dim: requested_values}, method="nearest"
                )
            logger.warning(f"{path} was computed on another grid or abundance cut; recomputing it")
        elif not compute_if_missing:
            msg = f"continuum cache {path} does not exist; run the preparation step first"
            raise FileNotFoundError(msg)
        logger.info(f"Computing the {backend} continuum for {abundance} on {grid.size} wavelengths")
        kwargs = {"workers": workers} if backend == "fiasco" else {}
        continuum = CONTINUUM_BACKENDS[backend](
            temperature,
            wavelength_grid,
            density=density,
            pressure=pressure,
            abundance=abundance,
            minimum_abundance=minimum_abundance,
            **kwargs,
        )
        _replace_atomically(path, functools.partial(save_spectral_response, continuum))
        return continuum


@u.quantity_input(doppler_velocity=u.km / u.s, wavelength_range=u.AA)
def generate_responses(
    *,
    output_directory: str | Path,
    author: str,
    effective_area_note: str,
    sg_effective_area_files: Sequence[str | Path] = (),
    ci_effective_area_files: Sequence[str | Path] = (),
    abundances: Sequence[str] = DEFAULT_ABUNDANCES,
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    doppler_velocity: u.Quantity | None = None,
    main_lines_by_band: Mapping[int, Sequence[str]] | None = None,
    spectral_order_by_band: Mapping[int, int] | None = None,
    instrumental_width_by_band: Mapping[int, u.Quantity] | None = None,
    maximum_contaminants: int = 10000,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    line_list_backend: str = "fiasco",
    wavelength_range: u.Quantity | None = None,
    line_list_file: str | Path | None = None,
    line_boosts: Mapping[tuple[str, int, int], float] | None = None,
    line_selection: Iterable[tuple[str, int, int]] | None = None,
    ci_fractional_width: float = 0.01,
    include_continuum: bool = True,
    continuum_backend: str = "fiasco",
    ci_units: Sequence[str] = ("ph", "DN"),
    sg_detector_kwargs: Mapping[str, object] | None = None,
    sg_units_kwargs: Mapping[str, object] | None = None,
    cache_directory: str | Path | None = None,
    compute_if_missing: bool = True,
    skip_existing: bool = False,
    workers: int = 1,
    threads: int = DEFAULT_THREADS,
) -> list[Path]:
    """
    Generate the response products of every effective-area file, abundance and plasma
    point.

    One line list per abundance covers every file. Per pressure, each SG band gets a waveband
    cube and a detector response for every main line, and the same pair for the brightest
    other lines summed. Each CI band gets a cube of its brightest lines and the continuum on
    bins of constant velocity width, and that cube summed into one imager response, in
    photons and in DN.

    Parameters
    ----------
    output_directory : `str` or `pathlib.Path`
        Where the products go, as zarr stores named
        ``respfunc_<band>_<product>_<abundance>_<note>_<plasma>_<author>_<date>.zarr``.
    author : `str`
        Initials recorded in the names and attributes.
    effective_area_note : `str`
        Tag of the effective areas in the names, e.g. the coating.
    sg_effective_area_files, ci_effective_area_files : `list` of path-like
        Effective-area files, one band each, as `create_effective_area` writes them. A CI
        area in cm2 DN / ph carries the camera gain, so only DN products exist for it. The
        file of another imager describes its detector in ``pixel_width``, ``pixel_height``,
        ``gain`` and ``pair_energy`` attributes, as `create_aia_effective_area` writes them,
        and is converted with those instead of the MUSE values.
    abundances : `list` of `str`, optional
        CHIANTI abundance names, by default the coronal and photospheric 2021 sets.
    temperature : `xarray.DataArray`, optional
        Temperatures with a ``logT`` dimension, by default ``logT`` 4.5 to 7.9 in steps of 0.1.
    pressure, density : `xarray.DataArray`, optional
        The plasma grid, by default the pressures 3e14, 3e15 and 3e16 K / cm3. A density grid
        is computed in one go and tagged ``density``.
    doppler_velocity : `astropy.units.Quantity`, optional
        By default -1000 to 1000 km/s in steps of 10.
    main_lines_by_band, spectral_order_by_band, instrumental_width_by_band : `dict`, optional
        Per SG band (its nominal wavelength in Angstrom as an `int`): the ``full_name`` values
        of the main lines, the spectral order and the instrumental width; by default the MUSE
        values from `~muse.variables.DEFAULTS_MUSE`.
    maximum_contaminants : `int`, optional
        The brightest lines kept per band, by default 10000: summed into the SG contaminant
        products, which 0 skips, and into the CI cube, which needs at least one.
    minimum_abundance : `float`, optional
        Elemental abundance cut of the line list and the continuum.
    line_list_backend : {"chianti", "fiasco"}, optional
        Builder of a missing line list, by default fiasco.
    wavelength_range : `astropy.units.Quantity`, optional
        The range of the line list, by default the one the effective-area files span. Pass the
        range of the whole run when its bands are processed in separate calls, so that they
        share one cached list.
    line_list_file : `str` or `pathlib.Path`, optional
        The line list of the one abundance in ``abundances``, to use instead of the cache.
    line_boosts : `dict`, optional
        Passed to `apply_line_boosts` after the list is loaded, so the cache stays unboosted.
    line_selection : iterable of `tuple`, optional
        Passed to `select_transitions` after the boosts.
    ci_fractional_width : `float`, optional
        Width of the CI bins relative to their lower edge, by default 0.01.
    include_continuum : `bool`, optional
        Fold the continuum into the CI products, by default `True`.
    continuum_backend : {"chianti", "fiasco"}, optional
        Calculator of a missing continuum, by default fiasco.
    ci_units : `list` of `str`, optional
        The CI products to write, a non-empty subset of ``("ph", "DN")``.
    sg_detector_kwargs : `dict`, optional
        Keyword arguments of `map_response_to_sg_detector` other than ``channel``.
    sg_units_kwargs : `dict`, optional
        Keyword arguments of `transform_response_units` other than ``new_units`` and
        ``channel``, for every SG band.
    cache_directory : `str` or `pathlib.Path`, optional
        Where the line lists and continua are cached, by default ``output_directory / "cache"``.
    compute_if_missing : `bool`, optional
        If `False`, a missing line list or continuum raises instead of being computed, so a
        task of a job array fails fast when the preparation step was skipped.
    skip_existing : `bool`, optional
        If `True`, a product that already exists, under any date, is kept and the others are
        written, so a run that stopped halfway can be finished. By default an existing product
        raises, as its name does not carry every setting of the run.
    workers : `int`, optional
        Processes for the fiasco backends.
    threads : `int`, optional
        BLAS and numexpr threads during the run, by default 4.

    Returns
    -------
    `list` of `pathlib.Path`
        The products, written or kept: ``detector_response(line, doppler_velocity, pressure or logD,
        logT, slit, detector_x_pixel)`` for the SG, without the detector axes for the CI, and
        ``spectral_response`` on ``wavelength_bin`` for the cubes, each with the band's
        ``effective_area`` curve.
    """
    with _limited_threads(threads):
        for name, value in (("author", author), ("effective_area_note", effective_area_note)):
            if not isinstance(value, str) or not value.strip():
                msg = f"{name} must be a non-empty string"
                raise ValueError(msg)
        ci_units = tuple(ci_units)
        if not ci_units or set(ci_units) - {"ph", "DN"}:
            msg = f"ci_units must be a non-empty subset of ('ph', 'DN'), got {ci_units!r}"
            raise ValueError(msg)
        for name, backend, backends in (
            ("line_list_backend", line_list_backend, LINE_LIST_BACKENDS),
            ("continuum_backend", continuum_backend, CONTINUUM_BACKENDS),
        ):
            if backend not in backends:
                msg = f"{name} must be one of {sorted(backends)}, got {backend!r}"
                raise ValueError(msg)
        if line_list_file is not None and len(abundances) != 1:
            msg = "line_list_file holds the list of one abundance; pass that one name in abundances"
            raise ValueError(msg)
        run = _RunInputs(output_directory, cache_directory, pressure, density)
        temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        doppler_velocity = DEFAULT_DOPPLER_VELOCITY if doppler_velocity is None else doppler_velocity
        sg_areas = [read_effective_area(path) for path in sg_effective_area_files]
        ci_areas = [read_effective_area(path) for path in ci_effective_area_files]
        if not sg_areas and not ci_areas:
            msg = "pass at least one effective-area file"
            raise ValueError(msg)
        area_range = effective_area_wavelength_range([*sg_areas, *ci_areas])
        if wavelength_range is None:
            wavelength_range = area_range
        elif wavelength_range[0] > area_range[0] or wavelength_range[-1] < area_range[-1]:
            msg = f"wavelength_range {wavelength_range} does not cover the effective areas, {area_range}"
            raise ValueError(msg)
        continuum_note = (
            f"{continuum_backend} free-free + free-bound + two-photon, minimum abundance {minimum_abundance}"
        )

        products = []
        for abundance in abundances:
            if line_list_file is not None:
                line_list = xr.load_dataset(line_list_file)
            else:
                line_list = load_or_create_line_list(
                    run.cache_directory,
                    abundance=abundance,
                    wavelength_range=wavelength_range,
                    temperature=temperature,
                    pressure=run.pressure,
                    density=run.density,
                    minimum_abundance=minimum_abundance,
                    backend=line_list_backend,
                    compute_if_missing=compute_if_missing,
                    workers=workers,
                    threads=threads,
                )
            if line_boosts:
                line_list = apply_line_boosts(line_list, line_boosts)
            if line_selection:
                line_list = select_transitions(line_list, line_selection)
            provenance = {
                "abundance": abundance,
                # A given list was made elsewhere: its file can be recorded, a backend cannot.
                **(
                    {"line_list_backend": line_list_backend}
                    if line_list_file is None
                    else {"line_list_file": str(line_list_file)}
                ),
                "author": author,
                "effective_area_note": effective_area_note,
                "minimum_abundance": minimum_abundance,
                **{
                    key: line_list.attrs[key]
                    for key in ("Chianti", "Chiantipy", "fiasco", "line_boosts", "line_selection")
                    if key in line_list.attrs
                },
            }
            name_tail = f"{abundance}_{effective_area_note}_{{plasma}}{{units}}_{author}_{run.date}"
            writer = functools.partial(_write_product, run=run, provenance=provenance, skip_existing=skip_existing)
            for effective_area in sg_areas:
                band = _nominal_band(effective_area)
                main_lines = _per_band(main_lines_by_band, band, "main_lines_by_band")
                spectral_order = _per_band(spectral_order_by_band, band, "spectral_order_by_band")
                instrumental_width = _per_band(instrumental_width_by_band, band, "instrumental_width_by_band")
                for plasma_index in range(run.number_of_plasma_points):
                    products += _sg_band_products(
                        run,
                        writer,
                        line_list,
                        effective_area,
                        plasma_index,
                        main_lines=list(main_lines),
                        spectral_order=int(spectral_order),
                        instrumental_width=u.Quantity(instrumental_width),
                        doppler_velocity=doppler_velocity,
                        maximum_contaminants=maximum_contaminants,
                        sg_detector_kwargs=sg_detector_kwargs or {},
                        sg_units_kwargs=sg_units_kwargs or {},
                        name_tail=name_tail,
                    )
            for effective_area in ci_areas:
                continuum = None
                if include_continuum:
                    continuum = load_or_create_continuum(
                        run.cache_directory,
                        abundance=abundance,
                        wavelength_grid=effective_area.wavelength.values * u.AA,
                        temperature=temperature,
                        pressure=run.pressure,
                        density=run.density,
                        minimum_abundance=minimum_abundance,
                        backend=continuum_backend,
                        compute_if_missing=compute_if_missing,
                        workers=workers,
                        threads=threads,
                    )
                for plasma_index in range(run.number_of_plasma_points):
                    products += _ci_band_products(
                        run,
                        writer,
                        line_list,
                        effective_area,
                        plasma_index,
                        continuum=continuum,
                        continuum_note=continuum_note,
                        doppler_velocity=doppler_velocity,
                        maximum_contaminants=maximum_contaminants,
                        ci_fractional_width=ci_fractional_width,
                        ci_units=ci_units,
                        name_tail=name_tail,
                    )
        return products


@u.quantity_input(doppler_velocity=u.km / u.s, wavelength_step=u.AA)
def generate_continuum_responses(
    *,
    output_directory: str | Path,
    author: str,
    effective_area_note: str,
    sg_effective_area_files: Sequence[str | Path],
    abundances: Sequence[str] = DEFAULT_ABUNDANCES,
    continuum_backend: str = "fiasco",
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    doppler_velocity: u.Quantity | None = None,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    wavelength_step: u.Quantity = 0.01 * u.AA,
    spectral_order_by_band: Mapping[int, int] | None = None,
    sg_detector_kwargs: Mapping[str, object] | None = None,
    sg_units_kwargs: Mapping[str, object] | None = None,
    cache_directory: str | Path | None = None,
    compute_if_missing: bool = True,
    skip_existing: bool = False,
    workers: int = 1,
    threads: int = DEFAULT_THREADS,
) -> list[Path]:
    """
    Generate the SG continuum response of every band: the continuum alone on the
    detector.

    Per band the rest-frame continuum on a fine grid over the waveband, cached by
    `load_or_create_continuum`, is folded through the effective area with `bin_continuum`,
    converted to photons and mapped onto the detector.

    Parameters
    ----------
    output_directory, author, effective_area_note, sg_effective_area_files, abundances
        As for `generate_responses`.
    continuum_backend : {"chianti", "fiasco"}, optional
        The continuum calculator, by default fiasco; it is part of the product names.
    temperature, pressure, density, minimum_abundance
        The plasma grid and the abundance cut, as for `generate_responses`.
    doppler_velocity : `astropy.units.Quantity`, optional
        By default -1000 to 800 km/s in steps of 200.
    wavelength_step : `astropy.units.Quantity`, optional
        Sampling of the fine grid, by default 0.01 Angstrom.
    spectral_order_by_band, sg_detector_kwargs, sg_units_kwargs
        The detector of each band, as for `generate_responses`.
    cache_directory, compute_if_missing, skip_existing, workers, threads
        As for `generate_responses`.

    Returns
    -------
    `list` of `pathlib.Path`
        The products written, named ``respfunc_<band>_continuum_<backend>_...`` like the line
        products, with the three continuum components as ``line`` and the SG detector axes.
    """
    with _limited_threads(threads):
        if continuum_backend not in CONTINUUM_BACKENDS:
            msg = f"continuum_backend must be one of {sorted(CONTINUUM_BACKENDS)}, got {continuum_backend!r}"
            raise ValueError(msg)
        run = _RunInputs(output_directory, cache_directory, pressure, density)
        temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        doppler_velocity = DEFAULT_CONTINUUM_DOPPLER_VELOCITY if doppler_velocity is None else doppler_velocity
        step = wavelength_step.to_value(u.AA)
        plasma_tag = run.plasma_tag(0) if run.number_of_plasma_points == 1 else "pressure"
        effective_areas = [read_effective_area(path) for path in sg_effective_area_files]
        products = []
        for abundance in abundances:
            provenance = {
                "abundance": abundance,
                "continuum_backend": continuum_backend,
                "author": author,
                "effective_area_note": effective_area_note,
                "minimum_abundance": minimum_abundance,
            }
            writer = functools.partial(_write_product, run=run, provenance=provenance, skip_existing=skip_existing)
            for effective_area in effective_areas:
                band = _nominal_band(effective_area)
                area_curve = select_effective_area_curve(effective_area)
                spectral_order = int(_per_band(spectral_order_by_band, band, "spectral_order_by_band"))
                grid = _sg_waveband_grid(band, spectral_order, step)
                continuum = load_or_create_continuum(
                    run.cache_directory,
                    abundance=abundance,
                    wavelength_grid=grid,
                    temperature=temperature,
                    pressure=run.pressure,
                    density=density,
                    minimum_abundance=minimum_abundance,
                    backend=continuum_backend,
                    compute_if_missing=compute_if_missing,
                    workers=workers,
                    threads=threads,
                )
                binned_continuum = bin_continuum(
                    continuum,
                    _centers_to_edges(grid.value) * u.AA,
                    effective_area=area_curve,
                    doppler_velocity=doppler_velocity,
                )
                photons = transform_response_units(
                    binned_continuum, _PHOTON_UNITS, band * u.AA, detector="SG", **(sg_units_kwargs or {})
                )
                detector_response = map_response_to_sg_detector(
                    photons, channel=band * u.AA, **(sg_detector_kwargs or {})
                ).assign(effective_area=area_curve)
                geometry_tag = _sg_geometry_tag(sg_detector_kwargs, spectral_order)
                tail = f"{abundance}_{effective_area_note}_{plasma_tag}_{author}_{run.date}"
                name = f"respfunc_{band}_continuum_{continuum_backend}_{geometry_tag}_{tail}.zarr"
                products.append(writer(detector_response, name, save_response))
        return products


class _RunInputs:
    """
    The directories and the plasma grid of one run, resolved once.
    """

    def __init__(
        self, output_directory, cache_directory, pressure: xr.DataArray | None, density: xr.DataArray | None
    ) -> None:
        if pressure is not None and density is not None:
            msg = "pressure and density are mutually exclusive"
            raise ValueError(msg)
        self.output_directory = Path(output_directory)
        self.cache_directory = self.output_directory / "cache" if cache_directory is None else Path(cache_directory)
        self.pressure = DEFAULT_PRESSURE if pressure is None and density is None else pressure
        self.density = density
        self.date = datetime.now(UTC).date()
        self._pressures = (
            None
            if self.pressure is None
            else _validate_positive_data_array(self.pressure, "pressure", u.K / u.cm**3).data.reshape(-1)
        )
        self.number_of_plasma_points = 1 if density is not None else self._pressures.size

    def plasma_tag(self, plasma_index: int) -> str:
        """
        ``density`` for a density run, else the plasma point's pressure as
        ``pressure_3e+15``.
        """
        return "density" if self.density is not None else f"pressure_{self._pressures[plasma_index]:.0e}"

    def at_plasma_point(self, dataset: xr.Dataset, plasma_index: int) -> xr.Dataset:
        """
        ``dataset`` at one plasma point; the whole dataset for a density run.
        """
        if self.density is not None:
            return dataset
        # By value, not by position: a cached or given list may hold other pressures than this run.
        pressure = self._pressures[plasma_index]
        selected = dataset.sel(pressure=[pressure], method="nearest")
        if not np.isclose(selected.pressure.values[0], pressure, rtol=1e-9, atol=0):
            msg = f"the pressure {pressure:g} K / cm3 is not among {dataset.pressure.values.tolist()}"
            raise ValueError(msg)
        return selected


def _sg_band_products(
    run: _RunInputs,
    writer,
    line_list: xr.Dataset,
    effective_area: xr.Dataset,
    plasma_index: int,
    *,
    main_lines: list[str],
    spectral_order: int,
    instrumental_width: u.Quantity,
    doppler_velocity: u.Quantity,
    maximum_contaminants: int,
    sg_detector_kwargs: Mapping[str, object],
    sg_units_kwargs: Mapping[str, object],
    name_tail: str,
) -> list[Path]:
    """
    The main-line and contaminant products of one SG band at one plasma point.
    """
    band = _nominal_band(effective_area)
    channel = band * u.AA
    area_curve = select_effective_area_curve(effective_area)
    step = DEFAULTS_MUSE.wavelength_step_SG.to_value(u.AA)
    grid = _sg_waveband_grid(band, spectral_order, step)
    geometry_tag = _sg_geometry_tag(sg_detector_kwargs, spectral_order)
    tail = name_tail.format(plasma=run.plasma_tag(plasma_index), units="")
    line_list = run.at_plasma_point(line_list, plasma_index)
    logger.info(f"SG band {band}, {tail}: main lines {main_lines}")

    main_response = create_spectral_response(
        line_list,
        grid,
        main_lines=main_lines,
        instrumental_width=instrumental_width,
        doppler_velocity=doppler_velocity,
        effective_area=area_curve,
    )
    main_response = transform_response_units(main_response, _PHOTON_UNITS, channel, detector="SG", **sg_units_kwargs)
    main_detector_response = map_response_to_sg_detector(main_response, channel=channel, **sg_detector_kwargs).assign(
        effective_area=area_curve
    )
    products = []
    for name in main_lines:
        line_tag = "".join(name.split())
        cube = main_response.sel(line=[name]).assign(effective_area=area_curve)
        products.append(writer(cube, f"respfunc_{band}_{line_tag}_waveband_{tail}.zarr", save_spectral_response))
        products.append(
            writer(
                main_detector_response.sel(line=[name]),
                f"respfunc_{band}_{line_tag}_{geometry_tag}_{tail}.zarr",
                save_response,
            )
        )
    if maximum_contaminants <= 0:
        return products

    in_band = (
        (line_list.wavelength >= grid[0].value)
        & (line_list.wavelength <= grid[-1].value)
        & ~line_list.full_name.isin(main_lines)
    )
    ranked = rank_line_list(line_list.isel(trans_index=in_band.values), area_curve).isel(
        trans_index=slice(maximum_contaminants)
    )
    if ranked.sizes["trans_index"] == 0:
        logger.warning(f"SG band {band}: no contaminant line in the waveband")
        return products
    contaminants = create_spectral_response(
        ranked,
        grid,
        main_lines=None,
        include_contaminants=True,
        instrumental_width=instrumental_width,
        doppler_velocity=doppler_velocity,
        effective_area=area_curve,
    )
    contaminants = transform_response_units(contaminants, _PHOTON_UNITS, channel, detector="SG", **sg_units_kwargs)
    wavelength_attrs = {"units": str(u.AA)}
    contaminant_cube = contaminants.assign_coords(
        line=[f"{band} contaminants"],
        line_wavelength=("line", [float(ranked.wavelength[0])], wavelength_attrs),
    ).assign_attrs(number_of_lines=ranked.sizes["trans_index"])
    products.append(
        writer(
            contaminant_cube.assign(effective_area=area_curve),
            f"respfunc_{band}_contaminants_waveband_{tail}.zarr",
            save_spectral_response,
        )
    )
    # On the detector the contaminants are referenced to the band's main line, as the line
    # products are, so a consumer derives one geometry per band.
    contaminants_at_main_line = contaminant_cube.assign_coords(
        line_wavelength=("line", [float(main_response.line_wavelength[0])], wavelength_attrs)
    )
    contaminant_detector_response = map_response_to_sg_detector(
        contaminants_at_main_line, channel=channel, **sg_detector_kwargs
    ).assign(effective_area=area_curve)
    products.append(
        writer(contaminant_detector_response, f"respfunc_{band}_contaminants_{geometry_tag}_{tail}.zarr", save_response)
    )
    return products


def _ci_band_products(
    run: _RunInputs,
    writer,
    line_list: xr.Dataset,
    effective_area: xr.Dataset,
    plasma_index: int,
    *,
    continuum: xr.Dataset | None,
    continuum_note: str,
    doppler_velocity: u.Quantity,
    maximum_contaminants: int,
    ci_fractional_width: float,
    ci_units: tuple[str, ...],
    name_tail: str,
) -> list[Path]:
    """
    The waveband cube and the summed imager response of one CI band at one plasma point.
    """
    band = _nominal_band(effective_area)
    channel = band * u.AA
    area_curve = select_effective_area_curve(effective_area)
    area_in_dn = u.Unit(area_curve.attrs["units"]).is_equivalent(_DN_PER_PHOTON)
    response_area = area_curve
    if area_in_dn:
        if "DN" not in ci_units:
            msg = f"CI band {band}: the area is in cm2 DN / ph, so only DN products exist; add 'DN' to ci_units"
            raise ValueError(msg)
        # The gain is already in the area: the response is built from it as a plain cm2 area
        # and what comes out per photon is DN. The products keep the area as it was given.
        response_area = area_curve.assign_attrs(units=str(u.Unit(area_curve.attrs["units"]) / (u.DN / u.ph)))
    wavelengths = area_curve.wavelength.values
    edges = create_log_wavelength_edges([wavelengths[0], wavelengths[-1]] * u.AA, ci_fractional_width)
    # A file that describes its own detector, as an AIA one does, is converted with it and
    # not with the MUSE pixel size, gain and pair energy.
    detector = {
        name: u.Quantity(effective_area.attrs[name]) for name in _DETECTOR_ATTRS if name in effective_area.attrs
    }
    plasma_tag = run.plasma_tag(plasma_index)
    line_list = run.at_plasma_point(line_list, plasma_index)
    ranked = rank_line_list(line_list, response_area).isel(trans_index=slice(maximum_contaminants))
    number_of_lines = ranked.sizes["trans_index"]
    logger.info(f"CI band {band}, {name_tail.format(plasma=plasma_tag, units='')}: {number_of_lines} lines")

    lines = create_spectral_response(
        ranked,
        main_lines=None,
        include_contaminants=True,
        doppler_velocity=doppler_velocity,
        effective_area=response_area,
        wavelength_edges=edges,
    )
    ci_line_name = f"{band} lines"
    lines = lines.assign_coords(
        line=[ci_line_name], line_wavelength=("line", [float(ranked.wavelength[0])], {"units": str(u.AA)})
    )
    parts = [lines]
    if continuum is not None:
        continuum_subset = run.at_plasma_point(continuum, plasma_index)
        binned_continuum = bin_continuum(
            continuum_subset, edges, effective_area=response_area, doppler_velocity=doppler_velocity
        )
        binned_continuum = binned_continuum.assign_coords(
            line=[f"{band} continuum {name}" for name in binned_continuum.line.values]
        )
        parts.append(binned_continuum.transpose(*lines.spectral_response.dims, "bound", missing_dims="ignore"))
        ci_line_name += " + continuum"
    ci_cube = xr.concat(parts, dim="line", data_vars="all", combine_attrs="override")
    ci_cube.attrs.update(lines.attrs, number_of_lines=number_of_lines)
    if continuum is not None:
        ci_cube.attrs["continuum"] = continuum_note
    converted = transform_response_units(ci_cube, _PHOTON_UNITS, channel, detector="CI", **detector)
    cubes_by_units = {}
    if area_in_dn:
        converted.spectral_response.attrs["units"] = _DN_UNITS
        cubes_by_units["_DN"] = converted
    else:
        if "ph" in ci_units:
            cubes_by_units[""] = converted
        if "DN" in ci_units:
            cubes_by_units["_DN"] = transform_response_units(ci_cube, _DN_UNITS, channel, detector="CI", **detector)
    products = []
    for units_tag, cube in cubes_by_units.items():
        tail = name_tail.format(plasma=plasma_tag, units=units_tag)
        products.append(
            writer(
                cube.assign(effective_area=area_curve),
                f"respfunc_ci_{band}_waveband_{tail}.zarr",
                save_spectral_response,
            )
        )
        summed = _sum_ci_response(map_response_to_ci_detector(cube, channel), ci_line_name).assign(
            effective_area=area_curve
        )
        products.append(writer(summed, f"respfunc_ci_{band}_{tail}.zarr", save_response))
    return products


def _requested_grid(
    temperature: xr.DataArray, density: xr.DataArray | None, pressure: xr.DataArray | None
) -> tuple[np.ndarray, str, np.ndarray]:
    """
    The requested ``logT`` values, and the name and the values of the plasma axis, as a
    cached line list or continuum stores them.
    """
    validated_temperature, plasma_grid = _validate_plasma_inputs(temperature, density, pressure)
    logT = np.log10(validated_temperature.data.reshape(-1))
    plasma_values = plasma_grid.data.reshape(-1)
    if density is not None:
        return logT, "logD", np.log10(plasma_values)
    return logT, plasma_grid.dims[0], plasma_values


def _line_list_cache_path(
    cache_directory, backend: str, abundance: str, wavelength_range: u.Quantity, *, density_dependent: bool
) -> Path:
    lower, upper = wavelength_range.to_value(u.AA)
    suffix = "_density" if density_dependent else ""
    # Not rounded down to whole Angstrom: 25.4-900.9 must not find the list of 25-900.
    return Path(cache_directory) / f"line_list_{backend}_{lower:g}-{upper:g}A_{abundance}{suffix}.nc"


def _continuum_cache_path(
    cache_directory, backend: str, abundance: str, wavelength_grid: u.Quantity, *, density_dependent: bool
) -> Path:
    values = wavelength_grid.to_value(u.AA)
    suffix = "_density" if density_dependent else ""
    return (
        Path(cache_directory)
        / f"continuum_{backend}_{int(values[0])}-{int(values[-1])}A_n{values.size}_{abundance}{suffix}.nc"
    )


def _cache_matches(
    cached: xr.Dataset,
    logT: np.ndarray,
    plasma_dim: str,
    requested_values: np.ndarray,
    minimum_abundance: float,
    wavelength_grid: np.ndarray | None = None,
) -> bool:
    """
    Whether a cached line list or continuum holds the requested temperatures, plasma
    points and abundance cut, and a continuum the requested wavelengths.
    """
    return (
        cached.sizes["logT"] == logT.size
        and np.allclose(cached.logT.values, logT, atol=1e-6)
        and plasma_dim in cached.dims
        and all(np.isclose(cached[plasma_dim].values, value, rtol=1e-9, atol=0).any() for value in requested_values)
        and cached.attrs.get("minimum_abundance") == minimum_abundance
        and (
            wavelength_grid is None
            or (
                cached.sizes["wavelength_bin"] == wavelength_grid.size
                and np.allclose(cached.wavelength_grid.values, wavelength_grid)
            )
        )
    )


def _nominal_band(effective_area: xr.Dataset) -> int:
    """
    The band of a one-band effective-area file, in Angstrom.
    """
    if effective_area.sizes["band"] != 1:
        msg = (
            f"the effective-area file holds {effective_area.sizes['band']} bands; the pipeline takes one band per file"
        )
        raise ValueError(msg)
    return round(float(effective_area.band[0]))


def _per_band(overrides: Mapping[int, object] | None, band: int, name: str):
    """
    The override for ``band``, else its MUSE default.
    """
    try:
        return {**_SG_DEFAULTS[name], **(overrides or {})}[band]
    except KeyError:
        msg = f"no MUSE default for band {band}; pass {name}={{{band}: ...}}"
        raise ValueError(msg) from None


def _sg_waveband_grid(band: int, spectral_order: int, step: float) -> u.Quantity:
    """
    The waveband of an SG band: the MUSE half width over the spectral order, closed at
    both ends, in Angstrom steps of ``step``.
    """
    half_width = DEFAULTS_MUSE.waveband_half_width_SG.to_value(u.AA) / spectral_order
    return np.arange(band - half_width, band + half_width + step, step) * u.AA


def _sg_geometry_tag(sg_detector_kwargs: Mapping[str, object] | None, spectral_order: int) -> str:
    """
    ``nslits<n>_sep<mA>`` of the product names: the number of slits and their spectral
    separation, stated at second order as the MUSE value is.
    """
    geometry = sg_detector_kwargs or {}
    number_of_slits = geometry.get("number_of_slits", DEFAULTS_MUSE.number_of_slits_SG)
    separation = DEFAULTS_MUSE.spectral_slit_separation_SG
    if "dispersion" in geometry:
        # map_response_to_sg_detector takes its default dispersion from the separation; this
        # is that relation the other way round.
        slit_spacing = geometry.get("slit_spacing", DEFAULTS_MUSE.pixels_between_slits_SG)
        separation = geometry["dispersion"] * slit_spacing * spectral_order / 2
    return f"nslits{number_of_slits}_sep{separation.to_value(u.mAA):g}"


def _sum_ci_response(response: xr.Dataset, line_name: str) -> xr.Dataset:
    """
    One imager response: every line of ``response`` summed, named ``line_name``.
    """
    summed = response.detector_response.sum("line", keep_attrs=True).expand_dims(line=[line_name])
    first = response.isel(line=0)
    wavelength_attrs = {"units": str(u.AA)}
    return (
        response.drop_dims("line")
        .assign(detector_response=summed)
        .assign_coords(
            line_wavelength=("line", [float(first.line_wavelength)], wavelength_attrs),
            channel=("line", [float(first.channel)], wavelength_attrs),
            detector_wavelength=("line", [float(first.detector_wavelength)], wavelength_attrs),
        )
    )


def _replace_atomically(path: Path, write) -> None:
    """
    Call ``write`` on a staging file beside ``path`` and rename it into place, so a task
    racing on the same missing file never reads a half-written one.
    """
    # netCDF4 reports a missing directory as "Permission denied".
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = path.with_name(f".{path.stem}.tmp{os.getpid()}{path.suffix}")
    write(staging_path)
    staging_path.replace(path)


def _write_product(
    dataset: xr.Dataset, name: str, save, *, run: _RunInputs, provenance: dict, skip_existing: bool
) -> Path:
    """
    Write ``dataset`` with ``save`` beside its destination and rename it into place,
    never over an existing one: that one is kept if ``skip_existing``, else it raises.
    """
    path = run.output_directory / name
    if skip_existing:
        # Under any date: a run finished after midnight UTC must find what it wrote before.
        existing = sorted(run.output_directory.glob(name.replace(str(run.date), "*")))
        if existing:
            # shortcut: the product was computed again before being skipped here; name the
            # products before computing them once a rerun of a long job costs too much.
            logger.info(f"Keeping the existing {existing[-1].name}")
            return existing[-1]
    elif path.exists():
        msg = f"Refusing to overwrite existing product: {path}; pass skip_existing=True to keep it"
        raise ValueError(msg)
    dataset = dataset.assign_attrs(provenance)
    _replace_atomically(path, functools.partial(save, dataset))
    logger.info(f"Wrote {path.name} {dict(dataset.sizes)}")
    return path
