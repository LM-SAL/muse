"""
Response-function generation: cached line lists and continua, and the SG and CI products
of every band, abundance and plasma point.
"""

import os
from pathlib import Path
from datetime import UTC, datetime
from collections.abc import Mapping, Iterable, Sequence

import numpy as np
import xarray as xr

import astropy.units as u

from muse.instrument.continuum import bin_continuum, create_chianti_continuum, create_fiasco_continuum
from muse.instrument.effective_area import (
    _DN_PER_PHOTON,
    effective_area_wavelength_range,
    read_effective_area,
    select_effective_area_curve,
)
from muse.instrument.linelist import (
    _validate_positive_data_array,
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
#: The grids of a baseline run.
DEFAULT_ABUNDANCES = ("sun_coronal_2021_chianti", "sun_photospheric_2021_asplund")
DEFAULT_PRESSURE = xr.DataArray([3e14, 3e15, 3e16] * u.K / u.cm**3, dims="pressure")
DEFAULT_TEMPERATURE = xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT")
DEFAULT_DOPPLER_VELOCITY = np.arange(-1000, 1010, 10) * u.km / u.s
DEFAULT_CONTINUUM_DOPPLER_VELOCITY = np.arange(-1000, 1000, 200) * u.km / u.s
PHOTON_UNITS = "1e-27 cm5 ph / (Angstrom s)"
DN_UNITS = "1e-27 cm5 DN / (Angstrom s)"
_SG_CHANNELS = [int(channel) for channel in DEFAULTS_MUSE.channel_spectral_order_SG.channel.values]
_MAIN_LINES_SG = dict(zip(_SG_CHANNELS, DEFAULTS_MUSE.main_lines_SG, strict=True))


def load_or_create_line_list(
    cache_directory: str | Path,
    *,
    abundance: str,
    wavelength_range: u.Quantity,
    temperature: xr.DataArray,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    backend: str = "chianti",
    compute_if_missing: bool = True,
    workers: int = 1,
) -> xr.Dataset:
    """
    The line list of one abundance over a wavelength range, cached in
    ``cache_directory``.

    Parameters
    ----------
    cache_directory : `str` or `pathlib.Path`
        Directory of the cache files.
    abundance, wavelength_range, temperature, pressure, density, minimum_abundance
        As for `create_chianti_line_list` and `create_fiasco_line_list`.
    backend : {"chianti", "fiasco"}, optional
        Which builder computes a missing list, by default ChiantiPy.
    compute_if_missing : `bool`, optional
        If `False`, a missing cache raises instead of being computed, so a cluster task
        fails fast when the preparation step was skipped.
    workers : `int`, optional
        Processes for the fiasco backend.

    Returns
    -------
    `xarray.Dataset`
        The line list.
    """
    if backend not in LINE_LIST_BACKENDS:
        msg = f"backend must be one of {sorted(LINE_LIST_BACKENDS)}, got {backend!r}"
        raise ValueError(msg)
    path = _line_list_cache_path(
        cache_directory, backend, abundance, wavelength_range, density_dependent=density is not None
    )
    if path.exists():
        logger.info(f"Loading the line list from {path}")
        return xr.load_dataset(path)
    if not compute_if_missing:
        msg = f"line-list cache {path} does not exist; run the preparation step first"
        raise FileNotFoundError(msg)
    logger.info(f"Computing the {backend} line list for {abundance} over {wavelength_range}")
    kwargs = {"workers": workers} if backend == "fiasco" else {}
    line_list = LINE_LIST_BACKENDS[backend](
        temperature,
        density=density,
        pressure=pressure,
        abundance=abundance,
        wavelength_range=wavelength_range,
        minimum_abundance=minimum_abundance,
        **kwargs,
    )
    # Written beside the destination and renamed, so a task racing on the same missing cache
    # never reads a half-written file.
    staging_path = _staging_path(path)
    encoding = {
        name: {"zlib": True, "complevel": 5}
        for name, variable in line_list.data_vars.items()
        if variable.dtype.kind in "fiu"
    }
    line_list.to_netcdf(staging_path, encoding=encoding)
    staging_path.replace(path)
    return line_list


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
) -> xr.Dataset:
    """
    The rest-frame continuum of one abundance on one wavelength grid, cached in
    ``cache_directory``.

    A cached cube whose temperatures, wavelengths or abundance cut differ from the request is
    recomputed; one holding more pressures or densities than requested serves the requested
    ones. Every band of a run folds the same cube through its own effective area with
    `bin_continuum`.

    Parameters
    ----------
    cache_directory : `str` or `pathlib.Path`
        Directory of the cache files.
    abundance, wavelength_grid, temperature, pressure, density, minimum_abundance
        As for `create_chianti_continuum` and `create_fiasco_continuum`.
    backend : {"chianti", "fiasco"}, optional
        Which calculator computes a missing cube, by default fiasco.
    compute_if_missing : `bool`, optional
        If `False`, a missing cache raises instead of being computed.
    workers : `int`, optional
        Processes for the fiasco backend.

    Returns
    -------
    `xarray.Dataset`
        The continuum on exactly the requested grids.
    """
    if backend not in CONTINUUM_BACKENDS:
        msg = f"backend must be one of {sorted(CONTINUUM_BACKENDS)}, got {backend!r}"
        raise ValueError(msg)
    if (pressure is None) == (density is None):
        msg = "Specify pressure or density"
        raise ValueError(msg)
    logT = np.log10(_validate_positive_data_array(temperature, "temperature", u.K, dimension="logT").data.reshape(-1))
    if density is None:
        plasma_dim = "pressure"
        requested_values = _validate_positive_data_array(pressure, "pressure", u.K / u.cm**3).data.reshape(-1)
    else:
        plasma_dim = "logD"
        requested_values = np.log10(_validate_positive_data_array(density, "density", u.cm**-3).data.reshape(-1))
    grid = np.asarray(wavelength_grid.to_value(u.AA), dtype=float)
    path = _continuum_cache_path(
        cache_directory, backend, abundance, wavelength_grid, density_dependent=density is not None
    )
    if path.exists():
        # Closed after loading, so a recompute can replace the file on Windows too.
        with read_spectral_response(path) as opened:
            cached = opened.load()
        if _continuum_cache_matches(cached, logT, grid, plasma_dim, requested_values, minimum_abundance):
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
    staging_path = _staging_path(path)
    save_spectral_response(continuum, staging_path)
    staging_path.replace(path)
    return continuum


@u.quantity_input(doppler_velocity=u.km / u.s)
def generate_responses(
    *,
    output_directory: str | Path,
    author: str,
    effective_area_note: str,
    sg_effective_areas: Sequence[str | Path] = (),
    ci_effective_areas: Sequence[str | Path] = (),
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
    line_list_backend: str = "chianti",
    line_list_file: str | Path | None = None,
    line_boosts: Mapping[tuple[str, int, int], float] | None = None,
    line_selection: Iterable[tuple[str, int, int]] | None = None,
    ci_fractional_width: float = 0.01,
    include_continuum: bool = True,
    continuum_backend: str = "fiasco",
    ci_units: Sequence[str] = ("ph", "DN"),
    sg_detector_kwargs: Mapping[str, object] | None = None,
    sg_units_kwargs: Mapping[str, object] | None = None,
    ci_units_kwargs: Mapping[str, object] | None = None,
    cache_directory: str | Path | None = None,
    workers: int = 1,
) -> list[Path]:
    """
    Generate the response products of every effective-area file, abundance and plasma
    point.

    For each abundance one line list over the wavelength range of every file is computed or
    read from the cache. Each SG band then gets, per pressure, a waveband cube and a detector
    response per main line, and a cube and a detector response of the brightest other lines
    summed. Each CI band gets, per pressure, a waveband cube of its brightest lines on bins
    of constant velocity width, with the continuum folded in, and that cube summed into one
    imager response, in photons and in DN.

    Parameters
    ----------
    output_directory : `str` or `pathlib.Path`
        Where the products go, as zarr stores named
        ``respfunc_<band>_<product>_<abundance>_<note>_<plasma>_<author>_<date>.zarr``.
    author : `str`
        Initials recorded in the names and attributes.
    effective_area_note : `str`
        Tag of the effective areas in the names, e.g. the coating.
    sg_effective_areas, ci_effective_areas : `list` of path-like
        Effective-area files, one band each, as `create_effective_area` writes them. A CI
        area in cm2 DN / ph carries the camera gain, so only DN products exist for it.
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
        Lines summed into the contaminant products, by default 10000; 0 skips them.
    minimum_abundance : `float`, optional
        Elemental abundance cut of the line list and the continuum.
    line_list_backend : {"chianti", "fiasco"}, optional
        Builder of a missing line list.
    line_list_file : `str` or `pathlib.Path`, optional
        A line list to use instead of the cache.
    line_boosts : `dict`, optional
        Passed to `apply_line_boosts` after the list is loaded, so the cache stays unboosted.
    line_selection : iterable of `tuple`, optional
        Passed to `select_transitions` after the boosts; pair it with ``include_continuum=False``
        and a distinct note when a product should hold those transitions alone.
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
    sg_units_kwargs, ci_units_kwargs : `dict`, optional
        Keyword arguments of `transform_response_units` other than ``new_units`` and
        ``channel``, such as the pixel size, gain and pair energy of another imager.
    cache_directory : `str` or `pathlib.Path`, optional
        Where the line lists and continua are cached, by default ``output_directory / "cache"``.
    workers : `int`, optional
        Processes for the fiasco backends.

    Returns
    -------
    `list` of `pathlib.Path`
        The products written. An SG detector response holds ``detector_response`` over
        ``line``, ``doppler_velocity``, the plasma axis (``pressure`` or ``logD``), ``logT``,
        ``slit`` and ``detector_x_pixel``; a CI one has no slit or pixel axis; a waveband cube
        holds ``spectral_response`` with ``wavelength_bin`` in place of the detector axes. Every
        product carries the band's ``effective_area`` curve.
    """
    for name, value in (("author", author), ("effective_area_note", effective_area_note)):
        if not isinstance(value, str) or not value.strip():
            msg = f"{name} must be a non-empty string"
            raise ValueError(msg)
    ci_units = tuple(ci_units)
    if not ci_units or set(ci_units) - {"ph", "DN"}:
        msg = f"ci_units must be a non-empty subset of ('ph', 'DN'), got {ci_units!r}"
        raise ValueError(msg)
    if line_list_backend not in LINE_LIST_BACKENDS:
        msg = f"line_list_backend must be one of {sorted(LINE_LIST_BACKENDS)}, got {line_list_backend!r}"
        raise ValueError(msg)
    if continuum_backend not in CONTINUUM_BACKENDS:
        msg = f"continuum_backend must be one of {sorted(CONTINUUM_BACKENDS)}, got {continuum_backend!r}"
        raise ValueError(msg)
    run = _RunInputs.resolve(output_directory, cache_directory, pressure, density)
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    doppler_velocity = DEFAULT_DOPPLER_VELOCITY if doppler_velocity is None else doppler_velocity
    sg_areas = [read_effective_area(path) for path in sg_effective_areas]
    ci_areas = [read_effective_area(path) for path in ci_effective_areas]
    if not sg_areas and not ci_areas:
        msg = "pass at least one effective-area file"
        raise ValueError(msg)
    wavelength_range = effective_area_wavelength_range([*sg_areas, *ci_areas])

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
                workers=workers,
            )
        if line_boosts:
            line_list = apply_line_boosts(line_list, line_boosts)
        if line_selection:
            line_list = select_transitions(line_list, line_selection)
        provenance = {
            "abundance": abundance,
            "line_list_backend": line_list_backend,
            "author": author,
            "effective_area_note": effective_area_note,
            "minimum_abundance": minimum_abundance,
            **{
                key: line_list.attrs[key]
                for key in ("Chianti", "Chiantipy", "fiasco", "line_boosts", "line_selection")
                if key in line_list.attrs
            },
        }
        for effective_area in sg_areas:
            band = _nominal_band(effective_area)
            band_settings = {
                "main_lines": list(_per_band(main_lines_by_band, band, _MAIN_LINES_SG, "main_lines_by_band")),
                "spectral_order": int(
                    _per_band(
                        spectral_order_by_band, band, DEFAULTS_MUSE.channel_spectral_order_SG, "spectral_order_by_band"
                    )
                ),
                "instrumental_width": u.Quantity(
                    _per_band(
                        instrumental_width_by_band,
                        band,
                        DEFAULTS_MUSE.instrumental_width_SG,
                        "instrumental_width_by_band",
                    )
                ),
            }
            for plasma_index in range(run.number_of_plasma_points):
                products += _sg_band_products(
                    run,
                    line_list,
                    effective_area,
                    plasma_index,
                    **band_settings,
                    doppler_velocity=doppler_velocity,
                    maximum_contaminants=maximum_contaminants,
                    sg_detector_kwargs=sg_detector_kwargs or {},
                    sg_units_kwargs=sg_units_kwargs or {},
                    name_tail=f"{abundance}_{effective_area_note}_{{plasma}}_{author}_{run.date}",
                    provenance=provenance,
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
                    workers=workers,
                )
            for plasma_index in range(run.number_of_plasma_points):
                products += _ci_band_products(
                    run,
                    line_list,
                    effective_area,
                    plasma_index,
                    continuum=continuum,
                    continuum_note=f"{continuum_backend} free-free + free-bound + two-photon, minimum abundance {minimum_abundance}",
                    doppler_velocity=doppler_velocity,
                    maximum_contaminants=maximum_contaminants,
                    ci_fractional_width=ci_fractional_width,
                    ci_units=ci_units,
                    ci_units_kwargs=ci_units_kwargs or {},
                    name_tail=f"{abundance}_{effective_area_note}_{{plasma}}_{author}_{run.date}",
                    provenance=provenance,
                )
    return products


@u.quantity_input(doppler_velocity=u.km / u.s, wavelength_step=u.AA)
def generate_continuum_responses(
    *,
    output_directory: str | Path,
    author: str,
    effective_area_note: str,
    sg_effective_areas: Sequence[str | Path],
    abundance: str = DEFAULT_ABUNDANCES[0],
    backend: str = "fiasco",
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
    workers: int = 1,
) -> list[Path]:
    """
    Generate the SG continuum response of every band: the continuum alone on the
    detector.

    The continuum counterpart of `generate_responses`: per band the rest-frame continuum on a
    fine grid over the waveband (cached by `load_or_create_continuum`) is folded through the
    band's effective area with `bin_continuum`, converted to photons and mapped onto the SG
    detector.

    Parameters
    ----------
    output_directory, author, effective_area_note, sg_effective_areas
        As for `generate_responses`.
    temperature, pressure, density, minimum_abundance, spectral_order_by_band, sg_detector_kwargs, sg_units_kwargs, cache_directory, workers
        As for `generate_responses`.
    abundance : `str`, optional
        One CHIANTI abundance name.
    backend : {"chianti", "fiasco"}, optional
        The continuum calculator, by default fiasco; it is part of the product names.
    doppler_velocity : `astropy.units.Quantity`, optional
        By default -1000 to 800 km/s in steps of 200.
    wavelength_step : `astropy.units.Quantity`, optional
        Sampling of the fine grid, by default 0.01 Angstrom.

    Returns
    -------
    `list` of `pathlib.Path`
        The products written, named ``respfunc_<band>_continuum_<backend>_...`` like the line
        products, with the three continuum components as ``line`` and the SG detector axes.
    """
    if backend not in CONTINUUM_BACKENDS:
        msg = f"backend must be one of {sorted(CONTINUUM_BACKENDS)}, got {backend!r}"
        raise ValueError(msg)
    run = _RunInputs.resolve(output_directory, cache_directory, pressure, density)
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    doppler_velocity = DEFAULT_CONTINUUM_DOPPLER_VELOCITY if doppler_velocity is None else doppler_velocity
    step = wavelength_step.to_value(u.AA)
    plasma_tag = run.plasma_tag(0) if run.number_of_plasma_points == 1 else "pressure"
    provenance = {
        "abundance": abundance,
        "continuum_backend": backend,
        "author": author,
        "effective_area_note": effective_area_note,
        "minimum_abundance": minimum_abundance,
    }
    products = []
    for path in sg_effective_areas:
        effective_area = read_effective_area(path)
        band = _nominal_band(effective_area)
        area_curve = select_effective_area_curve(effective_area)
        spectral_order = int(
            _per_band(spectral_order_by_band, band, DEFAULTS_MUSE.channel_spectral_order_SG, "spectral_order_by_band")
        )
        half_width = DEFAULTS_MUSE.waveband_half_width_SG.to_value(u.AA) / spectral_order
        grid = np.arange(band - half_width, band + half_width, step) * u.AA
        continuum = load_or_create_continuum(
            run.cache_directory,
            abundance=abundance,
            wavelength_grid=grid,
            temperature=temperature,
            pressure=run.pressure,
            density=density,
            minimum_abundance=minimum_abundance,
            backend=backend,
            workers=workers,
        )
        binned_continuum = bin_continuum(
            continuum,
            _centers_to_edges(grid.value) * u.AA,
            effective_area=area_curve,
            doppler_velocity=doppler_velocity,
        )
        photons = transform_response_units(
            binned_continuum, PHOTON_UNITS, band * u.AA, detector="SG", **(sg_units_kwargs or {})
        )
        detector_response = map_response_to_sg_detector(
            photons, channel=band * u.AA, **(sg_detector_kwargs or {})
        ).assign(effective_area=area_curve)
        name = f"respfunc_{band}_continuum_{backend}_{_sg_geometry_tag(sg_detector_kwargs)}_{abundance}_{effective_area_note}_{plasma_tag}_{author}_{run.date}.zarr"
        products.append(_write_product(detector_response, run.output_directory / name, provenance, save_response))
    return products


class _RunInputs:
    """
    The directories and the plasma grid of one run, resolved once.
    """

    def __init__(
        self, output_directory: Path, cache_directory: Path, pressure: xr.DataArray | None, density: xr.DataArray | None
    ) -> None:
        self.output_directory = output_directory
        self.cache_directory = cache_directory
        self.pressure = pressure
        self.density = density
        self.date = datetime.now(UTC).date()
        self._pressures = (
            None
            if pressure is None
            else _validate_positive_data_array(pressure, "pressure", u.K / u.cm**3).data.reshape(-1)
        )
        self.number_of_plasma_points = 1 if density is not None else self._pressures.size

    @classmethod
    def resolve(cls, output_directory, cache_directory, pressure, density) -> "_RunInputs":
        if pressure is not None and density is not None:
            msg = "pressure and density are mutually exclusive"
            raise ValueError(msg)
        if pressure is None and density is None:
            pressure = DEFAULT_PRESSURE
        output_directory = Path(output_directory)
        cache_directory = output_directory / "cache" if cache_directory is None else Path(cache_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        cache_directory.mkdir(parents=True, exist_ok=True)
        return cls(output_directory, cache_directory, pressure, density)

    def plasma_tag(self, plasma_index: int) -> str:
        """
        ``density`` for a density run, else the pressure of the plasma point as
        ``pressure_3e+15``.
        """
        return "density" if self.density is not None else f"pressure_{self._pressures[plasma_index]:.0e}"

    def line_list_at(self, line_list: xr.Dataset, plasma_index: int) -> xr.Dataset:
        """
        The line list restricted to one plasma point, the whole list for a density run.
        """
        return line_list if self.density is not None else line_list.isel(pressure=[plasma_index])


def _sg_band_products(
    run: _RunInputs,
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
    provenance: dict,
) -> list[Path]:
    """
    The main-line and contaminant products of one SG band at one plasma point.
    """
    band = _nominal_band(effective_area)
    channel = band * u.AA
    area_curve = select_effective_area_curve(effective_area)
    half_width = DEFAULTS_MUSE.waveband_half_width_SG.to_value(u.AA) / spectral_order
    step = DEFAULTS_MUSE.wavelength_step_SG.to_value(u.AA)
    grid = np.arange(band - half_width, band + half_width + step, step) * u.AA
    geometry_tag = _sg_geometry_tag(sg_detector_kwargs)
    tail = name_tail.format(plasma=run.plasma_tag(plasma_index))
    line_list = run.line_list_at(line_list, plasma_index)
    logger.info(f"SG band {band}, {tail}: main lines {main_lines}")

    main_response = create_spectral_response(
        line_list,
        grid,
        main_lines=main_lines,
        instrumental_width=instrumental_width,
        doppler_velocity=doppler_velocity,
        effective_area=area_curve,
    )
    main_response = transform_response_units(main_response, PHOTON_UNITS, channel, detector="SG", **sg_units_kwargs)
    main_detector_response = map_response_to_sg_detector(main_response, channel=channel, **sg_detector_kwargs).assign(
        effective_area=area_curve
    )
    products = []
    for name in main_lines:
        line_tag = "".join(name.split())
        products.append(
            _write_product(
                main_response.sel(line=[name]).assign(effective_area=area_curve),
                run.output_directory / f"respfunc_{band}_{line_tag}_waveband_{tail}.zarr",
                provenance,
                save_spectral_response,
            )
        )
        products.append(
            _write_product(
                main_detector_response.sel(line=[name]),
                run.output_directory / f"respfunc_{band}_{line_tag}_{geometry_tag}_{tail}.zarr",
                provenance,
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
    contaminants = transform_response_units(contaminants, PHOTON_UNITS, channel, detector="SG", **sg_units_kwargs)
    wavelength_attrs = {"units": str(u.AA)}
    contaminant_cube = contaminants.assign_coords(
        line=[f"{band} remaining {ranked.sizes['trans_index']} lines"],
        line_wavelength=("line", [float(ranked.wavelength[0])], wavelength_attrs),
    )
    products.append(
        _write_product(
            contaminant_cube.assign(effective_area=area_curve),
            run.output_directory / f"respfunc_{band}_nomainlines_waveband_{tail}.zarr",
            provenance,
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
        _write_product(
            contaminant_detector_response,
            run.output_directory / f"respfunc_{band}_L_{geometry_tag}_{tail}.zarr",
            provenance,
            save_response,
        )
    )
    return products


def _ci_band_products(
    run: _RunInputs,
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
    ci_units_kwargs: Mapping[str, object],
    name_tail: str,
    provenance: dict,
) -> list[Path]:
    """
    The waveband cube and the summed imager response of one CI band at one plasma point.
    """
    band = _nominal_band(effective_area)
    channel = band * u.AA
    area_curve = select_effective_area_curve(effective_area)
    area_in_dn = u.Unit(area_curve.attrs["units"]).is_equivalent(_DN_PER_PHOTON)
    if area_in_dn:
        if "DN" not in ci_units:
            msg = f"the CI area of band {band} carries the camera gain (cm2 DN / ph), so only DN products exist; add 'DN' to ci_units"
            raise ValueError(msg)
        # The gain is already in the area: build in cm2 and label the photon product DN.
        area_curve = area_curve.assign_attrs(units=str(u.Unit(area_curve.attrs["units"]) / (u.DN / u.ph)))
    wavelengths = area_curve.wavelength.values
    edges = create_log_wavelength_edges([wavelengths[0], wavelengths[-1]] * u.AA, ci_fractional_width)
    plasma_tag = run.plasma_tag(plasma_index)
    line_list = run.line_list_at(line_list, plasma_index)
    ranked = rank_line_list(line_list, area_curve).isel(trans_index=slice(maximum_contaminants))
    number_of_lines = ranked.sizes["trans_index"]
    logger.info(f"CI band {band}, {name_tail.format(plasma=plasma_tag)}: {number_of_lines} lines")

    lines = create_spectral_response(
        ranked,
        main_lines=None,
        include_contaminants=True,
        doppler_velocity=doppler_velocity,
        effective_area=area_curve,
        wavelength_edges=edges,
        bin_method="average",
    )
    ci_line_name = f"{band} remaining {number_of_lines} lines"
    lines = lines.assign_coords(
        line=[ci_line_name], line_wavelength=("line", [float(ranked.wavelength[0])], {"units": str(u.AA)})
    )
    parts = [lines]
    if continuum is not None:
        continuum_subset = continuum if run.density is not None else continuum.isel(pressure=[plasma_index])
        binned_continuum = bin_continuum(
            continuum_subset, edges, effective_area=area_curve, doppler_velocity=doppler_velocity
        )
        binned_continuum = binned_continuum.assign_coords(
            line=[f"{band} continuum {name}" for name in binned_continuum.line.values]
        )
        parts.append(binned_continuum.transpose(*lines.spectral_response.dims, "bound", missing_dims="ignore"))
        ci_line_name += " + continuum"
    ci_cube = xr.concat(parts, dim="line", data_vars="all", combine_attrs="override")
    ci_cube.attrs.update(lines.attrs)
    if continuum is not None:
        ci_cube.attrs["continuum"] = continuum_note
    photons = transform_response_units(ci_cube, PHOTON_UNITS, channel, detector="CI", **ci_units_kwargs)
    cubes_by_units = {}
    if area_in_dn:
        photons.spectral_response.attrs["units"] = DN_UNITS
        cubes_by_units["_DN"] = photons
    else:
        if "ph" in ci_units:
            cubes_by_units[""] = photons
        if "DN" in ci_units:
            cubes_by_units["_DN"] = transform_response_units(
                ci_cube, DN_UNITS, channel, detector="CI", **ci_units_kwargs
            )
    products = []
    for units_tag, cube in cubes_by_units.items():
        tail = name_tail.format(plasma=f"{plasma_tag}{units_tag}")
        products.append(
            _write_product(
                cube.assign(effective_area=area_curve),
                run.output_directory / f"respfunc_ci_{band}_waveband_{tail}.zarr",
                provenance,
                save_spectral_response,
            )
        )
        summed = _sum_ci_response(map_response_to_ci_detector(cube, channel), ci_line_name).assign(
            effective_area=area_curve
        )
        products.append(
            _write_product(summed, run.output_directory / f"respfunc_ci_{band}_{tail}.zarr", provenance, save_response)
        )
    return products


def _line_list_cache_path(
    cache_directory, backend: str, abundance: str, wavelength_range: u.Quantity, *, density_dependent: bool
) -> Path:
    lower, upper = (int(value) for value in wavelength_range.to_value(u.AA))
    suffix = "_density" if density_dependent else ""
    return Path(cache_directory) / f"line_list_{backend}_{lower}-{upper}A_{abundance}{suffix}.nc"


def _continuum_cache_path(
    cache_directory, backend: str, abundance: str, wavelength_grid: u.Quantity, *, density_dependent: bool
) -> Path:
    values = wavelength_grid.to_value(u.AA)
    suffix = "_density" if density_dependent else ""
    return (
        Path(cache_directory)
        / f"continuum_{backend}_{int(values[0])}-{int(values[-1])}A_n{values.size}_{abundance}{suffix}.nc"
    )


def _continuum_cache_matches(
    cached: xr.Dataset,
    logT: np.ndarray,
    grid: np.ndarray,
    plasma_dim: str,
    requested_values: np.ndarray,
    minimum_abundance: float,
) -> bool:
    return (
        cached.sizes["logT"] == logT.size
        and np.allclose(cached.logT.values, logT, atol=1e-6)
        and cached.sizes["wavelength_bin"] == grid.size
        and np.allclose(cached.wavelength_grid.values, grid)
        and plasma_dim in cached.dims
        and all(np.isclose(cached[plasma_dim].values, value, rtol=1e-9, atol=0).any() for value in requested_values)
        and cached.attrs.get("minimum_abundance") == minimum_abundance
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


def _per_band(overrides: Mapping[int, object] | None, band: int, defaults, name: str):
    """
    The override for ``band``, else its MUSE default (a mapping or a ``channel``-indexed
    array).
    """
    if overrides and band in overrides:
        return overrides[band]
    try:
        return defaults[band] if isinstance(defaults, Mapping) else defaults.sel(channel=band).data
    except KeyError:
        msg = f"no MUSE default for band {band}; pass {name}={{{band}: ...}}"
        raise ValueError(msg) from None


def _sg_geometry_tag(sg_detector_kwargs: Mapping[str, object] | None) -> str:
    """
    ``nslits<n>_sep<mA>`` of the product names, with the MUSE spectral slit separation.
    """
    number_of_slits = (sg_detector_kwargs or {}).get("number_of_slits", DEFAULTS_MUSE.number_of_slits_SG)
    separation = DEFAULTS_MUSE.spectral_slit_separation_SG.to_value(u.mAA)
    return f"nslits{number_of_slits}_sep{separation:g}"


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


def _staging_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{os.getpid()}{path.suffix}")


def _write_product(dataset: xr.Dataset, path: Path, provenance: dict, save) -> Path:
    """
    Write a product with ``save`` beside its destination and rename it into place, never
    over an existing one.
    """
    if path.exists():
        msg = f"Refusing to overwrite existing product: {path}"
        raise ValueError(msg)
    dataset = dataset.assign_attrs(provenance)
    staging_path = _staging_path(path)
    save(dataset, staging_path)
    staging_path.replace(path)
    logger.info(f"Wrote {path.name} {dict(dataset.sizes)}")
    return path
