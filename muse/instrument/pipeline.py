"""
Response-function generation: cached line lists and continua, and the SG and CI products
of every band, abundance and plasma point, in one run or one cluster task at a time.
"""

import os
from pathlib import Path
from datetime import UTC, datetime

import numpy as np
import xarray as xr

import astropy.units as u

from muse.instrument.continuum import bin_continuum, create_chianti_continuum, create_fiasco_continuum
from muse.instrument.effective_area import effective_area_curve, read_effective_area
from muse.instrument.linelist import (
    _validate_positive_data_array,
    apply_line_boosts,
    create_chianti_line_list,
    create_fiasco_line_list,
    rank_line_list,
    select_lines,
)
from muse.instrument.radiometry import transform_response_units
from muse.instrument.response import map_response_to_ci_detector, map_response_to_sg_detector
from muse.instrument.response_io import read_spectral_response, save_response, save_spectral_response
from muse.instrument.spectral import _centers_to_edges, create_spectral_response, log_wavelength_edges
from muse.log import logger
from muse.variables import DEFAULTS_MUSE

__all__ = [
    "CONTINUUM_BACKENDS",
    "LINE_LIST_BACKENDS",
    "continuum_cache_path",
    "continuum_response_pipeline",
    "enumerate_tasks",
    "get_continuum",
    "get_line_list",
    "line_list_cache_path",
    "response_pipeline",
    "task_id_from_environment",
]

#: Line-list builders by backend name.
LINE_LIST_BACKENDS = {"chianti": create_chianti_line_list, "fiasco": create_fiasco_line_list}
#: Continuum calculators by backend name.
CONTINUUM_BACKENDS = {"chianti": create_chianti_continuum, "fiasco": create_fiasco_continuum}
PHOTON_UNITS = "1e-27 cm5 ph / (Angstrom s)"
DN_UNITS = "1e-27 cm5 DN / (Angstrom s)"
DEFAULT_ABUNDANCES = ("sun_coronal_2021_chianti", "sun_photospheric_2021_asplund")
_SG_WAVELENGTH_STEP = 0.0049  # Angstrom, the SG detector pixel over three
_SG_HALF_WIDTH = 35.0  # Angstrom at first order: the waveband each SG band covers
_DN_PER_PHOTON = u.cm**2 * u.DN / u.ph


def enumerate_tasks(*, sg_effective_areas, ci_effective_areas, abundances, n_pressure: int) -> list[dict]:
    """
    The independent units of work of `response_pipeline`, in its iteration order.

    One task is one effective-area file at one pressure index for one abundance, so a
    cluster job array maps its index straight to ``tasks[index]``.

    Parameters
    ----------
    sg_effective_areas, ci_effective_areas : sequence of path-like
        The SG and CI effective-area files.
    abundances : sequence of `str`
        CHIANTI abundance names.
    n_pressure : `int`
        Number of pressures, 1 for a density run.

    Returns
    -------
    `list` of `dict`
        Each with ``detector`` ("SG" or "CI"), ``abundance``, ``effective_area`` and ``pressure_index``.
    """
    return [
        {"detector": detector, "abundance": abundance, "effective_area": Path(path), "pressure_index": i_pressure}
        for abundance in abundances
        for detector, paths in (("SG", sg_effective_areas), ("CI", ci_effective_areas))
        for path in paths
        for i_pressure in range(n_pressure)
    ]


def task_id_from_environment() -> int | None:
    """
    The job-array index from ``SLURM_ARRAY_TASK_ID`` or ``PBS_ARRAY_INDEX``, or `None`
    outside an array job.
    """
    for name in ("SLURM_ARRAY_TASK_ID", "PBS_ARRAY_INDEX"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return None


@u.quantity_input(wavelength_range=u.AA)
def line_list_cache_path(
    cache_dir, *, backend: str, abundance: str, wavelength_range: u.Quantity, density_dependent: bool = False
) -> Path:
    """
    The file `get_line_list` caches one abundance's line list in.
    """
    lower, upper = (int(value) for value in wavelength_range.to_value(u.AA))
    suffix = "_density" if density_dependent else ""
    return Path(cache_dir) / f"line_list_{backend}_{lower}-{upper}A_{abundance}{suffix}.nc"


@u.quantity_input(wavelength_grid=u.AA)
def continuum_cache_path(
    cache_dir, *, backend: str, abundance: str, wavelength_grid: u.Quantity, density_dependent: bool = False
) -> Path:
    """
    The file `get_continuum` caches one abundance's continuum on one wavelength grid in.
    """
    values = wavelength_grid.to_value(u.AA)
    suffix = "_density" if density_dependent else ""
    return (
        Path(cache_dir)
        / f"continuum_{backend}_{int(values[0])}-{int(values[-1])}A_n{values.size}_{abundance}{suffix}.nc"
    )


def get_line_list(
    cache_dir,
    *,
    abundance: str,
    wavelength_range: u.Quantity,
    temperature: xr.DataArray,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    backend: str = "chianti",
    line_list_file=None,
    compute_if_missing: bool = True,
    workers: int = 1,
) -> xr.Dataset:
    """
    The line list of one abundance over the whole wavelength range, cached in
    ``cache_dir``.

    Parameters
    ----------
    cache_dir : path-like
        Directory of the cache files, named by `line_list_cache_path`.
    abundance, wavelength_range, temperature, pressure, density, minimum_abundance
        As for `create_chianti_line_list` and `create_fiasco_line_list`.
    backend : {"chianti", "fiasco"}, optional
        Which builder computes a missing list, by default ChiantiPy.
    line_list_file : path-like, optional
        A line list to load instead, bypassing the cache.
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
    if line_list_file is not None:
        return xr.load_dataset(line_list_file)
    path = line_list_cache_path(
        cache_dir,
        backend=backend,
        abundance=abundance,
        wavelength_range=wavelength_range,
        density_dependent=density is not None,
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
    staged = _staged(path)
    encoding = {
        name: {"zlib": True, "complevel": 5} for name, var in line_list.data_vars.items() if var.dtype.kind in "fiu"
    }
    line_list.to_netcdf(staged, encoding=encoding)
    staged.replace(path)
    return line_list


def get_continuum(
    cache_dir,
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
    ``cache_dir``.

    A cached cube whose temperatures, wavelengths or abundance cut differ from the request is
    recomputed; one holding more pressures or densities than requested serves the requested
    ones. Every band of a run folds the same cube through its own effective area with
    `bin_continuum`.

    Parameters
    ----------
    cache_dir : path-like
        Directory of the cache files, named by `continuum_cache_path`.
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
        plasma, requested = (
            "pressure",
            _validate_positive_data_array(pressure, "pressure", u.K / u.cm**3).data.reshape(-1),
        )
    else:
        plasma, requested = (
            "logD",
            np.log10(_validate_positive_data_array(density, "density", u.cm**-3).data.reshape(-1)),
        )
    grid = np.asarray(wavelength_grid.to_value(u.AA), dtype=float)
    path = continuum_cache_path(
        cache_dir,
        backend=backend,
        abundance=abundance,
        wavelength_grid=wavelength_grid,
        density_dependent=density is not None,
    )
    if path.exists():
        cached = read_spectral_response(path).load()
        matches = (
            cached.sizes["logT"] == logT.size
            and np.allclose(cached.logT.values, logT, atol=1e-6)
            and cached.sizes["wavelength_bin"] == grid.size
            and np.allclose(cached.wavelength_grid.values, grid)
            and plasma in cached.dims
            and all(np.isclose(cached[plasma].values, value, rtol=1e-9, atol=0).any() for value in requested)
            and cached.attrs.get("minimum_abundance") == minimum_abundance
        )
        if matches:
            logger.info(f"Loading the continuum from {path}")
            return cached.assign_coords(logT=cached.logT.copy(data=logT)).sel({plasma: requested}, method="nearest")
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
    staged = _staged(path)
    save_spectral_response(continuum, staged)
    staged.replace(path)
    return continuum


@u.quantity_input(doppler_velocity=u.km / u.s)
def response_pipeline(
    *,
    output_dir,
    author: str,
    effective_area_note: str,
    sg_effective_areas=(),
    ci_effective_areas=(),
    abundances=DEFAULT_ABUNDANCES,
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    doppler_velocity: u.Quantity | None = None,
    main_lines: dict | None = None,
    spectral_order: dict | None = None,
    instrumental_width: dict | None = None,
    n_contaminants: int = 10000,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    line_list_backend: str = "chianti",
    line_list_file=None,
    line_boosts: dict | None = None,
    line_selection=None,
    ci_fractional_width: float | None = 0.01,
    include_continuum: bool = True,
    continuum_backend: str = "fiasco",
    ci_units=("ph", "DN"),
    sg_geometry: dict | None = None,
    sg_radiometry: dict | None = None,
    ci_radiometry: dict | None = None,
    cache_dir=None,
    workers: int = 1,
) -> list[Path]:
    """
    Generate the response products of every effective-area file, abundance and plasma
    point.

    For each abundance one line list over the wavelength range of every file is computed or
    read from the cache. Each SG band then gets, per pressure, a waveband cube and a detector
    response per main line, and a cube and a detector response of the brightest
    ``n_contaminants`` other lines summed. Each CI band gets, per pressure, a waveband cube
    of its brightest lines on bins of constant velocity width, with the continuum folded in,
    and that cube summed into one imager response, in photons and in DN.

    Parameters
    ----------
    output_dir : path-like
        Where the products go, as zarr stores named
        ``respfunc_<band>_<product>_<abundance>_<note>_<plasma>_<author>_<date>.zarr``.
    author : `str`
        Initials recorded in the names and attributes.
    effective_area_note : `str`
        Tag of the effective areas in the names, e.g. the coating.
    sg_effective_areas, ci_effective_areas : sequence of path-like
        Effective-area files, one band each, as `create_effective_area` writes them. A CI
        area in cm2 DN / ph carries the camera gain, so only DN products exist for it.
    abundances : sequence of `str`, optional
        CHIANTI abundance names, by default the coronal and photospheric 2021 sets.
    temperature : `xarray.DataArray`, optional
        Temperatures with a ``logT`` dimension, by default ``logT`` 4.5 to 7.9 in steps of 0.1.
    pressure, density : `xarray.DataArray`, optional
        The plasma grid, by default the pressures 3e14, 3e15 and 3e16 K / cm3. A density grid
        is computed in one go and tagged ``density``.
    doppler_velocity : `astropy.units.Quantity`, optional
        By default -1000 to 1000 km/s in steps of 10.
    main_lines, spectral_order, instrumental_width : `dict`, optional
        Per SG band (its nominal wavelength in Angstrom as an `int`): the ``full_name`` values
        of the main lines, the spectral order and the instrumental width; by default the MUSE
        values from `~muse.variables.DEFAULTS_MUSE`.
    n_contaminants : `int`, optional
        Lines summed into the contaminant products, by default 10000; 0 skips them.
    minimum_abundance : `float`, optional
        Elemental abundance cut of the line list and the continuum.
    line_list_backend : {"chianti", "fiasco"}, optional
        Builder of a missing line list.
    line_list_file : path-like, optional
        A line list to use instead of the cache.
    line_boosts : `dict`, optional
        Passed to `apply_line_boosts` after the list is loaded, so the cache stays unboosted.
    line_selection : iterable of `tuple`, optional
        Passed to `select_lines` after the boosts; pair it with ``include_continuum=False``
        and a distinct note when a product should hold those lines alone.
    ci_fractional_width : `float`, optional
        Width of the CI bins relative to their lower edge, by default 0.01; `None` uses the
        fine uniform SG grid.
    include_continuum : `bool`, optional
        Fold the continuum into the CI products, by default `True`.
    continuum_backend : {"chianti", "fiasco"}, optional
        Calculator of a missing continuum, by default fiasco.
    ci_units : sequence of `str`, optional
        The CI products to write, a non-empty subset of ``("ph", "DN")``.
    sg_geometry : `dict`, optional
        Keyword arguments of `map_response_to_sg_detector` other than ``channel``.
    sg_radiometry, ci_radiometry : `dict`, optional
        Keyword arguments of `transform_response_units` other than ``new_units`` and
        ``channel``, such as the pixel size and gain of another imager.
    cache_dir : path-like, optional
        Where the line lists and continua are cached, by default ``output_dir / "cache"``.
    workers : `int`, optional
        Processes for the fiasco backends.

    Returns
    -------
    `list` of `pathlib.Path`
        The products written.
    """
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache" if cache_dir is None else Path(cache_dir)
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
    if pressure is not None and density is not None:
        msg = "pressure and density are mutually exclusive"
        raise ValueError(msg)
    if pressure is None and density is None:
        pressure = xr.DataArray([3e14, 3e15, 3e16] * u.K / u.cm**3, dims="pressure")
    if temperature is None:
        temperature = xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT")
    if doppler_velocity is None:
        doppler_velocity = np.arange(-1000, 1010, 10) * u.km / u.s
    sg_areas = [read_effective_area(path) for path in sg_effective_areas]
    ci_areas = [read_effective_area(path) for path in ci_effective_areas]
    if not sg_areas and not ci_areas:
        msg = "pass at least one effective-area file"
        raise ValueError(msg)
    wavelength_range = [
        min(float(area.wavelength.min()) for area in (*sg_areas, *ci_areas)),
        max(float(area.wavelength.max()) for area in (*sg_areas, *ci_areas)),
    ] * u.AA
    n_plasma = 1 if density is not None else pressure.sizes[pressure.dims[0]]
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(UTC).date()
    written = []

    for abundance in abundances:
        line_list = get_line_list(
            cache_dir,
            abundance=abundance,
            wavelength_range=wavelength_range,
            temperature=temperature,
            pressure=pressure,
            density=density,
            minimum_abundance=minimum_abundance,
            backend=line_list_backend,
            line_list_file=line_list_file,
            workers=workers,
        )
        if line_boosts:
            line_list = apply_line_boosts(line_list, line_boosts)
        if line_selection:
            line_list = select_lines(line_list, line_selection)
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

        for area in sg_areas:
            band = _band(area)
            channel = band * u.AA
            curve = effective_area_curve(area)
            order = _per_band(spectral_order, band, DEFAULTS_MUSE.channel_spectral_order_SG, "spectral_order")
            width = u.Quantity(
                _per_band(instrumental_width, band, DEFAULTS_MUSE.instrumental_width_SG, "instrumental_width")
            )
            names = list((main_lines or {}).get(band) or _default_main_lines(band))
            grid = (
                np.arange(
                    band - _SG_HALF_WIDTH / order,
                    band + _SG_HALF_WIDTH / order + _SG_WAVELENGTH_STEP,
                    _SG_WAVELENGTH_STEP,
                )
                * u.AA
            )
            geometry = _sg_geometry_tag(sg_geometry)
            for i_plasma in range(n_plasma):
                subset = line_list if density is not None else line_list.isel(pressure=[i_plasma])
                stem = f"{abundance}_{effective_area_note}_{_plasma_tag(pressure, i_plasma, density)}_{author}_{date}"
                logger.info(
                    f"SG band {band}, {abundance}, {_plasma_tag(pressure, i_plasma, density)}: main lines {names}"
                )
                mains = create_spectral_response(
                    subset,
                    grid,
                    main_lines=names,
                    instrumental_width=width,
                    doppler_velocity=doppler_velocity,
                    effective_area=curve,
                )
                mains = transform_response_units(mains, PHOTON_UNITS, channel, detector="SG", **(sg_radiometry or {}))
                detector = map_response_to_sg_detector(mains, channel=channel, **(sg_geometry or {}))
                for name in names:
                    label = "".join(name.split())
                    written.append(
                        _write(
                            mains.sel(line=[name]),
                            output_dir / f"respfunc_{band}_{label}_waveband_{stem}.zarr",
                            provenance,
                            spectral=True,
                        )
                    )
                    written.append(
                        _write(
                            detector.sel(line=[name]).assign(effective_area=curve),
                            output_dir / f"respfunc_{band}_{label}_{geometry}_{stem}.zarr",
                            provenance,
                        )
                    )
                if n_contaminants <= 0:
                    continue
                in_band = (
                    (subset.wavelength >= grid[0].value)
                    & (subset.wavelength <= grid[-1].value)
                    & ~subset.full_name.isin(names)
                )
                ranked = rank_line_list(subset.isel(trans_index=in_band.values), curve).isel(
                    trans_index=slice(n_contaminants)
                )
                if ranked.sizes["trans_index"] == 0:
                    logger.warning(f"SG band {band}: no contaminant line in the waveband")
                    continue
                contaminants = create_spectral_response(
                    ranked,
                    grid,
                    main_lines=None,
                    include_contaminants=True,
                    instrumental_width=width,
                    doppler_velocity=doppler_velocity,
                    effective_area=curve,
                )
                contaminants = transform_response_units(
                    contaminants, PHOTON_UNITS, channel, detector="SG", **(sg_radiometry or {})
                )
                contaminants = contaminants.assign_coords(
                    line=[f"{band} remaining {ranked.sizes['trans_index']} lines"]
                )
                cube = contaminants.assign_coords(
                    line_wavelength=("line", [float(ranked.wavelength[0])], {"units": str(u.AA)})
                )
                written.append(
                    _write(
                        cube,
                        output_dir / f"respfunc_{band}_nomainlines_waveband_{stem}.zarr",
                        provenance,
                        spectral=True,
                    )
                )
                # On the detector the contaminants are referenced to the band's main line, as the
                # line products are, so a consumer derives one geometry per band.
                referenced = contaminants.assign_coords(
                    line_wavelength=("line", [float(mains.line_wavelength[0])], {"units": str(u.AA)})
                )
                detector = map_response_to_sg_detector(referenced, channel=channel, **(sg_geometry or {})).assign(
                    effective_area=curve
                )
                written.append(_write(detector, output_dir / f"respfunc_{band}_L_{geometry}_{stem}.zarr", provenance))

        for area in ci_areas:
            band = _band(area)
            channel = band * u.AA
            curve = effective_area_curve(area)
            area_in_dn = u.Unit(curve.attrs["units"]).is_equivalent(_DN_PER_PHOTON)
            if area_in_dn:
                if "DN" not in ci_units:
                    msg = f"the CI area of band {band} carries the camera gain (cm2 DN / ph), so only DN products exist; add 'DN' to ci_units"
                    raise ValueError(msg)
                # The gain is already in the area: build in cm2 and label the photon product DN.
                curve = curve.assign_attrs(units=str(u.Unit(curve.attrs["units"]) / (u.DN / u.ph)))
            wavelengths = curve.wavelength.values
            if ci_fractional_width is None:
                edges = (
                    _centers_to_edges(
                        np.arange(wavelengths[0], wavelengths[-1] + _SG_WAVELENGTH_STEP, _SG_WAVELENGTH_STEP)
                    )
                    * u.AA
                )
            else:
                edges = log_wavelength_edges(wavelengths[0] * u.AA, wavelengths[-1] * u.AA, ci_fractional_width)
            continuum = None
            if include_continuum:
                continuum = get_continuum(
                    cache_dir,
                    abundance=abundance,
                    wavelength_grid=wavelengths * u.AA,
                    temperature=temperature,
                    pressure=pressure,
                    density=density,
                    minimum_abundance=minimum_abundance,
                    backend=continuum_backend,
                    workers=workers,
                )
            for i_plasma in range(n_plasma):
                subset = line_list if density is not None else line_list.isel(pressure=[i_plasma])
                plasma_tag = _plasma_tag(pressure, i_plasma, density)
                ranked = rank_line_list(subset, curve).isel(trans_index=slice(n_contaminants))
                n_lines = ranked.sizes["trans_index"]
                logger.info(f"CI band {band}, {abundance}, {plasma_tag}: {n_lines} lines")
                lines = create_spectral_response(
                    ranked,
                    main_lines=None,
                    include_contaminants=True,
                    doppler_velocity=doppler_velocity,
                    effective_area=curve,
                    wavelength_edges=edges,
                    profile="integrate",
                )
                label = f"{band} remaining {n_lines} lines"
                lines = lines.assign_coords(
                    line=[label], line_wavelength=("line", [float(ranked.wavelength[0])], {"units": str(u.AA)})
                )
                parts = [lines]
                if continuum is not None:
                    cube = continuum if density is not None else continuum.isel(pressure=[i_plasma])
                    folded = bin_continuum(cube, curve, edges, doppler_velocity=doppler_velocity)
                    folded = folded.assign_coords(line=[f"{band} continuum {name}" for name in folded.line.values])
                    parts.append(folded.transpose(*lines.spectral_response.dims, "bound", missing_dims="ignore"))
                    label += " + continuum"
                cube = xr.concat(parts, dim="line", data_vars="all", combine_attrs="override")
                cube.attrs.update(lines.attrs)
                if continuum is not None:
                    cube.attrs["continuum"] = (
                        f"{continuum_backend} free-free + free-bound + two-photon, minimum abundance {minimum_abundance}"
                    )
                photons = transform_response_units(cube, PHOTON_UNITS, channel, detector="CI", **(ci_radiometry or {}))
                products = {}
                if area_in_dn:
                    photons.spectral_response.attrs["units"] = DN_UNITS
                    products["_DN"] = photons
                else:
                    if "ph" in ci_units:
                        products[""] = photons
                    if "DN" in ci_units:
                        products["_DN"] = transform_response_units(
                            cube, DN_UNITS, channel, detector="CI", **(ci_radiometry or {})
                        )
                for units_tag, product in products.items():
                    stem = f"{abundance}_{effective_area_note}_{plasma_tag}{units_tag}_{author}_{date}"
                    written.append(
                        _write(
                            product, output_dir / f"respfunc_ci_{band}_waveband_{stem}.zarr", provenance, spectral=True
                        )
                    )
                    summed = _sum_ci_response(map_response_to_ci_detector(product, channel), label).assign(
                        effective_area=curve
                    )
                    written.append(_write(summed, output_dir / f"respfunc_ci_{band}_{stem}.zarr", provenance))
    return written


@u.quantity_input(doppler_velocity=u.km / u.s, wavelength_step=u.AA)
def continuum_response_pipeline(
    *,
    output_dir,
    author: str,
    effective_area_note: str,
    sg_effective_areas,
    abundance: str = DEFAULT_ABUNDANCES[0],
    backend: str = "fiasco",
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    doppler_velocity: u.Quantity | None = None,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    wavelength_step: u.Quantity = 0.01 * u.AA,
    spectral_order: dict | None = None,
    sg_geometry: dict | None = None,
    sg_radiometry: dict | None = None,
    cache_dir=None,
    workers: int = 1,
) -> list[Path]:
    """
    Generate the SG continuum response of every band: the continuum alone on the
    detector.

    The continuum counterpart of `response_pipeline`: per band the rest-frame continuum on a
    fine grid over the waveband (cached by `get_continuum`) is folded through the band's
    effective area with `bin_continuum`, converted to photons and mapped onto the SG detector.

    Parameters
    ----------
    output_dir, author, effective_area_note, sg_effective_areas, temperature, pressure, density,
    minimum_abundance, spectral_order, sg_geometry, sg_radiometry, cache_dir, workers
        As for `response_pipeline`.
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
        The products written, ``respfunc_<band>_continuum_<backend>_<geometry>_<abundance>_<note>_<plasma>_<author>_<date>.zarr``.
    """
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache" if cache_dir is None else Path(cache_dir)
    if backend not in CONTINUUM_BACKENDS:
        msg = f"backend must be one of {sorted(CONTINUUM_BACKENDS)}, got {backend!r}"
        raise ValueError(msg)
    if pressure is not None and density is not None:
        msg = "pressure and density are mutually exclusive"
        raise ValueError(msg)
    if pressure is None and density is None:
        pressure = xr.DataArray([3e14, 3e15, 3e16] * u.K / u.cm**3, dims="pressure")
    if temperature is None:
        temperature = xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT")
    if doppler_velocity is None:
        doppler_velocity = np.arange(-1000, 1000, 200) * u.km / u.s
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(UTC).date()
    step = wavelength_step.to_value(u.AA)
    if density is not None:
        plasma_tag = "density"
    elif pressure.size == 1:
        plasma_tag = _plasma_tag(pressure, 0, None)
    else:
        plasma_tag = "pressure"
    written = []
    for path in sg_effective_areas:
        area = read_effective_area(path)
        band = _band(area)
        channel = band * u.AA
        curve = effective_area_curve(area)
        order = _per_band(spectral_order, band, DEFAULTS_MUSE.channel_spectral_order_SG, "spectral_order")
        grid = np.arange(band - _SG_HALF_WIDTH / order, band + _SG_HALF_WIDTH / order, step) * u.AA
        continuum = get_continuum(
            cache_dir,
            abundance=abundance,
            wavelength_grid=grid,
            temperature=temperature,
            pressure=pressure,
            density=density,
            minimum_abundance=minimum_abundance,
            backend=backend,
            workers=workers,
        )
        folded = bin_continuum(
            continuum, curve, _centers_to_edges(grid.value) * u.AA, doppler_velocity=doppler_velocity
        )
        photons = transform_response_units(folded, PHOTON_UNITS, channel, detector="SG", **(sg_radiometry or {}))
        detector = map_response_to_sg_detector(photons, channel=channel, **(sg_geometry or {})).assign(
            effective_area=curve
        )
        provenance = {
            "abundance": abundance,
            "continuum_backend": backend,
            "author": author,
            "effective_area_note": effective_area_note,
            "minimum_abundance": minimum_abundance,
        }
        name = f"respfunc_{band}_continuum_{backend}_{_sg_geometry_tag(sg_geometry)}_{abundance}_{effective_area_note}_{plasma_tag}_{author}_{date}.zarr"
        written.append(_write(detector, output_dir / name, provenance))
    return written


def _band(area: xr.Dataset) -> int:
    return round(float(area.band[0]))


def _per_band(values: dict | None, band: int, defaults: xr.DataArray, name: str):
    if values and band in values:
        return values[band]
    try:
        return defaults.sel(channel=band).data
    except KeyError:
        msg = f"no MUSE default {name} for band {band}; pass {name}={{{band}: ...}}"
        raise ValueError(msg) from None


def _default_main_lines(band: int) -> tuple[str, ...]:
    channels = [int(channel) for channel in DEFAULTS_MUSE.channel_spectral_order_SG.channel.values]
    if band not in channels:
        msg = f"no MUSE default main lines for band {band}; pass main_lines={{{band}: [...]}}"
        raise ValueError(msg)
    return DEFAULTS_MUSE.main_lines_SG[channels.index(band)]


def _plasma_tag(pressure: xr.DataArray | None, index: int, density: xr.DataArray | None) -> str:
    if density is not None:
        return "density"
    value = _validate_positive_data_array(pressure, "pressure", u.K / u.cm**3).data.reshape(-1)[index]
    return f"pressure_{value:.0e}"


def _sg_geometry_tag(sg_geometry: dict | None) -> str:
    geometry = sg_geometry or {}
    number_of_slits = geometry.get("number_of_slits", DEFAULTS_MUSE.number_of_slits_SG)
    separation = DEFAULTS_MUSE.spectral_slit_separation_SG.to_value(u.mAA)
    return f"nslits{number_of_slits}_sep{separation:g}"


def _sum_ci_response(response: xr.Dataset, label: str) -> xr.Dataset:
    """
    One imager response: every line of ``response`` summed, labelled ``label``.
    """
    summed = response.detector_response.sum("line", keep_attrs=True).expand_dims(line=[label])
    first = response.isel(line=0)
    return (
        response.drop_dims("line")
        .assign(detector_response=summed)
        .assign_coords(
            line_wavelength=("line", [float(first.line_wavelength)], {"units": str(u.AA)}),
            channel=("line", [float(first.channel)], {"units": str(u.AA)}),
            detector_wavelength=("line", [float(first.detector_wavelength)], {"units": str(u.AA)}),
        )
    )


def _staged(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{os.getpid()}{path.suffix}")


def _write(dataset: xr.Dataset, path: Path, provenance: dict, *, spectral: bool = False) -> Path:
    """
    Write a product beside its destination and rename it into place, never over an
    existing one.
    """
    if path.exists():
        msg = f"Refusing to overwrite existing product: {path}"
        raise ValueError(msg)
    dataset = dataset.assign_attrs(provenance)
    staged = _staged(path)
    (save_spectral_response if spectral else save_response)(dataset, staged)
    staged.replace(path)
    logger.info(f"Wrote {path.name} {dict(dataset.sizes)}")
    return path
