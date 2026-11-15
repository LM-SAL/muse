"""
The response products of a set of effective-area files: named, generated and written.
"""

import re
from numbers import Integral
from pathlib import Path
from datetime import UTC, datetime
from collections import Counter
from collections.abc import Mapping, Callable, Iterable, Sequence

import numpy as np
import xarray as xr

import astropy.units as u

from muse.instrument.atomic.cache import (
    _CONTINUUM_BACKENDS,
    _LINE_LIST_BACKENDS,
    _limited_threads,
    _PlasmaRequest,
    _replace_atomically,
    _require_name_part,
    load_or_create_continuum,
    load_or_create_line_list,
)
from muse.instrument.atomic.linelist import boost_transitions, rank_line_list, select_transitions
from muse.instrument.effective_area import (
    _DETECTOR_ATTRS,
    _DN_PER_PHOTON,
    effective_area_wavelength_range,
    read_effective_area,
)
from muse.instrument.response.detector import map_response_to_ci_detector, map_response_to_sg_detector
from muse.instrument.response.io import _open_response_file, save_response, save_spectral_response
from muse.instrument.response.radiometry import transform_response_units
from muse.instrument.response.spectral import (
    _centers_to_edges,
    create_continuum_response,
    create_log_wavelength_edges,
    create_spectral_response,
)
from muse.log import logger
from muse.utils.utils import _inherit_history, add_history
from muse.variables import DEFAULTS_MUSE

__all__ = ["generate_ci_responses", "generate_sg_continuum_responses", "generate_sg_responses"]

_PHOTON_UNITS = "1e-27 cm5 ph / (Angstrom s)"
_DN_UNITS = "1e-27 cm5 DN / (Angstrom s)"
_ANGSTROM = {"units": str(u.AA)}
# What a line list or a continuum says of the software and the database that made it.
_VERSION_ATTRS = ("Chianti", "Chiantipy", "fiasco")
_SG_CHANNELS = tuple(int(channel) for channel in DEFAULTS_MUSE.channel_spectral_order_SG.channel.values)
# The channels of the MUSE imager, whose gain and pair energy the defaults hold.
_CI_CHANNELS = tuple(int(channel) for channel in DEFAULTS_MUSE.ccd_gain_CI.channel.values)
# The slits and their spectral separation, as the names of the SG detector products state them.
_SG_GEOMETRY = (
    f"nslits{DEFAULTS_MUSE.number_of_slits_SG}_sep{DEFAULTS_MUSE.spectral_slit_separation_SG.to_value(u.mAA):g}"
)


@u.quantity_input(doppler_velocity=u.km / u.s, line_list_wavelength_range=u.AA)
def generate_sg_responses(
    *,
    output_directory: str | Path,
    author: str,
    effective_area_label: str,
    effective_area_files: Sequence[str | Path],
    abundances: Sequence[str] = DEFAULTS_MUSE.response_abundances,
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    doppler_velocity: u.Quantity | None = None,
    main_lines_by_channel: Mapping[int | u.Quantity, Sequence[str]] | None = None,
    instrumental_width_by_channel: Mapping[int | u.Quantity, u.Quantity] | None = None,
    maximum_contaminants: int = 10000,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    line_list_backend: str = "fiasco",
    line_list_wavelength_range: u.Quantity | None = None,
    line_list_file: str | Path | None = None,
    transition_boosts: Mapping[tuple[str, int, int], float] | None = None,
    transition_selection: Iterable[tuple[str, int, int]] | None = None,
    cache_directory: str | Path | None = None,
    create_if_missing: bool = True,
    skip_existing: bool = False,
    workers: int = 1,
    threads: int = 4,
) -> list[Path]:
    """
    Generate the spectrograph (SG) line responses of a set of effective-area files.

    One set of products is written per file, abundance and plasma point. A plasma point is
    one pressure; a density grid is one plasma point as a whole. One line list per abundance
    serves every file. Every main line of a channel gets two products: its response in
    wavelength space (the waveband cube) and that cube mapped onto the 35 slits of the
    detector. The brightest other lines of the waveband, summed, get the same pair as the
    ``contaminants``; if the waveband holds no other line, a warning is logged and the channel
    gets none.

    Each product is one file, a zarr store packed into a zip, named
    ``respfunc_<channel>_<line>_<kind>_<abundance>_<label>_<plasma>_<author>_<date>.zarr.zip``:
    ``<line>`` is the name of the main line without spaces, or ``contaminants``; ``<kind>`` is
    ``waveband`` for a cube and ``nslits35_sep390`` (the number of slits and their separation
    in milliangstrom) for a detector response; ``<plasma>`` is ``pressure_<value>`` in K / cm3
    to six significant digits, e.g. ``pressure_3e+15``, or ``density``; ``<date>`` is the UTC
    date at the start of the call, as ``YYYY-MM-DD``. Every name is fixed before any work is
    done, so two products with one name, or a product that is already there under any date,
    stop the call at its start.

    On the default grids the detector response of one line takes 2 GB of memory, and making it
    about 12 GB.

    Parameters
    ----------
    output_directory : `str` or `pathlib.Path`
        Directory of the products; created if needed.
    author : `str`
        Initials for the names and the attributes: letters, digits, ``.``, ``+`` and ``-``.
    effective_area_label : `str`
        What tells these effective areas from others in the names, e.g. the coating. Letters,
        digits, ``.``, ``+`` and ``-``, as for ``author``; no underscore, so ``Al-Zr`` and not
        ``Al_Zr``.
    effective_area_files : `list` of `str` or `pathlib.Path`
        Files of `muse.instrument.save_effective_area`, each of one of the channels 108, 171
        and 284 Angstrom, in cm2. A file that describes its detector in the attributes
        ``pixel_width``, ``pixel_height``, ``gain`` and ``pair_energy`` is converted with
        those instead of the MUSE values.
    abundances : `list` of `str`, optional
        CHIANTI abundance names, by default ``DEFAULTS_MUSE.response_abundances``.
    temperature : `xarray.DataArray`, optional
        Temperatures along ``logT`` with units convertible to K, by default
        ``DEFAULTS_MUSE.response_temperature``.
    pressure, density : `xarray.DataArray`, optional
        The plasma points: pressures with units convertible to K / cm3, or densities with
        units convertible to cm-3; at most one of the two, by default
        ``DEFAULTS_MUSE.response_pressure``.
    doppler_velocity : `astropy.units.Quantity`, optional
        Doppler velocities of the products, by default
        ``DEFAULTS_MUSE.response_doppler_velocity``.
    main_lines_by_channel : `dict`, optional
        The ``full_name`` of the main lines of a channel, keyed by the channel as an `int` in
        Angstrom or as a wavelength. A channel that is not a key keeps
        ``DEFAULTS_MUSE.main_lines_SG``.
    instrumental_width_by_channel : `dict`, optional
        The 1 sigma instrumental width of a channel, keyed alike. A channel that is not a key
        keeps ``DEFAULTS_MUSE.instrumental_width_SG``.
    maximum_contaminants : `int`, optional
        How many of the brightest other lines of a waveband are summed into the contaminant
        products, by default 10000; below 1 there are none.
    minimum_abundance : `float`, optional
        Elements less abundant than this are left out of the line list, by default
        ``DEFAULTS_MUSE.minimum_abundance``.
    line_list_backend : {"chianti", "fiasco"}, optional
        What creates a missing line list, by default ``"fiasco"``.
    line_list_wavelength_range : `astropy.units.Quantity`, optional
        The range of the line list, by default one range from the lowest to the highest end
        of the wavebands of the channels of the files (90.5 to 319 Angstrom for the three
        channels), the gaps between the wavebands included. Pass a wider one, e.g. one that
        covers the imager's files too, so that separate calls share one cached line list.
    line_list_file : `str` or `pathlib.Path`, optional
        A line list of the one abundance in ``abundances``, on ``temperature`` and holding
        the plasma points, to use instead of the cache: a netCDF file or a zarr store. ``line_list_backend``,
        ``minimum_abundance``, ``create_if_missing`` and ``workers`` then have no effect.
    transition_boosts : `dict`, optional
        Passed to `muse.instrument.boost_transitions` after the line list is read, so the
        cache stays as it was made.
    transition_selection : iterable of `tuple`, optional
        Passed to `muse.instrument.select_transitions` after the boosts.
    cache_directory : `str` or `pathlib.Path`, optional
        Where `muse.instrument.load_or_create_line_list` stores the line lists, by default
        ``output_directory / "cache"``.
    create_if_missing : `bool`, optional
        If `False`, a missing line list raises instead of being created. Pass `False` when
        several processes each handle one pressure: a line list created by one of them holds
        that pressure alone and is refused by the others. Create it first, on every pressure,
        with `muse.instrument.load_or_create_line_list`.
    skip_existing : `bool`, optional
        If `True`, a product that is already there, under any date, is kept and only the
        others are made, so that a call that stopped halfway can be finished. By default such
        a product raises, as its name does not carry every setting.
    workers : `int`, optional
        Processes the fiasco backend splits the ions over, by default 1; ``-1`` uses every CPU
        this process may run on. The ChiantiPy backend does not use it.
    threads : `int`, optional
        Upper limit on the BLAS, OpenMP and numexpr threads during the call, by default 4. A
        pool that is set lower stays as it is, and the limit is never more than the CPUs this
        process may use.

    Returns
    -------
    `list` of `pathlib.Path`
        The products, made or kept, in the order they were named. A cube holds
        ``spectral_response(line, doppler_velocity, logT, pressure or logD, wavelength_bin)``
        in 1e-27 cm5 ph / (Angstrom s) and a detector response
        ``detector_response(line, doppler_velocity, logT, pressure or logD, slit,
        detector_x_pixel)`` in 1e-27 cm5 ph / s, each with the ``effective_area(wavelength)``
        it was made with. The ``line`` of a contaminant product is ``"<channel> contaminants"``.

    Raises
    ------
    TypeError
        If ``maximum_contaminants`` is not an integer.
    ValueError
        If an option is not valid, two products would get one name, a product exists and
        ``skip_existing`` is `False`, a file is of no SG channel or carries the camera gain,
        or a line list does not hold the requested temperatures and plasma points.
    FileNotFoundError
        If a line list is missing and ``create_if_missing`` is `False`.
    FileExistsError
        If another process is creating the line list; see
        `muse.instrument.load_or_create_line_list`.
    """
    call = dict(locals())
    settings = {
        "maximum_contaminants": maximum_contaminants,
        **_line_list_settings(line_list_file, line_list_backend, minimum_abundance),
    }
    with _limited_threads(threads):
        run = _Run(generate_sg_responses, settings, call)
        _require_sg_files(run.effective_areas)
        if not isinstance(maximum_contaminants, Integral):
            msg = f"maximum_contaminants must be an integer, got {maximum_contaminants!r}"
            raise TypeError(msg)
        main_lines = _main_lines_by_sg_channel(main_lines_by_channel)
        widths = _instrumental_width_by_sg_channel(instrumental_width_by_channel)
        wavebands = u.Quantity([_sg_waveband(_channel(area)) for area in run.effective_areas])
        load_line_list = _line_list_loader(
            run,
            # One range over the wavebands of the files, not over the effective areas, which may
            # span far more: a line list of 25 to 900 Angstrom takes hours.
            u.Quantity([wavebands.min(), wavebands.max()]),
            wavelength_range=line_list_wavelength_range,
            line_list_file=line_list_file,
            backend=line_list_backend,
            transition_boosts=transition_boosts,
            transition_selection=transition_selection,
        )
        doppler_velocity = _or_default(doppler_velocity, DEFAULTS_MUSE.response_doppler_velocity)

        def names_of(area: xr.Dataset) -> dict:
            channel = _channel(area)
            tags = [*map(_line_tag, main_lines[channel]), *(["contaminants"] if maximum_contaminants >= 1 else [])]
            return {(tag, kind): (f"{channel}_{tag}_{kind}", "") for tag in tags for kind in ("waveband", _SG_GEOMETRY)}

        for abundance, tasks in run.plan(names_of).items():
            line_list = load_line_list(abundance) if tasks else None
            for area, plasma_point, names in tasks:
                _write_sg_products(
                    run,
                    names,
                    line_list.isel({run.plasma.axis: plasma_point}),
                    area,
                    main_lines=main_lines[_channel(area)],
                    instrumental_width=widths[_channel(area)],
                    doppler_velocity=doppler_velocity,
                    maximum_contaminants=maximum_contaminants,
                    abundance=abundance,
                )
        return run.products()


@u.quantity_input(doppler_velocity=u.km / u.s, wavelength_step=u.AA)
def generate_sg_continuum_responses(
    *,
    output_directory: str | Path,
    author: str,
    effective_area_label: str,
    effective_area_files: Sequence[str | Path],
    abundances: Sequence[str] = DEFAULTS_MUSE.response_abundances,
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    doppler_velocity: u.Quantity | None = None,
    wavelength_step: u.Quantity = 0.01 * u.AA,
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    continuum_backend: str = "fiasco",
    cache_directory: str | Path | None = None,
    create_if_missing: bool = True,
    skip_existing: bool = False,
    workers: int = 1,
    threads: int = 4,
) -> list[Path]:
    """
    Generate the spectrograph (SG) continuum responses of a set of effective-area files.

    One product is written per file, abundance and plasma point: the continuum alone on the
    detector. The rest-frame continuum of a channel, on a fine grid over its waveband, is
    folded through the effective area with `muse.instrument.create_continuum_response`,
    converted to photons and mapped onto the detector.

    Each product is one file, a zarr store packed into a zip, named
    ``respfunc_<channel>_continuum_<backend>_nslits35_sep390_<abundance>_<label>_<plasma>_<author>_<date>.zarr.zip``:
    ``<backend>`` is ``continuum_backend``, and the other parts of the names and the rules for
    them are as in `generate_sg_responses`.

    Parameters
    ----------
    output_directory, author, effective_area_label, effective_area_files, abundances
        As for `generate_sg_responses`.
    temperature, pressure, density
        The temperatures and the plasma points, as for `generate_sg_responses`.
    doppler_velocity : `astropy.units.Quantity`, optional
        Doppler velocities of the products, by default
        ``DEFAULTS_MUSE.continuum_response_doppler_velocity``, which is coarser than that of
        the lines and spans the same velocities.
    wavelength_step : `astropy.units.Quantity`, optional
        Sampling of the fine grid, by default 0.01 Angstrom.
    minimum_abundance : `float`, optional
        Elements less abundant than this are left out of the continuum, by default
        ``DEFAULTS_MUSE.minimum_abundance``.
    continuum_backend : {"chianti", "fiasco"}, optional
        What creates a missing continuum, by default ``"fiasco"``.
    cache_directory : `str` or `pathlib.Path`, optional
        Where `muse.instrument.load_or_create_continuum` stores the continua, by default
        ``output_directory / "cache"``.
    create_if_missing, skip_existing, workers, threads
        As for `generate_sg_responses`, for the continuum in place of the line list.

    Returns
    -------
    `list` of `pathlib.Path`
        The products, made or kept: ``detector_response(line, doppler_velocity, logT, pressure
        or logD, slit, detector_x_pixel)`` in 1e-27 cm5 ph / s, with ``"free-free"``,
        ``"free-bound"`` and ``"two-photon"`` along ``line`` and the
        ``effective_area(wavelength)`` it was made with.

    Raises
    ------
    ValueError, FileNotFoundError, FileExistsError
        As for `generate_sg_responses`, for the continuum in place of the line list.
    """
    call = dict(locals())
    settings = {
        "wavelength_step": wavelength_step,
        "minimum_abundance": minimum_abundance,
        "continuum_backend": continuum_backend,
    }
    with _limited_threads(threads):
        if not wavelength_step > 0:
            msg = f"wavelength_step must be positive, got {wavelength_step}"
            raise ValueError(msg)
        run = _Run(generate_sg_continuum_responses, settings, call)
        _require_sg_files(run.effective_areas)
        load_continuum = _continuum_loader(run, continuum_backend)
        doppler_velocity = _or_default(doppler_velocity, DEFAULTS_MUSE.continuum_response_doppler_velocity)

        def names_of(area: xr.Dataset) -> dict:
            return {"continuum": (f"{_channel(area)}_continuum_{continuum_backend}_{_SG_GEOMETRY}", "")}

        for abundance, tasks in run.plan(names_of).items():
            for area, plasma_point, names in tasks:
                channel = _channel(area) * u.AA
                grid = _sg_waveband_grid(_channel(area), wavelength_step)
                folded = create_continuum_response(
                    load_continuum(abundance, grid).isel({run.plasma.axis: plasma_point}),
                    _centers_to_edges(grid.value) * u.AA,
                    effective_area=_curve(area),
                    doppler_velocity=doppler_velocity,
                )
                photons = transform_response_units(folded, _PHOTON_UNITS, channel, detector="SG", **_detector(area))
                run.write(
                    map_response_to_sg_detector(photons, channel=channel).assign(effective_area=_curve(area)),
                    names["continuum"],
                    save_response,
                    abundance=abundance,
                    effective_area_file=area.attrs["effective_area_file"],
                )
        return run.products()


@u.quantity_input(doppler_velocity=u.km / u.s, line_list_wavelength_range=u.AA)
def generate_ci_responses(
    *,
    output_directory: str | Path,
    author: str,
    effective_area_label: str,
    effective_area_files: Sequence[str | Path],
    abundances: Sequence[str] = DEFAULTS_MUSE.response_abundances,
    temperature: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    density: xr.DataArray | None = None,
    doppler_velocity: u.Quantity | None = None,
    maximum_lines: int = 10000,
    bin_fractional_width: float = 0.01,
    include_continuum: bool = True,
    continuum_backend: str = "fiasco",
    units: Sequence[str] = ("ph", "DN"),
    minimum_abundance: float = DEFAULTS_MUSE.minimum_abundance,
    line_list_backend: str = "fiasco",
    line_list_wavelength_range: u.Quantity | None = None,
    line_list_file: str | Path | None = None,
    transition_boosts: Mapping[tuple[str, int, int], float] | None = None,
    transition_selection: Iterable[tuple[str, int, int]] | None = None,
    cache_directory: str | Path | None = None,
    create_if_missing: bool = True,
    skip_existing: bool = False,
    workers: int = 1,
    threads: int = 4,
) -> list[Path]:
    """
    Generate the imager responses of a set of effective-area files.

    The files are of the MUSE context imager (CI) or of another imager. An imager sums
    everything its passband lets through, so per file, abundance and plasma point there are
    two products in every unit: a waveband cube that holds the brightest lines summed and the
    three continuum components, on bins of constant velocity width, and that cube summed over
    wavelength and over its parts into one imager response.

    Each product is one file, a zarr store packed into a zip, named
    ``respfunc_ci_<channel>[_waveband]_<abundance>_<label>_<plasma>[_DN]_<author>_<date>.zarr.zip``:
    ``_waveband`` marks the cube and ``_DN`` the products in DN instead of photons; the other
    parts of the names and the rules for them are as in `generate_sg_responses`.

    Parameters
    ----------
    output_directory, author, effective_area_label, abundances
        As for `generate_sg_responses`.
    effective_area_files : `list` of `str` or `pathlib.Path`
        Files of `muse.instrument.save_effective_area`, one channel each. An area in
        cm2 DN / ph carries the camera gain, so only DN products exist for it. A file of
        another imager than the MUSE one must give ``gain`` and ``pair_energy`` in its
        attributes, and should give ``pixel_width`` and ``pixel_height``, which are otherwise
        those of the MUSE imager; `muse.instrument.create_aia_effective_area` writes all four.
    temperature, pressure, density
        The temperatures and the plasma points, as for `generate_sg_responses`.
    doppler_velocity : `astropy.units.Quantity`, optional
        Doppler velocities of the products, by default
        ``DEFAULTS_MUSE.response_doppler_velocity``.
    maximum_lines : `int`, optional
        How many of the brightest lines of a channel go into its products, by default 10000;
        at least 1.
    bin_fractional_width : `float`, optional
        Width of the wavelength bins relative to their lower edge, by default 0.01.
    include_continuum : `bool`, optional
        Whether the continuum goes into the products, by default `True`.
    continuum_backend : {"chianti", "fiasco"}, optional
        What creates a missing continuum, by default ``"fiasco"``.
    units : `list` of `str`, optional
        The units the products are written in: ``"ph"``, ``"DN"`` or both, the default.
    minimum_abundance : `float`, optional
        Elements less abundant than this are left out of the line list and the continuum, by
        default ``DEFAULTS_MUSE.minimum_abundance``.
    line_list_backend, line_list_file
        The line list, as for `generate_sg_responses`.
    line_list_wavelength_range : `astropy.units.Quantity`, optional
        The range of the line list, by default the one the effective-area files span. Pass
        the range of every file when they are processed in separate calls, so that the calls
        share one cached line list.
    transition_boosts, transition_selection
        As for `generate_sg_responses`.
    cache_directory : `str` or `pathlib.Path`, optional
        Where the line lists and the continua are stored, by default
        ``output_directory / "cache"``.
    create_if_missing, skip_existing, workers, threads
        As for `generate_sg_responses`, for the line list and the continuum alike.

    Returns
    -------
    `list` of `pathlib.Path`
        The products, made or kept. A cube holds ``spectral_response(line, doppler_velocity,
        logT, pressure or logD, wavelength_bin)`` in 1e-27 cm5 ph / (Angstrom s), or DN in
        place of ph, and an imager response ``detector_response(line, doppler_velocity, logT,
        pressure or logD)`` in 1e-27 cm5 ph / s or DN / s, each with the
        ``effective_area(wavelength)`` it was made with. Along ``line`` a cube holds
        ``"<channel> lines"`` and ``"<channel> continuum <component>"``; an imager response
        holds one entry, ``"<channel> lines"`` or ``"<channel> lines + continuum"``.

    Raises
    ------
    TypeError
        If ``maximum_lines`` is not an integer.
    ValueError, FileNotFoundError, FileExistsError
        As for `generate_sg_responses`, for the line list and the continuum alike.
    """
    call = dict(locals())
    settings = {
        "maximum_lines": maximum_lines,
        "bin_fractional_width": bin_fractional_width,
        "include_continuum": include_continuum,
        **({"continuum_backend": continuum_backend} if include_continuum else {}),
        **_line_list_settings(line_list_file, line_list_backend, minimum_abundance),
        # The continuum is cut at the minimum abundance also when the line list is a given file.
        **({"minimum_abundance": minimum_abundance} if include_continuum else {}),
    }
    with _limited_threads(threads):
        units = (units,) if isinstance(units, str) else tuple(units)
        if not units or set(units) - {"ph", "DN"}:
            msg = f"units must be 'ph', 'DN' or both, got {units!r}"
            raise ValueError(msg)
        if not isinstance(maximum_lines, Integral):
            msg = f"maximum_lines must be an integer, got {maximum_lines!r}"
            raise TypeError(msg)
        if maximum_lines < 1:
            msg = f"maximum_lines must be at least 1, got {maximum_lines}"
            raise ValueError(msg)
        if not bin_fractional_width > 0:
            msg = f"bin_fractional_width must be positive, got {bin_fractional_width!r}"
            raise ValueError(msg)
        run = _Run(generate_ci_responses, settings, call)
        _require_ci_files(run.effective_areas, units)
        load_line_list = _line_list_loader(
            run,
            effective_area_wavelength_range(run.effective_areas),
            wavelength_range=line_list_wavelength_range,
            line_list_file=line_list_file,
            backend=line_list_backend,
            transition_boosts=transition_boosts,
            transition_selection=transition_selection,
        )
        load_continuum = _continuum_loader(run, continuum_backend)
        doppler_velocity = _or_default(doppler_velocity, DEFAULTS_MUSE.response_doppler_velocity)

        def names_of(area: xr.Dataset) -> dict:
            tags = (
                ["_DN"] if _carries_gain(area) else [tag for unit, tag in (("ph", ""), ("DN", "_DN")) if unit in units]
            )
            return {
                (tag, kind): (f"ci_{_channel(area)}{suffix}", tag)
                for tag in tags
                for kind, suffix in (("waveband", "_waveband"), ("summed", ""))
            }

        for abundance, tasks in run.plan(names_of).items():
            line_list = load_line_list(abundance) if tasks else None
            for area, plasma_point, names in tasks:
                continuum = None
                if include_continuum:
                    # At the wavelength samples of the effective area: the continuum is smooth, so
                    # samples that resolve the area resolve their product.
                    continuum = load_continuum(abundance, area.wavelength.values * u.AA).isel(
                        {run.plasma.axis: plasma_point}
                    )
                _write_ci_products(
                    run,
                    names,
                    line_list.isel({run.plasma.axis: plasma_point}),
                    area,
                    continuum=continuum,
                    doppler_velocity=doppler_velocity,
                    maximum_lines=maximum_lines,
                    bin_fractional_width=bin_fractional_width,
                    abundance=abundance,
                )
        return run.products()


class _Run:
    """
    What the generators share: the checked inputs of a call, the names of its products
    and their writing.

    Every product is named before any work is done, so two products with one name, or a
    product that is already there, stop the call at its start.
    """

    def __init__(self, generator: Callable, settings: dict, call: dict) -> None:
        """
        ``call`` holds the arguments of ``generator`` by name, as ``locals()`` gives
        them at its start; ``settings`` those that go into the history and the
        attributes of every product.
        """
        self.generator, self.settings = generator, settings
        self.author, self.effective_area_label = call["author"], call["effective_area_label"]
        _require_name_part("author", self.author)
        _require_name_part("effective_area_label", self.effective_area_label)
        abundances = call["abundances"]
        # A repeated abundance is taken once.
        self.abundances = [abundances] if isinstance(abundances, str) else list(dict.fromkeys(abundances))
        if not self.abundances:
            msg = "pass at least one abundance"
            raise ValueError(msg)
        for abundance in self.abundances:
            _require_name_part("abundance", abundance, "._+-")
        self.output_directory = Path(call["output_directory"])
        cache_directory = call["cache_directory"]
        self.cache_directory = self.output_directory / "cache" if cache_directory is None else Path(cache_directory)
        self.skip_existing = call["skip_existing"]
        self.date = datetime.now(UTC).date()
        pressure, density = call["pressure"], call["density"]
        if pressure is None and density is None:
            pressure = DEFAULTS_MUSE.response_pressure
        self.plasma = _PlasmaRequest(
            _or_default(call["temperature"], DEFAULTS_MUSE.response_temperature), pressure, density
        )
        # What every read of a cache in the call shares.
        self._cache_options = {
            "temperature": self.plasma.temperature,
            "pressure": self.plasma.pressure,
            "density": self.plasma.density,
            **{name: call[name] for name in ("minimum_abundance", "create_if_missing", "workers", "threads")},
        }
        # One product per pressure; a density grid goes into one product whole.
        self.plasma_points = (
            [("density", slice(None))]
            if density is not None
            else [(f"pressure_{value:g}", [index]) for index, value in enumerate(self.plasma.values)]
        )
        files = call["effective_area_files"]
        self.effective_areas = [
            read_effective_area(path).assign_attrs(effective_area_file=Path(path).name)
            for path in ([files] if isinstance(files, str | Path) else files)
        ]
        if not self.effective_areas:
            msg = "pass at least one effective-area file"
            raise ValueError(msg)
        self._paths = {}
        self._done = {}

    def cached(self, load: Callable[..., xr.Dataset], **kwargs) -> xr.Dataset:
        """
        ``load``, one of the two cache functions, in the cache directory of the call, on
        its temperatures and plasma points and with its options.
        """
        return load(self.cache_directory, **self._cache_options, **kwargs)

    def plan(self, names_of: Callable[[xr.Dataset], dict]) -> dict[str, list[tuple]]:
        """
        Name every product of the call, and return what is still to make: per abundance,
        the ``(effective area, plasma point, names)`` of every group with a product
        missing.

        ``names_of(effective_area)`` gives what is particular to each product of one
        effective area, under a key of the caller's: the ``(prefix, units tag)`` of its
        name. A name here is a file name without its date.
        """
        tasks = {
            abundance: [
                (
                    area,
                    plasma_point,
                    {
                        key: f"respfunc_{prefix}_{abundance}_{self.effective_area_label}_{tag}{units_tag}_{self.author}"
                        for key, (prefix, units_tag) in names_of(area).items()
                    },
                )
                for area in self.effective_areas
                for tag, plasma_point in self.plasma_points
            ]
            for abundance in self.abundances
        }
        every_name = [name for group in tasks.values() for *_, names in group for name in names.values()]
        if duplicates := sorted(name for name, count in Counter(every_name).items() if count > 1):
            msg = (
                f"{len(duplicates)} product name(s) would be used twice, the first of them {duplicates[0]}: check "
                "for two effective-area files of one channel, or pressures that agree to six significant digits"
            )
            raise ValueError(msg)
        present = sorted(path.name for path in self.output_directory.glob("respfunc_*.zarr.zip"))
        for name in every_name:
            self._paths[name] = self.output_directory / f"{name}_{self.date}.zarr.zip"
            # Under any date: a call that goes on after midnight UTC must find what it wrote before,
            # and a call on a later day must not write a second copy beside the first.
            dated = re.compile(re.escape(name) + r"_\d{4}-\d{2}-\d{2}\.zarr\.zip")
            found = [file_name for file_name in present if dated.fullmatch(file_name)]
            if found and not self.skip_existing:
                msg = (
                    f"the product {self.output_directory / found[-1]} is already there; pass skip_existing=True to "
                    "keep it and make only what is missing, or use another output directory"
                )
                raise ValueError(msg)
            if found:
                self._done[name] = self.output_directory / found[-1]
        return {
            abundance: [task for task in group if not all(name in self._done for name in task[2].values())]
            for abundance, group in tasks.items()
        }

    def write(self, dataset: xr.Dataset, name: str, save: Callable, **attrs) -> None:
        """
        Write ``dataset`` with ``save`` as the product ``name``, with the call that made
        it in its history and ``attrs`` among its attributes.

        A product that is kept is not written.
        """
        if name in self._done:
            logger.info(f"Keeping the existing {self._done[name].name}")
            return
        # One order of the dimensions for every product, whatever order its parts came in.
        dataset = dataset.transpose(
            "line", "doppler_velocity", "logT", self.plasma.axis, ..., missing_dims="ignore"
        ).assign_attrs(author=self.author, effective_area_label=self.effective_area_label, **attrs)
        add_history(dataset, self.settings, self.generator)
        _replace_atomically(self._paths[name], lambda staging: save(dataset, staging))
        self._done[name] = self._paths[name]
        logger.info(f"Wrote {self._paths[name].name} {dict(dataset.sizes)}")

    def products(self) -> list[Path]:
        """
        The products of the call that exist, made or kept, in the order they were named.
        """
        return [self._done[name] for name in self._paths if name in self._done]


def _write_sg_products(
    run: _Run,
    names: dict,
    line_list: xr.Dataset,
    area: xr.Dataset,
    *,
    main_lines: list[str],
    instrumental_width: u.Quantity,
    doppler_velocity: u.Quantity,
    maximum_contaminants: int,
    abundance: str,
) -> None:
    """
    The cube and the detector response of every main line of one SG channel, and of its
    brightest other lines summed, at one plasma point.
    """
    channel = _channel(area) * u.AA
    curve = _curve(area)
    grid = _sg_waveband_grid(_channel(area))

    def photon_response(lines: xr.Dataset, **kwargs) -> xr.Dataset:
        response = create_spectral_response(
            lines,
            grid,
            instrumental_width=instrumental_width,
            doppler_velocity=doppler_velocity,
            effective_area=curve,
            **kwargs,
        )
        return transform_response_units(response, _PHOTON_UNITS, channel, detector="SG", **_detector(area))

    def write(cube: xr.Dataset, on_detector: xr.Dataset, tag: str) -> None:
        attrs = {"abundance": abundance, "effective_area_file": area.attrs["effective_area_file"]}
        run.write(cube.assign(effective_area=curve), names[tag, "waveband"], save_spectral_response, **attrs)
        run.write(on_detector.assign(effective_area=curve), names[tag, _SG_GEOMETRY], save_response, **attrs)

    main = photon_response(line_list, main_lines=main_lines)
    for line in main_lines:
        # One line at a time: the mapping needs several times the memory of its result, which
        # is 2 GB on the default grids.
        one = main.sel(line=[line])
        write(one, map_response_to_sg_detector(one, channel=channel), _line_tag(line))
    if maximum_contaminants < 1:
        return

    # The contaminants are the lines of the waveband itself, not of the grid, whose last step
    # may reach beyond it: a wider line list then gives the same product.
    lower, upper = _sg_waveband(_channel(area)).to_value(u.AA)
    in_waveband = (
        (line_list.wavelength >= lower) & (line_list.wavelength <= upper) & ~line_list.full_name.isin(main_lines)
    )
    ranked = rank_line_list(line_list.isel(trans_index=in_waveband.values), curve).isel(
        trans_index=slice(maximum_contaminants)
    )
    if ranked.sizes["trans_index"] == 0:
        logger.warning(f"SG channel {_channel(area)}: no line but the main lines in the waveband, so no contaminants")
        return
    contaminants = (
        photon_response(ranked, main_lines=None, include_contaminants=True)
        .assign_coords(
            line=[f"{_channel(area)} contaminants"],
            line_wavelength=("line", [float(ranked.wavelength[0])], _ANGSTROM),
        )
        .assign_attrs(number_of_lines=ranked.sizes["trans_index"])
    )
    # On the detector the contaminants are referenced to the channel's first main line, as its
    # line products are, so that a reader derives one geometry per channel.
    at_main_line = contaminants.assign_coords(line_wavelength=("line", [float(main.line_wavelength[0])], _ANGSTROM))
    write(contaminants, map_response_to_sg_detector(at_main_line, channel=channel), "contaminants")


def _write_ci_products(
    run: _Run,
    names: dict,
    line_list: xr.Dataset,
    area: xr.Dataset,
    *,
    continuum: xr.Dataset | None,
    doppler_velocity: u.Quantity,
    maximum_lines: int,
    bin_fractional_width: float,
    abundance: str,
) -> None:
    """
    The waveband cube and the imager response of one channel at one plasma point, in
    every unit ``names`` holds.
    """
    channel = _channel(area) * u.AA
    curve = _curve(area)
    response_area = curve
    if _carries_gain(area):
        # The gain is in the area already: the response is built from it as a plain cm2 area,
        # and what comes out per photon is DN. The products keep the area as it was given.
        response_area = curve.assign_attrs(units=str(u.Unit(curve.attrs["units"]) / (u.DN / u.ph)))
    edges = create_log_wavelength_edges(curve.wavelength.values[[0, -1]] * u.AA, bin_fractional_width)
    ranked = rank_line_list(line_list, response_area).isel(trans_index=slice(maximum_lines))
    line = f"{_channel(area)} lines"
    cube = create_spectral_response(
        ranked,
        main_lines=None,
        include_contaminants=True,
        doppler_velocity=doppler_velocity,
        effective_area=response_area,
        wavelength_edges=edges,
    ).assign_coords(line=[line], line_wavelength=("line", [float(ranked.wavelength[0])], _ANGSTROM))
    cube.attrs["number_of_lines"] = ranked.sizes["trans_index"]
    if continuum is not None:
        folded = create_continuum_response(
            continuum, edges, effective_area=response_area, doppler_velocity=doppler_velocity
        )
        folded = folded.assign_coords(
            line=[f"{_channel(area)} continuum {component}" for component in folded.line.values]
        )
        versions = {f"continuum_{key}": value for key, value in continuum.attrs.items() if key in _VERSION_ATTRS}
        # join="exact": a continuum on other temperatures or plasma points than the lines stops
        # here instead of filling the cube with NaN.
        cube = xr.concat(
            [cube, folded.transpose(*cube.spectral_response.dims, "bound", missing_dims="ignore")],
            dim="line",
            data_vars="all",
            combine_attrs="override",
            join="exact",
        ).assign_attrs(versions)
        # The attributes are those of the lines: the calls that made the continuum join the history here.
        _inherit_history(cube, folded)
        line += " + continuum"

    detector = _detector(area)
    photons = transform_response_units(cube, _PHOTON_UNITS, channel, detector="CI", **detector)
    if _carries_gain(area):
        in_dn = photons.assign(spectral_response=photons.spectral_response.assign_attrs(units=_DN_UNITS))
    else:
        in_dn = transform_response_units(cube, _DN_UNITS, channel, detector="CI", **detector)
    attrs = {"abundance": abundance, "effective_area_file": area.attrs["effective_area_file"]}
    for tag, converted in (("", photons), ("_DN", in_dn)):
        if (tag, "waveband") not in names:
            continue
        run.write(converted.assign(effective_area=curve), names[tag, "waveband"], save_spectral_response, **attrs)
        summed = _sum_lines(map_response_to_ci_detector(converted, channel), line)
        run.write(summed.assign(effective_area=curve), names[tag, "summed"], save_response, **attrs)


def _line_list_loader(
    run: _Run,
    needed_range: u.Quantity,
    *,
    wavelength_range: u.Quantity | None,
    line_list_file: str | Path | None,
    backend: str,
    transition_boosts: Mapping | None,
    transition_selection: Iterable | None,
) -> Callable[[str], xr.Dataset]:
    """
    Check the line-list options of a call, and return the function that gives the line
    list of one abundance at the plasma points of the call; ``needed_range`` is what the
    products of the call use of it.
    """
    _require_choice("line_list_backend", backend, _LINE_LIST_BACKENDS)
    if line_list_file is not None and len(run.abundances) != 1:
        msg = "line_list_file holds the line list of one abundance; pass that one name in abundances"
        raise ValueError(msg)
    if wavelength_range is None:
        wavelength_range = needed_range
    elif wavelength_range.shape != (2,):
        msg = f"line_list_wavelength_range must hold a lower and an upper wavelength, got {wavelength_range}"
        raise ValueError(msg)
    elif wavelength_range[0] > needed_range[0] or wavelength_range[1] < needed_range[1]:
        msg = f"line_list_wavelength_range {wavelength_range} does not cover what the products need, {needed_range}"
        raise ValueError(msg)
    transition_selection = list(transition_selection or [])
    # Checked here, before a line list is created, and again where a transition is matched.
    for transition in (*(transition_boosts or {}), *transition_selection):
        if not isinstance(transition, tuple | list) or len(transition) != 3:
            msg = (
                "a transition is (ion_name, lower_level_index, upper_level_index), such as ('fe_9', 1, 13); "
                f"got {transition!r}"
            )
            raise ValueError(msg)

    def load(abundance: str) -> xr.Dataset:
        if line_list_file is None:
            line_list = run.cached(
                load_or_create_line_list, abundance=abundance, wavelength_range=wavelength_range, backend=backend
            )
        else:
            # A netCDF file or a zarr store, such as a cache file of load_or_create_line_list.
            with _open_response_file(Path(line_list_file)) as opened:
                line_list = opened.load()
            problem = run.plasma.mismatch(line_list)
            if line_list.attrs.get("abundance") not in (None, abundance):
                problem = f"is of the abundance {line_list.attrs['abundance']}, not {abundance}"
            if problem:
                msg = f"the line list {line_list_file} {problem}"
                raise ValueError(msg)
            line_list = run.plasma.select(line_list)
        # After the line list is read, so that the cache stays as the backend made it.
        if transition_boosts:
            line_list = boost_transitions(line_list, transition_boosts)
        if transition_selection:
            line_list = select_transitions(line_list, transition_selection)
        return line_list

    return load


def _continuum_loader(run: _Run, backend: str) -> Callable[[str, u.Quantity], xr.Dataset]:
    """
    Check the continuum backend, and return the function that gives the continuum of one
    abundance on one wavelength grid at the plasma points of the call, read once.
    """
    _require_choice("continuum_backend", backend, _CONTINUUM_BACKENDS)
    loaded = {}

    def load(abundance: str, wavelength_grid: u.Quantity) -> xr.Dataset:
        key = (abundance, wavelength_grid.to_value(u.AA).tobytes())
        if key not in loaded:
            loaded[key] = run.cached(
                load_or_create_continuum, abundance=abundance, wavelength_grid=wavelength_grid, backend=backend
            )
        return loaded[key]

    return load


def _line_list_settings(line_list_file: str | Path | None, line_list_backend: str, minimum_abundance: float) -> dict:
    """
    What a product records of its line list.

    A given line list was made elsewhere: its file can be recorded, by name alone as the
    effective area is, but neither a backend nor a minimum abundance can be claimed for it.
    """
    if line_list_file is None:
        return {"line_list_backend": line_list_backend, "minimum_abundance": minimum_abundance}
    return {"line_list_file": Path(line_list_file).name}


def _or_default(value, default):
    return default if value is None else value


def _require_choice(name: str, value: str, choices: Iterable[str]) -> None:
    if value not in choices:
        msg = f"{name} must be one of {sorted(choices)}, got {value!r}"
        raise ValueError(msg)


def _channel(effective_area: xr.Dataset) -> int:
    """
    The channel of an effective area as the names and the MUSE defaults state it: whole
    Angstrom.
    """
    return round(float(effective_area.channel))


def _curve(effective_area: xr.Dataset) -> xr.DataArray:
    """
    The area along ``wavelength`` alone: its scalar ``channel`` would clash with the
    ``channel`` a detector response carries along ``line``.
    """
    return effective_area.effective_area.reset_coords("channel", drop=True)


def _carries_gain(effective_area: xr.Dataset) -> bool:
    return u.Unit(effective_area.effective_area.attrs["units"]).is_equivalent(_DN_PER_PHOTON)


def _detector(effective_area: xr.Dataset) -> dict[str, u.Quantity]:
    """
    The detector a file describes in its attributes, for `transform_response_units`;
    what it leaves out is the MUSE value.
    """
    return {name: u.Quantity(effective_area.attrs[name]) for name in _DETECTOR_ATTRS if name in effective_area.attrs}


def _line_tag(line: str) -> str:
    return "".join(line.split())


def _require_sg_files(effective_areas: list[xr.Dataset]) -> None:
    for area in effective_areas:
        file_name = area.attrs["effective_area_file"]
        if _channel(area) not in _SG_CHANNELS:
            msg = (
                f"{file_name} is of channel {_channel(area)} Angstrom; SG products exist for the MUSE channels "
                f"{', '.join(map(str, _SG_CHANNELS))}"
            )
            raise ValueError(msg)
        if _carries_gain(area):
            msg = f"{file_name} is in cm2 DN / ph; an SG product needs the area in cm2, without the camera gain"
            raise ValueError(msg)


def _require_ci_files(effective_areas: list[xr.Dataset], units: tuple[str, ...]) -> None:
    for area in effective_areas:
        file_name = area.attrs["effective_area_file"]
        if _carries_gain(area) and "DN" not in units:
            msg = f"{file_name} is in cm2 DN / ph, so only DN products exist for it; add 'DN' to units"
            raise ValueError(msg)
        missing = [name for name in ("gain", "pair_energy") if name not in area.attrs]
        if missing and _channel(area) not in _CI_CHANNELS:
            msg = (
                f"{file_name} is of channel {_channel(area)} Angstrom, which is no channel of the MUSE imager "
                f"({', '.join(map(str, _CI_CHANNELS))}), so it must give {' and '.join(missing)} in its attributes, "
                "as create_aia_effective_area does"
            )
            raise ValueError(msg)


def _by_sg_channel(overrides: Mapping | None, name: str, defaults: Iterable) -> dict[int, object]:
    """
    The MUSE value of every SG channel, replaced by the caller's where one is given; a
    key is a channel in Angstrom as a number or as a wavelength.
    """
    values = dict(zip(_SG_CHANNELS, defaults, strict=True))
    for key, value in (overrides or {}).items():
        channel = round(float(u.Quantity(key, u.AA).value))
        if channel not in values:
            msg = f"{name} has the key {key!r}; the SG channels are {', '.join(map(str, _SG_CHANNELS))}"
            raise ValueError(msg)
        values[channel] = value
    return values


def _main_lines_by_sg_channel(overrides: Mapping | None) -> dict[int, list[str]]:
    main_lines = {}
    for channel, lines in _by_sg_channel(overrides, "main_lines_by_channel", DEFAULTS_MUSE.main_lines_SG).items():
        # One name alone is one line, not its characters.
        main_lines[channel] = [lines] if isinstance(lines, str) else list(lines)
        tags = [_line_tag(line) for line in main_lines[channel]]
        if len(set(tags)) != len(tags):
            msg = (
                f"main_lines_by_channel: two of the main lines {main_lines[channel]} of channel {channel} differ "
                "by spaces alone and would get one file name"
            )
            raise ValueError(msg)
    return main_lines


def _instrumental_width_by_sg_channel(overrides: Mapping | None) -> dict[int, u.Quantity]:
    widths = {}
    defaults = DEFAULTS_MUSE.instrumental_width_SG.data
    for channel, width in _by_sg_channel(overrides, "instrumental_width_by_channel", defaults).items():
        widths[channel] = u.Quantity(width)
        if not widths[channel].isscalar or not widths[channel].unit.is_equivalent(u.AA):
            msg = (
                f"instrumental_width_by_channel: the width of channel {channel} must be one wavelength, such as "
                f"0.02 * u.AA, got {width!r}"
            )
            raise ValueError(msg)
    return widths


def _sg_waveband(channel: int) -> u.Quantity:
    """
    The two ends of what an SG channel covers: the MUSE half width over the spectral
    order to either side of the channel.
    """
    order = int(DEFAULTS_MUSE.channel_spectral_order_SG.sel(channel=channel))
    half_width = DEFAULTS_MUSE.waveband_half_width_SG / order
    return u.Quantity([channel * u.AA - half_width, channel * u.AA + half_width])


def _sg_waveband_grid(channel: int, step: u.Quantity = DEFAULTS_MUSE.wavelength_step_SG) -> u.Quantity:
    """
    The waveband of an SG channel in steps of ``step`` from its lower end; the last
    sample is the one nearest its upper end, and may lie just beyond it.
    """
    lower, upper = _sg_waveband(channel).to_value(u.AA)
    step = step.to_value(u.AA)
    # By the number of steps: np.arange up to the upper end gains or loses a sample to round-off.
    return (lower + step * np.arange(round((upper - lower) / step) + 1)) * u.AA


def _sum_lines(response: xr.Dataset, name: str) -> xr.Dataset:
    """
    One imager response: every ``line`` of ``response`` summed and called ``name``.
    """
    summed = response.detector_response.sum("line", keep_attrs=True).expand_dims(line=[name])
    first = response.isel(line=0)
    return (
        response.drop_dims("line")
        .assign(detector_response=summed)
        .assign_coords(
            line_wavelength=("line", [float(first.line_wavelength)], _ANGSTROM),
            channel=("line", [float(first.channel)], _ANGSTROM),
            detector_wavelength=("line", [float(first.detector_wavelength)], _ANGSTROM),
        )
    )
