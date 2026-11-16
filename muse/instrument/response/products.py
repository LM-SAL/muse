"""
The response products of a set of effective-area files: named, generated and written.
"""

import hashlib
from numbers import Integral
from pathlib import Path
from collections import Counter
from collections.abc import Mapping, Callable, Iterable, Sequence, Collection

import numpy as np
import xarray as xr
import zarr
from zarr.storage import ZipStore

import astropy.units as u

from muse.instrument.atomic.backends import _MINIMUM_ABUNDANCE
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
    effective_area_wavelength_range,
    read_effective_area,
)
from muse.instrument.response.detector import map_response_to_ci_detector, map_response_to_sg_detector
from muse.instrument.response.io import _open_dataset_file, save_response, save_spectral_response
from muse.instrument.response.radiometry import transform_response_units
from muse.instrument.response.spectral import (
    _centers_to_edges,
    create_continuum_response,
    create_log_wavelength_edges,
    create_spectral_response,
)
from muse.log import logger
from muse.utils.utils import _inherit_history, _instrument_defaults_label, _written_in_full, add_history
from muse.variables import DEFAULTS_MUSE
from muse.variables_schema import InstrumentDefaults, _required

__all__ = ["generate_ci_responses", "generate_sg_continuum_responses", "generate_sg_responses"]

_PHOTON_UNITS = "1e-27 cm5 ph / (Angstrom s)"
_DN_UNITS = "1e-27 cm5 DN / (Angstrom s)"
_ABUNDANCES = ("sun_coronal_2021_chianti", "sun_photospheric_2021_asplund")
_TEMPERATURE = 10 ** np.arange(4.5, 8.0, 0.1) * u.K
_PRESSURE = [3e14, 3e15, 3e16] * u.K / u.cm**3
_DOPPLER_VELOCITY = np.arange(-1000, 1010, 10) * u.km / u.s
# A continuum varies slowly with wavelength, so its velocities are coarser.
_CONTINUUM_DOPPLER_VELOCITY = np.arange(-1000, 1001, 200) * u.km / u.s
_ANGSTROM = {"units": str(u.AA)}
# The attributes that record the software and database that made a line list or a continuum.
_VERSION_ATTRS = ("chianti_version", "chiantipy_version", "fiasco_version")


@u.quantity_input(doppler_velocity=u.km / u.s, line_list_wavelength_range=u.AA)
def generate_sg_responses(
    *,
    output_directory: str | Path,
    author: str,
    effective_area_label: str,
    effective_area_files: Sequence[str | Path],
    instrument_defaults: InstrumentDefaults | None = None,
    abundances: Sequence[str] = _ABUNDANCES,
    temperature: u.Quantity | None = None,
    pressure: u.Quantity | None = None,
    density: u.Quantity | None = None,
    doppler_velocity: u.Quantity | None = None,
    main_lines_by_channel: Mapping[int | u.Quantity, Sequence[str]] | None = None,
    instrumental_fwhm_by_channel: Mapping[int | u.Quantity, u.Quantity] | None = None,
    maximum_contaminants: int = 10000,
    minimum_abundance: float = _MINIMUM_ABUNDANCE,
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

    Per file, abundance and plasma point (one pressure, or a whole density grid), every
    main line of the channel gets a waveband cube and that cube mapped onto the slits of
    the detector, one line at a time so that one detector response is in memory. The
    brightest other lines of the waveband, summed, get the same pair as the
    ``contaminants``; a waveband with no other line logs a warning and gets none. One line
    list per abundance serves every file.

    Each product is a zipped zarr store,
    ``<abundance>/response_sg_<channel>_<line>_<kind>_<label>_<plasma>.zarr.zip``:
    ``<line>`` is the main line without spaces, or ``contaminants``; ``<kind>`` is
    ``waveband`` or ``detector``; ``<plasma>`` is ``density``, or ``pressure_<value>`` in
    K / cm3 to six significant digits, e.g. ``pressure_3e+15``. How a product was made,
    such as the author, the date and the slits, is in its attributes. All names are
    checked before any work is done.

    Parameters
    ----------
    output_directory : `str` or `pathlib.Path`
        Created if needed.
    author : `str`
        Initials for the attributes.
    effective_area_label : `str`
        Tells these effective areas from others in the names, e.g. the coating: letters,
        digits, ``.``, ``+`` and ``-`` (``Al-Zr``, not ``Al_Zr``).
    effective_area_files : `list` of `str` or `pathlib.Path`
        Files of `muse.instrument.save_effective_area` in cm2, one SG channel of
        ``instrument_defaults`` each. Their attributes ``pixel_width``, ``pixel_height``,
        ``gain`` and ``pair_energy``, where given, replace those of
        ``instrument_defaults``.
    instrument_defaults : `~muse.variables_schema.InstrumentDefaults`, optional
        Detector geometry and calibration, spectral orders, wavebands, instrumental FWHM
        and main lines, by default `~muse.variables.DEFAULTS_MUSE`. The products record
        whether these were the MUSE values, and a digest of them.
    abundances : `list` of `str`, optional
        CHIANTI abundance names, by default ``sun_coronal_2021_chianti`` and
        ``sun_photospheric_2021_asplund``.
    temperature : `astropy.units.Quantity`, optional
        Convertible to K, by default 10**4.5 to 10**7.9 K in steps of 0.1 dex.
    pressure, density : `astropy.units.Quantity`, optional
        The plasma points, as for `muse.instrument.create_chianti_line_list`; at most one
        of the two, by default 3e14, 3e15 and 3e16 K / cm3.
    doppler_velocity : `astropy.units.Quantity`, optional
        By default -1000 to 1000 km / s in steps of 10 km / s.
    main_lines_by_channel : `dict`, optional
        The ``full_name`` of a channel's main lines, keyed by the channel in Angstrom (an
        `int`) or as a wavelength; other channels keep those of ``instrument_defaults``.
    instrumental_fwhm_by_channel : `dict`, optional
        The instrumental FWHM of a channel, keyed alike; other channels keep that of
        ``instrument_defaults``.
    maximum_contaminants : `int`, optional
        How many of the brightest other lines of a waveband are summed into the contaminant
        products, by default 10000; below 1 there are none.
    minimum_abundance : `float`, optional
        Elements less abundant than this are left out of the line list, by default 1e-21.
    line_list_backend : {"chianti", "fiasco"}, optional
        What creates a missing line list, by default ``"fiasco"``.
    line_list_wavelength_range : `astropy.units.Quantity`, optional
        The range of the line list, by default from the lowest to the highest end of the
        files' wavebands. A wider one, e.g. covering the imager's files too, lets separate
        calls share one cached line list.
    line_list_file : `str` or `pathlib.Path`, optional
        A netCDF file or zarr store to use instead of the cache: the line list of the one
        abundance in ``abundances``, on ``temperature`` and holding the plasma points.
        ``line_list_backend``, ``minimum_abundance``, ``create_if_missing`` and ``workers``
        then do not apply to the line list.
    transition_boosts : `dict`, optional
        Passed to `muse.instrument.boost_transitions` once the line list is read; the cache
        is left as made.
    transition_selection : iterable of `tuple`, optional
        Passed to `muse.instrument.select_transitions` after the boosts.
    cache_directory : `str` or `pathlib.Path`, optional
        Where `muse.instrument.load_or_create_line_list` stores the line lists, by default
        ``output_directory / "cache"``.
    create_if_missing : `bool`, optional
        If `False`, a missing line list raises instead of being created. Pass `False` when
        several processes each take one pressure, after creating the line list on every
        pressure with `muse.instrument.load_or_create_line_list`: one created by a process
        holds its pressure alone and the others refuse it.
    skip_existing : `bool`, optional
        If `True`, existing products are kept and only the missing ones made, to finish a
        call that stopped halfway; one made with other settings raises. By default an
        existing product raises, as its name does not carry every setting.
    workers : `int`, optional
        Processes the fiasco backend splits the ions over, by default 1; ``-1`` uses every
        CPU this process may run on. Unused by ChiantiPy.
    threads : `int`, optional
        Upper limit on the BLAS, OpenMP and numexpr threads during the call, by default 4,
        and at most the CPUs this process may use; a pool set lower stays as it is.

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
        ``skip_existing`` is `False`, a file is of no SG channel, or a line list does not
        hold the requested temperatures and plasma points.
    FileNotFoundError
        If a line list is missing and ``create_if_missing`` is `False`.
    FileExistsError
        If another process is creating the line list; see
        `muse.instrument.load_or_create_line_list`.
    """
    instrument_defaults = DEFAULTS_MUSE if instrument_defaults is None else instrument_defaults
    if transition_selection is not None and not isinstance(transition_selection, Collection):
        # a generator is read once and would be recorded by its address, not its contents
        transition_selection = list(transition_selection)
    call_inputs = dict(locals())
    settings = {
        "instrument_defaults": instrument_defaults,
        "maximum_contaminants": maximum_contaminants,
        **_line_list_settings(line_list_file, line_list_backend, minimum_abundance),
    }
    with _limited_threads(threads):
        run = _Run(generate_sg_responses, settings, call_inputs)
        _require_sg_files(run.effective_areas, instrument_defaults)
        if not isinstance(maximum_contaminants, Integral):
            msg = f"maximum_contaminants must be an integer, got {maximum_contaminants!r}"
            raise TypeError(msg)
        main_lines_by_sg_channel = _main_lines_by_sg_channel(main_lines_by_channel, instrument_defaults)
        instrumental_fwhm_by_sg_channel = _instrumental_fwhm_by_sg_channel(
            instrumental_fwhm_by_channel, instrument_defaults
        )
        wavebands = u.Quantity([_sg_waveband(_channel(area), instrument_defaults) for area in run.effective_areas])
        load_line_list = _line_list_loader(
            run,
            # The wavebands, not the effective areas: those may span far more and slow the line list.
            u.Quantity([wavebands.min(), wavebands.max()]),
            wavelength_range=line_list_wavelength_range,
            line_list_file=line_list_file,
            backend=line_list_backend,
            transition_boosts=transition_boosts,
            transition_selection=transition_selection,
        )
        doppler_velocity = _or_default(doppler_velocity, _DOPPLER_VELOCITY)

        def names_of(area: xr.Dataset) -> dict:
            channel = _channel(area)
            tags = [
                *map(_line_tag, main_lines_by_sg_channel[channel]),
                *(["contaminants"] if maximum_contaminants >= 1 else []),
            ]
            return {
                (tag, kind): (f"sg_{channel}_{tag}_{kind}", "") for tag in tags for kind in ("waveband", "detector")
            }

        def settings_of(area: xr.Dataset) -> dict:
            channel = _channel(area)
            return {
                "main_lines": main_lines_by_sg_channel[channel],
                "instrumental_fwhm": instrumental_fwhm_by_sg_channel[channel],
            }

        for abundance, tasks in run.plan(names_of, settings_of).items():
            line_list = load_line_list(abundance) if tasks else None
            for area, plasma_point, names in tasks:
                _write_sg_products(
                    run,
                    names,
                    line_list.isel({run.request.plasma_dim: plasma_point}),
                    area,
                    main_lines=main_lines_by_sg_channel[_channel(area)],
                    instrumental_fwhm=instrumental_fwhm_by_sg_channel[_channel(area)],
                    doppler_velocity=doppler_velocity,
                    maximum_contaminants=maximum_contaminants,
                    abundance=abundance,
                    instrument_defaults=instrument_defaults,
                )
        return run.products()


@u.quantity_input(doppler_velocity=u.km / u.s, wavelength_step=u.AA)
def generate_sg_continuum_responses(
    *,
    output_directory: str | Path,
    author: str,
    effective_area_label: str,
    effective_area_files: Sequence[str | Path],
    instrument_defaults: InstrumentDefaults | None = None,
    abundances: Sequence[str] = _ABUNDANCES,
    temperature: u.Quantity | None = None,
    pressure: u.Quantity | None = None,
    density: u.Quantity | None = None,
    doppler_velocity: u.Quantity | None = None,
    wavelength_step: u.Quantity = 0.01 * u.AA,
    minimum_abundance: float = _MINIMUM_ABUNDANCE,
    continuum_backend: str = "fiasco",
    cache_directory: str | Path | None = None,
    create_if_missing: bool = True,
    skip_existing: bool = False,
    workers: int = 1,
    threads: int = 4,
) -> list[Path]:
    """
    Generate the spectrograph (SG) continuum responses of a set of effective-area files.

    Per file, abundance and plasma point, the rest-frame continuum on a fine grid over the
    channel's waveband is folded through the effective area with
    `muse.instrument.create_continuum_response`, converted to photons and mapped onto the
    detector.

    Each product is named
    ``<abundance>/response_sg_<channel>_continuum_<backend>_detector_<label>_<plasma>.zarr.zip``,
    with ``continuum_backend`` as ``<backend>`` and the rest as in `generate_sg_responses`.

    Parameters
    ----------
    output_directory, author, effective_area_label, effective_area_files, instrument_defaults, abundances
        As for `generate_sg_responses`.
    temperature, pressure, density
        As for `generate_sg_responses`.
    doppler_velocity : `astropy.units.Quantity`, optional
        By default -1000 to 1000 km / s in steps of 200 km / s.
    wavelength_step : `astropy.units.Quantity`, optional
        Sampling of the fine grid, by default 0.01 Angstrom.
    minimum_abundance : `float`, optional
        Elements less abundant than this are left out of the continuum, by default 1e-21.
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
    instrument_defaults = DEFAULTS_MUSE if instrument_defaults is None else instrument_defaults
    call_inputs = dict(locals())
    settings = {
        "instrument_defaults": instrument_defaults,
        "wavelength_step": wavelength_step,
        "minimum_abundance": minimum_abundance,
        "continuum_backend": continuum_backend,
    }
    with _limited_threads(threads):
        if not wavelength_step > 0:
            msg = f"wavelength_step must be positive, got {wavelength_step}"
            raise ValueError(msg)
        run = _Run(generate_sg_continuum_responses, settings, call_inputs)
        _require_sg_files(run.effective_areas, instrument_defaults)
        load_continuum = _continuum_loader(run, continuum_backend)
        doppler_velocity = _or_default(doppler_velocity, _CONTINUUM_DOPPLER_VELOCITY)

        def names_of(area: xr.Dataset) -> dict:
            return {"continuum": (f"sg_{_channel(area)}_continuum_{continuum_backend}_detector", "")}

        for abundance, tasks in run.plan(names_of).items():
            for area, plasma_point, names in tasks:
                channel = _channel(area) * u.AA
                grid = _sg_waveband_grid(_channel(area), instrument_defaults, wavelength_step)
                folded = create_continuum_response(
                    load_continuum(abundance, grid).isel({run.request.plasma_dim: plasma_point}),
                    _centers_to_edges(grid.value) * u.AA,
                    effective_area=area,
                    doppler_velocity=doppler_velocity,
                )
                photons = transform_response_units(
                    folded,
                    _PHOTON_UNITS,
                    channel=channel,
                    detector="SG",
                    **_detector_kwargs(area, "SG", instrument_defaults),
                )
                mapped = map_response_to_sg_detector(photons, channel=channel, instrument_defaults=instrument_defaults)
                run.write(
                    mapped.assign(effective_area=_curve(area)),
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
    instrument_defaults: InstrumentDefaults | None = None,
    abundances: Sequence[str] = _ABUNDANCES,
    temperature: u.Quantity | None = None,
    pressure: u.Quantity | None = None,
    density: u.Quantity | None = None,
    doppler_velocity: u.Quantity | None = None,
    maximum_lines: int = 10000,
    bin_fractional_width: float = 0.01,
    include_continuum: bool = True,
    continuum_backend: str = "fiasco",
    minimum_abundance: float = _MINIMUM_ABUNDANCE,
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

    The files are of the MUSE context imager (CI) or another imager. Per file, abundance
    and plasma point there are three products: waveband cubes in photons and in DN, holding
    the brightest lines summed and the continuum components on bins of constant velocity
    width, and the imager response in DN, the DN cube summed over wavelength and its parts.

    Each product is named
    ``<abundance>/response_ci_<channel>[_waveband]_<label>_<plasma>[_DN].zarr.zip``, with
    ``_waveband`` marking a cube and ``_DN`` a product in DN; the rest is as in
    `generate_sg_responses`.

    Parameters
    ----------
    output_directory, author, effective_area_label, abundances
        As for `generate_sg_responses`.
    effective_area_files : `list` of `str` or `pathlib.Path`
        Files of `muse.instrument.save_effective_area`, one channel each. A file of another
        imager must give ``gain`` and ``pair_energy`` in its attributes, and should give
        ``pixel_width`` and ``pixel_height``, else taken from ``instrument_defaults``;
        `muse.instrument.create_aia_effective_area` writes all four.
    instrument_defaults : `~muse.variables_schema.InstrumentDefaults`, optional
        As for `generate_sg_responses`.
    temperature, pressure, density
        As for `generate_sg_responses`.
    doppler_velocity : `astropy.units.Quantity`, optional
        By default -1000 to 1000 km / s in steps of 10 km / s.
    maximum_lines : `int`, optional
        How many of the brightest lines of a channel go into its products, by default 10000;
        at least 1.
    bin_fractional_width : `float`, optional
        Width of the wavelength bins relative to their lower edge, by default 0.01.
    include_continuum : `bool`, optional
        Whether the continuum goes into the products, by default `True`.
    continuum_backend : {"chianti", "fiasco"}, optional
        What creates a missing continuum, by default ``"fiasco"``.
    minimum_abundance : `float`, optional
        Elements less abundant than this are left out of the line list and the continuum, by
        default 1e-21.
    line_list_backend, line_list_file
        As for `generate_sg_responses`.
    line_list_wavelength_range : `astropy.units.Quantity`, optional
        The range of the line list, by default the one the files span. Pass that of every
        file to separate calls, so that they share one cached line list.
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
        place of ph, and the imager response ``detector_response(line, doppler_velocity, logT,
        pressure or logD)`` in 1e-27 cm5 DN / s, each with the
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
    instrument_defaults = DEFAULTS_MUSE if instrument_defaults is None else instrument_defaults
    if transition_selection is not None and not isinstance(transition_selection, Collection):
        # a generator is read once and would be recorded by its address, not its contents
        transition_selection = list(transition_selection)
    call_inputs = dict(locals())
    settings = {
        "instrument_defaults": instrument_defaults,
        "maximum_lines": maximum_lines,
        "bin_fractional_width": bin_fractional_width,
        "include_continuum": include_continuum,
        **({"continuum_backend": continuum_backend} if include_continuum else {}),
        **_line_list_settings(line_list_file, line_list_backend, minimum_abundance),
        # The continuum is cut at the minimum abundance also when the line list is a given file.
        **({"minimum_abundance": minimum_abundance} if include_continuum else {}),
    }
    with _limited_threads(threads):
        if not isinstance(maximum_lines, Integral):
            msg = f"maximum_lines must be an integer, got {maximum_lines!r}"
            raise TypeError(msg)
        if maximum_lines < 1:
            msg = f"maximum_lines must be at least 1, got {maximum_lines}"
            raise ValueError(msg)
        if not bin_fractional_width > 0:
            msg = f"bin_fractional_width must be positive, got {bin_fractional_width!r}"
            raise ValueError(msg)
        run = _Run(generate_ci_responses, settings, call_inputs)
        _require_ci_files(run.effective_areas, instrument_defaults)
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
        doppler_velocity = _or_default(doppler_velocity, _DOPPLER_VELOCITY)

        def names_of(area: xr.Dataset) -> dict:
            return {
                "waveband": (f"ci_{_channel(area)}_waveband", ""),
                "waveband_DN": (f"ci_{_channel(area)}_waveband", "_DN"),
                "summed_DN": (f"ci_{_channel(area)}", "_DN"),
            }

        for abundance, tasks in run.plan(names_of).items():
            line_list = load_line_list(abundance) if tasks else None
            for area, plasma_point, names in tasks:
                continuum = None
                if include_continuum:
                    # On the area's samples: the continuum is smooth, so they resolve the product too.
                    continuum = load_continuum(abundance, area.wavelength.values * u.AA).isel(
                        {run.request.plasma_dim: plasma_point}
                    )
                _write_ci_products(
                    run,
                    names,
                    line_list.isel({run.request.plasma_dim: plasma_point}),
                    area,
                    continuum=continuum,
                    doppler_velocity=doppler_velocity,
                    maximum_lines=maximum_lines,
                    bin_fractional_width=bin_fractional_width,
                    abundance=abundance,
                    instrument_defaults=instrument_defaults,
                )
        return run.products()


class _Run:
    """
    What the generators share: the checked inputs of a call, its product names and their
    writing.
    """

    def __init__(self, generator: Callable, settings: dict, call_inputs: dict) -> None:
        """
        ``call_inputs`` is ``locals()`` at the start of ``generator``; ``settings`` go
        into the history and the attributes of every product.
        """
        self.generator, self.settings = generator, settings
        self.author, self.effective_area_label = call_inputs["author"], call_inputs["effective_area_label"]
        _require_name_part("effective_area_label", self.effective_area_label)
        abundances = call_inputs["abundances"]
        self.abundances = [abundances] if isinstance(abundances, str) else list(dict.fromkeys(abundances))
        if not self.abundances:
            msg = "pass at least one abundance"
            raise ValueError(msg)
        for abundance in self.abundances:
            _require_name_part("abundance", abundance, "._+-")
        self.output_directory = Path(call_inputs["output_directory"])
        cache_directory = call_inputs["cache_directory"]
        self.cache_directory = self.output_directory / "cache" if cache_directory is None else Path(cache_directory)
        self.skip_existing = call_inputs["skip_existing"]
        pressure, density = call_inputs["pressure"], call_inputs["density"]
        if pressure is None and density is None:
            pressure = _PRESSURE
        self.request = _PlasmaRequest(_or_default(call_inputs["temperature"], _TEMPERATURE), pressure, density)
        self._cache_options = {
            "temperature": self.request.temperature,
            "pressure": self.request.pressure,
            "density": self.request.density,
            **{name: call_inputs[name] for name in ("minimum_abundance", "create_if_missing", "workers", "threads")},
        }
        self.plasma_points = (
            [("density", slice(None))]
            if density is not None
            else [(f"pressure_{value:g}", [index]) for index, value in enumerate(self.request.plasma_values)]
        )
        files = call_inputs["effective_area_files"]
        self.effective_areas = [
            read_effective_area(path).assign_attrs(effective_area_file=Path(path).name)
            for path in ([files] if isinstance(files, str | Path) else files)
        ]
        if not self.effective_areas:
            msg = "pass at least one effective-area file"
            raise ValueError(msg)
        # The area and its channel's settings join per product, a line-list file once the names pass.
        self._line_list_file = call_inputs.get("line_list_file")
        self._shaping = {
            **settings,
            **{name: call_inputs[name] for name in _SHAPING_INPUTS if name in call_inputs},
            # Without a file, the range selects the lines of the cache.
            **(
                {"line_list_wavelength_range": call_inputs.get("line_list_wavelength_range")}
                if self._line_list_file is None
                else {}
            ),
        }
        self._paths = {}
        self._done = {}
        self._settings_digests = {}

    def cached(self, load: Callable[..., xr.Dataset], **kwargs) -> xr.Dataset:
        """
        Call the cache function ``load`` with the call's cache directory and options.
        """
        return load(self.cache_directory, **self._cache_options, **kwargs)

    def plan(
        self, names_of: Callable[[xr.Dataset], dict], settings_of: Callable[[xr.Dataset], dict] | None = None
    ) -> dict[str, list[tuple]]:
        """
        Name every product, and return per abundance the ``(effective area, plasma
        point, names)`` of every group with a product missing.

        ``names_of(area)`` maps a key of the caller's choice to the ``(prefix, units
        tag)`` of each product; ``settings_of(area)`` gives what else shapes them, such
        as the channel's main lines. A name is a product's path below the output
        directory, without suffix.
        """
        tasks = {
            abundance: [
                (
                    area,
                    plasma_point,
                    {
                        key: f"{abundance}/response_{prefix}_{self.effective_area_label}_{plasma_tag}{units_tag}"
                        for key, (prefix, units_tag) in names_of(area).items()
                    },
                )
                for area in self.effective_areas
                for plasma_tag, plasma_point in self.plasma_points
            ]
            for abundance in self.abundances
        }
        if earlier := sorted(self.output_directory.glob("respfunc_*.zarr.zip")):
            msg = (
                f"{self.output_directory} holds products of an earlier muse, such as {earlier[0].name}, which this "
                "one names otherwise and would make again beside them: use another output directory"
            )
            raise ValueError(msg)
        every_name = [name for group in tasks.values() for *_, names in group for name in names.values()]
        if duplicates := sorted(name for name, count in Counter(every_name).items() if count > 1):
            msg = (
                f"{len(duplicates)} product name(s) would be used twice, the first of them {duplicates[0]}: check "
                "for two effective-area files of one channel, or pressures that agree to six significant digits"
            )
            raise ValueError(msg)
        if self._line_list_file is not None:
            self._shaping["line_list"] = _line_list_identity(self._line_list_file)
        for group in tasks.values():
            for area, _, names in group:
                digest = self._settings_digest(area, settings_of(area) if settings_of else {})
                self._settings_digests.update(dict.fromkeys(names.values(), digest))
        for name in every_name:
            kept = self._paths[name] = self.output_directory / f"{name}.zarr.zip"
            if not kept.exists():
                continue
            if not self.skip_existing:
                msg = (
                    f"the product {kept} is already there; pass skip_existing=True to keep it and make only "
                    "what is missing, or use another output directory"
                )
                raise ValueError(msg)
            made_with = _attribute_of_product(kept, "instrument_defaults")
            wanted = _instrument_defaults_label(self.settings["instrument_defaults"])
            if made_with != wanted:
                msg = (
                    f"the product {kept} was made with {made_with or 'unrecorded'} instrument defaults and "
                    f"this call uses {wanted} ones: use another output directory"
                )
                raise ValueError(msg)
            if _attribute_of_product(kept, "settings_digest") != self._settings_digests[name]:
                msg = (
                    f"the product {kept} was made with other settings than this call's, such as another grid, "
                    "effective area, main line or line list: use another output directory"
                )
                raise ValueError(msg)
            self._done[name] = kept
        return {
            abundance: [task for task in group if not all(name in self._done for name in task[2].values())]
            for abundance, group in tasks.items()
        }

    def _settings_digest(self, area: xr.Dataset, per_area: dict) -> str:
        """
        A digest of all that shapes a product of ``area``.
        """
        detector = {name: area.attrs.get(name) for name in _DETECTOR_ATTRS}
        values = {
            **self._shaping,
            **per_area,
            "effective_area": [area.effective_area, area.effective_area.attrs.get("units"), detector],
        }
        digest = hashlib.sha256()
        for name in sorted(values):
            digest.update(f"{name}={_written_in_full(values[name])};".encode())
        return digest.hexdigest()[:12]

    def write(self, dataset: xr.Dataset, name: str, save: Callable, **attrs) -> None:
        """
        Write ``dataset`` with ``save`` as the product ``name``, unless it is kept.
        """
        if name in self._done:
            logger.info(f"Keeping the existing {self._done[name].name}")
            return
        # One dimension order for every product, whatever order its parts came in.
        dataset = dataset.transpose(
            "line", "doppler_velocity", "logT", self.request.plasma_dim, ..., missing_dims="ignore"
        ).assign_attrs(
            author=self.author,
            effective_area_label=self.effective_area_label,
            settings_digest=self._settings_digests[name],
            **attrs,
        )
        add_history(dataset, self.settings, self.generator)
        _replace_atomically(self._paths[name], lambda staging: save(dataset, staging))
        self._done[name] = self._paths[name]
        logger.info(f"Wrote {self._paths[name].name} {dict(dataset.sizes)}")

    def products(self) -> list[Path]:
        """
        The products made or kept, in the order they were named.
        """
        return [self._done[name] for name in self._paths if name in self._done]


# Inputs besides the settings that shape a product; the others only place, sign or run it, or are in its name.
_SHAPING_INPUTS = ("temperature", "density", "doppler_velocity", "transition_boosts", "transition_selection")


def _line_list_identity(line_list_file: str | Path) -> dict:
    """
    Tell a line-list file from another of its name without reading it, by the size and
    last write of its files, which any rewrite changes.
    """
    path = Path(line_list_file)
    statuses = [part.stat() for part in ([path] if path.is_file() else path.rglob("*")) if part.is_file()]
    return {
        "name": path.name,
        "size": sum(status.st_size for status in statuses),
        "written": max((status.st_mtime_ns for status in statuses), default=0),
    }


def _write_sg_products(
    run: _Run,
    names: dict,
    line_list: xr.Dataset,
    area: xr.Dataset,
    *,
    main_lines: list[str],
    instrumental_fwhm: u.Quantity,
    doppler_velocity: u.Quantity,
    maximum_contaminants: int,
    abundance: str,
    instrument_defaults: InstrumentDefaults,
) -> None:
    """
    Write the products of one SG channel at one plasma point.
    """
    channel = _channel(area) * u.AA
    curve = _curve(area)
    grid = _sg_waveband_grid(_channel(area), instrument_defaults)

    def photon_response(lines: xr.Dataset, **kwargs) -> xr.Dataset:
        response = create_spectral_response(
            lines,
            grid,
            instrumental_fwhm=instrumental_fwhm,
            doppler_velocity=doppler_velocity,
            effective_area=area,  # the Dataset, so that the product inherits its history
            **kwargs,
        )
        detector_kwargs = _detector_kwargs(area, "SG", instrument_defaults)
        return transform_response_units(response, _PHOTON_UNITS, channel=channel, detector="SG", **detector_kwargs)

    def write(cube: xr.Dataset, on_detector: xr.Dataset, tag: str) -> None:
        attrs = {"abundance": abundance, "effective_area_file": area.attrs["effective_area_file"]}
        run.write(cube.assign(effective_area=curve), names[tag, "waveband"], save_spectral_response, **attrs)
        run.write(on_detector.assign(effective_area=curve), names[tag, "detector"], save_response, **attrs)

    main = photon_response(line_list, main_lines=main_lines)
    for line in main_lines:
        # One line at a time, so that only one detector response is in memory.
        one = main.sel(line=[line])
        write(
            one,
            map_response_to_sg_detector(one, channel=channel, instrument_defaults=instrument_defaults),
            _line_tag(line),
        )
    if maximum_contaminants < 1:
        return

    # Lines of the waveband, not the grid, whose last sample may pass it: a wider line list gives the same product.
    lower, upper = _sg_waveband(_channel(area), instrument_defaults).to_value(u.AA)
    in_waveband = (
        (line_list.wavelength >= lower) & (line_list.wavelength <= upper) & ~line_list.full_name.isin(main_lines)
    )
    ranked = rank_line_list(line_list.isel(transition=in_waveband.values), curve).isel(
        transition=slice(maximum_contaminants)
    )
    if ranked.sizes["transition"] == 0:
        logger.warning(f"SG channel {_channel(area)}: no line but the main lines in the waveband, so no contaminants")
        return
    contaminants = (
        photon_response(ranked, main_lines=None, include_contaminants=True)
        .assign_coords(
            line=[f"{_channel(area)} contaminants"],
            line_wavelength=("line", [float(ranked.wavelength[0])], _ANGSTROM),
        )
        .assign_attrs(number_of_lines=ranked.sizes["transition"])
    )
    # At the first main line, not the brightest contaminant, so that a reader derives one geometry per channel.
    at_main_line = contaminants.assign_coords(line_wavelength=("line", [float(main.line_wavelength[0])], _ANGSTROM))
    on_detector = map_response_to_sg_detector(at_main_line, channel=channel, instrument_defaults=instrument_defaults)
    write(contaminants, on_detector, "contaminants")


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
    instrument_defaults: InstrumentDefaults,
) -> None:
    """
    The waveband cubes and the imager response of one channel at one plasma point.
    """
    channel = _channel(area) * u.AA
    curve = _curve(area)
    edges = create_log_wavelength_edges(curve.wavelength.values[[0, -1]] * u.AA, bin_fractional_width)
    ranked = rank_line_list(line_list, area).isel(transition=slice(maximum_lines))
    line_label = f"{_channel(area)} lines"
    cube = create_spectral_response(
        ranked,
        main_lines=None,
        include_contaminants=True,
        doppler_velocity=doppler_velocity,
        effective_area=area,  # the Dataset, so that the products inherit its history
        wavelength_edges=edges,
    ).assign_coords(line=[line_label], line_wavelength=("line", [float(ranked.wavelength[0])], _ANGSTROM))
    cube.attrs["number_of_lines"] = ranked.sizes["transition"]
    if continuum is not None:
        # Without its history: the lines already bring it into the cube.
        folded = create_continuum_response(
            continuum, edges, effective_area=area.drop_attrs(deep=False), doppler_velocity=doppler_velocity
        )
        folded = folded.assign_coords(
            line=[f"{_channel(area)} continuum {component}" for component in folded.line.values]
        )
        versions = {f"continuum_{key}": value for key, value in continuum.attrs.items() if key in _VERSION_ATTRS}
        # join="exact": a continuum on another grid than the lines raises instead of filling NaN.
        cube = xr.concat(
            [cube, folded.transpose(*cube.spectral_response.dims, "bound", missing_dims="ignore")],
            dim="line",
            data_vars="all",
            combine_attrs="override",
            join="exact",
        ).assign_attrs(versions)
        # The attributes are the lines'; the calls that made the continuum join the history here.
        _inherit_history(cube, folded)
        line_label += " + continuum"

    detector_kwargs = _detector_kwargs(area, "CI", instrument_defaults)
    photons = transform_response_units(cube, _PHOTON_UNITS, channel=channel, detector="CI", **detector_kwargs)
    in_dn = transform_response_units(cube, _DN_UNITS, channel=channel, detector="CI", **detector_kwargs)
    attrs = {"abundance": abundance, "effective_area_file": area.attrs["effective_area_file"]}
    for name, converted in (("waveband", photons), ("waveband_DN", in_dn)):
        run.write(converted.assign(effective_area=curve), names[name], save_spectral_response, **attrs)
    # In DN alone: a sum in photons has lost the wavelengths its DN depend on.
    summed = _sum_lines(map_response_to_ci_detector(in_dn, channel=channel), line_label)
    run.write(summed.assign(effective_area=curve), names["summed_DN"], save_response, **attrs)


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
    Check the line-list options, and return the loader of one abundance's line list;
    ``needed_range`` is the range the products use.
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
    # Before a line list is created; checked again where a transition is matched.
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
            with _open_dataset_file(Path(line_list_file)) as opened:
                line_list = opened.load()
            problem = run.request.mismatch(line_list)
            if line_list.attrs.get("abundance") not in (None, abundance):
                problem = f"is of the abundance {line_list.attrs['abundance']}, not {abundance}"
            if problem:
                msg = f"the line list {line_list_file} {problem}"
                raise ValueError(msg)
            line_list = run.request.select(line_list)
        if transition_boosts:
            line_list = boost_transitions(line_list, transition_boosts)
        if transition_selection:
            line_list = select_transitions(line_list, transition_selection)
        return line_list

    return load


def _continuum_loader(run: _Run, backend: str) -> Callable[[str, u.Quantity], xr.Dataset]:
    """
    Check the continuum backend, and return the loader of one abundance's continuum on
    one wavelength grid, read once.
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
    What a product records of its line list: of a given file, its name alone, as no
    backend or minimum abundance can be claimed for it.
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


def _channel(area: xr.Dataset) -> int:
    """
    The channel of an effective area in whole Angstrom, as the names and defaults give
    it.
    """
    return round(float(area.channel))


def _curve(area: xr.Dataset) -> xr.DataArray:
    """
    The area along ``wavelength`` alone: its scalar ``channel`` would clash with the
    ``channel`` a detector response carries along ``line``.
    """
    return area.effective_area.reset_coords("channel", drop=True)


def _attribute_of_product(product: Path, name: str):
    """
    An attribute of a written product, read from its metadata alone.
    """
    store = ZipStore(product, mode="r")
    try:
        return zarr.open_group(store, mode="r").attrs.get(name)
    finally:
        store.close()


def _detector_kwargs(area: xr.Dataset, detector: str, instrument_defaults: InstrumentDefaults) -> dict[str, u.Quantity]:
    """
    The detector of an effective area for `transform_response_units`: its file's
    attributes, else ``instrument_defaults``.
    """
    detector = detector.upper()
    channel = _channel(area)
    kwargs = {name: getattr(instrument_defaults, f"{name}_{detector}") for name in ("pixel_width", "pixel_height")}
    for name in ("gain", "pair_energy"):
        table = getattr(instrument_defaults, f"{name}_{detector}")
        # A channel of another instrument has no entry; its file gives the value.
        if table is not None and channel in table.channel.values:
            kwargs[name] = table.sel(channel=channel).data
    kwargs |= {name: u.Quantity(area.attrs[name]) for name in _DETECTOR_ATTRS if name in area.attrs}
    # Without this check transform_response_units would take what is missing from DEFAULTS_MUSE.
    if missing := [name for name in _DETECTOR_ATTRS if kwargs.get(name) is None]:
        msg = (
            f"{area.attrs['effective_area_file']}: neither the file nor instrument_defaults gives the "
            f"{', '.join(missing)} of the {detector} channel {channel}"
        )
        raise ValueError(msg)
    return kwargs


def _line_tag(line: str) -> str:
    return "".join(line.split())


def _sg_channels(instrument_defaults: InstrumentDefaults) -> tuple[int, ...]:
    return tuple(int(channel) for channel in _required(instrument_defaults, "channel_spectral_order_SG").channel.values)


def _ci_channels(instrument_defaults: InstrumentDefaults) -> tuple[int, ...]:
    """
    The imager channels: those with a gain or pair energy in the defaults.
    """
    tables = (instrument_defaults.gain_CI, instrument_defaults.pair_energy_CI)
    return tuple(sorted({int(channel) for table in tables if table is not None for channel in table.channel.values}))


def _require_sg_files(effective_areas: list[xr.Dataset], instrument_defaults: InstrumentDefaults) -> None:
    sg_channels = _sg_channels(instrument_defaults)
    for area in effective_areas:
        file_name = area.attrs["effective_area_file"]
        if _channel(area) not in sg_channels:
            msg = (
                f"{file_name} is of channel {_channel(area)} Angstrom; SG products exist for the MUSE channels "
                f"{', '.join(map(str, sg_channels))}"
            )
            raise ValueError(msg)


def _require_ci_files(effective_areas: list[xr.Dataset], instrument_defaults: InstrumentDefaults) -> None:
    ci_channels = _ci_channels(instrument_defaults)
    for area in effective_areas:
        file_name = area.attrs["effective_area_file"]
        missing = [name for name in ("gain", "pair_energy") if name not in area.attrs]
        if missing and _channel(area) not in ci_channels:
            msg = (
                f"{file_name} is of channel {_channel(area)} Angstrom, which is no channel of the MUSE imager "
                f"({', '.join(map(str, ci_channels))}), so it must give {' and '.join(missing)} in its attributes, "
                "as create_aia_effective_area does"
            )
            raise ValueError(msg)


def _by_sg_channel(overrides: Mapping | None, name: str, defaults: Mapping[int, object]) -> dict[int, object]:
    """
    ``defaults`` with the caller's ``overrides``, keyed by channel in Angstrom or as a
    wavelength.
    """
    values = dict(defaults)
    for key, value in (overrides or {}).items():
        channel = round(float(u.Quantity(key, u.AA).value))
        if channel not in values:
            msg = f"{name} has the key {key!r}; the SG channels are {', '.join(map(str, values))}"
            raise ValueError(msg)
        values[channel] = value
    return values


def _main_lines_by_sg_channel(
    overrides: Mapping | None, instrument_defaults: InstrumentDefaults
) -> dict[int, list[str]]:
    main_lines = {}
    for channel, lines in _by_sg_channel(
        overrides, "main_lines_by_channel", _required(instrument_defaults, "main_lines_by_channel_SG")
    ).items():
        # One name alone is one line, not its characters.
        main_lines[channel] = [lines] if isinstance(lines, str) else list(lines)
        tags = [_line_tag(line) for line in main_lines[channel]]
        if directory := next((line for line, tag in zip(main_lines[channel], tags, strict=True) if "/" in tag), None):
            msg = f"main_lines_by_channel: the main line {directory!r} of channel {channel} would make a directory"
            raise ValueError(msg)
        if len(set(tags)) != len(tags):
            msg = (
                f"main_lines_by_channel: two of the main lines {main_lines[channel]} of channel {channel} differ "
                "by spaces alone and would get one file name"
            )
            raise ValueError(msg)
    return main_lines


def _instrumental_fwhm_by_sg_channel(
    overrides: Mapping | None, instrument_defaults: InstrumentDefaults
) -> dict[int, u.Quantity]:
    widths = {}
    fwhm = _required(instrument_defaults, "instrumental_fwhm_SG")
    defaults = dict(zip(fwhm.channel.values.tolist(), fwhm.data, strict=True))
    for channel, width in _by_sg_channel(overrides, "instrumental_fwhm_by_channel", defaults).items():
        widths[channel] = u.Quantity(width)
        if not widths[channel].isscalar or not widths[channel].unit.is_equivalent(u.AA):
            msg = (
                f"instrumental_fwhm_by_channel: the FWHM of channel {channel} must be one wavelength, such as "
                f"0.04 * u.AA, got {width!r}"
            )
            raise ValueError(msg)
    return widths


def _sg_waveband(channel: int, instrument_defaults: InstrumentDefaults) -> u.Quantity:
    """
    The two ends of the waveband of an SG channel.
    """
    order = int(_required(instrument_defaults, "channel_spectral_order_SG").sel(channel=channel))
    half_width = _required(instrument_defaults, "waveband_half_width_SG") / order
    return u.Quantity([channel * u.AA - half_width, channel * u.AA + half_width])


def _sg_waveband_grid(
    channel: int, instrument_defaults: InstrumentDefaults, step: u.Quantity | None = None
) -> u.Quantity:
    """
    The waveband of an SG channel in steps of ``step``, by default that of
    ``instrument_defaults``, from its lower end; the last sample may lie just beyond the
    upper end.
    """
    step = _required(instrument_defaults, "wavelength_step_SG") if step is None else step
    lower, upper = _sg_waveband(channel, instrument_defaults).to_value(u.AA)
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
