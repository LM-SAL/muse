"""
Effective-area files: the wavelength-dependent area of one channel, with its components.
"""

from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

import astropy.constants as const
import astropy.units as u
from astropy.time import Time

from muse.instrument.response.io import _ZARR_SUFFIXES, _check_destination, _open_dataset_file
from muse.utils.utils import _require_increasing_axis, _to_zarr, add_history, coord_as_unit, require_unit

__all__ = [
    "create_aia_effective_area",
    "create_effective_area",
    "effective_area_wavelength_range",
    "read_effective_area",
    "save_effective_area",
]

_CM2_DN_PER_PHOTON = u.cm**2 * u.DN / u.ph
# Attributes with which an effective-area file describes its own detector: the keywords of
# transform_response_units, as "<value> <unit>" strings.
_DETECTOR_ATTRS = ("pixel_width", "pixel_height", "gain", "pair_energy")
# SolarSoft's default aia_get_response epoch, the start of the science mission.
_AIA_MISSION_START = "2010-05-01T00:00:00.000"
# Mean energy that frees one electron-hole pair in silicon, as the SolarSoft AIA tables take it.
_AIA_PAIR_ENERGY = 3.65 * u.eV / u.electron


@u.quantity_input(area=[u.cm**2, _CM2_DN_PER_PHOTON], wavelength=u.AA, channel=u.AA)
def create_effective_area(
    area: u.Quantity,
    wavelength: u.Quantity,
    *,
    channel: u.Quantity,
    components: Mapping[str, u.Quantity] | None = None,
    attrs: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """
    Build the effective-area dataset of one channel from arrays.

    Parameters
    ----------
    area : `astropy.units.Quantity`
        The area at each wavelength, in cm2, or in cm2 DN / ph for an area that includes
        the camera's conversion from photons to DN.
    wavelength : `astropy.units.Quantity`
        Increasing wavelengths of the samples.
    channel : `astropy.units.Quantity`
        Nominal wavelength of the channel, a scalar such as ``171 * u.AA``.
    components : `dict`, optional
        Further curves of any unit, such as ``mirror``, ``filter``, ``ccd`` or ``grating``,
        stored alongside under their names. Each has one value per wavelength, or a single
        value that holds at every wavelength.
    attrs : `dict`, optional
        Attributes to record, e.g. the coating. To describe the detector of an imager other
        than MUSE, give ``pixel_width``, ``pixel_height``, ``gain`` and ``pair_energy`` as
        strings of a value and a unit, such as ``"0.6 arcsec"`` and ``"18 electron / DN"``;
        they are the keywords of `muse.instrument.transform_response_units`.

    Returns
    -------
    `xarray.Dataset`
        ``effective_area(wavelength)`` with its units, the components, ``wavelength`` in
        Angstrom and the scalar coordinate ``channel`` in Angstrom.

    Raises
    ------
    ValueError
        If ``wavelength`` is not increasing, ``channel`` is not a scalar, a component is
        named ``effective_area``, or ``area`` or a component does not have one value per
        wavelength.
    """
    call_inputs = dict(locals())
    wavelengths = np.asarray(wavelength.to_value(u.AA), dtype=float).reshape(-1)
    _require_increasing_axis(wavelengths, "wavelength", positive=True)
    if not channel.isscalar:
        msg = "channel must be a scalar wavelength, such as 171 * u.AA: an effective-area dataset holds one channel"
        raise ValueError(msg)

    if "effective_area" in (components or {}):
        msg = "a component cannot be named effective_area: that is the area itself"
        raise ValueError(msg)

    def _as_curve(name: str, values: u.Quantity, *, broadcast: bool) -> xr.DataArray:
        data = np.asarray(values.value, dtype=float)
        if broadcast and data.size == 1:
            data = np.full(wavelengths.shape, data.item())
        if data.shape != wavelengths.shape:
            msg = f"{name} must have one value per wavelength, {wavelengths.shape}, got {data.shape}"
            raise ValueError(msg)
        return xr.DataArray(data, dims="wavelength", attrs={"units": str(values.unit)})

    dataset = xr.Dataset(
        {
            "effective_area": _as_curve("area", area, broadcast=False),
            **{name: _as_curve(name, values, broadcast=True) for name, values in (components or {}).items()},
        },
        coords={
            "wavelength": ("wavelength", wavelengths, {"units": str(u.AA)}),
            "channel": ((), channel.to_value(u.AA), {"units": str(u.AA)}),
        },
        attrs=dict(attrs or {}),
    )
    # The components and the attributes are in the dataset itself; as dicts they would only
    # fill the history with array dumps and cannot be stored as inputs.
    recorded_inputs = {name: value for name, value in call_inputs.items() if name not in ("components", "attrs")}
    add_history(dataset, recorded_inputs, create_effective_area)
    return dataset


@u.quantity_input(channel=u.AA)
def create_aia_effective_area(
    channel: u.Quantity,
    *,
    obstime: Time | str | None = None,
    include_crosstalk: bool = True,
    include_degradation: bool = False,
    include_eve_correction: bool = True,
    include_dn_per_photon: bool = True,
) -> xr.Dataset:
    """
    Build the effective-area dataset of one SDO/AIA channel from aiapy.

    The area is ``(A_eff + A_cross) * C_T(obstime) * C_E(obstime) [* G(wavelength)]``, the
    terms of ``aiapy.response.Channel.wavelength_response``. The defaults are those of
    SolarSoft's ``aia_get_response(/dn, /evenorm)``, which DEM codes use.

    Parameters
    ----------
    channel : `astropy.units.Quantity`
        The AIA channel, e.g. ``304 * u.AA``.
    obstime : `astropy.time.Time` or `str`, optional
        Epoch of the degradation and of the EVE normalization, by default 1 May 2010, the
        start of the AIA science mission, as in SolarSoft.
    include_crosstalk : `bool`, optional
        Add the light of the other channel on the same telescope that leaks through this
        channel's focal-plane filter (94 with 304, 131 with 335), by default `True`. For 335
        the leak rivals the channel's own peak.
    include_degradation : `bool`, optional
        Apply ``aiapy.calibrate.degradation`` at ``obstime``, by default `False`, as aiapy's
        degradation at the default ``obstime`` is not exactly 1. Set it, with ``obstime``, to
        compare with observations.
    include_eve_correction : `bool`, optional
        Apply the EVE cross-calibration factor of ``obstime``, SolarSoft's ``/evenorm``, by
        default `True`.
    include_dn_per_photon : `bool`, optional
        Multiply the area by the DN that one photon gives, aiapy's ``Channel.gain``, which
        varies with wavelength, by default `True`. The area is then in cm2 DN / ph, and
        `muse.instrument.create_spectral_response` and
        `muse.instrument.create_continuum_response` refuse it. Pass `False` for an area in
        cm2, which they take. This factor is not the camera gain in electron / DN, which is
        always recorded in the ``gain`` attribute.

    Returns
    -------
    `xarray.Dataset`
        The dataset `create_effective_area` returns, with the curves ``mirror``, ``filter``,
        ``ccd``, ``contamination``, ``crosstalk`` and ``dn_per_photon`` alongside. The
        attributes ``pixel_width`` and ``pixel_height`` (arcsec), ``gain`` (electron / DN)
        and ``pair_energy`` (eV / electron) describe the AIA detector, as strings of a value
        and a unit.

    Notes
    -----
    Needs aiapy, which downloads the AIA instrument file on first use, and the correction
    table when a degradation or an EVE factor is applied.
    """
    call_inputs = dict(locals())
    try:
        import aiapy  # noqa: PLC0415
        from aiapy.calibrate import degradation  # noqa: PLC0415
        from aiapy.response import Channel  # noqa: PLC0415
    except ImportError:
        msg = "aiapy is required for this function, install it with `pip install muse[aia]`"
        raise ImportError(msg) from None

    obstime = Time(_AIA_MISSION_START if obstime is None else obstime)
    aia_channel = Channel(channel)
    crosstalk = aia_channel.crosstalk * int(include_crosstalk)
    degradation_factor = degradation(channel, obstime).to_value(u.one).item() if include_degradation else 1.0
    eve_factor = aia_channel.eve_correction(obstime).to_value(u.one).item() if include_eve_correction else 1.0
    area = (aia_channel.effective_area + crosstalk) * degradation_factor * eve_factor
    if include_dn_per_photon:
        area = area * aia_channel.gain
    # aiapy's Channel.gain is DN per photon: the photon energy over the pair energy and the
    # camera gain (electron / DN). So the camera gain follows from any one wavelength.
    photon_energy = (const.h * const.c / aia_channel.wavelength[0]).to(u.eV) / u.ph
    camera_gain = (photon_energy / _AIA_PAIR_ENERGY / aia_channel.gain[0]).to(u.electron / u.DN)
    pixel_size = ((aia_channel.plate_scale * u.pix) ** 0.5).to(u.arcsec)
    effective_area = create_effective_area(
        area,
        aia_channel.wavelength,
        channel=channel,
        components={
            "mirror": aia_channel.primary_reflectance * aia_channel.secondary_reflectance,
            "filter": aia_channel.entrance_filter_efficiency * aia_channel.focal_plane_filter_efficiency,
            "ccd": aia_channel.quantum_efficiency,
            "contamination": aia_channel.contamination,
            "crosstalk": crosstalk,
            "dn_per_photon": aia_channel.gain,
        },
        attrs={
            "instrument": "SDO/AIA",
            "telescope": int(aia_channel.telescope_number),
            "aiapy": aiapy.__version__,
            "degradation": degradation_factor,
            "eve_correction": eve_factor,
            **dict(
                zip(_DETECTOR_ATTRS, map(str, (pixel_size, pixel_size, camera_gain, _AIA_PAIR_ENERGY)), strict=True)
            ),
        },
    )
    add_history(effective_area, {**call_inputs, "obstime": obstime.utc.isot}, create_aia_effective_area)
    return effective_area


def _validate_effective_area(effective_area: xr.Dataset, label: str) -> xr.Dataset:
    """
    ``effective_area`` with its wavelength and channel in Angstrom, after checking the
    layout `create_effective_area` gives.
    """
    if "effective_area" not in effective_area.data_vars or "channel" not in effective_area.coords:
        msg = f"{label} must carry effective_area and a channel coordinate"
        raise ValueError(msg)
    if effective_area.effective_area.dims != ("wavelength",) or effective_area.channel.ndim != 0:
        msg = (
            f"{label}: effective_area must lie along wavelength alone and channel must be a scalar coordinate: "
            "a dataset holds one channel"
        )
        raise ValueError(msg)
    unit = require_unit(effective_area, "effective_area", f"{label}: effective_area")
    if not (unit.is_equivalent(u.cm**2) or unit.is_equivalent(_CM2_DN_PER_PHOTON)):
        msg = f"{label}: effective_area units {unit} must be convertible to cm2 or cm2 DN / ph"
        raise ValueError(msg)
    # Converted here, once: every later step may take the numbers as Angstrom.
    wavelength = coord_as_unit(effective_area, "wavelength", u.AA, f"{label}: wavelength")
    channel = coord_as_unit(effective_area, "channel", u.AA, f"{label}: channel")
    return effective_area.assign_coords(wavelength=wavelength, channel=channel)


def save_effective_area(effective_area: xr.Dataset, effective_area_file: str | Path) -> None:
    """
    Save an effective-area dataset, e.g. from `create_effective_area`.

    Parameters
    ----------
    effective_area : `xarray.Dataset`
        ``effective_area(wavelength)`` with units convertible to cm2 or cm2 DN / ph, and the
        scalar coordinate ``channel``.
    effective_area_file : `str` or `pathlib.Path`
        Destination ending in ``.zarr``, ``.zarr.zip``, ``.nc``, ``.ncdf``, or ``.netcdf``.
        A ``.zarr.zip`` destination is one file: the zarr store packed into a zip. Existing
        destinations are not overwritten.

    Raises
    ------
    ValueError
        If the dataset does not have that layout, or the destination exists or does not end
        in one of those suffixes.
    """
    validated = _validate_effective_area(effective_area, "effective_area").drop_encoding()
    destination = _check_destination(effective_area_file, what="effective area", argument="effective_area_file")
    if destination.suffix.lower() in _ZARR_SUFFIXES:
        _to_zarr(validated, destination)
    else:
        validated.to_netcdf(destination)


def read_effective_area(effective_area_file: str | Path) -> xr.Dataset:
    """
    Read an effective-area file written by `save_effective_area`.

    Parameters
    ----------
    effective_area_file : `str` or `pathlib.Path`
        A zarr store or a netCDF file.

    Returns
    -------
    `xarray.Dataset`
        The dataset in memory: ``effective_area(wavelength)``, its units convertible to cm2 or
        cm2 DN / ph, with ``wavelength`` and the scalar ``channel`` converted to Angstrom.

    Raises
    ------
    ValueError
        If the file does not have that layout or lacks a unit.
    """
    effective_area_file = Path(effective_area_file)
    with _open_dataset_file(effective_area_file) as opened:
        effective_area = opened.load()
    effective_area = _validate_effective_area(effective_area, effective_area_file.name)
    # zarr hands back arrays with an explicit byte order, which netCDF then writes with a warning.
    native = {
        name: variable.astype(variable.dtype.newbyteorder("="))
        for name, variable in effective_area.variables.items()
        if variable.dtype.byteorder in "<>"
    }
    effective_area = effective_area.assign(
        {name: variable for name, variable in native.items() if name not in effective_area.coords}
    )
    return effective_area.assign_coords(
        {name: variable for name, variable in native.items() if name in effective_area.coords}
    )


def effective_area_wavelength_range(effective_areas: Sequence[xr.Dataset]) -> u.Quantity:
    """
    Return the wavelength range that a set of effective areas spans together.

    A line list or a continuum that serves all of them must cover this range.

    Parameters
    ----------
    effective_areas : `list` of `xarray.Dataset`
        Effective-area datasets from `create_effective_area` or `read_effective_area`.

    Returns
    -------
    `astropy.units.Quantity`
        The lowest and the highest wavelength of all of them, in Angstrom.

    Raises
    ------
    ValueError
        If ``effective_areas`` is empty, or a ``wavelength`` coordinate lacks a unit.
    """
    if not effective_areas:
        msg = "effective_areas must hold at least one effective area"
        raise ValueError(msg)
    wavelengths = [np.asarray(coord_as_unit(area, "wavelength", u.AA, "wavelength")) for area in effective_areas]
    return [min(w.min() for w in wavelengths), max(w.max() for w in wavelengths)] * u.AA
