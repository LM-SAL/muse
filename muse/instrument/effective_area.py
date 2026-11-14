"""
Effective-area files: the wavelength-dependent area of one band, with its components.
"""

from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

import astropy.constants as const
import astropy.units as u
from astropy.time import Time

from muse.utils.utils import _require_increasing_axis, add_history, coord_as_unit, require_unit

__all__ = [
    "create_aia_effective_area",
    "create_effective_area",
    "effective_area_wavelength_range",
    "read_effective_area",
    "select_effective_area_curve",
]

_DN_PER_PHOTON = u.cm**2 * u.DN / u.ph
# Attributes with which an effective-area file describes its own detector: the keywords of
# transform_response_units, as "<value> <unit>" strings. generate_responses reads them.
_DETECTOR_ATTRS = ("pixel_width", "pixel_height", "gain", "pair_energy")
# SolarSoft's default aia_get_response epoch: the instrument at launch and its EVE factors.
_AIA_LAUNCH_EPOCH = "2010-05-01T00:00:00.000"
# Mean energy that frees one electron-hole pair in silicon, as the SolarSoft AIA tables take it.
_AIA_PAIR_ENERGY = 3.65 * u.eV / u.electron


@u.quantity_input(area=[u.cm**2, _DN_PER_PHOTON], wavelength=u.AA, band=u.AA)
def create_effective_area(
    area: u.Quantity,
    wavelength: u.Quantity,
    *,
    band: u.Quantity,
    components: Mapping[str, u.Quantity] | None = None,
    attrs: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """
    Build an effective-area dataset from arrays.

    Parameters
    ----------
    area : `astropy.units.Quantity`
        The area at each wavelength, in cm2, or in cm2 DN / ph for an area that carries the
        camera gain. One value per wavelength, or a ``(wavelength, band)`` table.
    wavelength : `astropy.units.Quantity`
        Increasing wavelengths of the samples.
    band : `astropy.units.Quantity`
        Nominal wavelength of the band, or one per band column.
    components : `dict`, optional
        Further curves of the same shape and any unit, such as ``mirror``, ``filter``, ``ccd``
        or ``grating``, stored alongside under their names.
    attrs : `dict`, optional
        Attributes to record, e.g. the coating, or the detector of another imager as
        `generate_responses` reads it.

    Returns
    -------
    `xarray.Dataset`
        ``effective_area(wavelength, band)`` with its units, plus the components.
    """
    call_inputs = dict(locals())
    wavelengths = np.asarray(wavelength.to_value(u.AA), dtype=float).reshape(-1)
    _require_increasing_axis(wavelengths, "wavelength", positive=True)
    bands = np.atleast_1d(band.to_value(u.AA))
    shape = (wavelengths.size, bands.size)

    def _as_table(values: u.Quantity) -> xr.DataArray:
        return xr.DataArray(
            np.asarray(values.value, dtype=float).reshape(shape),
            dims=("wavelength", "band"),
            attrs={"units": str(values.unit)},
        )

    dataset = xr.Dataset(
        {"effective_area": _as_table(area), **{name: _as_table(values) for name, values in (components or {}).items()}},
        coords={
            "wavelength": ("wavelength", wavelengths, {"units": str(u.AA)}),
            "band": ("band", bands, {"units": str(u.AA)}),
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
    units: str = "DN",
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
        Epoch of the degradation and of the EVE normalization, by default the launch epoch,
        1 May 2010, as in SolarSoft.
    include_crosstalk : `bool`, optional
        Add the light of the other channel on the same telescope that leaks through this
        channel's focal-plane filter (94 with 304, 131 with 335), by default `True`. For 335
        the leak rivals the channel's own peak.
    include_degradation : `bool`, optional
        Apply ``aiapy.calibrate.degradation`` at ``obstime``, by default `False`, as aiapy's
        degradation at the launch epoch is not exactly 1. Set it, with ``obstime``, to compare
        with observations.
    include_eve_correction : `bool`, optional
        Apply the EVE cross-calibration factor of ``obstime``, SolarSoft's ``/evenorm``, by
        default `True`.
    units : {"DN", "ph"}, optional
        ``"DN"`` (default) folds the camera gain in: the area is in cm2 DN / ph and a response
        built from it integrates DN over wavelength. ``"ph"`` leaves the area in cm2.

    Returns
    -------
    `xarray.Dataset`
        As `create_effective_area` returns, with the curves ``mirror``, ``filter``, ``ccd``,
        ``contamination``, ``crosstalk`` and ``dn_per_photon`` alongside. The attributes
        ``pixel_width``, ``pixel_height``, ``gain`` and ``pair_energy`` describe the AIA
        detector, and `generate_responses` uses them in place of the MUSE values.

    Notes
    -----
    Needs aiapy, which downloads the AIA instrument file on first use, and the correction
    table when a degradation or an EVE factor is applied.
    """
    call_inputs = dict(locals())
    if units not in ("DN", "ph"):
        msg = f"units must be 'DN' or 'ph', got {units!r}"
        raise ValueError(msg)
    try:
        import aiapy  # noqa: PLC0415
        from aiapy.calibrate import degradation  # noqa: PLC0415
        from aiapy.response import Channel  # noqa: PLC0415
    except ImportError:
        msg = "aiapy is required for this function, install it with `pip install muse[aia]`"
        raise ImportError(msg) from None

    obstime = Time(_AIA_LAUNCH_EPOCH if obstime is None else obstime)
    aia_channel = Channel(channel)
    crosstalk = aia_channel.crosstalk * int(include_crosstalk)
    degradation_factor = degradation(channel, obstime).to_value(u.one).item() if include_degradation else 1.0
    eve_factor = aia_channel.eve_correction(obstime).to_value(u.one).item() if include_eve_correction else 1.0
    area = (aia_channel.effective_area + crosstalk) * degradation_factor * eve_factor
    if units == "DN":
        area = area * aia_channel.gain
    # aiapy's gain is the photon energy over the pair energy and the camera gain, so the
    # camera gain follows from any one wavelength.
    photon_energy = (const.h * const.c / aia_channel.wavelength[0]).to(u.eV) / u.ph
    camera_gain = (photon_energy / _AIA_PAIR_ENERGY / aia_channel.gain[0]).to(u.electron / u.DN)
    pixel_size = ((aia_channel.plate_scale * u.pix) ** 0.5).to(u.arcsec)
    effective_area = create_effective_area(
        area,
        aia_channel.wavelength,
        band=channel,
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


def read_effective_area(effective_area_file: str | Path) -> xr.Dataset:
    """
    Read an effective-area file written by `create_effective_area`.

    Parameters
    ----------
    effective_area_file : `str` or `pathlib.Path`
        A zarr store or a netCDF file.

    Returns
    -------
    `xarray.Dataset`
        The dataset in memory, with ``effective_area(wavelength, band)``, its units convertible
        to cm2 or cm2 DN / ph, and ``wavelength`` in a wavelength unit.
    """
    effective_area_file = Path(effective_area_file)
    opened = (
        xr.open_zarr(effective_area_file, consolidated=False)
        if effective_area_file.is_dir()
        else xr.open_dataset(effective_area_file)
    )
    with opened:
        effective_area = opened.load()
    label = effective_area_file.name
    if "effective_area" not in effective_area or "band" not in effective_area.coords:
        msg = f"{label} must carry effective_area and a band coordinate"
        raise ValueError(msg)
    unit = require_unit(effective_area, "effective_area", f"{label}: effective_area")
    if not (unit.is_equivalent(u.cm**2) or unit.is_equivalent(_DN_PER_PHOTON)):
        msg = f"{label}: effective_area units {unit} must be convertible to cm2 or cm2 DN / ph"
        raise ValueError(msg)
    coord_as_unit(effective_area, "wavelength", u.AA, f"{label}: wavelength")
    # zarr hands back arrays with an explicit byte order, which netCDF then writes with a warning.
    native = {
        name: variable.astype(variable.dtype.newbyteorder("="))
        for name, variable in effective_area.variables.items()
        if variable.dtype.byteorder in "<>"
    }
    effective_area = effective_area.assign(
        {name: variable for name, variable in native.items() if name not in effective_area.coords}
    )
    effective_area = effective_area.assign_coords(
        {name: variable for name, variable in native.items() if name in effective_area.coords}
    )
    return effective_area.assign(effective_area=effective_area.effective_area.transpose("wavelength", "band"))


@u.quantity_input(band=u.AA)
def select_effective_area_curve(effective_area: xr.Dataset, *, band: u.Quantity | None = None) -> xr.DataArray:
    """
    The one-dimensional area curve of one band, as `create_spectral_response` takes it.

    Parameters
    ----------
    effective_area : `xarray.Dataset`
        From `create_effective_area` or `read_effective_area`.
    band : `astropy.units.Quantity`, optional
        The band to select; by default the only one.

    Returns
    -------
    `xarray.DataArray`
        ``effective_area(wavelength)`` with its units and the ``wavelength`` coordinate in Angstrom.
    """
    if band is None:
        if effective_area.sizes["band"] != 1:
            msg = f"the dataset holds {effective_area.sizes['band']} bands; pass band"
            raise ValueError(msg)
        curve = effective_area.effective_area.isel(band=0)
    else:
        curve = effective_area.effective_area.sel(band=band.to_value(u.AA))
    wavelength = coord_as_unit(effective_area, "wavelength", u.AA, "wavelength")
    return curve.drop_vars("band").assign_coords(wavelength=wavelength)


def effective_area_wavelength_range(effective_areas: Sequence[xr.Dataset]) -> u.Quantity:
    """
    The wavelength range every effective area spans together, as a line list must cover.

    Parameters
    ----------
    effective_areas : `list` of `xarray.Dataset`
        From `create_effective_area` or `read_effective_area`.

    Returns
    -------
    `astropy.units.Quantity`
        The lowest and the highest wavelength of all of them.
    """
    if not effective_areas:
        msg = "pass at least one effective area"
        raise ValueError(msg)
    wavelengths = [np.asarray(coord_as_unit(area, "wavelength", u.AA, "wavelength")) for area in effective_areas]
    return [min(w.min() for w in wavelengths), max(w.max() for w in wavelengths)] * u.AA
