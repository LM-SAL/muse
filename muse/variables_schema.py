"""
Validated attrs schemas for package-wide variables.
"""

import numbers
from contextlib import suppress
from collections.abc import Mapping

import numpy as np
import xarray as xr
from attrs import Converter, cmp_using, converters, define, field, fields

import astropy.units as u

__all__ = [
    "FrozenDict",
    "InstrumentDefaults",
]


class FrozenDict(dict):
    """
    A read-only `dict`.

    Unlike `types.MappingProxyType` it pickles, so frozen defaults work with
    multiprocessing. Array values are writable again after unpickling, which is harmless
    as each process has its own copy.
    """

    def _readonly(self, *_args, **_kwargs):
        msg = f"{type(self).__name__} is read-only"
        raise TypeError(msg)

    __setitem__ = _readonly
    __delitem__ = _readonly
    __or__ = _readonly
    __ror__ = _readonly
    __ior__ = _readonly
    clear = _readonly
    pop = _readonly
    popitem = _readonly
    setdefault = _readonly
    update = _readonly

    def __reduce__(self):
        return (type(self), (dict(self),))


def _required_quantity(unit):
    """
    Converter to a read-only `astropy.units.Quantity` in ``unit``.
    """

    def to_unit(value):
        return _readonly_quantity(value, unit)

    return to_unit


def _quantity(unit):
    """
    Like `_required_quantity`, but `None` passes through.
    """
    return converters.optional(_required_quantity(unit))


def _data_array(unit=None):
    """
    Converter to a read-only copy of an `xarray.DataArray`, in ``unit`` when given.

    `None` passes through.
    """

    def to_unit(value):
        return _readonly_data_array(value, unit)

    return converters.optional(to_unit)


def _optional_int(value, attribute):
    """
    Converter: accept integer values, including numpy integer scalars, but reject bools.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        msg = f"{attribute.name} must be an integer, not {type(value).__name__}"
        raise TypeError(msg)
    return int(value)


_int_converter = Converter(_optional_int, takes_field=True)


def _frozen_lines(value):
    if not isinstance(value, Mapping):
        msg = "main_lines_by_channel_SG must be a mapping from channel to line names"
        raise TypeError(msg)
    # Keyed by the channel in Angstrom, as the generators' main_lines_by_channel is.
    return FrozenDict(
        {
            round(u.Quantity(channel, u.AA).value): lines if isinstance(lines, str) else tuple(lines)
            for channel, lines in value.items()
        }
    )


def _set_readonly(value):
    with suppress(AttributeError, ValueError):
        value.setflags(write=False)


def _readonly_quantity(value, unit):
    quantity = u.Quantity(value).to(unit)
    _set_readonly(quantity)
    return quantity


def _readonly_data_array(value, unit=None):
    data_array = value.copy(deep=True)
    if unit is not None:
        quantity = u.Quantity(data_array.data)
        if quantity.unit == u.dimensionless_unscaled:
            msg = f"DataArray values must have units convertible to {unit}"
            raise u.UnitsError(msg)
        data_array.data = quantity.to(unit)
    _set_readonly(data_array.data)
    for coord in data_array.coords.values():
        _set_readonly(coord.data)
    return data_array


def _eq_optional(equal):
    """
    Wrap an equality function so `None` only equals `None`.
    """

    def compare(a, b):
        if a is None or b is None:
            return a is b
        return bool(equal(a, b))

    return compare


def _channel_coordinates(data_array, name, dimension="channel"):
    if data_array is None:
        return None
    if dimension not in data_array.dims:
        msg = f"{name} must have a {dimension!r} dimension"
        raise ValueError(msg)
    if dimension not in data_array.coords:
        msg = f"{name} must have a {dimension!r} coordinate"
        raise ValueError(msg)
    if data_array.coords[dimension].dims != (dimension,):
        msg = f"{name}.{dimension} must be one-dimensional along the {dimension!r} dimension"
        raise ValueError(msg)
    return tuple(np.asarray(data_array.coords[dimension].values).tolist())


_data_array_eq = cmp_using(eq=_eq_optional(lambda a, b: a.equals(b)), class_name="_DataArrayEq")


@define(frozen=True, kw_only=True, hash=False)
class InstrumentDefaults:
    """
    Instrument properties the muse functions use.

    Every field is optional and defaults to `None`; set the ones an instrument needs
    (`muse.variables` holds the MUSE values). On construction each field is validated,
    copied and normalized to the unit it documents.

    Per-channel fields have a ``channel`` dimension. The ``_SG`` and ``_CI`` suffixes
    name the detector, and the two channel sets are independent. CI fields may cover
    different channels from one another.

    Fields cannot be reassigned; make a modified copy with `attrs.evolve`. Array buffers
    are read-only, but xarray can still change the ``.data``, coordinates, ``.attrs``
    and ``.encoding`` of an `xarray.DataArray` field, so instances are unhashable.
    """

    # Context Imager (CI) parameters
    pixel_width_CI: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    CI pixel size along x, in arcseconds.
    """

    pixel_height_CI: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    CI pixel size along y, in arcseconds.
    """

    # Spectrograph (SG) parameters
    pixel_width_SG: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    SG pixel size along x, in arcseconds.
    """

    pixel_height_SG: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    SG pixel size along y, in arcseconds.
    """

    detector_x_pixels_SG: int | None = field(default=None, converter=_int_converter)
    """
    SG detector size along the spectral dimension, a number of pixels.
    """

    number_of_slits_SG: int | None = field(default=None, converter=_int_converter)
    """
    Number of SG slits.
    """

    pixels_between_slits_SG: u.Quantity | None = field(default=None, converter=_quantity(u.pixel))
    """
    Detector distance between adjacent SG slits, in pixels.
    """

    spectral_slit_separation_SG: xr.DataArray | None = field(
        default=None, converter=_data_array(u.AA), eq=_data_array_eq
    )
    """
    Nominal spectral separation of adjacent SG slits on the detector for each channel,
    at the order of the channel, in Angstroms.
    """

    waveband_half_width_SG: u.Quantity | None = field(default=None, converter=_quantity(u.AA))
    """
    Half the wavelength range of an SG channel at first order, in Angstroms; at order n
    it is this over n.
    """

    wavelength_step_SG: u.Quantity | None = field(default=None, converter=_quantity(u.AA))
    """
    Step of the wavelength grid SG spectral responses are built on, in Angstroms.
    """

    steps_per_raster_SG: int | None = field(default=None, converter=_int_converter)
    """
    Number of steps in an SG raster.
    """

    # Effective areas
    main_line_effective_area_SG: xr.DataArray | None = field(
        default=None, converter=_data_array(u.cm**2), eq=_data_array_eq
    )
    """
    SG effective area at the main line of each channel, in square centimeters.
    """

    main_line_effective_area_CI: xr.DataArray | None = field(
        default=None, converter=_data_array(u.cm**2), eq=_data_array_eq
    )
    """
    CI effective area at the main line of each channel, in square centimeters.
    """

    # Other Properties
    gain_SG: xr.DataArray | None = field(default=None, converter=_data_array(u.electron / u.DN), eq=_data_array_eq)
    """
    SG CCD gain of each channel, in electrons per DN.
    """

    gain_CI: xr.DataArray | None = field(default=None, converter=_data_array(u.electron / u.DN), eq=_data_array_eq)
    """
    CI CCD gain of each channel, in electrons per DN.
    """

    pair_energy_SG: xr.DataArray | None = field(
        default=None, converter=_data_array(u.eV / u.electron), eq=_data_array_eq
    )
    """
    Mean energy that frees one electron-hole pair in the SG detector for each channel,
    in eV per electron.
    """

    pair_energy_CI: xr.DataArray | None = field(
        default=None, converter=_data_array(u.eV / u.electron), eq=_data_array_eq
    )
    """
    Mean energy that frees one electron-hole pair in the CI detector for each channel,
    in eV per electron.
    """

    main_lines_by_channel_SG: FrozenDict | None = field(default=None, converter=converters.optional(_frozen_lines))
    """
    Names of the main SG lines, keyed by channel.
    """

    wavelength_start_SG: xr.DataArray | None = field(default=None, converter=_data_array(u.AA), eq=_data_array_eq)
    """
    Wavelength at detector pixel zero of slit zero for each channel, in Angstroms.
    """

    channel_spectral_order_SG: xr.DataArray | None = field(default=None, converter=_data_array(), eq=_data_array_eq)
    """
    Spectral order at which each SG channel observes its main line.
    """

    instrumental_fwhm_SG: xr.DataArray | None = field(default=None, converter=_data_array(u.AA), eq=_data_array_eq)
    """
    Instrumental FWHM of each SG channel in Angstroms, the plate scale and order of the
    channel included.
    """

    def __repr__(self) -> str:
        lines = []
        for attribute in fields(type(self)):
            value = getattr(self, attribute.name)
            if value is None:
                continue
            text = "\n        ".join(repr(value).splitlines())
            lines.append(f"    {attribute.name}={text},")
        return f"{type(self).__name__}(\n" + "\n".join(lines) + "\n)"

    def __attrs_post_init__(self):
        self._validate_channel_fields()

    def _validate_channel_fields(self):
        # CI fields share the dimension name, not the channels, so they skip the SG matching.
        array_fields = {
            a.name: value for a in fields(type(self)) if isinstance(value := getattr(self, a.name), xr.DataArray)
        }
        for name, value in array_fields.items():
            if name.endswith("_CI"):
                _channel_coordinates(value, name, dimension="channel")

        channels_by_field = {
            name: _channel_coordinates(value, name)
            for name, value in array_fields.items()
            if not name.endswith("_CI") and "channel" in value.dims
        }
        reference, channels = next(iter(channels_by_field.items()), (None, None))
        for name, field_channels in channels_by_field.items():
            if field_channels != channels:
                msg = f"{name} and {reference} must have matching channel coordinates"
                raise ValueError(msg)
        lines = self.main_lines_by_channel_SG
        if lines is not None and channels is not None and set(lines) != set(channels):
            msg = f"main_lines_by_channel_SG must be keyed by the channels of {reference}, {list(channels)}"
            raise ValueError(msg)


def _required(defaults: InstrumentDefaults, name: str):
    """
    The field ``name`` of ``defaults``, or a `ValueError` naming it when it is unset.
    """
    value = getattr(defaults, name)
    if value is None:
        msg = f"instrument_defaults has no {name}; give it one, e.g. attrs.evolve(DEFAULTS_MUSE, {name}=...)"
        raise ValueError(msg)
    return value
