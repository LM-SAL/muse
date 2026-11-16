"""
Validated attrs schemas for package-wide variables.
"""

import numbers
from contextlib import suppress

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
    A read-only `dict` subclass.

    Unlike `types.MappingProxyType`, instances are picklable, which keeps frozen
    defaults usable with multiprocessing. The mapping itself stays read-only across a
    pickle round-trip, but numpy or `astropy.units.Quantity` values it holds regain
    their writable flag on unpickle; this is harmless because each process operates on
    its own copy.
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
    Converter factory: coerce the value to `astropy.units.Quantity` and normalize it to
    ``unit``.
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
    Converter factory: copy an `xarray.DataArray`, make its data read-only, and
    optionally normalize quantity-valued data to ``unit``.

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
    return FrozenDict({channel: lines if isinstance(lines, str) else tuple(lines) for channel, lines in value.items()})


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
    Container of instrument properties used by functions and methods within the muse
    library; not a general-purpose instrument defaults class.

    Every parameter is optional and defaults to `None`. Populate the ones a given
    instrument needs (see `muse.variables` for the MUSE instance). Each field documents
    its meaning and the unit it is normalized to.

    Per-channel calibrations for both detectors are keyed by a ``channel`` dimension:
    the ``_SG``/``_CI`` field suffix identifies the detector, and the two channel sets
    are independent of each other.

    Fields are validated and normalized on construction. Instances prevent top-level
    attribute reassignment; create modified copies with `attrs.evolve`.

    This is not a deeply immutable container. The converters copy input values and make
    array buffers read-only, but `xarray.DataArray` fields can still have ``.data``,
    coordinates, ``.attrs`` and ``.encoding`` reassigned or mutated through xarray APIs.
    Because nested state can change, instances are intentionally unhashable.
    """

    # Context Imager (CI) parameters
    pixel_width_CI: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    Spatial pixel size along x-axis for the CI.

    Normalized to arcseconds.
    """

    pixel_height_CI: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    Spatial pixel size along y-axis for the CI.

    Normalized to arcseconds.
    """

    # Spectrograph (SG) parameters
    pixel_width_SG: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    Spatial pixel size along x-axis for the SG.

    Normalized to arcseconds.
    """

    pixel_height_SG: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    Spatial pixel size along y-axis for the SG.

    Normalized to arcseconds.
    """

    detector_x_pixels_SG: int | None = field(default=None, converter=_int_converter)
    """
    Number of pixels along the spectral dimension for the SG.
    """

    number_of_slits_SG: int | None = field(default=None, converter=_int_converter)
    """
    Number of slits in the SG.
    """

    pixels_between_slits_SG: u.Quantity | None = field(default=None, converter=_quantity(u.pixel))
    """
    Number of pixels between slits for the SG.
    """

    spectral_slit_separation_SG: xr.DataArray | None = field(
        default=None, converter=_data_array(u.AA), eq=_data_array_eq
    )
    """
    Nominal spectral separation of adjacent SG slits on the detector, for each channel
    at its own spectral order.

    Values normalized to Angstroms.
    """

    waveband_half_width_SG: u.Quantity | None = field(default=None, converter=_quantity(u.AA))
    """
    Half of the wavelength range an SG channel covers at first order; at spectral order
    n the half width is this value over n.

    Normalized to Angstroms.
    """

    wavelength_step_SG: u.Quantity | None = field(default=None, converter=_quantity(u.AA))
    """
    Sampling of the wavelength grid an SG spectral response is built on.

    Normalized to Angstroms.
    """

    steps_per_raster_SG: int | None = field(default=None, converter=_int_converter)
    """
    Number of steps per raster for the SG.
    """

    # Effective areas
    main_line_effective_area_SG: xr.DataArray | None = field(
        default=None, converter=_data_array(u.cm**2), eq=_data_array_eq
    )
    """
    Effective area of the main line for the SG.

    Normalized to square centimeters.
    """

    main_line_effective_area_CI: xr.DataArray | None = field(
        default=None, converter=_data_array(u.cm**2), eq=_data_array_eq
    )
    """
    Effective area of the main line for the CI.

    Normalized to square centimeters and keyed by ``channel``.
    """

    # Other Properties
    gain_SG: xr.DataArray | None = field(default=None, converter=_data_array(u.electron / u.DN), eq=_data_array_eq)
    """
    SG CCD gain in electrons per DN, keyed by ``channel``.
    """

    gain_CI: xr.DataArray | None = field(default=None, converter=_data_array(u.electron / u.DN), eq=_data_array_eq)
    """
    CI CCD gain in electrons per DN, keyed by ``channel``.
    """

    pair_energy_SG: xr.DataArray | None = field(
        default=None, converter=_data_array(u.eV / u.electron), eq=_data_array_eq
    )
    """
    Mean energy that frees one electron-hole pair in the SG detector, keyed by
    ``channel``.
    """

    pair_energy_CI: xr.DataArray | None = field(
        default=None, converter=_data_array(u.eV / u.electron), eq=_data_array_eq
    )
    """
    Mean energy that frees one electron-hole pair in the CI detector, keyed by
    ``channel``.

    CI calibration fields may cover different channel sets.
    """

    main_lines_by_channel_SG: FrozenDict | None = field(default=None, converter=converters.optional(_frozen_lines))
    """
    Main spectral lines for the SG, keyed by channel.
    """

    wavelength_start_SG: xr.DataArray | None = field(default=None, converter=_data_array(u.AA), eq=_data_array_eq)
    """
    Wavelength at detector pixel zero for slit zero, in Angstroms.

    Values normalized to Angstroms.
    """

    channel_spectral_order_SG: xr.DataArray | None = field(default=None, converter=_data_array(), eq=_data_array_eq)
    """
    Spectral order of main line for each band (channel)
    """

    instrumental_fwhm_SG: xr.DataArray | None = field(default=None, converter=_data_array(u.AA), eq=_data_array_eq)
    """
    Instrumental FWHM of each SG channel, with the spectral plate scale and the spectral
    order of the channel baked in.

    Values normalized to Angstroms.
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
        # SG and CI calibrations share the "channel" dimension name but carry
        # independent channel sets. The _CI suffix identifies a detector's fields:
        # CI DataArrays are checked for the dimension and excluded from the SG
        # cross-field matching below.
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
    The field ``name`` of ``defaults``, which the caller cannot do without: a custom
    object that leaves it unset is refused with the name of the field.
    """
    value = getattr(defaults, name)
    if value is None:
        msg = f"instrument_defaults has no {name}; give it one, e.g. attrs.evolve(DEFAULTS_MUSE, {name}=...)"
        raise ValueError(msg)
    return value
