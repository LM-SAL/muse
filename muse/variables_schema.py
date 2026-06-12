"""
Validated attrs schemas for package-wide variables.
"""

import numbers
from contextlib import suppress
from collections.abc import Mapping

import numpy as np
import xarray as xr
from attrs import cmp_using, converters, define, field, validators

import astropy.units as u
from astropy.units import imperial

__all__ = [
    "FrozenDict",
    "InstrumentDefaults",
]


class FrozenDict(dict):
    """
    A read-only `dict` subclass.

    Unlike `types.MappingProxyType`, instances are picklable, which keeps frozen
    defaults usable with multiprocessing.
    """

    def _readonly(self, *_args, **_kwargs):
        msg = f"{type(self).__name__} is read-only"
        raise TypeError(msg)

    __setitem__ = _readonly
    __delitem__ = _readonly
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


def _instance(type_):
    """
    Validator: value is `None` or an instance of ``type_``.
    """
    return validators.optional(validators.instance_of(type_))


def _nested_tuple(value):
    return tuple(tuple(item) if isinstance(item, (list, tuple)) else item for item in value)


def _set_readonly(value):
    with suppress(AttributeError, ValueError):
        value.setflags(write=False)


def _readonly_quantity(value, unit):
    # dtype=None preserves the input dtype; the default casts integers to float.
    quantity = u.Quantity(value, copy=True, dtype=None)
    # ``.to`` always converts to float; skip it for matching units to preserve dtype.
    if quantity.unit != unit:
        quantity = quantity.to(unit)
    _set_readonly(quantity)
    return quantity


def _readonly_array(value):
    array = np.asanyarray(value).copy()
    array.setflags(write=False)
    return array


def _readonly_data_array(value, unit=None):
    data_array = value.copy(deep=True)
    if unit is not None:
        quantity = u.Quantity(data_array.data, copy=True, dtype=None)
        if quantity.unit == u.dimensionless_unscaled:
            msg = f"DataArray values must have units convertible to {unit}"
            raise u.UnitsError(msg)
        if quantity.unit != unit:
            quantity = quantity.to(unit)
        data_array.data = quantity
    _set_readonly(data_array.data)
    for coord in data_array.coords.values():
        _set_readonly(coord.data)
    return data_array


def _immutable_value(value):
    if isinstance(value, u.Quantity):
        quantity = value.copy()
        _set_readonly(quantity)
        return quantity
    if isinstance(value, xr.DataArray):
        return _readonly_data_array(value)
    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    if isinstance(value, Mapping):
        return _immutable_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    return value


def _immutable_mapping(mapping):
    return FrozenDict({key: _immutable_value(value) for key, value in mapping.items()})


def _quantity_mapping(unit):
    """
    Converter factory: coerce every value of a mapping to `astropy.units.Quantity` and
    normalize it to ``unit``.

    `None` passes through.
    """

    def to_unit(mapping):
        return FrozenDict({key: _readonly_quantity(value, unit) for key, value in mapping.items()})

    return converters.optional(to_unit)


def _eq_optional(equal):
    """
    Wrap an equality function so `None` only equals `None`.
    """

    def compare(a, b):
        if a is None or b is None:
            return a is b
        return bool(equal(a, b))

    return compare


def _mapping_values_equal(a, b):
    """
    Compare two mappings whose values may be arrays.

    Values are compared with `numpy.array_equal`, which ignores units; safe here
    because the converters normalize units before storage.
    """
    return a.keys() == b.keys() and all(np.array_equal(a[key], b[key]) for key in a)


def _channel_coordinates(data_array, name):
    if data_array is None:
        return None
    if "channel" not in data_array.dims:
        msg = f"{name} must have a 'channel' dimension"
        raise ValueError(msg)
    if "channel" not in data_array.coords:
        msg = f"{name} must have a 'channel' coordinate"
        raise ValueError(msg)
    if data_array.coords["channel"].dims != ("channel",):
        msg = f"{name}.channel must be one-dimensional along the 'channel' dimension"
        raise ValueError(msg)
    return tuple(np.asarray(data_array.coords["channel"].values).tolist())


def _quantity_set(quantity, unit):
    return set(np.asarray(quantity.to_value(unit)).tolist())


def _flat_line_names(main_lines):
    return tuple(line for channel_lines in main_lines for line in channel_lines)


def _validate_matching_keys(first, first_name, second, second_name):
    if first is None or second is None:
        return
    if first.keys() != second.keys():
        msg = f"{first_name} and {second_name} must use matching keys"
        raise ValueError(msg)


_array_eq = cmp_using(eq=_eq_optional(np.array_equal), class_name="_ArrayEq")
_data_array_eq = cmp_using(eq=_eq_optional(lambda a, b: a.equals(b)), class_name="_DataArrayEq")
_mapping_eq = cmp_using(eq=_eq_optional(_mapping_values_equal), class_name="_MappingEq")


@define(frozen=True, kw_only=True)
class InstrumentDefaults:
    """
    Base class which lists the required parameters and documentation for each parameter.
    These parameters should be set in subclasses.

    Furthermore, these parameters are used for functions and methods within the muse
    library, this is not a general instrument defaults class.

    All fields are validated and normalized on construction. Instances are immutable;
    create modified copies with `attrs.evolve`.
    """

    FWHM_TO_SIGMA = 2.355  # This was 2.35482 in guasslobes.py
    """
    Conversion factor between FWHM and Gaussian sigma, 2 * sqrt(2 * ln 2), truncated.
    """

    # Context Imager (CI) parameters
    dx_pixel_CI: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    Spatial pixel size along x-axis for the CI.

    Normalized to arcseconds.
    """

    dy_pixel_CI: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    Spatial pixel size along y-axis for the CI.

    Normalized to arcseconds.
    """

    slit_sep_CI: u.Quantity | None = field(default=None, converter=_quantity(u.pixel))
    """
    Slit separation in pixels to convert the CI into SG format.
    """

    full_well_depth_CI: u.Quantity | None = field(default=None, converter=_quantity(u.DN))
    """
    Full well depth in DN for the CI.
    """

    # Spectrograph (SG) parameters
    dx_pixel_SG: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    Spatial pixel size along x-axis for the SG.

    Normalized to arcseconds.
    """

    dy_pixel_SG: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    Spatial pixel size along y-axis for the SG.

    Normalized to arcseconds.
    """

    slit_sep_SG: u.Quantity | None = field(default=None, converter=_quantity(u.pixel))
    """
    Slit separation in pixels for the SG.
    """

    pixels_SG: u.Quantity | None = field(default=None, converter=_quantity(u.pix))
    """
    Number of pixels along the spectral dimension for the SG.
    """

    number_of_slits_SG: int | None = field(default=None, validator=_instance(numbers.Integral))
    """
    Number of slits in the SG.
    """

    pixels_between_slits: u.Quantity | None = field(default=None, converter=_quantity(u.pixel))
    """
    Number of pixels between slits for the SG.
    """

    spectral_slit_separation_SG: u.Quantity | None = field(default=None, converter=_quantity(u.AA))
    """
    Spectral slit separation for the SG.

    Normalized to Angstroms.
    """

    steps_per_raster_SG: int | None = field(default=None, validator=_instance(numbers.Integral))
    """
    Number of steps per raster for the SG.
    """

    # Diffraction parameters
    mesh_transmission: Mapping | None = field(default=None, converter=converters.optional(_immutable_mapping))
    """
    Mesh transmission coefficient, keyed by diffraction channel.
    """

    oversample_x_SG: int | None = field(default=None, validator=_instance(numbers.Integral))
    """
    Oversampling factor along x-axis for the SG.
    """

    oversample_y_SG: int | None = field(default=None, validator=_instance(numbers.Integral))
    """
    Oversampling factor along y-axis for the SG.
    """

    center_diffraction: bool | None = field(default=None, validator=_instance((bool, np.bool_)))
    """
    Centers the peak of the PSF before convolution.
    """

    lpi: Mapping | None = field(default=None, converter=_quantity_mapping(1 / imperial.inch), eq=_mapping_eq)
    """
    Lines per inch of mesh grid, keyed by diffraction channel.

    The set of keys defines which channels have diffraction patterns.
    """

    psf_fwhm: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    FWHM of the core PSF.

    Normalized to arcseconds.
    """

    psf_fwhm_x: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    FWHM of the core PSF in x direction.

    Normalized to arcseconds.
    """

    psf_fwhm_y: u.Quantity | None = field(default=None, converter=_quantity(u.arcsec))
    """
    FWHM of the core PSF in y direction.

    Normalized to arcseconds.
    """

    # Other Properties
    data_compression: int | None = field(default=None, validator=_instance(numbers.Integral))
    """
    Data compression level.
    """

    ccd_gain: u.Quantity | None = field(default=None, converter=_quantity(u.electron / u.DN))
    """
    CCD gain in electrons per DN.
    """

    # Synthesis/inversions
    sum_over_dims_synthesis: tuple | None = field(default=None, converter=converters.optional(tuple))
    """
    Dimensions to sum over during synthesis/inversions.
    """

    main_lines_SG: tuple | None = field(default=None, converter=converters.optional(_nested_tuple))
    """
    Main spectral lines for the SG, grouped per channel.
    """

    main_lines_SG_wavelength: Mapping | None = field(default=None, converter=_quantity_mapping(u.AA), eq=_mapping_eq)
    """
    Center of the main spectral lines in angstrom, keyed by line name.

    Values normalized to Angstroms.
    """

    bands_SG: u.Quantity | None = field(default=None, converter=_quantity(u.AA), eq=_array_eq)
    """
    Wavelength bands for the SG.

    Normalized to Angstroms.
    """

    fov_mode: str | None = field(default=None, validator=_instance(str))
    """
    This is the pad method used by `xarray.DataArray.pad`
    """

    fov_restype: str | None = field(default=None, validator=_instance(str))
    """
    Type of tiling and resolution matching.
    """

    fov_sub_interpolation: int | None = field(default=None, validator=_instance(numbers.Integral))
    """
    Does a subgrid interpolation.
    """

    exposure_times_SG: Mapping | None = field(default=None, converter=_quantity_mapping(u.s), eq=_mapping_eq)
    """
    Typical exposure times for the SG, keyed by solar condition.

    Values normalized to seconds.
    """

    exposure_times_CI: Mapping | None = field(default=None, converter=_quantity_mapping(u.s), eq=_mapping_eq)
    """
    Typical exposure times for the CI, keyed by solar condition.

    Values normalized to seconds.
    """

    # Response creation
    electron_density: u.Quantity = field(default=1e9 / u.cm**3, converter=_required_quantity(1 / u.cm**3))
    """
    Effective density for response creation.

    Normalized to inverse cubic centimeters.
    """

    electron_pressure: u.Quantity = field(default=3e15 * u.K / u.cm**3, converter=_required_quantity(u.K / u.cm**3))
    """
    Effective pressure for response creation.

    Normalized to Kelvin per cubic centimeter.
    """

    response_logT_min: float = field(default=4.8, converter=float)
    """
    Minimum logT for response creation.
    """

    target_logT: Mapping | None = field(default=None, converter=converters.optional(_immutable_mapping), eq=_mapping_eq)
    """
    Target logT values for different solar conditions.
    """

    target_vdop: Mapping | None = field(default=None, converter=_quantity_mapping(u.km / u.s), eq=_mapping_eq)
    """
    Target vdop values for different solar conditions.

    Values normalized to kilometers per second.
    """

    minimum_abundance: float | None = field(default=None, converter=converters.optional(float))
    """
    Minimum abundance considered for response creation.
    """

    response_method: str = field(default="linear", validator=validators.instance_of(str))
    """
    Type of interpolation in the response creation.
    """

    normalization: float = field(default=1e-27, converter=float)
    """
    Normalization in the response function.
    """

    num_lines_keep: int | None = field(default=None, validator=_instance(numbers.Integral))
    """
    Number of lines conserved independently for the response creation.
    """

    sum_lines: bool | None = field(default=None, validator=_instance((bool, np.bool_)))
    """
    Sum all lines for response creation.
    """

    initial_wavelength_SG: xr.DataArray | None = field(default=None, converter=_data_array(u.AA), eq=_data_array_eq)
    """
    Wavelength at SG_xpixel=0 for slit=0, in Angstroms.

    Values normalized to Angstroms.
    """

    channel_spectral_order: xr.DataArray | None = field(default=None, converter=_data_array(), eq=_data_array_eq)
    """
    Spectral order of main line for each band (channel)
    """

    def __attrs_post_init__(self):
        self._validate_channel_fields()
        self._validate_line_fields()
        _validate_matching_keys(self.target_logT, "target_logT", self.target_vdop, "target_vdop")
        _validate_matching_keys(
            self.exposure_times_SG, "exposure_times_SG", self.exposure_times_CI, "exposure_times_CI"
        )
        _validate_matching_keys(self.lpi, "lpi", self.mesh_transmission, "mesh_transmission")

    def _validate_channel_fields(self):
        initial_channels = _channel_coordinates(self.initial_wavelength_SG, "initial_wavelength_SG")
        order_channels = _channel_coordinates(self.channel_spectral_order, "channel_spectral_order")
        if initial_channels is not None and order_channels is not None and initial_channels != order_channels:
            msg = "initial_wavelength_SG and channel_spectral_order must have matching channel coordinates"
            raise ValueError(msg)

        channels = initial_channels or order_channels
        if channels is None:
            return
        channel_set = set(channels)
        if self.bands_SG is not None and _quantity_set(self.bands_SG, u.AA) != channel_set:
            msg = "bands_SG unique channels must match the SG channel coordinates"
            raise ValueError(msg)

        for mapping_name in ("lpi", "mesh_transmission"):
            mapping = getattr(self, mapping_name)
            if mapping is not None and not set(mapping).issubset(channel_set):
                msg = f"{mapping_name} channel keys must be present in the SG channel coordinates"
                raise ValueError(msg)

    def _validate_line_fields(self):
        if self.main_lines_SG is None:
            return

        line_names = _flat_line_names(self.main_lines_SG)
        if self.bands_SG is not None and len(self.bands_SG) != len(line_names):
            msg = "bands_SG must contain one entry for each line in main_lines_SG"
            raise ValueError(msg)

        if self.main_lines_SG_wavelength is None:
            return
        if missing := sorted(set(line_names) - set(self.main_lines_SG_wavelength)):
            msg = f"main_lines_SG_wavelength is missing entries for {missing}"
            raise ValueError(msg)

    @property
    def instrumental_width_sg(self):
        """
        Instrumental width sigma in Angstroms.

        FWHM in pixels converted to 1 sigma in Angstroms. Value, .0815, provided by Paul
        B. Value has spectral plate scale baked in and should be calculated using a
        future property.
        """
        if self.channel_spectral_order is None:
            msg = "instrumental_width_sg requires channel_spectral_order"
            raise ValueError(msg)
        return 0.0815 * u.AA / self.FWHM_TO_SIGMA / self.channel_spectral_order
