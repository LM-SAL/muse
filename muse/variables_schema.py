"""
Validated attrs schemas for package-wide variables.
"""

from types import MappingProxyType
from contextlib import suppress
from collections.abc import Mapping

import numpy as np
import xarray as xr
from attrs import converters, define, field, validators

import astropy.units as u

__all__ = [
    "InstrumentDefaults",
]


def _quantity(unit):
    """
    Converter factory: coerce the value to `astropy.units.Quantity` and normalize it to
    ``unit``.

    `None` passes through.
    """

    def to_unit(value):
        return _readonly_quantity(value, unit)

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
    with suppress(AttributeError, ValueError):
        np.asarray(value).setflags(write=False)


def _readonly_quantity(value, unit):
    quantity = u.Quantity(value, copy=True).to(unit, copy=True)
    _set_readonly(quantity)
    return quantity


def _readonly_array(value):
    array = np.asanyarray(value).copy()
    array.setflags(write=False)
    return array


def _readonly_data_array(value):
    data_array = value.copy(deep=True)
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
    return MappingProxyType({key: _immutable_value(value) for key, value in mapping.items()})


def _quantity_mapping(unit):
    """
    Converter factory: coerce every value of a mapping to `astropy.units.Quantity` and
    normalize it to ``unit``.

    `None` passes through.
    """

    def to_unit(mapping):
        return MappingProxyType({key: _readonly_quantity(value, unit) for key, value in mapping.items()})

    return converters.optional(to_unit)


@define(frozen=True, kw_only=True, eq=False)
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

    number_of_slits_SG: int | None = field(default=None, validator=_instance(int))
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

    steps_per_raster_SG: int | None = field(default=None, validator=_instance(int))
    """
    Number of steps per raster for the SG.
    """

    # Diffraction parameters
    mesh_transmission: Mapping | None = field(default=None, converter=converters.optional(_immutable_mapping))
    """
    Mesh transmission coefficient, keyed by diffraction channel.
    """

    oversample_x_SG: int | None = field(default=None, validator=_instance(int))
    """
    Oversampling factor along x-axis for the SG.
    """

    oversample_y_SG: int | None = field(default=None, validator=_instance(int))
    """
    Oversampling factor along y-axis for the SG.
    """

    center_diffraction: bool | None = field(default=None, validator=_instance(bool))
    """
    Centers the peak of the PSF before convolution.
    """

    lpi: Mapping | None = field(default=None, converter=converters.optional(_immutable_mapping))
    """
    Line Per Inch of mesh grid, keyed by diffraction channel.

    The set of keys defines which channels have diffraction patterns.
    """

    psf_fwhm: float | None = field(default=None, converter=converters.optional(float))
    """
    FWHM of the core PSF.
    """

    psf_fwhm_x: float | None = field(default=None, converter=converters.optional(float))
    """
    FWHM of the core PSF in x direction.
    """

    psf_fwhm_y: float | None = field(default=None, converter=converters.optional(float))
    """
    FWHM of the core PSF in y direction.
    """

    # Other Properties
    data_compression: int | None = field(default=None, validator=_instance(int))
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

    main_lines_SG_wavelength: Mapping | None = field(default=None, converter=converters.optional(_immutable_mapping))
    """
    Center of the main spectral lines in angstrom, keyed by line name.
    """

    bands_SG: np.ndarray | None = field(default=None, converter=converters.optional(_readonly_array))
    """
    Wavelength bands for the SG.
    """

    fov_mode: str | None = field(default=None, validator=_instance(str))
    """
    This is the pad method used by `xarray.DataArray.pad`
    """

    fov_restype: str | None = field(default=None, validator=_instance(str))
    """
    Type of tiling and resolution matching.
    """

    fov_sub_interpolation: int | None = field(default=None, validator=_instance(int))
    """
    Does a subgrid interpolation.
    """

    exposure_times_SG: Mapping | None = field(default=None, converter=_quantity_mapping(u.s))
    """
    Typical exposure times for the SG, keyed by solar condition.

    Values normalized to seconds.
    """

    exposure_times_CI: Mapping | None = field(default=None, converter=_quantity_mapping(u.s))
    """
    Typical exposure times for the CI, keyed by solar condition.

    Values normalized to seconds.
    """

    # Response creation
    electron_density: float = field(default=1e9, converter=float)
    """
    Effective density for response creation.
    """

    electron_pressure: float = field(default=3e15, converter=float)
    """
    Effective pressure for response creation.
    """

    response_logT_min: float = field(default=4.8, converter=float)
    """
    Minimum logT for response creation.
    """

    target_logT: Mapping | None = field(default=None, converter=converters.optional(_immutable_mapping))
    """
    Target logT values for different solar conditions.
    """

    target_vdop: Mapping | None = field(default=None, converter=converters.optional(_immutable_mapping))
    """
    Target vdop values for different solar conditions.
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

    num_lines_keep: int | None = field(default=None, validator=_instance(int))
    """
    Number of lines conserved independently for the response creation.
    """

    sum_lines: bool | None = field(default=None, validator=_instance(bool))
    """
    Sum all lines for response creation.
    """

    initial_wavelength_SG: xr.DataArray | None = field(
        default=None, converter=converters.optional(_readonly_data_array)
    )
    """
    Wavelength at SG_xpixel=0 for slit=0, in Angstroms.
    """

    channel_spectral_order: xr.DataArray | None = field(
        default=None, converter=converters.optional(_readonly_data_array)
    )
    """
    Spectral order of main line for each band (channel)
    """

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
        return 0.0815 / self._FWHM_TO_SIGMA / self.channel_spectral_order
