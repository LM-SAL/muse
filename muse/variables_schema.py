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
    "GFATDefaults",
    "InstrumentDefaults",
    "SDCBenchmarkDefaults",
    "SDCDefaults",
    "SVDDefaults",
]

_FWHM_TO_SIGMA = 2.355  # This was 2.35482 in guasslobes.py
"""
Conversion factor between FWHM and Gaussian sigma, 2 * sqrt(2 * ln 2), truncated.
"""

_DATA_DRIVEN_MASK_KEYWORD = {"fill_value": 1.0, "threshold": 2.24}


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
    return tuple(tuple(item) if isinstance(item, list | tuple) else item for item in value)


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
    if isinstance(value, list | tuple):
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


def _data_driven_mask_keyword():
    return dict(_DATA_DRIVEN_MASK_KEYWORD)


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
        return 0.0815 / _FWHM_TO_SIGMA / self.channel_spectral_order


@define(kw_only=True)
class GFATDefaults:
    """
    Class to hold default values for fitting GFAT to datasets.
    """

    kwargs_obs: dict | None = field(default=None, validator=_instance(dict))
    """
    Additional keyword arguments passed to the Gaussian fitting analysis function.
    """

    npix: int = field(default=2, validator=validators.instance_of(int))
    """
    Number of pixels (plus/minus) to consider around the main peak for the GFAT fitting.
    """

    vrange: u.Quantity | None = field(default=None, converter=_quantity(u.km / u.s))
    """
    Velocity range to find the main line.

    Normalized to km/s.
    """

    flag_pix_slac: int = field(default=0, validator=validators.instance_of(int))
    """
    Flag to identify which pixels to exclude in the slack algorithm (0 = no
    exclusion).
    """

    npix_localmax_slac: int = field(default=1, validator=validators.instance_of(int))
    """
    Number of pixels around local maximum to consider in the slack algorithm.
    """

    npix_left: int | None = field(default=None, validator=_instance(int))
    """
    Number of pixels to the left of the peak to include in spectral fitting.
    """

    npix_right: int | None = field(default=None, validator=_instance(int))
    """
    Number of pixels to the right of the peak to include in spectral fitting.
    """

    ic: tuple = field(default=(1, 0, 10), converter=tuple)
    """
    Initial conditions for Gaussian fitting: (amplitude, center, width) starting values.
    """

    width_min: float = field(default=0, converter=float)
    """
    Minimum Gaussian width (sigma) allowed in the fit in km/s.
    """

    ic_user: object | None = None
    """
    User-provided initial conditions for Gaussian fitting parameters.
    """

    factor_int: float = field(default=4, converter=float)
    """
    Intensity factor for generating outliers.
    """

    bc_user: object | None = None
    """
    User-provided boundary conditions for Gaussian fitting parameters.
    """

    keep_only_local_maxima: bool | None = field(default=None, validator=_instance(bool))
    """
    If True, only fit spectral peaks at local maxima; if False, fit all peaks.
    """

    line_maxlogT: object | None = None
    """
    Line identifier list used to select which lines to analyze (None = all
    lines).
    """

    cal_moment: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, calculate statistical moments of the spectrum in addition to Gaussian fit.
    """

    return_spec: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, return the fitted spectrum in addition to the moments.
    """

    clip_at_vrange: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, clip spectral data at the velocity range boundaries before fitting.
    """

    velocity_array: object | None = None
    """
    Pre-computed velocity array to guide the Gaussian fitting (used for guided fitting).
    """

    spec_noise: object | None = None
    """
    Noise spectrum or noise standard deviation to use for weighted fitting.
    """

    boolean_spec_mask: bool = field(default=True, validator=validators.instance_of(bool))
    """
    If True, use a boolean mask to exclude bad pixels; if False, use weighted masking.
    """

    method: str = field(default="scipy", validator=validators.instance_of(str))
    """
    Method for Gaussian fitting: "scipy" uses scipy.optimize, other methods may be
    available.
    """

    serial: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, process data serially (pixel-by-pixel); if False, use vectorized
    operations.
    """

    # For maskers
    thresholds_ph: object | None = None
    """
    Intensity threshold(s) in photons for masking low signal pixels.
    """

    thresholds_dn: object | None = None
    """
    Intensity threshold(s) in data numbers (DN) for masking low signal pixels.
    """

    # For outliers
    mask_outlier: object | None = None
    """
    Mask to apply when identifying outliers in the fitted parameters.
    """

    mask_gt_outlier: object | None = None
    """
    Mask for ground truth data when comparing to identify outliers.
    """

    mask_obs_outlier: object | None = None
    """
    Mask for observational data when comparing to identify outliers.
    """

    consider_intensity_condition: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, include intensity thresholds when computing outlier masks.
    """

    sigma: float = field(default=1.0, converter=float)
    """
    Sigma level of the promised uncertainty to consider for outlier detection.
    """

    # SDC benchmark:
    velocity_range: u.Quantity = field(default=50 * u.km / u.s, converter=_quantity(u.km / u.s))
    """
    Velocity range to find the main line.

    Normalized to km/s.
    """

    velocity_range_gt: u.Quantity | None = field(default=None, converter=_quantity(u.km / u.s))
    """
    Velocity range to find the main line in ground truth data.

    Normalized to km/s.
    """

    velocity_range_guided: u.Quantity | None = field(default=None, converter=_quantity(u.km / u.s))
    """
    Velocity range for guided fitting using inverted main line parameters.

    Normalized to km/s.
    """

    velocity_range_gfat: u.Quantity = field(default=100 * u.km / u.s, converter=_quantity(u.km / u.s))
    """
    Velocity range to find the main line in observation fitting (step 0).

    Normalized to km/s.
    """

    intensity_scaling: str | None = field(default=None, validator=_instance(str))
    """
    Intensity scaling method for SDC: None, "no_noise", "noise", "sigma_1", etc.
    """

    velocity_range_list: dict = field(
        factory=lambda: {"QS": 100, "plage": 100, "AR": 100, "M-flare": 400, "X-flare": 400},
        validator=validators.instance_of(dict),
    )
    """
    Velocity range (km/s) used in forward synthesis, keyed by solar condition.
    """

    vmax_plot: dict = field(
        factory=lambda: {"QS": 40, "plage": 40, "AR": 40, "M-flare": 100, "X-flare": 200},
        validator=validators.instance_of(dict),
    )
    """
    Maximum velocity magnitude for plotting axes, keyed by solar condition.
    """

    # PLOTS:
    param_list: tuple = field(default=("net_flux", "velocity", "linewidth"), converter=tuple)
    """
    Parameters to plot from GFAT moments: net flux/intensity, velocity, linewidth.
    """

    response: str = field(default="0.390/0.780 A", validator=validators.instance_of(str))
    """
    Response function wavelength range string for plot labeling (informational).
    """

    rf_info: str = field(default="", validator=validators.instance_of(str))
    """
    Response function metadata string for plot annotations.
    """

    gt_title: str = field(default="GT", validator=validators.instance_of(str))
    """
    Title label for ground truth (GT) data in comparison plots.
    """

    sg_channels: np.ndarray = field(factory=lambda: np.asanyarray([108, 171, 284]))
    """
    Spectrograph channel identifiers corresponding to the main spectral lines.
    """

    sg_main_lines: np.ndarray = field(
        factory=lambda: np.asanyarray(["Fe XIX 108.355", "Fe IX 171.073", "Fe XV 284.163"])
    )
    """
    Main spectral line identifiers for each SG channel.
    """

    cont_title: str = field(default="Contaminated", validator=validators.instance_of(str))
    """
    Title label for contaminated data in comparison plots.
    """

    xaxis_gt: bool = field(default=True, validator=validators.instance_of(bool))
    """
    No idea.

    .. todo:: Unknown meaning; document or remove.
    """

    suptitle_string: str = field(default="", validator=validators.instance_of(str))
    """
    Main super-title string for plots (empty by default, user-customizable).
    """

    dpi: int = field(default=300, validator=validators.instance_of(int))
    """
    Dots per inch (DPI) resolution for saved plot figures.
    """

    nticks_net_flux: int = field(default=5, validator=validators.instance_of(int))
    """
    Number of tick marks on the net flux (intensity) colorbar axis.
    """

    gamma_net_flux: float = field(default=0.1, converter=float)
    """
    Power-law exponent for non-linear scaling of intensity tick marks (0-1 range).
    """

    min_ratio_int: float = field(default=0.1, converter=float)
    """
    Minimum intensity ratio threshold for identifying valid pixels in comparisons.
    """

    save_id: str = field(default="", validator=validators.instance_of(str))
    """
    Identifier string appended to saved figure filenames.
    """

    savepath: str = field(default="../", validator=validators.instance_of(str))
    """
    Directory path where diagnostic plots are saved.
    """

    max_vel: u.Quantity = field(default=200 * u.km / u.s, converter=_quantity(u.km / u.s))
    """
    Maximum velocity for colorbar limits in velocity heatmaps.
    """

    min_vel: u.Quantity = field(default=-200 * u.km / u.s, converter=_quantity(u.km / u.s))
    """
    Minimum velocity for colorbar limits in velocity heatmaps.
    """

    max_error_chisq: float = field(default=100, converter=float)
    """
    Maximum chi-squared error threshold for masking bad pixels in plots.
    """

    gridsize: tuple = field(default=(80, 1200), converter=tuple)
    """
    Hexbin grid dimensions (height, width) for 2D scatter plot histograms.
    """

    ylabels: object | None = None
    """
    Custom Y-axis labels for parameter plots (None uses default labels).
    """

    pwnorm: float = field(default=0.3, converter=float)
    """
    Power-law exponent for non-linear colorbar scaling (used in matplotlib PowerNorm).
    """

    outliers_plot: object | None = None
    """
    Outlier mask to highlight anomalous pixels in diagnostic plots.
    """

    sigmas_plot: object | None = None
    """
    Sigma/uncertainty values to display as error bars in scatter plots.
    """

    # This is for plot_4
    mask_gt: object | None = None
    """
    Ground truth mask for plot_4 to highlight valid regions.
    """

    ylims: object | None = None
    """
    Y-axis limits for scatter/comparison plots (e.g., [ymin, ymax]).
    """

    # This is for plot_9
    linear_plot_nine_conts: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, use linear color scaling for contaminant line plots; if False, use log.
    """

    # This is for plot_slit_cont_maps
    mask_all: object | None = None
    """
    Combined mask for plot_slit_cont_maps marking valid pixels across all criteria.
    """

    # This is for plot_gfat_4
    response_plot_gfat_4: object | None = None
    """
    Response function metadata to display in plot_gfat_4.
    """

    exp_time_plot_gfat_4: object | None = None
    """
    Exposure time value to display in plot_gfat_4 title.
    """


@define(kw_only=True)
class SDCDefaults:
    """
    Class to hold default values for doing the SDC.
    """

    alpha: float = field(default=1e-3, converter=float)
    """
    Regularization parameter (penalty strength) for Lasso/Ridge inversion.
    """

    positive: bool = field(default=True, validator=validators.instance_of(bool))
    """
    If True, constrain inversion coefficients to be non-negative.
    """

    fit_intercept: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, fit an intercept/offset term in linear inversion.
    """

    tol: float = field(default=1e-4, converter=float)
    """
    Convergence tolerance for iterative solvers and optimizers.
    """

    svd_low_threshold: float = field(default=-20, converter=float)
    """
    Negative intensity threshold for masking wrong SVD solutions.
    """

    chi_threshold: float = field(default=20, converter=float)
    """
    Chi-squared threshold for masking pixels with poor inversion fit quality.
    """

    portion: int = field(default=60, validator=validators.instance_of(int))
    """
    No idea.

    .. todo:: Unknown meaning; document or remove.
    """

    model2: object | None = None
    """
    Secondary SKlearn model for refining inversion results.
    """

    # Priors:
    _not_vdem_coords: tuple = field(default=("line", "SG_xpixel"), converter=tuple)
    """
    Coordinate dimensions excluded when applying VDEM priors.
    """

    prior_start: int = field(default=100, validator=validators.instance_of(int))
    """
    No idea.

    .. todo:: Unknown meaning; document or remove.
    """

    prior_jumps: int = field(default=1, validator=validators.instance_of(int))
    """
    Step size for sampling prior points (1=use all, >1=subsample every Nth).
    """

    prior_velocity_list: tuple = field(default=(300, 400, 500), converter=tuple)
    """
    List of velocity thresholds (km/s) for generating velocity-based priors.
    """

    prior_temperature_list: tuple = field(default=(5.9, 6.3, 6.9, 7.1), converter=tuple)
    """
    List of temperature thresholds (log10 K) for generating temperature-based priors.
    """

    prior_method: str = field(default="log", validator=validators.instance_of(str))
    """
    Prior generation method: "log" (logarithmic) or "linear".
    """

    prior_steps_in_sound_speed: tuple = field(default=(1, 2, 3), converter=tuple)
    """
    Velocity steps relative to sound speed for generating acoustic velocity priors.
    """

    prior_threshold: float = field(default=1, converter=float)
    """
    Intensity threshold for identifying valid prior pixels (relative units/photons).
    """

    prior_multiplier: float = field(default=500, converter=float)
    """
    Multiplicative scaling factor for prior weight/strength in the inversion.
    """

    prior_multiplier_sound_speed: float = field(default=30, converter=float)
    """
    Scaling factor for sound-speed-based velocity priors.
    """

    prior_normalize: str | None = field(default=None, validator=_instance(str))
    """
    Normalization method for priors: None, "sum" (normalize to sum=1), "max", etc.
    """

    prior_retain_slit: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, retain slit-specific information in priors; if False, combine across slits.
    """

    prior_slope: float = field(default=1, converter=float)
    """
    Slope parameter for prior weighting function (linear or power-law).
    """

    prior_no_response: bool = field(default=True, validator=validators.instance_of(bool))
    """
    If True, discard response information in priors.
    """

    prior_quantile_value: float = field(default=0.5, converter=float)
    """
    Quantile max value for prior.
    """

    prior_floor_value: float = field(default=0.0, converter=float)
    """
    Floor value for prior.
    """

    prior_normalize_response: str = field(default="sum", validator=validators.instance_of(str))
    """
    Normalization of temperature response for imager prior.
    """

    prior_p: float = field(default=1, converter=float)
    """
    Power exponent for prior generation (1=linear, 2=quadratic, etc.).
    """

    prior_custom_prior_list: object | None = None
    """
    User-defined list of custom prior arrays for advanced inversion control.
    """

    prior_composition_method: str = field(default="multiply", validator=validators.instance_of(str))
    """
    Method for combining multiple priors: "multiply" (product), "add" (sum), etc.
    """

    prior_ci_channel: int = field(default=195, validator=validators.instance_of(int))
    """
    Context Imager channel wavelength (Ångströms) for CI-based priors.
    """

    prior_scale: float = field(default=1.0, converter=float)
    """
    Overall scaling factor applied to all prior weights.
    """

    alphas: tuple = field(default=(0.0, 0.1, 0.2), converter=tuple)
    """
    No idea.

    .. todo:: Unknown meaning; document or remove.
    """

    var1: str = field(default="logT", validator=validators.instance_of(str))
    """
    No idea.

    .. todo:: Unknown meaning; document or remove.
    """

    var2: str = field(default="basis_logT", validator=validators.instance_of(str))
    """
    No idea.

    .. todo:: Unknown meaning; document or remove.
    """

    # inversion
    mask: object | None = None
    """
    No idea.

    .. todo:: Unknown meaning; document or remove.
    """

    scaling: str | None = field(default=None, validator=_instance(str))
    """
    Scaling method for response functions: None, "intensity", "spectrum", etc.
    """

    drop_pix: object | None = None
    """
    Indices or boolean mask of pixels to exclude from inversion.
    """

    disable: bool = field(default=False, validator=validators.instance_of(bool))
    """
    No idea.

    .. todo:: Unknown meaning; document or remove.
    """

    serial: bool = field(default=True, validator=validators.instance_of(bool))
    """
    If True, process inversions pixel-by-pixel serially; if False, use vectorized.
    """

    response_scaling: object | None = None
    """
    Pre-computed response function scaling factors to apply during inversion.
    """

    randomize: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, use randomized algorithms (faster SVD decomposition but slightly less
    accurate).
    """

    using_weights: bool = field(default=False, validator=validators.instance_of(bool))
    """
    No idea.

    .. todo:: Unknown meaning; document or remove.
    """

    vdem_prior: object | None = None
    """
    No idea.

    .. todo:: Unknown meaning; document or remove.
    """

    # Sparse method defaults
    sparse_epsilon: float = field(default=1e-4, converter=float)
    """
    Small regularization constant to avoid singular matrices in sparse methods.
    """

    sparse_slit_pos_vdem: int = field(default=0, validator=validators.instance_of(int))
    """
    Slit position index for sparse VDEM prior (0=first slit).
    """

    # Mask response
    mask_response_cuda_device: int = field(default=0, validator=validators.instance_of(int))
    """
    CUDA GPU device ID (0, 1, 2, ...) for GPU-accelerated response masking.
    """

    data_driven_mask_keyword: dict = field(factory=_data_driven_mask_keyword)
    """
    Keyword for data-driven mask generation applied to response functions.
    """


@define(kw_only=True)
class SVDDefaults:
    """
    Class to hold default values for doing the SVD.
    """

    ncomponents: int = field(default=1000, validator=validators.instance_of(int))
    """
    Number of components to keep in the SVD truncation.
    """

    n_iter_svd: int = field(default=5, validator=validators.instance_of(int))
    """
    Number of iterations for the randomized SVD solver (if used).
    """

    lr: float = field(default=1e-3, converter=float)
    """
    Learning rate for the PyTorch optimizer (if used).
    """

    epochs: int = field(default=100, validator=validators.instance_of(int))
    """
    Number of epochs for the PyTorch optimizer (if used).
    """

    positive: bool = field(default=False, validator=validators.instance_of(bool))
    """
    Whether to enforce signs of coefficients to be positive.
    """

    fit_intercept: bool = field(default=False, validator=validators.instance_of(bool))
    """
    Whether to calculate the intercept for the model.
    """

    space: str = field(default="S", validator=validators.in_(("S", "V")))
    """
    The space in which to perform the inversion ('S' or 'V').

    'S' uses the singular values directly (diagonal system). 'V' uses the right singular
    vectors basis.
    """

    lambda_reg: float = field(default=1.0, converter=float)
    """
    Regularization parameter (strength of the prior).
    """

    pytorch: bool = field(default=True, validator=validators.instance_of(bool))
    """
    Whether to use PyTorch implementation for the solver.
    """

    device: str = field(default="cpu", validator=validators.instance_of(str))
    """
    Sample device to run on ('cpu' or 'cuda').
    """

    alpha: int | None = field(default=None, validator=_instance(int))
    """
    Alias for `ncomponents` (backward compatibility).
    """


@define(kw_only=True)
class SDCBenchmarkDefaults:
    """
    Class to hold default values for the SDC benchmark pipeline.
    """

    # Model and Response Function Paths
    model_path: str | None = field(default=None, validator=_instance(str))
    """
    Path to directory containing VDEM model file.
    """
    model_name: str | None = field(default=None, validator=_instance(str))
    """
    Filename of the VDEM model file.
    """

    resp_path_sg: str | None = field(default=None, validator=_instance(str))
    """
    Path to directory containing spectrograph (SG) response function files.
    """

    resp_path_ci: str | None = field(default=None, validator=_instance(str))
    """
    Path to directory containing context imager (CI) response function files.
    """

    # Response Function Files
    FeXIX_108_resp: str = field(
        default="respfunc_108_FeXIX_nslits35_sep390_coronal_2021_chianti_coatingOptixFab_JMS_24Sept24.zarr",
        validator=validators.instance_of(str),
    )
    """
    Filename of Fe XIX 108 Å response function (main line only).
    """

    FeXXI_108_resp: str = field(
        default="respfunc_108_FeXXI_nslits35_sep390_coronal_2021_chianti_coatingOptixFab_JMS_24Sept24.zarr",
        validator=validators.instance_of(str),
    )
    """
    Filename of Fe XXI 108 Å response function (contaminant line).
    """

    contam_108_resp: str = field(
        default="respfunc_108-L_nslits35_sep390_coronal_2021_chianti_coatingOptixFab_JMS_24Sept24.zarr",
        validator=validators.instance_of(str),
    )
    """
    Filename of 108 Å contaminant lines combined response function.
    """

    FeIX_171_resp: str = field(
        default="respfunc_171_FeIX_nslits35_sep390_coronal_2021_chianti_coatingOptixFab_JMS_24Sept24.zarr",
        validator=validators.instance_of(str),
    )
    """
    Filename of Fe IX 171 Å response function (main line only).
    """

    contam_171_resp: str = field(
        default="respfunc_171-L_nslits35_sep390_coronal_2021_chianti_coatingOptixFab_JMS_24Sept24.zarr",
        validator=validators.instance_of(str),
    )
    """
    Filename of 171 Å contaminant lines combined response function.
    """

    FeXV_284_resp: str = field(
        default="respfunc_284_FeXV_nslits35_sep390_coronal_2021_chianti_coatingOptixFab_JMS_24Sept24.zarr",
        validator=validators.instance_of(str),
    )
    """
    Filename of Fe XV 284 Å response function (main line only).
    """

    contam_284_resp: str = field(
        default="respfunc_284-L_nslits35_sep390_coronal_2021_chianti_coatingOptixFab_JMS_24Sept24.zarr",
        validator=validators.instance_of(str),
    )
    """
    Filename of 284 Å contaminant lines combined response function.
    """

    ci_195_resp: str = field(
        default="respfunc_195_CI_coronal_2021_chianti_coatingOptixFab_JMS_24Sept24.zarr",
        validator=validators.instance_of(str),
    )
    """
    Filename of Context Imager 195 Å response function.
    """

    # Inversion Response Function Paths (default to None to use synthesis ones if not provided)
    resp_path_sg_inv: str | None = field(default=None, validator=_instance(str))
    """
    Path to inversion-specific spectrograph response functions (optional, overrides
    synthesis responses).
    """

    resp_path_ci_inv: str | None = field(default=None, validator=_instance(str))
    """
    Path to inversion-specific context imager response functions (optional).
    """

    FeXIX_108_resp_inv: str | None = field(default=None, validator=_instance(str))
    """
    Inversion-specific Fe XIX 108 Å response function filename (optional).
    """

    FeXXI_108_resp_inv: str | None = field(default=None, validator=_instance(str))
    """
    Inversion-specific Fe XXI 108 Å response function filename (optional).
    """

    contam_108_resp_inv: str | None = field(default=None, validator=_instance(str))
    """
    Inversion-specific 108 Å contaminant response function filename (optional).
    """

    FeIX_171_resp_inv: str | None = field(default=None, validator=_instance(str))
    """
    Inversion-specific Fe IX 171 Å response function filename (optional).
    """

    contam_171_resp_inv: str | None = field(default=None, validator=_instance(str))
    """
    Inversion-specific 171 Å contaminant response function filename (optional).
    """

    FeXV_284_resp_inv: str | None = field(default=None, validator=_instance(str))
    """
    Inversion-specific Fe XV 284 Å response function filename (optional).
    """

    contam_284_resp_inv: str | None = field(default=None, validator=_instance(str))
    """
    Inversion-specific 284 Å contaminant response function filename (optional).
    """

    ci_195_resp_inv: str | None = field(default=None, validator=_instance(str))
    """
    Inversion-specific Context Imager 195 Å response function filename (optional).
    """

    # Synthesis and Effective Area Paths
    syn_path: str | None = field(default=None, validator=_instance(str))
    """
    Path to directory containing pre-computed synthetic spectra or observations.
    """

    effect_path: str | None = field(default=None, validator=_instance(str))
    """
    Path to directory containing effective area files for radiometric calibration.
    """

    effect_area_108: str = field(
        default="effective_area_channel_108_JMS_2024-10-31.zarr", validator=validators.instance_of(str)
    )
    """
    Filename of 108 Å channel effective area file (photon collection efficiency).
    """

    effect_area_171: str = field(
        default="effective_area_channel_171_JMS_2024-10-31.zarr", validator=validators.instance_of(str)
    )
    """
    Filename of 171 Å channel effective area file (photon collection efficiency).
    """

    effect_area_284: str = field(
        default="effective_area_channel_284_JMS_2024-10-31.zarr", validator=validators.instance_of(str)
    )
    """
    Filename of 284 Å channel effective area file (photon collection efficiency).
    """

    # Input/Output Files
    obs_filename_ci: str | None = field(default=None, validator=_instance(str))
    """
    Filename of context imager observation or simulated data (zarr format).
    """

    obs_filename_sg: str | None = field(default=None, validator=_instance(str))
    """
    Filename of spectrograph observation or simulated data (zarr format).
    """

    gt_filename_sg: str | None = field(default=None, validator=_instance(str))
    """
    Filename of ground-truth spectrograph data without noise (zarr format).
    """

    # Processing Flags
    serial_gfat: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, run GFAT serially (pixel-by-pixel); if False, use vectorized processing.
    """

    run_SDC: bool = field(default=True, validator=validators.instance_of(bool))
    """
    If True, run spectral decontamination (SDC) inversion; if False, skip SDC.
    """

    serial_SDC: bool = field(default=True, validator=validators.instance_of(bool))
    """
    If True, run SDC serially; if False, use parallel processing.
    """

    run_mask: str | None = field(default=None, validator=_instance(str))
    """
    Masking strategy for SDC: None, "gt_vdem" (ground truth), "redundant", "intensity",
    or "data_driven".
    """

    redundant_mask: object | None = None
    """
    Pre-computed mask to remove redundant temperature/velocity combinations.
    """

    velbin_inv: float | None = field(default=None, converter=converters.optional(float))
    """
    Velocity bin width for inversion grid refinement (km/s).

    If set, velocity grid is rebinned.
    """

    tgbin_inv: float | None = field(default=None, converter=converters.optional(float))
    """
    Temperature bin width for inversion grid refinement (log10 K).

    If set, temperature grid is rebinned.
    """

    ntgbin_inv: int | None = field(default=None, validator=_instance(int))
    """
    Number of temperature bins for inversion (alternative to tgbin_inv).
    """

    rotate: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, rotate VDEM model 90 degrees to match MUSE field of view orientation.
    """

    saturation: float | None = field(default=None, converter=converters.optional(float))
    """
    Saturation level in DN.

    Pixels exceeding this value are clipped (None = no saturation).
    """

    # Data Selection
    KEYS: str | None = field(default=None, validator=_instance(str))  # [None,"middle","bottom", "all"]
    """
    Spatial regions to analyze: "top", "middle", "bottom", or "all" FOV.
    """

    target: str | None = field(default=None, validator=_instance(str))  # ["QS","AR","FL"]
    """
    Solar target type: "QS" (quiet sun), "AR" (active region), "FL" (flare).
    """

    jump_pix: tuple | None = field(
        default=None, converter=converters.optional(tuple)
    )  # [10,10] Jump in y-axis and t-step
    """
    Spatial/temporal subsampling: [dy, dt] to skip every N pixels/steps (None = no
    subsampling).
    """
    # Coordinate Specifications
    logT: object | None = None
    """
    Temperature grid (log10 K) for response functions.

    If None, uses default target-based grid.
    """

    vdop: object | None = None
    """
    Velocity grid (km/s) for response functions.

    If None, uses default target-based grid.
    """

    logT_inv: object | None = None
    """
    Temperature grid specifically for inversion (log10 K).

    If None, uses logT.
    """

    vdop_inv: object | None = None
    """
    Velocity grid specifically for inversion (km/s).

    If None, uses vdop.
    """

    # Summation Dimensions
    sum_over_syn: tuple | None = field(default=None, converter=converters.optional(tuple))
    """
    Dimensions to sum over in synthesis (e.g., ["logT", "vdop"]).

    None defaults to ["logT", "vdop"].
    """

    sum_over_inv: tuple | None = field(default=None, converter=converters.optional(tuple))
    """
    Dimensions to sum over in inversion (e.g., ["logT", "vdop"]).

    None defaults to ["logT", "vdop"].
    """

    order: np.ndarray | None = field(default=None, converter=converters.optional(np.asanyarray))
    """
    Spectral diffraction order for each channel (array of integers, e.g., [2, 2, 1]).
    """

    # Models
    model: object | None = None
    """
    Scikit-learn estimator for SDC inversion (e.g., Lasso, Ridge, ElasticNet).
    """

    model2: object | None = None
    """
    Second scikit-learn estimator for refining the inversion (optional).
    """

    data_driven_mask_keyword: dict = field(factory=_data_driven_mask_keyword)
    """
    Keyword argument to use for data-driven mask generation.
    """

    # Synthesis Parameters
    exp_time: float = field(default=0.1, converter=float)
    """
    Exposure time in seconds for forward synthesis.
    """

    noise: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, add photon noise and readout noise to synthetic spectra.
    """

    noise_seed: int = field(default=123456789, validator=validators.instance_of(int))
    """
    Random seed for reproducible noise generation.
    """

    readoutnoise: float = field(default=2.24, converter=float)
    """
    Readout noise level in DN (depends on compression scheme used).
    """

    compression: int = field(default=1, validator=validators.instance_of(int))
    """
    Data compression level (affects readout noise): 1=lossless (1.66 DN), 2=lossy 3.5bpp
    (2.24 DN), 3=lossy 2.5bpp (3.87 DN).
    """

    # Output Parameters
    savefigs: bool = field(default=True, validator=validators.instance_of(bool))
    """
    If True, save diagnostic plots to disk.
    """

    savelogfile: str = field(default="sdc_benchmark.json", validator=validators.instance_of(str))
    """
    Filename for saving benchmark parameters and results as JSON.
    """

    save_output: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, save intermediate results (moments, spectra) as zarr files.
    """

    save_output_netcdf: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, save intermediate results as NetCDF files (slower but more portable).
    """

    author: str = field(default="JMS", validator=validators.instance_of(str))
    """
    Author name appended to output filenames and metadata.
    """

    appendlog: dict | None = field(default=None, validator=_instance(dict))
    """
    Existing log dictionary to append results to (for batch processing).
    """

    disable_pbar: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, disable progress bars during processing.
    """

    # Effective Parameters
    run_cont_id: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, identify and analyze contaminant lines using CHIANTI database.
    """

    dens_eff: float = field(default=1e9, converter=float)
    """
    Effective electron density (cm^-3) used for contaminant analysis.
    """

    pres_eff: float = field(default=3e15, converter=float)
    """
    Effective electron pressure (erg/cm^3) used for contaminant analysis.
    """

    lgtgmin_eff: float = field(default=4.8, converter=float)
    """
    Minimum log10 temperature (K) for contaminant line identification.
    """

    int_cont_ratio: float = field(default=1e2, converter=float)
    """
    Intensity ratio threshold to identify contaminants vs.

    main lines.
    """

    main_lines_cont: object | None = None
    """
    Main emission lines to consider for contaminant analysis (defaults to all lines).
    """

    n_cont_top: int = field(default=20, validator=validators.instance_of(int))
    """
    Number of top contaminant lines to include in detailed analysis.
    """

    cuda_device: int = field(default=0, validator=validators.instance_of(int))
    """
    CUDA GPU device ID (0, 1, 2, ...) for GPU acceleration.

    -1 = CPU only.
    """

    muse_fov_mode: str = field(default="wrap", validator=validators.instance_of(str))
    """
    Padding mode for MUSE FOV transformation: "wrap" or "edge".
    """

    passchannels_to_solve: object | None = None
    """
    Subset of spectral channels for SDC inversion (None = all channels).
    """

    dropping: object | None = None
    """
    Pixels to drop/mask before SDC inversion (None = no pixels dropped).
    """

    weight_prior: str | None = field(default=None, validator=_instance(str))
    """
    Prior weighting scheme: None, "uniform", or other schemes.
    """

    prior_kwargs: object | None = None
    """
    Keyword arguments for prior generation (list of dicts for multiple priors).
    """

    # Masking and Inversion Options
    unmask_outliers: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, unmask outlier pixels in plots (show all data including bad pixels).
    """

    only_outliers: bool = field(default=True, validator=validators.instance_of(bool))
    """
    If True, plot only outlier regions to focus on problem areas.
    """

    randomizeSVD: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, use randomized SVD for faster computation (slightly less accurate).
    """

    basis: bool | None = field(default=None, validator=_instance(bool))
    """
    If True, use polynomial basis functions to reduce degrees of freedom in inversion.
    """

    SVD_mask: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, apply mask based on SVD singular values (mask low-signal components).
    """

    svd_low_threshold_list: tuple | None = field(default=None, converter=converters.optional(tuple))
    """
    List of thresholds to test for SVD masking (e.g., [-10, -50, -100]).
    """

    max_error: float = field(default=100, converter=float)
    """
    Maximum allowed fitting error threshold for masking bad pixels.
    """

    chi_mask: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, apply mask based on chi-squared error between inversion and observation.
    """

    chi_threshold_list: tuple | None = field(default=None, converter=converters.optional(tuple))
    """
    List of chi-squared thresholds to test for masking (e.g., [10, 50, 100]).
    """

    chirel_mask: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, apply mask based on relative chi-squared error.
    """

    chirel_threshold_list: tuple | None = field(default=None, converter=converters.optional(tuple))
    """
    List of relative chi-squared thresholds to test for masking.
    """

    chisq_mask: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, apply mask based on chi-squared goodness-of-fit.
    """

    chisq_threshold_list: tuple | None = field(default=None, converter=converters.optional(tuple))
    """
    List of chi-squared goodness-of-fit thresholds to test for masking.
    """

    noise_without_slitsum: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, add plot of outliers generate by the noise without summing over slits.
    """

    # binning_axis parameters
    binning_axisname: str = field(default="logT", validator=validators.instance_of(str))
    """
    Axis name for binning: "logT" (temperature) or "vdop" (velocity).
    """

    binning_nbin: int = field(default=2, validator=validators.instance_of(int))
    """
    Number of output bins when binning_axis is applied.
    """

    # Velocity masking parameters
    velocity_mask: bool = field(default=False, validator=validators.instance_of(bool))
    """
    If True, apply mask based on absolute velocity exceeding threshold.
    """

    velocity_threshold_list: tuple | None = field(default=None, converter=converters.optional(tuple))  # [250] km/s
    """
    List of velocity thresholds (km/s) to test for masking high-velocity pixels.
    """

    # Prior AIA parameters
    aia_ci_response: object | None = None
    """
    Prior AIA: Need to synthesize AIA or not.

    .. todo:: Description recovered from a misaligned docstring block; verify.
    """

    aia_ci_resolution: tuple = field(default=(0.6, 0.6), converter=tuple)
    """
    Prior AIA: CI resolution.
    """

    aia_psf: bool = field(default=False, validator=validators.instance_of(bool))
    """
    Prior AIA: CI with psf or not.
    """

    aia_error_table_path: str | None = field(default=None, validator=_instance(str))
    """
    Prior AIA: CI with noise or not.

    .. todo:: Description recovered from a misaligned docstring block; verify
       (likely: path to the AIA error table used for CI noise).
    """

    aia_prior_norm: bool = field(default=False, validator=validators.instance_of(bool))
    """
    Prior AIA: Normalize AIA prior with mean.
    """

    parallel_workers: int = field(default=4, validator=validators.instance_of(int))
    """
    Number of parallel workers for AIA prior.
    """
