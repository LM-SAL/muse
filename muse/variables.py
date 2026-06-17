"""
This module stores package wide constants and variables.
"""

import numpy as np
import xarray as xr

import astropy.units as u
from astropy.units import imperial

from muse import variables_schema as _schema

__all__ = [
    "DEFAULTS_AIA",
    "DEFAULTS_MUSE",
]


MUSE_DEFAULTS_DICT = {
    # CI
    "dx_pixel_CI": 0.143 * u.arcsec,
    "dy_pixel_CI": 0.143 * u.arcsec,
    "slit_sep_CI": 27 * u.pixel,
    "full_well_depth_CI": 120000 * u.DN,
    # SG
    "dx_pixel_SG": 0.4 * u.arcsec,
    "dy_pixel_SG": 0.167 * u.arcsec,
    "slit_sep_SG": None,
    "pixels_SG": u.Quantity(1024, unit=u.pix, dtype=int),
    "number_of_slits_SG": 35,
    "pixels_between_slits": 26.53 * u.pixel,
    "spectral_slit_separation_SG": 390.0 * u.mAA,
    "steps_per_raster_SG": 11,
    # Diffraction
    "mesh_transmission": {284: 0.81},
    "oversample_x_SG": 7,
    "oversample_y_SG": 3,
    "center_diffraction": False,
    "lpi": {284: 70 / imperial.inch},
    "psf_fwhm_x": 0.25 * u.arcsec,  # 0.25 in x and 0.5 in y.
    "psf_fwhm_y": 0.5 * u.arcsec,
    "psf_fwhm": 0.5 * u.arcsec,
    # Other
    "data_compression": 1,
    "ccd_gain": 10 * u.electron / u.DN,
    # Synthesis/inversions
    "sum_over_dims_synthesis": ("logT", "vdop", "slit"),
    "main_lines_SG": [["Fe XIX 108.355", "Fe XXI 108.117"], ["Fe IX 171.073"], ["Fe XV 284.163"]],
    "main_lines_SG_wavelength": {
        "Fe XIX 108.355": 108.355 * u.AA,
        "Fe XXI 108.117": 108.117 * u.AA,
        "Fe IX 171.073": 171.073 * u.AA,
        "Fe XV 284.163": 284.163 * u.AA,
    },
    "bands_SG": np.asanyarray([108, 108, 171, 284]) * u.AA,
    "fov_mode": "wrap",
    "fov_restype": "match_res_tile",
    "fov_sub_interpolation": 2,
    "target_logT": {
        "QS": np.arange(4.8, 6.7, 0.1),
        "AR": np.arange(4.8, 7.2, 0.1),
        "FL": np.arange(4.8, 7.6, 0.1),
    },
    "target_vdop": {
        "QS": np.arange(-200, 210, 20) * u.km / u.s,
        "AR": np.arange(-300, 310, 20) * u.km / u.s,
        "FL": np.arange(-500, 510, 20) * u.km / u.s,
    },
    # Response
    "minimum_abundance": 1e-21,
    "num_lines_keep": 2,  # nervous about this since there are 3 mainlines in 108 due to repeat wavelength.  Not used in pipeline
    "sum_lines": False,
    "initial_wavelength_SG": xr.DataArray(
        np.array([107.68034, 170.62314, 283.01608]) * u.AA,
        coords={"channel": [108, 171, 284]},
        dims="channel",
    ),
    "channel_spectral_order": xr.DataArray(np.array([2, 2, 1]), coords={"channel": [108, 171, 284]}, dims="channel"),
    # Exposures
    "exposure_times_SG": {
        "QS": (2, 6, 18, 60.0) * u.s,
        "plage": (1, 2, 8, 32) * u.s,
        "AR": (1, 2, 8, 32) * u.s,
        "M-flare": (0.1, 0.6, 1.8, 6.0) * u.s,
        "X-flare": (0.1, 0.6, 1.8, 6.0) * u.s,
    },
    "exposure_times_CI": {
        "QS": (1.5, 3.0, 6.0, 12) * u.s,
        "plage": (0.6, 1.2, 2.5, 5) * u.s,
        "AR": (0.6, 1.2, 2.5, 5) * u.s,
        "M-flare": (0.06, 0.15, 0.3, 1.2) * u.s,
        "X-flare": (0.06, 0.15, 0.3, 1.2) * u.s,
    },
}

AIA_DEFAULTS_DICT = {
    "dx_pixel_CI": 0.6 * u.arcsec,
    "dy_pixel_CI": 0.6 * u.arcsec,
    "full_well_depth_CI": 290000 * u.DN,
    "ccd_gain": 18 * u.electron / u.DN,
    "sum_over_dims_synthesis": ("logT", "vdop"),
}


DEFAULTS_MUSE = _schema.InstrumentDefaults(**MUSE_DEFAULTS_DICT)
DEFAULTS_AIA = _schema.InstrumentDefaults(**AIA_DEFAULTS_DICT)


def _conversion_ph2dn(wvl, gain):
    """
    Convert photons to DN or electrons.

    Parameters
    ----------
    wvl : `float` or array-like
        Wavelength in Angstroms.
    gain : `float`
        e->DN gain.

    Returns
    -------
    conversion factor : `float`
    """
    return 12398.0 / wvl / 3.65 / gain
