"""
Response functions: building them in wavelength space, mapping them onto a detector,
their units and their files.
"""

# products.py is exported by muse.instrument alone: importing it here would be circular.
from muse.instrument.response.detector import map_response_to_ci_detector, map_response_to_sg_detector
from muse.instrument.response.io import (
    align_response_and_vdem,
    read_response,
    read_responses,
    read_spectral_response,
    save_response,
    save_spectral_response,
)
from muse.instrument.response.radiometry import transform_flux_units, transform_response_units
from muse.instrument.response.spectral import (
    create_continuum_response,
    create_log_wavelength_edges,
    create_spectral_response,
)

__all__ = [
    "align_response_and_vdem",
    "create_continuum_response",
    "create_log_wavelength_edges",
    "create_spectral_response",
    "map_response_to_ci_detector",
    "map_response_to_sg_detector",
    "read_response",
    "read_responses",
    "read_spectral_response",
    "save_response",
    "save_spectral_response",
    "transform_flux_units",
    "transform_response_units",
]
