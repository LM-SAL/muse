import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.stats import gaussian_sigma_to_fwhm

from muse.instrument.response.spectral import create_spectral_response
from muse.tests.helpers import fake_line_list, figure_test

pytest.importorskip("numexpr")
pytest.importorskip("periodictable")

WAVELENGTH_GRID = np.linspace(170.75, 171.25, 501) * u.AA


def _line_list_and_name():
    line_list = fake_line_list(wavelength=[171.0])
    return line_list, [line_list.full_name.item()]


@figure_test
def test_spectral_response_broadening_profiles():
    """
    Thermal, instrumental, and nonthermal broadening produce distinct line widths.
    """
    line_list, main_lines = _line_list_and_name()
    kwargs = {"line_list": line_list, "wavelength_grid": WAVELENGTH_GRID, "main_lines": main_lines}
    responses = {
        "thermal only": create_spectral_response(**kwargs),
        "instrumental sigma=0.05 A": create_spectral_response(
            **kwargs, instrumental_fwhm=0.05 * u.AA * gaussian_sigma_to_fwhm
        ),
        "nonthermal=80 km/s": create_spectral_response(**kwargs, nonthermal_velocity=80 * u.km / u.s),
    }

    fig, ax = plt.subplots(constrained_layout=True)
    for label, response in responses.items():
        profile = response.spectral_response.sel(logT=6.0).isel(line=0).squeeze(drop=True)
        ax.plot(response.wavelength_grid, profile, label=label)
    ax.set(xlabel="wavelength [Angstrom]", ylabel="spectral response", title="Line-broadening comparison")
    ax.legend()
    return fig
