import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from muse.instrument.spectral import create_spectral_response
from muse.instrument.tests.test_spectral import synthetic_effective_area, synthetic_line_list
from muse.tests.helpers import figure_test

WAVELENGTH_GRID = np.linspace(170.75, 171.25, 501) * u.AA


def _line_list_and_name():
    line_list = synthetic_line_list(wavelength=[171.0])
    return line_list, [line_list.full_name.item()]


@figure_test
def test_spectral_response_doppler_profiles():
    """
    A single line shifts across wavelength for negative, zero, and positive velocities.
    """
    line_list, main_lines = _line_list_and_name()
    response = create_spectral_response(
        line_list,
        WAVELENGTH_GRID,
        main_lines=main_lines,
        instrumental_width=0.02 * u.AA,
        doppler_velocity=np.array([-200.0, 0.0, 200.0]) * u.km / u.s,
    )
    profile = response.spectral_response.sel(logT=6.0).isel(line=0)

    fig, ax = plt.subplots(constrained_layout=True)
    for velocity in response.doppler_velocity.values:
        ax.plot(
            response.wavelength_grid,
            profile.sel(doppler_velocity=velocity),
            label=f"{velocity:+.0f} km/s",
        )
    ax.axvline(171.0, ls="--", color="k", alpha=0.4, label="rest wavelength")
    ax.set(xlabel="wavelength [Angstrom]", ylabel="spectral response", title="Doppler-shifted line profiles")
    ax.legend()
    return fig


@figure_test
def test_spectral_response_broadening_profiles():
    """
    Thermal, instrumental, and nonthermal broadening produce distinct line widths.
    """
    line_list, main_lines = _line_list_and_name()
    kwargs = {"line_list": line_list, "wavelength_grid": WAVELENGTH_GRID, "main_lines": main_lines}
    responses = {
        "thermal only": create_spectral_response(**kwargs),
        "instrumental sigma=0.05 A": create_spectral_response(**kwargs, instrumental_width=0.05 * u.AA),
        "nonthermal=80 km/s": create_spectral_response(**kwargs, nonthermal_velocity=80 * u.km / u.s),
    }

    fig, ax = plt.subplots(constrained_layout=True)
    for label, response in responses.items():
        profile = response.spectral_response.sel(logT=6.0).isel(line=0).squeeze(drop=True)
        ax.plot(response.wavelength_grid, profile, label=label)
    ax.set(xlabel="wavelength [Angstrom]", ylabel="spectral response", title="Line-broadening comparison")
    ax.legend()
    return fig


@figure_test
def test_spectral_response_effective_area_cutoff():
    """
    A finite effective-area range clips an otherwise broad line profile to zero.
    """
    line_list, main_lines = _line_list_and_name()
    effective_area = synthetic_effective_area(values=[1.0, 1.0], wavelength=[170.9, 171.1])
    kwargs = {
        "line_list": line_list,
        "wavelength_grid": WAVELENGTH_GRID,
        "main_lines": main_lines,
        "instrumental_width": 0.06 * u.AA,
    }
    raw = create_spectral_response(**kwargs).spectral_response.sel(logT=6.0).isel(line=0)
    filtered = (
        create_spectral_response(**kwargs, effective_area=effective_area).spectral_response.sel(logT=6.0).isel(line=0)
    )

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(WAVELENGTH_GRID.value, raw / raw.max(), label="without effective area")
    ax.plot(WAVELENGTH_GRID.value, filtered / filtered.max(), label="with effective area")
    ax.axvspan(170.9, 171.1, color="0.9", label="effective-area coverage")
    ax.set(xlabel="wavelength [Angstrom]", ylabel="normalized response", title="Effective-area coverage")
    ax.legend()
    return fig
