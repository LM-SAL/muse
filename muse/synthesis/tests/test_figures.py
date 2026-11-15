import matplotlib.pyplot as plt
import numpy as np

from muse.synthesis.synthesis import vdem_synthesis
from muse.synthesis.utils import calculate_moments, wavelength_to_doppler
from muse.tests.helpers import figure_test


@figure_test
def test_vdem_synthesis_detector_spectrum(response, raster):
    """
    Synthesized detector spectrum per line at the brightest y/step.
    """
    flux = vdem_synthesis(raster, response).flux
    brightest = flux.sum(dim=["line", "detector_x_pixel"])
    it, istep = (int(i) for i in np.unravel_index(int(brightest.values.argmax()), brightest.shape))
    fig, ax = plt.subplots()
    flux.isel(y=it, step=istep).plot.line(x="detector_x_pixel", hue="line", ax=ax)
    ax.set_title(f"synthesized spectrum at y={it}, step={istep}")
    return fig


@figure_test
def test_vdem_synthesis_fov(response, raster):
    """
    Synthesized total-flux FOV: slit-summed (y, step) vs full-resolution (y, x).
    """
    # Default sum_over collapses slit -> coarse (y, step) field.
    collapsed = vdem_synthesis(raster, response).flux.sum(dim=["line", "detector_x_pixel"])
    # Keep slit, then restack (slit, step) -> x to recover the full spatial resolution.
    kept = vdem_synthesis(raster, response, sum_over=("logT", "doppler_velocity")).flux.sum(
        dim=["line", "detector_x_pixel"]
    )
    # reset_index drops the (slit, step) MultiIndex, leaving a plain 0..384 x axis to plot.
    full = kept.stack(x=("slit", "step")).transpose("y", "x").reset_index("x", drop=True)
    fig, (ax_coarse, ax_full) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    collapsed.plot(ax=ax_coarse)
    ax_coarse.set_title("slit-summed (y, step)")
    full.plot(ax=ax_full)
    ax_full.set_title("full resolution (y, x pixel)")
    return fig


@figure_test
def test_calculate_moments_maps(response, raster):
    """
    0th/1st/2nd moment maps over (y, step) for line 0 at the central slit.
    """
    spectrum = wavelength_to_doppler(vdem_synthesis(raster, response, sum_over=("logT", "doppler_velocity")))
    moments = calculate_moments(spectrum).isel(line=0, slit=17)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    titles = ("0th: intensity [ph / s]", "1st: velocity [km / s]", "2nd: width [km / s]")
    for ax, name, title in zip(axes, ("0th", "1st", "2nd"), titles, strict=True):
        moments[name].plot(ax=ax)
        ax.set_title(title)
    return fig
