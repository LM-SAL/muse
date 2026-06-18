import matplotlib.pyplot as plt
import numpy as np

from muse.synthesis.synthesis import vdem_synthesis
from muse.tests.helpers import figure_test
from muse.transforms.transforms import reshape_x_to_slit_step

__all__ = []


@figure_test
def test_vdem_synthesis_detector_spectrum(response, vdem):
    """Synthesized detector spectrum (flux vs SG_xpixel) per line at the brightest y/step."""
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    flux = vdem_synthesis(reshaped_vdem, response).flux
    brightest = flux.sum(dim=["line", "SG_xpixel"])
    it, istep = (int(i) for i in np.unravel_index(int(brightest.values.argmax()), brightest.shape))
    fig, ax = plt.subplots()
    flux.isel(y=it, step=istep).plot.line(x="SG_xpixel", hue="line", ax=ax)
    ax.set_title(f"synthesized spectrum at y={it}, step={istep}")
    return fig


@figure_test
def test_vdem_synthesis_fov(response, vdem):
    """Synthesized total-flux FOV: slit-summed (y, step) vs full-resolution (y, x)."""
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    # Default sum_over collapses slit -> coarse (y, step) field.
    collapsed = vdem_synthesis(reshaped_vdem, response).flux.sum(dim=["line", "SG_xpixel"])
    # Keep slit, then restack (slit, step) -> x to recover the full spatial resolution.
    kept = vdem_synthesis(reshaped_vdem, response, sum_over=("logT", "vdop")).flux.sum(dim=["line", "SG_xpixel"])
    # reset_index drops the (slit, step) MultiIndex, leaving a plain 0..384 x axis to plot.
    full = kept.stack(x=("slit", "step")).transpose("y", "x").reset_index("x", drop=True)
    fig, (ax_coarse, ax_full) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    collapsed.plot(ax=ax_coarse)
    ax_coarse.set_title("slit-summed (y, step)")
    full.plot(ax=ax_full)
    ax_full.set_title("full resolution (y, x pixel)")
    return fig
