import matplotlib.pyplot as plt
import numpy as np

from muse.synthesis.synthesis import vdem_synthesis
from muse.synthesis.utils import calculate_moments, wavelength_to_doppler
from muse.tests.helpers import fake_vdem_single_vdop, figure_test
from muse.transforms.transforms import reshape_x_to_slit_step


@figure_test
def test_vdem_synthesis_detector_spectrum(response, vdem):
    """
    Synthesized detector spectrum (flux vs SG_xpixel) per line at the brightest y/step.
    """
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
    """
    Synthesized total-flux FOV: slit-summed (y, step) vs full-resolution (y, x).
    """
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


@figure_test
def test_vdem_synthesis_doppler_shift(response):
    """
    Line-0 spectra synthesized at vdop -300/0/+300 km/s march across wavelength about
    the rest line.
    """
    fig, ax = plt.subplots()
    for vdop_kms in (-300.0, 0.0, 300.0):
        reshaped_vdem = reshape_x_to_slit_step(fake_vdem_single_vdop(vdop_kms), nslits=35, nraster=11)
        flux = vdem_synthesis(reshaped_vdem, response, sum_over=("logT", "vdop")).flux
        spectrum = flux.isel(line=0, slit=17).sum(dim=["y", "step"])
        wavelength = flux.SG_wvl.isel(line=0, slit=17).values
        ax.plot(wavelength, spectrum.values, label=f"vdop={vdop_kms:+.0f} km/s")
    ax.axvline(108.355, ls="--", color="k", alpha=0.4, label="rest 108.355 A")
    ax.set_xlim(107.0, 109.5)  # zoom on the line so the ~0.1 A shift is visible
    ax.set_xlabel("SG_wvl [Angstrom]")
    ax.set_ylabel("flux [ph / s]")
    ax.legend()
    return fig


@figure_test
def test_calculate_moments_maps(response, vdem):
    """
    0th/1st/2nd moment maps over (y, step) for line 0 at the central slit.
    """
    reshaped_vdem = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    spectrum = wavelength_to_doppler(vdem_synthesis(reshaped_vdem, response, sum_over=("logT", "vdop")))
    moments = calculate_moments(spectrum).isel(line=0, slit=17)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    titles = ("0th: intensity [ph / s]", "1st: velocity [km / s]", "2nd: width [km / s]")
    for ax, name, title in zip(axes, ("0th", "1st", "2nd"), titles, strict=True):
        moments[name].plot(ax=ax)
        ax.set_title(title)
    return fig
