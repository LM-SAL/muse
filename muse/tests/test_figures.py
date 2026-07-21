import matplotlib.pyplot as plt
import numpy as np

from muse.tests.helpers import figure_test


@figure_test
def test_vdem_intensity_map(vdem):
    """
    VDEM integrated over logT and vdop -> (y, x) intensity map.
    """
    fig, ax = plt.subplots()
    vdem.vdem.sum(dim=["logT", "vdop"]).plot(ax=ax)
    return fig


@figure_test
def test_response_logt_profile(response):
    """
    Detector response versus temperature for each line at vdop=0 and mid-slit.
    """
    fig, ax = plt.subplots()
    response.detector_response.sel(vdop=0.0).isel(slit=17).sum(dim="detector_x_pixel").plot.line(x="logT", ax=ax)
    return fig


@figure_test
def test_response_line_profiles(response):
    """
    Spectral line profile vs wavelength per channel, with peak wavelength marked.
    """
    channels = [108, 171, 284]
    fig, axes = plt.subplots(1, len(channels), figsize=(12, 4), constrained_layout=True)
    # vdop=0 rest frame, mid-slit, integrated over temperature -> spectral profile vs pixel.
    profiles = response.detector_response.sel(vdop=0.0).isel(slit=17).sum(dim="logT")
    wavelengths = response.detector_wavelength.isel(slit=17)
    for ax, channel in zip(axes, channels, strict=True):
        in_channel = response.channel == channel
        for line_index in np.flatnonzero(in_channel.values):
            profile = profiles.isel(line=line_index)
            wavelength = wavelengths.isel(line=line_index)
            ax.plot(wavelength, profile, label=str(response.line.values[line_index]))
            peak_wavelength = float(wavelength.values[profile.values.argmax()])
            ax.axvline(peak_wavelength, ls="--", color="k", alpha=0.4)
        ax.set_title(f"channel {channel}")
        ax.set_xlabel("detector wavelength [Angstrom]")
        ax.legend(fontsize="x-small")
    axes[0].set_ylabel("detector response [1e-27 ph cm5 / s]")
    return fig
