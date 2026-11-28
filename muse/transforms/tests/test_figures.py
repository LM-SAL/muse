import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import astropy.units as u

from muse.tests.helpers import figure_test
from muse.transforms.transforms import match_fov


@figure_test
def test_match_fov_letter():
    """
    The letter F on a fine grid (left), on MUSE pixels (middle), and on MUSE pixels
    after a quarter turn counter-clockwise (right).

    It reads as an F in the first two, in the same place, and lies on its back in the
    third.
    """
    x, y = np.meshgrid(np.arange(0, 12.01, 0.1), np.arange(0, 10.01, 0.05))
    stem = (x >= 2) & (x <= 3.5) & (y >= 1) & (y <= 9)
    top = (x >= 2) & (x <= 9) & (y >= 7.5) & (y <= 9)
    middle = (x >= 2) & (x <= 7) & (y >= 4.5) & (y <= 6)
    vdem = xr.Dataset(
        {
            "vdem": (
                ("logT", "doppler_velocity", "y", "x"),
                (stem | top | middle).astype(float)[None, None],
                {"units": "1e27 / cm5"},
            )
        },
        coords={
            "logT": ("logT", [6.0], {"units": "dex(K)"}),
            "doppler_velocity": ("doppler_velocity", [0.0], {"units": "km/s"}),
            "y": ("y", y[:, 0], {"units": "arcsec"}),
            "x": ("x", x[0], {"units": "arcsec"}),
        },
    )
    panels = (
        ("Input: 0.1 x 0.05 arcsec pixels", vdem),
        ("match_fov: 0.4 x 0.167 arcsec MUSE pixels", match_fov(vdem, extent="keep")),
        ("match_fov, a quarter turn\ncounter-clockwise", match_fov(vdem, extent="keep", rotation=90 * u.deg)),
    )
    # NaN in grey, so it cannot pass for the black of no emission.
    cmap = plt.get_cmap("inferno").with_extremes(bad="0.6")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for ax, (title, scene) in zip(axes, panels, strict=True):
        scene.vdem.squeeze(drop=True).plot(ax=ax, x="x", y="y", cmap=cmap, vmin=0, vmax=1, add_colorbar=False)
        ax.set(title=title, xlabel="x [arcsec]", ylabel="y [arcsec]", aspect="equal")
    return fig
