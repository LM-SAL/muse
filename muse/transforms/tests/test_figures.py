import matplotlib.pyplot as plt

from muse.tests.helpers import figure_test
from muse.transforms.transforms import muse_fov


@figure_test
def test_muse_fov_resampled_intensity_map(vdem_offgrid):
    """Before/after muse_fov: off-grid input vs MUSE-grid output, (y, x) maps over logT+vdop."""
    out = muse_fov(vdem_offgrid)
    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    vdem_offgrid.vdem.sum(dim=["logT", "vdop"]).plot(ax=ax_in)
    ax_in.set_title("input (off-grid)")
    out.vdem.sum(dim=["logT", "vdop"]).plot(ax=ax_out)
    ax_out.set_title("muse_fov output (MUSE grid)")
    return fig
