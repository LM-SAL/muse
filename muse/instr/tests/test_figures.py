import matplotlib.pyplot as plt

from muse.instr.gausslobes import gausslobes
from muse.tests.helpers import figure_test


@figure_test
def test_gausslobes_psf():
    """Default Gausslobe PSF (x, y); vmax saturates the core so the side lobes show."""
    fig, ax = plt.subplots()
    gausslobes().plot(ax=ax, vmin=0, vmax=0.002)
    return fig
