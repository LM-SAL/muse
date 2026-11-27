import matplotlib.pyplot as plt
import numpy as np

from muse.synthesis.synthesis import vdem_synthesis
from muse.synthesis.utils import calculate_moments, wavelength_to_doppler
from muse.tests.helpers import LOOP_SLITS, fake_loop_response, fake_loop_vdem, figure_test
from muse.transforms.transforms import reshape_slit_step_to_x, reshape_x_to_slit_step


@figure_test
def test_vdem_synthesis_fov():
    """
    The flux of both lines with the slits kept and restacked onto x (left): slit k
    images the 11 pixels between its dashed lines.

    The default sum over slits folds these strips onto the 11 steps (middle), the apex
    in slit 3 onto step 5 and the legs in slits 1 and 5 onto steps 2 and 8, and equals
    the kept slits summed (right, a ratio of 1).
    """
    raster = reshape_x_to_slit_step(fake_loop_vdem(), number_of_slits=LOOP_SLITS)
    summed = vdem_synthesis(raster, fake_loop_response()).flux.sum(["line", "detector_x_pixel"])
    kept = vdem_synthesis(raster, fake_loop_response(), sum_over=("logT", "doppler_velocity"))
    full = reshape_slit_step_to_x(kept).flux.sum(["line", "detector_x_pixel"])
    # One row of x per slit, as reshape_x_to_slit_step cuts them.
    strips = full.x.values.reshape(LOOP_SLITS, -1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), width_ratios=(3, 1, 1), constrained_layout=True)
    full.plot(ax=axes[0], cmap="inferno", cbar_kwargs={"label": "flux [ph / s]"})
    for edge in (strips[1:, 0] + strips[:-1, -1]) / 2:
        axes[0].axvline(edge, color="white", linestyle="--", linewidth=0.8)
    axes[0].secondary_xaxis("top").set_xticks(strips.mean(axis=1), labels=[f"slit {k}" for k in range(LOOP_SLITS)])
    axes[0].set_title("Fe IX 171.073 + Fe XV 284.163, slits kept, restacked onto x")
    summed.plot(ax=axes[1], cmap="inferno", cbar_kwargs={"label": "flux [ph / s]"})
    axes[1].set_title("slits summed (default)")
    # Rounded, so float noise cannot flip pixels between the two middle colors; a tight
    # range, so a slit left out of the sum, 1/7 of the background, turns it blue.
    ratio = (summed / kept.flux.sum(["slit", "line", "detector_x_pixel"])).round(3)
    ratio.plot(ax=axes[2], cmap="RdBu_r", vmin=0.9, vmax=1.1, cbar_kwargs={"label": "ratio"})
    axes[2].set_title("summed / kept slits summed")
    return fig


@figure_test
def test_vdem_synthesis_detector_spectrum():
    """
    Fe XV on the detector at raster step 5, its slits summed (top): each slit images the
    scene 4.4" further along x, its copy of the line next to the dashed pixel of its
    rest wavelength.

    Below, slits 1, 3 and 5 in their own Doppler velocity: the loop crosses slit 1 at
    about -35 km/s, slit 3 at rest at its apex and slit 5 at about +70 km/s.
    """
    raster = reshape_x_to_slit_step(fake_loop_vdem(), number_of_slits=LOOP_SLITS)
    response = fake_loop_response().sel(line="Fe XV 284.163")
    spectrum = vdem_synthesis(raster, response).isel(step=5)
    # Read off the detector wavelengths, not through wavelength_to_doppler, which the panels below check.
    rest = [
        np.interp(response.line_wavelength, wavelength, response.detector_x_pixel)
        for wavelength in response.detector_wavelength
    ]
    fig, axes = plt.subplot_mosaic([["detector"] * 3, [1, 3, 5]], figsize=(11, 7), constrained_layout=True)
    spectrum.flux.plot(ax=axes["detector"], x="detector_x_pixel", cmap="inferno")
    for pixel in rest:
        axes["detector"].axvline(pixel, color="white", linestyle="--", linewidth=0.8)
    axes["detector"].secondary_xaxis("top").set_xticks(rest, labels=[f"slit {slit}" for slit in range(LOOP_SLITS)])
    axes["detector"].set(title="Fe XV 284.163 at raster step 5, all slits summed", xlabel="detector pixel")
    for slit in (1, 3, 5):
        view = wavelength_to_doppler(
            spectrum.assign_coords(detector_wavelength=response.detector_wavelength.sel(slit=slit))
        )
        view.flux.plot(
            ax=axes[slit], x="detector_doppler_velocity", y="y", xlim=(-150, 150), cmap="inferno", add_colorbar=False
        )
        axes[slit].axvline(0, color="white", linestyle="--")
        axes[slit].set(title=f"slit {slit}", xlabel="Doppler velocity [km / s]")
    return fig


@figure_test
def test_calculate_moments_maps():
    """
    Moments of each slit's spectrum, restacked onto x.

    Fe IX shows the footpoints and the background, Fe XV the loop, blueshifted in its
    left leg and twice as redshifted in its right. The widths are the instrumental ones,
    30 km/s for Fe IX and 37 for Fe XV, over the background, and broader on the loop,
    most where its flow overlaps plasma at rest.
    """
    raster = reshape_x_to_slit_step(fake_loop_vdem(), number_of_slits=LOOP_SLITS)
    spectrum = vdem_synthesis(raster, fake_loop_response(), sum_over=("logT", "doppler_velocity"))
    moments = reshape_slit_step_to_x(calculate_moments(wavelength_to_doppler(spectrum)))
    # Fixed ranges, so a wrong scale changes the colors and not only the colorbar ticks.
    columns = (
        ("0th", "intensity [ph / s]", {}),
        ("1st", "velocity [km / s]", {"cmap": "RdBu_r", "vmin": -80, "vmax": 80}),
        ("2nd", "width $\\sigma$ [km / s]", {"vmin": 30, "vmax": 60}),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), constrained_layout=True)
    for row, line in zip(axes, moments.line.values, strict=True):
        for ax, (name, label, kwargs) in zip(row, columns, strict=True):
            moments[name].sel(line=line).plot(ax=ax, cbar_kwargs={"label": label}, **kwargs)
            ax.set_title(line)
    return fig
