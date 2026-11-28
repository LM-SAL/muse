import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import astropy.units as u

from muse.synthesis.synthesis import vdem_synthesis
from muse.synthesis.utils import calculate_moments, create_simple_vdem, wavelength_to_doppler
from muse.tests.helpers import (
    EXACT_STYLE,
    LOOP_SLITS,
    TESTED_STYLE,
    fake_loop_regions,
    fake_loop_response,
    fake_loop_vdem,
    figure_test,
)
from muse.transforms.transforms import reshape_slit_step_to_x, reshape_x_to_slit_step


@figure_test
def test_vdem_synthesis_fov():
    """
    The flux of both lines with each slit's image side by side (left): slit k images the
    11 pixels between its dashed lines.

    The default sum over slits folds these images onto the 11 raster steps (middle), the
    footpoints in slits 1 and 5 onto steps 2 and 8. Along the dashed row through the
    footpoints (right), its dots lie on the line of the kept slits added afterwards.
    """
    raster = reshape_x_to_slit_step(fake_loop_vdem(), number_of_slits=LOOP_SLITS)
    summed = vdem_synthesis(raster, fake_loop_response()).flux.sum(["line", "detector_x_pixel"])
    kept = vdem_synthesis(raster, fake_loop_response(), sum_over=("logT", "doppler_velocity"))
    full = reshape_slit_step_to_x(kept).flux.sum(["line", "detector_x_pixel"])
    # One row of x per slit, as reshape_x_to_slit_step cuts them.
    strips = full.x.values.reshape(LOOP_SLITS, -1)
    cut = 3.5
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), width_ratios=(3, 1, 1.5), constrained_layout=True)
    full.plot(ax=axes[0], cmap="inferno", cbar_kwargs={"label": "flux [ph / s]"})
    for edge in (strips[1:, 0] + strips[:-1, -1]) / 2:
        axes[0].axvline(edge, color="white", linestyle="--", linewidth=0.8)
    axes[0].secondary_xaxis("top").set_xticks(strips.mean(axis=1), labels=[f"slit {k}" for k in range(LOOP_SLITS)])
    axes[0].set_title("Both lines, each slit's image side by side")
    summed.plot(ax=axes[1], cmap="inferno", cbar_kwargs={"label": "flux [ph / s]"})
    axes[1].axhline(cut, color="white", linestyle="--", linewidth=0.8)
    axes[1].set_title("All slits summed onto the steps")
    reference = kept.flux.sum(["slit", "line", "detector_x_pixel"])
    for flux, label, style in (
        (reference, "Reference: slits kept, then added", {"color": "black"}),
        (summed, "Tested: slits summed by default", {"color": "tab:blue", "marker": "o", "linestyle": "none"}),
    ):
        axes[2].plot(flux.step, flux.sel(y=cut, method="nearest"), label=label, **style)
    axes[2].set(xlabel="step", ylabel="flux [ph / s]", title="Flux along the dashed line", ylim=(0, None))
    axes[2].legend(loc="lower center", fontsize=8)
    return fig


@figure_test
def test_vdem_synthesis_detector_spectrum():
    """
    Fe XV on the detector at raster step 5, its slits summed (top): each slit images the
    scene 4.4" further along x, its copy of the line next to the dashed pixel of its
    rest wavelength.

    Below, slits 1, 3 and 5 in their own Doppler velocity: the loop's emission lies on
    the dotted line of its own flow, -37 km/s where slit 1 crosses its left leg, at rest
    at its apex in slit 3 and +68 km/s in its right leg in slit 5.
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
    axes["detector"].set(
        title="Fe XV 284.163 on the detector at raster step 5 (dashed: each slit's rest wavelength)",
        xlabel="detector pixel",
    )
    # The loop's flow where each slit crosses it at step 5, cos(angle) * (60 + 20 cos(angle)) km/s.
    crossings = {1: ("left leg, blueshifted", -37), 3: ("apex, at rest", 0), 5: ("right leg, redshifted", 68)}
    for slit, (part, flow) in crossings.items():
        view = wavelength_to_doppler(
            spectrum.assign_coords(detector_wavelength=response.detector_wavelength.sel(slit=slit))
        )
        view.flux.plot(
            ax=axes[slit], x="detector_doppler_velocity", y="y", xlim=(-150, 150), cmap="inferno", add_colorbar=False
        )
        axes[slit].axvline(0, color="white", linestyle="--", label="at rest")
        axes[slit].axvline(flow, color="cyan", linestyle=":", label="the loop's flow")
        axes[slit].set(title=f"Slit {slit}: {part}", xlabel="Doppler velocity [km / s]")
    axes[1].legend(loc="lower right", fontsize=8)
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
        ("0th", "intensity", "[ph / s]", {"cmap": "viridis"}),
        ("1st", "Doppler velocity", "[km / s]", {"cmap": "RdBu_r", "vmin": -80, "vmax": 80}),
        ("2nd", "line width", "$\\sigma$ [km / s]", {"cmap": "viridis", "vmin": 30, "vmax": 60}),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), constrained_layout=True)
    for row, line in zip(axes, moments.line.values, strict=True):
        for ax, (name, heading, unit, kwargs) in zip(row, columns, strict=True):
            # NaN in grey: drawn blank, it would pass for the white of 0 km/s.
            cmap = plt.get_cmap(kwargs["cmap"]).with_extremes(bad="0.6")
            moments[name].sel(line=line).plot(ax=ax, cbar_kwargs={"label": unit}, **{**kwargs, "cmap": cmap})
            ax.set_title(f"{line}: {heading}")
    return fig


@figure_test
def test_vdem_synthesis_with_density():
    """
    The flux of both lines along a cut through the footpoints of a scene with three
    densities, synthesized three ways.

    Exact (black line): the background, the loop and the footpoints are synthesized
    separately, each at its own density, and added. Tested (blue dots): the three are
    added into one VDEM and synthesized with ``density``, the emission-weighted density
    of each bin. For contrast (orange line): the same VDEM at the background's 1e9 cm^-3
    everywhere.

    The dots lie on the black line, within 2%. The orange line is up to 2.5 times too
    high at the footpoints in Fe IX and 1.4 times in Fe XV.
    """
    regions, vdem = fake_loop_regions(), fake_loop_vdem()
    spectrograph = fake_loop_response().isel(slit=0, drop=True)
    logD = xr.DataArray(np.arange(8.0, 12.1, 0.5), dims="logD", attrs={"units": "dex(1 / cm3)"})
    # An imager's response that falls as 1 / (1 + n / 1e10 cm^-3): the line integrated, at its wavelength.
    imager = spectrograph.detector_response.sum("detector_x_pixel") / (1 + 10 ** (logD - 10))
    response = (
        imager.assign_attrs(spectrograph.detector_response.attrs)
        .to_dataset(name="detector_response")
        .assign_coords(logD=logD, detector_wavelength=spectrograph.line_wavelength)
    )
    density = sum(region.vdem * region.density for region in regions) / vdem.vdem
    exact = sum(vdem_synthesis(region, response, density=region.density).flux for region in regions)
    tested = vdem_synthesis(vdem, response, density=density.assign_attrs(units="cm-3")).flux
    contrast = vdem_synthesis(vdem, response, density=regions[-1].density).flux
    curves = (
        (exact, "Exact: each region at its own density", {"color": "black", "zorder": 2}),
        (
            tested,
            "Tested: density=, one density per VDEM bin",
            {"color": "tab:blue", "marker": "o", "markersize": 3, "linestyle": "none", "zorder": 3},
        ),
        (contrast, "For contrast: 1e9 cm$^{-3}$ everywhere", {"color": "tab:orange", "zorder": 1}),
    )
    cut = 3.5
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), height_ratios=(1, 1.2), constrained_layout=True)
    for (top, bottom), line in zip(axes.T, exact.line.values, strict=True):
        exact.sel(line=line).plot(ax=top, cmap="inferno", cbar_kwargs={"label": "exact flux [ph / s]"})
        top.axhline(cut, color="white", linestyle="--", linewidth=0.8)
        top.set_title(line)
        for flux, label, style in curves:
            bottom.plot(flux.x, flux.sel(line=line).sel(y=cut, method="nearest"), label=label, **style)
        bottom.set(xlabel="x [arcsec]", ylabel="flux [ph / s]", title="Flux along the dashed line", ylim=(0, None))
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("Background 1e9, loop 5e9, footpoints 3e10 cm$^{-3}$")
    return fig


@figure_test
def test_create_simple_vdem():
    """
    Left: three cells; each cross, at a cell's true log T and Doppler velocity, lies in
    a bright patch.

    Middle and right: one cell at a time; the mean Doppler velocity and the mean log T of
    its VDEM (dots) lie on the cell's own speed, its sign flipped as it is positive
    towards the observer, and on its own log T (lines).
    """
    velocity_grid = np.arange(-100.0, 101.0, 25.0) * u.km / u.s
    logT_grid = np.arange(5.5, 6.85, 0.1) * u.dex(u.K)

    def vdem_of(logT, velocity):
        # One cell per line of sight, the lines of sight along x.
        shape = (np.size(logT), 1, 1)
        return create_simple_vdem(
            temperature=np.reshape(10.0 ** np.asarray(logT), shape) * u.K,
            velocity=np.reshape(velocity, shape) * u.km / u.s,
            ne_nh=np.full(shape, 1e18) * u.cm**-6,
            cell_length=[1e9] * u.cm,
            x=np.arange(shape[0]) * 1e8 * u.cm,
            y=[0.0] * u.cm,
            velocity_grid=velocity_grid,
            logT_grid=logT_grid,
            temperature_bin_method="linear",
        ).vdem.isel(y=0)

    def mean(vdem, coordinate):
        return (vdem * vdem[coordinate]).sum(["logT", "doppler_velocity"]) / vdem.sum(["logT", "doppler_velocity"])

    cells = np.array([(5.8, 40.0), (6.2, -10.0), (6.5, -75.0)])  # log T, velocity towards the observer
    speeds, temperatures = np.arange(-95.0, 96.0, 10.0), np.arange(5.55, 6.76, 0.05)
    fig, (ax_map, ax_velocity, ax_logT) = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    vdem_of(cells[:, 0], cells[:, 1]).sum("x").plot(
        ax=ax_map, x="doppler_velocity", y="logT", cmap="inferno", cbar_kwargs={"label": "VDEM [1e27 cm$^{-5}$]"}
    )
    ax_map.plot(-cells[:, 1], cells[:, 0], "+", color="cyan", markersize=14, markeredgewidth=1.5)
    ax_map.set(title="Three cells (crosses: their true values)", xlabel="Doppler velocity [km / s]", ylabel="log T")
    ax_velocity.plot(speeds, -speeds, label="Exact: the same speed, sign flipped", **EXACT_STYLE)
    ax_velocity.plot(
        speeds,
        mean(vdem_of(np.full(speeds.size, 6.0), speeds), "doppler_velocity"),
        label="Tested: create_simple_vdem",
        **TESTED_STYLE,
    )
    ax_velocity.set(
        title="One cell: velocity",
        xlabel="velocity towards the observer [km / s]",
        ylabel="mean Doppler velocity of its VDEM [km / s]",
    )
    ax_logT.plot(temperatures, temperatures, label="Exact: the cell's log T", **EXACT_STYLE)
    ax_logT.plot(
        temperatures,
        mean(vdem_of(temperatures, np.zeros(temperatures.size)), "logT"),
        label="Tested: create_simple_vdem",
        **TESTED_STYLE,
    )
    ax_logT.set(title="One cell: temperature", xlabel="log T of the cell", ylabel="mean log T of its VDEM")
    for ax in (ax_velocity, ax_logT):
        ax.legend(fontsize=8)
    return fig
