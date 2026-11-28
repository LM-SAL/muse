import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.instrument.response.detector import map_response_to_sg_detector
from muse.instrument.response.io import align_response_and_vdem
from muse.instrument.response.spectral import create_spectral_response
from muse.tests.helpers import (
    EXACT_STYLE,
    TESTED_STYLE,
    fake_effective_area,
    fake_line_list,
    fake_response,
    figure_test,
)
from muse.variables import DEFAULTS_MUSE

pytest.importorskip("numexpr")
pytest.importorskip("periodictable")

SPEED_OF_LIGHT = const.c.to_value(u.km / u.s)
COLORS = ("tab:blue", "tab:green", "tab:red")
# Under the axes, where it covers no curve.
LEGEND = {"fontsize": 8, "loc": "upper center", "bbox_to_anchor": (0.5, -0.16), "ncols": 2, "frameon": False}


def _thermal_sigma(wavelength):
    """
    The thermal width, as a sigma in the units of ``wavelength``, of an iron line at 1
    MK.
    """
    return wavelength * np.sqrt(const.k_B * 1e6 * u.K / (55.845 * u.u)).to_value(u.km / u.s) / SPEED_OF_LIGHT


@figure_test
def test_create_spectral_response_profiles():
    """
    Left: one iron line at 1 MK, alone, with an instrumental width and with a nonthermal
    width. Right: the same line Doppler shifted, its expected centers dashed.

    The response (dots) lies on the Gaussian worked out from the widths and the shift
    (lines).
    """
    rest = 171.0
    line_list = fake_line_list(wavelength=[rest])
    grid = np.linspace(170.75, 171.25, 501)
    kwargs = {"line_list": line_list, "wavelength_grid": grid * u.AA, "main_lines": [line_list.full_name.item()]}
    thermal, instrumental, xi = _thermal_sigma(rest), 0.05, 80.0
    fwhm = instrumental * np.sqrt(8 * np.log(2)) * u.AA

    def profile(**extra):
        return create_spectral_response(**kwargs, **extra).spectral_response.sel(logT=6.0).squeeze(drop=True)

    def gaussian(center, sigma, like):
        # With the area of the response it is compared with.
        area = float(like.sum()) * (grid[1] - grid[0])
        return area * np.exp(-0.5 * ((grid - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    widths = (
        ("thermal only", thermal, {}),
        ("with the instrument, $\\sigma$ = 0.05 Å", np.hypot(thermal, instrumental), {"instrumental_fwhm": fwhm}),
        (
            "with nonthermal motions, $\\xi$ = 80 km/s",
            np.hypot(thermal, rest * xi / (SPEED_OF_LIGHT * np.sqrt(2))),
            {"nonthermal_velocity": xi * u.km / u.s},
        ),
    )
    fig, (ax_width, ax_shift) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for (label, sigma, extra), color in zip(widths, COLORS, strict=True):
        tested = profile(**extra)
        exact = gaussian(rest, sigma, tested)
        # Each over the peak of its exact Gaussian, as the narrow line is seven times the height of the others.
        ax_width.plot(grid, exact / exact.max(), **EXACT_STYLE)
        style = {**TESTED_STYLE, "color": color, "markersize": 3}
        ax_width.plot(grid[::2], tested[::2] / exact.max(), label=f"Tested: {label}", **style)
    ax_width.plot([], [], label="Exact: Gaussian of the expected width", **EXACT_STYLE)
    ax_width.set(title="Width", xlabel="wavelength [Å]", ylabel="spectral response / exact peak", xlim=(170.85, 171.15))
    ax_width.legend(**LEGEND)

    shifted = profile(instrumental_fwhm=fwhm, doppler_velocity=[-100.0, 0.0, 100.0] * u.km / u.s)
    for velocity, color in zip(shifted.doppler_velocity.values, COLORS, strict=True):
        tested = shifted.sel(doppler_velocity=velocity)
        center = rest * (1 + velocity / SPEED_OF_LIGHT)
        ax_shift.axvline(center, color=color, linestyle="--", linewidth=0.8)
        ax_shift.plot(grid, gaussian(center, np.hypot(thermal, instrumental), tested), **EXACT_STYLE)
        ax_shift.plot(
            grid[::8],
            tested[::8],
            label=f"Tested: {velocity:+.0f} km/s".replace("+0", "0"),
            **{**TESTED_STYLE, "color": color},
        )
    ax_shift.plot([], [], label="Exact: Gaussian at the shifted wavelength", **EXACT_STYLE)
    ax_shift.set(
        title="Doppler shift (negative is towards the observer)",
        xlabel="wavelength [Å]",
        ylabel="spectral response",
    )
    ax_shift.legend(**LEGEND)
    return fig


@figure_test
def test_map_response_to_sg_detector_slits():
    """
    Left: the response of one line on the detector, one row per slit; each bright tick
    lies in the circle at the pixel worked out from the dispersion and the slit
    separation.

    Right: three slits against the distance from that pixel (dots); all lie on the line
    as its thermal and instrumental widths draw it there (line).
    """
    rest, channel = 171.073, 171
    line_list = fake_line_list(wavelength=[rest])
    instrumental_fwhm = DEFAULTS_MUSE.instrumental_fwhm_SG.sel(channel=channel).item()
    response = create_spectral_response(
        line_list,
        np.arange(170.8, 171.35, 0.002) * u.AA,
        main_lines=[line_list.full_name.item()],
        instrumental_fwhm=instrumental_fwhm,
        effective_area=fake_effective_area(),
    )
    detector = map_response_to_sg_detector(response, channel=channel * u.AA).detector_response
    detector = detector.sel(logT=6.0).squeeze(drop=True)
    start = DEFAULTS_MUSE.wavelength_start_SG.sel(channel=channel).item().to_value(u.AA)
    separation = DEFAULTS_MUSE.spectral_slit_separation_SG.sel(channel=channel).item().to_value(u.AA)
    between = u.Quantity(DEFAULTS_MUSE.pixels_between_slits_SG).value
    dispersion = separation / between
    expected = (rest - start) / dispersion + between * detector.slit

    fig, (ax_map, ax_cut) = plt.subplots(1, 2, figsize=(13, 4.6), width_ratios=(1.6, 1), constrained_layout=True)
    detector.plot(ax=ax_map, x="detector_x_pixel", y="slit", cmap="inferno", cbar_kwargs={"label": "detector response"})
    ax_map.plot(expected[::2], detector.slit[::2], "o", markerfacecolor="none", markeredgecolor="cyan", markersize=9)
    ax_map.set(
        title="One line on the detector (circles: the pixel expected for each slit)",
        xlabel="detector pixel",
        ylabel="slit",
    )
    sigma = np.hypot(_thermal_sigma(rest), instrumental_fwhm.to_value(u.AA) / np.sqrt(8 * np.log(2))) / dispersion
    offset = np.linspace(-12, 12, 481)
    exact = float(detector.sel(slit=0).sum()) * np.exp(-0.5 * (offset / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax_cut.plot(offset, exact, label="Exact: the line at the expected pixel", **EXACT_STYLE)
    for slit, color, marker in zip((0, 17, 34), COLORS, ("o", "s", "^"), strict=True):
        row = detector.sel(slit=slit)
        style = {**TESTED_STYLE, "color": color, "marker": marker, "markersize": 5}
        ax_cut.plot(row.detector_x_pixel - float(expected.sel(slit=slit)), row, label=f"Tested: slit {slit}", **style)
    ax_cut.set(
        title="Three slits, each against its expected pixel",
        xlabel="detector pixel - expected pixel",
        ylabel="detector response",
        xlim=(-12, 12),
    )
    ax_cut.legend(**LEGEND)
    return fig


@figure_test
def test_align_response_and_vdem_velocities():
    """
    The response of one line at three detector pixels against Doppler velocity (lines),
    and the same after alignment to a VDEM with a finer velocity grid (dots, on the
    lines).
    """
    response = fake_response()
    velocity, logT = np.arange(-300.0, 301.0, 37.5), 6.4
    vdem = xr.Dataset(
        {"vdem": (("logT", "doppler_velocity"), np.ones((1, velocity.size)), {"units": "1e27 / cm5"})},
        coords={
            "logT": ("logT", [logT], {"units": "dex(K)"}),
            "doppler_velocity": ("doppler_velocity", velocity, {"units": "km/s"}),
        },
    )
    aligned, _ = align_response_and_vdem(response, vdem)
    # The line and slit whose response changes most with velocity, and its three pixels that change most.
    spread = response.detector_response.sel(logT=logT).std("doppler_velocity")
    where = {dim: int(index) for dim, index in spread.max("detector_x_pixel").argmax(...).items()}
    pixels = np.sort(spread.isel(where).argsort().values[-3:])

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for pixel, color in zip(pixels, COLORS, strict=True):
        before, after = (
            dataset.detector_response.isel({**where, "detector_x_pixel": pixel}).sel(logT=logT)
            for dataset in (response, aligned)
        )
        ax.plot(before.doppler_velocity, before, **EXACT_STYLE)
        label = f"Tested: aligned, detector pixel {int(response.detector_x_pixel[pixel])}"
        ax.plot(after.doppler_velocity, after, label=label, **{**TESTED_STYLE, "color": color})
    ax.plot([], [], label="Exact: the response as given, joined by straight lines", **EXACT_STYLE)
    ax.set(
        title=f"{response.line.values[where['line']]}, slit {where['slit']}, log T = {logT}",
        xlabel="Doppler velocity [km / s]",
        ylabel="detector response",
    )
    ax.legend(**LEGEND)
    return fig
