"""
Draw the diagnostic plots of a run directory from its saved products::

    python plot_response_functions.py --run-dir RUN --effective-area-dir AREAS

Per band and abundance: the main-line cubes against velocity, the main and contaminant
detector responses, the detector wavelengths per slit, a logT summary across bands, the
effective areas, and the bright-line scatter plots from the cached line lists. The plots land
in RUN/plots.
"""

import os
import re
import sys
import pathlib
import argparse
from collections import defaultdict

import numpy as np
import xarray as xr

import astropy.units as u

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_config import DEFAULT_CI_BANDS, DEFAULT_SG_BANDS, effective_area_file

from muse.instrument import (
    rank_line_list,
    read_effective_area,
    read_response,
    read_spectral_response,
    select_effective_area_curve,
)
from muse.log import change_logging_level
from muse.variables import DEFAULTS_MUSE

PRODUCT = re.compile(r"respfunc_(?P<band>\d+|ci)_(?P<rest>.+)\.zarr$")


def discover(run_dir: pathlib.Path) -> dict:
    """
    Group the products by (band, abundance, plasma tag) and kind, from the file names
    and attributes.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted((run_dir / "SG").glob("respfunc_*.zarr")) + sorted((run_dir / "CI").glob("respfunc_*.zarr")):
        match = PRODUCT.search(path.name)
        if not match:
            continue
        band, rest = match.group("band"), match.group("rest")
        abundance = xr.open_zarr(path, consolidated=False).attrs.get("abundance", "unknown")
        plasma = re.search(r"(pressure_[0-9.eE+\-]+|density)", rest)
        if rest.startswith("nomainlines_waveband"):
            kind = "contaminant_cube"
        elif "_waveband_" in rest:
            kind = "main_cube" if band != "ci" else "ci_cube"
        elif rest.startswith("L_nslits"):
            kind = "contaminants"
        elif "_nslits" in rest:
            kind = "main"
        else:
            kind = "ci"
        groups[(band, abundance, plasma.group(1) if plasma else "?")][kind].append(path)
    return groups


def _peak(values, keep):
    """
    Collapse every dimension not in ``keep`` by its maximum.
    """
    return values.max(dim=[dim for dim in values.dims if dim not in keep])


def _load(path):
    response = read_response(path).compute()
    return response.isel(pressure=0) if "pressure" in response.dims else response


def velocity_plots(groups, plot_dir):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    for (band, _abundance, _plasma), kinds in groups.items():
        if band == "ci" or not kinds.get("main_cube"):
            continue
        cubes = xr.concat(
            [read_spectral_response(path).load() for path in kinds["main_cube"]], dim="line", data_vars="all"
        )
        fig, ax = plt.subplots(figsize=(15, 8))
        for velocity in np.arange(-1000, 1200, 200):
            _peak(
                cubes.spectral_response.sel(doppler_velocity=velocity, method="nearest"), {"wavelength_bin", "line"}
            ).sum("line").plot(ax=ax, x="wavelength_grid", label=velocity)
        ax.legend()
        ax.set_xlim(int(band) - 2, int(band) + 2)
        fig.savefig(plot_dir / f"{kinds['main_cube'][0].stem}_velocity.png")
        plt.close(fig)


def main_contaminant_and_plate_scale(groups, plot_dir):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    for (band, _abundance, _plasma), kinds in groups.items():
        if band == "ci" or not kinds.get("main") or not kinds.get("contaminants"):
            continue
        mains = xr.concat([_load(path) for path in kinds["main"]], dim="line", data_vars="all")
        contaminants = _load(kinds["contaminants"][0])
        stem = kinds["main"][0].stem
        fig, ax = plt.subplots(figsize=(15, 8))
        for label, response in (("main", mains), ("contaminants", contaminants)):
            for slit, velocity in ((16, 0), (15, 0), (16, 500)):
                _peak(
                    response.detector_response.sel(slit=slit, doppler_velocity=velocity, method="nearest"),
                    {"detector_x_pixel", "line"},
                ).sum("line").plot(ax=ax, label=f"{label}, slit {slit}, {velocity} km/s")
        ax.set_title(f"Main and contaminant response of band {band} at the peak over logT")
        ax.legend()
        fig.savefig(plot_dir / f"{stem}_main_contaminants.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(15, 8))
        for slit in (15, 16):
            mains.detector_wavelength.sel(slit=slit).plot(ax=ax, label=f"main, slit {slit}")
            contaminants.detector_wavelength.sel(slit=slit).plot(ax=ax, label=f"contaminants, slit {slit}", ls="--")
        ax.legend()
        fig.savefig(plot_dir / f"{stem}_detector_wavelength.png")
        plt.close(fig)


def logT_summary(groups, plot_dir, run_label):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    bands = sorted(
        {band for (band, _abundance, _plasma), kinds in groups.items() if kinds.get("main_cube") or kinds.get("ci")}
    )
    if not bands:
        return
    fig, axes = plt.subplots(len(bands), 1, figsize=(15, 6 * len(bands)), squeeze=False)
    for ax, band in zip(axes.flatten(), bands, strict=True):
        for (group_band, abundance, plasma), kinds in groups.items():
            if group_band != band:
                continue
            for path in kinds.get("ci", []):
                _peak(_load(path).detector_response.sel(doppler_velocity=0, method="nearest"), {"logT"}).plot(
                    ax=ax, label=f"{abundance} {plasma}"
                )
            for kind, label in (("main_cube", ""), ("contaminant_cube", "no main ")):
                for path in kinds.get(kind, []):
                    cube = read_spectral_response(path).load()
                    width = cube.wavelength_bounds.isel(bound=1) - cube.wavelength_bounds.isel(bound=0)
                    _peak(
                        (cube.spectral_response * width)
                        .sel(doppler_velocity=0, method="nearest")
                        .sum(["wavelength_bin", "line"]),
                        {"logT"},
                    ).plot(ax=ax, label=f"{label}{abundance} {plasma}")
        ax.set_title(str(band))
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"logT_summary_{run_label}.png")
    plt.close(fig)


def effective_area_overview(area_files, plot_dir, run_label):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(15, 8))
    for path in area_files:
        effective_area = read_effective_area(path)
        select_effective_area_curve(effective_area).plot(ax=ax, label=int(effective_area.band[0]))
    ax.legend()
    ax.set_title("Effective area")
    ax.set_yscale("log")
    fig.savefig(plot_dir / f"effective_area_{run_label}.png")
    plt.close(fig)


def dem_weight(logT, dem_name="active_region"):
    """
    ``DEM(T) dT`` on the ``logT`` grid from a CHIANTI DEM, or `None` without ChiantiPy
    and XUVTOP.
    """
    if "XUVTOP" not in os.environ:
        return None
    try:
        import ChiantiPy.tools.io as chio  # noqa: PLC0415
    except ImportError:
        return None
    dem = chio.demRead(dem_name)
    dem_on_grid = np.interp(logT, np.log10(np.asarray(dem["temperature"])), np.asarray(dem["dem"]), left=0.0, right=0.0)
    return dem_on_grid * np.gradient(logT) * np.log(10) * 10 ** np.asarray(logT)


def _annotate(ax, x, y, names, n=20):
    for xi, yi, name in list(zip(np.ravel(x), np.ravel(y), np.ravel(names), strict=False))[:n]:
        ax.annotate(str(name), (float(xi), float(yi)), fontsize=6)


def _folded_scatter(ax, wavelengths, brightness, names, separation):
    """
    Scatter the lines folded into the slit of the brightest one, coloured by the slits
    away.
    """
    offset = wavelengths - wavelengths[0]
    folds = np.round(offset / separation)
    scatter = ax.scatter(offset - folds * separation, brightness, c=np.abs(folds), cmap="viridis", s=24)
    _annotate(ax, offset - folds * separation, brightness, names)
    ax.figure.colorbar(scatter, ax=ax, label="slits away")
    ax.set_yscale("log")
    ax.set_xlim(-separation / 2, separation / 2)
    ax.set_xlabel("offset within the slit of the brightest line [A]")


def bright_line_scatters(run_dir, area_files, groups, plot_dir):
    """
    The brightest lines of every band from the cached line lists: by logT, by
    wavelength, and folded into one slit.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    bands_present = {band for (band, _abundance, _plasma) in groups}
    area_curves = {}
    for path in area_files:
        effective_area = read_effective_area(path)
        area_curves[int(effective_area.band[0])] = select_effective_area_curve(effective_area)
    separation = DEFAULTS_MUSE.spectral_slit_separation_SG.to_value(u.AA)
    half_width = DEFAULTS_MUSE.waveband_half_width_SG.to_value(u.AA)
    for list_file in sorted((run_dir / "cache").glob("line_list_*.nc")):
        line_list = xr.load_dataset(list_file)
        for band, area_curve in area_curves.items():
            if str(band) not in bands_present or band not in DEFAULTS_MUSE.channel_spectral_order_SG.channel.values:
                continue
            half = half_width / int(DEFAULTS_MUSE.channel_spectral_order_SG.sel(channel=band))
            # one plasma point, so every per-line field is a scalar
            plasma = next((dim for dim in line_list.gofnt.dims if dim not in ("logT", "trans_index")), None)
            subset = line_list.isel({plasma: line_list.sizes[plasma] // 2}) if plasma else line_list
            subset = subset.isel(
                trans_index=((subset.wavelength > band - half) & (subset.wavelength < band + half)).values
            )
            if subset.sizes["trans_index"] == 0:
                continue
            ranked = rank_line_list(subset, area_curve)
            bright = ranked.isel(trans_index=(ranked.brightness > ranked.brightness.max() / 1e3).values)
            stem = f"{band}_{list_file.stem}"
            names = bright.full_name.values
            brightness_label = "peak gofnt x effective area (brightness proxy)"
            logT_peak = bright.logT.values[np.asarray(bright.gofnt.argmax("logT"))]
            for x, x_label, suffix in (
                (logT_peak, "logT of the peak", "logT"),
                (bright.wavelength.values, "wavelength [A]", "wavelength"),
            ):
                fig, ax = plt.subplots()
                ax.scatter(x, bright.brightness.values)
                _annotate(ax, x, bright.brightness.values, names)
                ax.set_yscale("log")
                ax.set_xlabel(x_label)
                ax.set_ylabel(brightness_label)
                ax.set_title(str(band))
                fig.savefig(plot_dir / f"{stem}_lines_{suffix}.png")
                plt.close(fig)
            fig, ax = plt.subplots()
            _folded_scatter(ax, bright.wavelength.values, bright.brightness.values, names, separation)
            ax.set_ylabel(brightness_label)
            ax.set_title(f"{band}: bright lines folded into one slit")
            fig.savefig(plot_dir / f"{stem}_lines_folded.png")
            plt.close(fig)
            weight = dem_weight(subset.logT.values)
            if weight is None:
                continue
            radiance = (subset.gofnt * xr.DataArray(weight, dims="logT")).sum("logT") * area_curve.interp(
                wavelength=subset.wavelength
            ).fillna(0).values
            top = np.argsort(-radiance.values)[:20]
            fig, ax = plt.subplots()
            _folded_scatter(
                ax, subset.wavelength.values[top], radiance.values[top], subset.full_name.values[top], separation
            )
            ax.set_ylabel("active-region DEM radiance x effective area [erg / (s sr)]")
            ax.set_title(f"{band}: brightest lines weighted by the active-region DEM")
            fig.savefig(plot_dir / f"{stem}_lines_folded_dem.png")
            plt.close(fig)


def main(argv=None):
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    change_logging_level("WARNING")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--effective-area-dir", type=pathlib.Path, default=None)
    parser.add_argument("--sg-band", type=int, nargs="*", default=None)
    parser.add_argument("--ci-band", type=int, nargs="*", default=None)
    parser.add_argument("--skip-scatters", action="store_true", help="Skip the line-list scatter plots.")
    args = parser.parse_args(argv)
    plot_dir = args.run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = discover(args.run_dir)
    if not groups:
        print(f"no respfunc products under {args.run_dir}/SG or /CI")
        return
    velocity_plots(groups, plot_dir)
    main_contaminant_and_plate_scale(groups, plot_dir)
    logT_summary(groups, plot_dir, args.run_dir.name)
    if args.effective_area_dir is not None:
        sg_bands = args.sg_band if args.sg_band is not None else DEFAULT_SG_BANDS
        ci_bands = args.ci_band if args.ci_band is not None else DEFAULT_CI_BANDS
        area_files = [effective_area_file(args.effective_area_dir, band) for band in (*sg_bands, *ci_bands)]
        if area_files:
            effective_area_overview(area_files, plot_dir, args.run_dir.name)
            if not args.skip_scatters:
                bright_line_scatters(args.run_dir, area_files, groups, plot_dir)
    print(f"plots written to {plot_dir}")


if __name__ == "__main__":
    main()
