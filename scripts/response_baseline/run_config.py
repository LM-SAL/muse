"""
Shared configuration of the baseline drivers: the effective-area files, the grids and
the task table, so the preparation step, the array tasks and a local run all agree.

The effective-area files are not part of the package: pass ``--effective-area-dir``
holding ``effective_area_band_<band>_*.zarr`` (or ``.nc``) files, one per band.
"""

import pathlib
import argparse

import numpy as np
import xarray as xr

import astropy.units as u

from muse.instrument import enumerate_tasks

DEFAULT_ABUNDANCES = ["sun_coronal_2021_chianti", "sun_photospheric_2021_asplund"]
DEFAULT_SG_BANDS = [108, 171, 284]
DEFAULT_CI_BANDS = [195]
PRESSURE = xr.DataArray([3e14, 3e15, 3e16] * u.K / u.cm**3, dims="pressure")
DENSITY = xr.DataArray([1e8, 1e9, 1e10, 1e11] * u.cm**-3, dims="density")
TEMPERATURE = xr.DataArray(10 ** np.arange(4.5, 8.0, 0.1) * u.K, dims="logT")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir", type=pathlib.Path, required=True, help="Run root: products, cache/, logs/, manifest/, plots/."
    )
    parser.add_argument(
        "--effective-area-dir",
        type=pathlib.Path,
        required=True,
        help="Directory of effective_area_band_<band>_*.zarr files.",
    )
    parser.add_argument(
        "--sg-band",
        type=int,
        nargs="*",
        default=None,
        help=f"SG bands, by default {DEFAULT_SG_BANDS}; give none for no SG band.",
    )
    parser.add_argument(
        "--ci-band",
        type=int,
        nargs="*",
        default=None,
        help=f"CI bands, by default {DEFAULT_CI_BANDS}; give none for no CI band.",
    )
    parser.add_argument(
        "--abundance",
        action="append",
        default=None,
        help=f"CHIANTI abundance (repeatable), by default {DEFAULT_ABUNDANCES}.",
    )
    parser.add_argument(
        "--density", action="store_true", help="Density-dependent run on the density grid instead of the pressures."
    )
    parser.add_argument("--minimum-abundance", type=float, default=1e-21)
    parser.add_argument("--line-list-backend", choices=["chianti", "fiasco"], default="chianti")
    parser.add_argument("--workers", type=int, default=1, help="Processes for the fiasco backends; -1 uses every core.")


def effective_area_file(directory: pathlib.Path, band: int) -> pathlib.Path:
    matches = sorted(directory.glob(f"effective_area_band_{band}_*.zarr")) + sorted(
        directory.glob(f"effective_area_band_{band}_*.nc")
    )
    if len(matches) != 1:
        msg = f"expected one effective_area_band_{band}_* file in {directory}, found {len(matches)}"
        raise FileNotFoundError(msg)
    return matches[0]


def effective_area_files(args) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    sg = [
        effective_area_file(args.effective_area_dir, band)
        for band in (args.sg_band if args.sg_band is not None else DEFAULT_SG_BANDS)
    ]
    ci = [
        effective_area_file(args.effective_area_dir, band)
        for band in (args.ci_band if args.ci_band is not None else DEFAULT_CI_BANDS)
    ]
    return sg, ci


def plasma_grid(args) -> dict:
    return {"density": DENSITY} if args.density else {"pressure": PRESSURE}


def tasks(args) -> list[dict]:
    sg, ci = effective_area_files(args)
    return enumerate_tasks(
        sg_effective_areas=sg,
        ci_effective_areas=ci,
        abundances=args.abundance or DEFAULT_ABUNDANCES,
        n_pressure=1 if args.density else PRESSURE.size,
    )


def wavelength_range(files) -> u.Quantity:
    """
    The range every effective-area file spans, as the pipeline sizes the shared line
    list.
    """
    from muse.instrument import read_effective_area  # noqa: PLC0415

    areas = [read_effective_area(path) for path in files]
    return [min(float(a.wavelength.min()) for a in areas), max(float(a.wavelength.max()) for a in areas)] * u.AA
