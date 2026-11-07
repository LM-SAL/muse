"""
Shared configuration of the baseline drivers: the effective-area files, the grids and
the task table, so the preparation step, the array tasks and a local run all agree.

The effective-area files are not part of the package: pass ``--effective-area-dir``
holding ``effective_area_band_<band>_*.zarr`` (or ``.nc``) files, one per band.
"""

import os
import pathlib
import argparse

import astropy.units as u

from muse.instrument import effective_area_wavelength_range, read_effective_area
from muse.instrument.pipeline import (
    DEFAULT_ABUNDANCES,
    DEFAULT_PRESSURE,
    DEFAULT_TEMPERATURE,
    DEFAULT_THREADS,
)
from muse.variables import DEFAULTS_MUSE

DEFAULT_SG_BANDS = [108, 171, 284]
DEFAULT_CI_BANDS = [195]
DEFAULT_DENSITY_GRID = [1e8, 1e9, 1e10, 1e11] * u.cm**-3


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
        help=f"CHIANTI abundance (repeatable), by default {list(DEFAULT_ABUNDANCES)}.",
    )
    parser.add_argument(
        "--density", action="store_true", help="Density-dependent run on the density grid instead of the pressures."
    )
    parser.add_argument("--minimum-abundance", type=float, default=DEFAULTS_MUSE.minimum_abundance)
    parser.add_argument("--line-list-backend", choices=["chianti", "fiasco"], default="chianti")
    parser.add_argument("--workers", type=int, default=1, help="Processes for the fiasco backends; -1 uses every core.")
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", DEFAULT_THREADS)),
        help="Threads the BLAS and numexpr pools may use, by default the job's SLURM_CPUS_PER_TASK or 4.",
    )


def effective_area_file(directory: pathlib.Path, band: int) -> pathlib.Path:
    matches = sorted(directory.glob(f"effective_area_band_{band}_*.zarr")) + sorted(
        directory.glob(f"effective_area_band_{band}_*.nc")
    )
    if len(matches) != 1:
        msg = f"expected one effective_area_band_{band}_* file in {directory}, found {len(matches)}"
        raise FileNotFoundError(msg)
    return matches[0]


def effective_area_files(args) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    sg_bands = args.sg_band if args.sg_band is not None else DEFAULT_SG_BANDS
    ci_bands = args.ci_band if args.ci_band is not None else DEFAULT_CI_BANDS
    return [effective_area_file(args.effective_area_dir, band) for band in sg_bands], [
        effective_area_file(args.effective_area_dir, band) for band in ci_bands
    ]


def plasma_grid(args) -> dict:
    """
    The ``pressure`` or ``density`` keyword of the pipeline for this run.
    """
    import xarray as xr  # noqa: PLC0415

    return (
        {"density": xr.DataArray(DEFAULT_DENSITY_GRID, dims="density")}
        if args.density
        else {"pressure": DEFAULT_PRESSURE}
    )


def wavelength_range(files) -> u.Quantity:
    """
    The range every effective-area file spans, as the pipeline sizes the shared line
    list.
    """
    return effective_area_wavelength_range([read_effective_area(path) for path in files])


def enumerate_tasks(*, sg_files, ci_files, abundances, number_of_pressures: int) -> list[dict]:
    """
    The independent units of work, in the pipeline's iteration order: one effective-area
    file at one pressure index for one abundance, so a job array maps its index to a
    task.
    """
    return [
        {
            "detector": detector,
            "abundance": abundance,
            "effective_area": pathlib.Path(path),
            "pressure_index": pressure_index,
        }
        for abundance in abundances
        for detector, paths in (("SG", sg_files), ("CI", ci_files))
        for path in paths
        for pressure_index in range(number_of_pressures)
    ]


def tasks(args) -> list[dict]:
    sg_files, ci_files = effective_area_files(args)
    return enumerate_tasks(
        sg_files=sg_files,
        ci_files=ci_files,
        abundances=args.abundance or DEFAULT_ABUNDANCES,
        number_of_pressures=1 if args.density else DEFAULT_PRESSURE.size,
    )


def task_id_from_environment() -> int | None:
    """
    The job-array index from ``SLURM_ARRAY_TASK_ID`` or ``PBS_ARRAY_INDEX``, or `None`
    outside an array job.
    """
    for name in ("SLURM_ARRAY_TASK_ID", "PBS_ARRAY_INDEX"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return None


__all__ = [
    "DEFAULT_CI_BANDS",
    "DEFAULT_SG_BANDS",
    "DEFAULT_TEMPERATURE",
    "add_common_arguments",
    "effective_area_file",
    "effective_area_files",
    "enumerate_tasks",
    "plasma_grid",
    "task_id_from_environment",
    "tasks",
    "wavelength_range",
]
