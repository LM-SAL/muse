"""
Generate the whole baseline locally, in one serial process.

The local counterpart of the cluster drivers: the per-abundance line-list caches, every task
of the table through the same run_task the array driver runs, then the plots, so the run
directory is identical to a cluster run (cache/ SG/ CI/ manifest/ logs/ plots/)::

    python generate_baseline.py --run-dir RUN --effective-area-dir AREAS --author ABC --effective-area-note coating
"""

import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_config import (
    DEFAULT_TEMPERATURE,
    add_common_arguments,
    effective_area_files,
    plasma_grid,
    tasks,
    wavelength_range,
)
from run_response_pipeline import add_task_arguments, run_task

from muse.instrument import load_or_create_line_list
from muse.instrument.pipeline import DEFAULT_ABUNDANCES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    add_task_arguments(parser)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    sg_files, ci_files = effective_area_files(args)
    if args.line_list_file is None:
        for abundance in args.abundance or DEFAULT_ABUNDANCES:
            load_or_create_line_list(
                args.run_dir / "cache",
                abundance=abundance,
                wavelength_range=wavelength_range(sg_files + ci_files),
                temperature=DEFAULT_TEMPERATURE,
                minimum_abundance=args.minimum_abundance,
                backend=args.line_list_backend,
                workers=args.workers,
                **plasma_grid(args),
            )
    for task_id, task in enumerate(tasks(args)):
        run_task(task, task_id, args)
    if not args.skip_plots:
        from plot_response_functions import main as make_plots  # noqa: PLC0415

        bands = [] if args.sg_band is None else ["--sg-band", *map(str, args.sg_band)]
        bands += [] if args.ci_band is None else ["--ci-band", *map(str, args.ci_band)]
        make_plots(["--run-dir", str(args.run_dir), "--effective-area-dir", str(args.effective_area_dir), *bands])


if __name__ == "__main__":
    main()
