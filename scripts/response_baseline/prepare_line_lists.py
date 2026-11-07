"""
Compute and cache the per-abundance line lists before a response run, so the array tasks
all load the same caches instead of racing to compute them.

Idempotent: existing caches are kept. With a job array of size len(abundances) each task
computes one abundance; without an array every abundance is computed in turn::

    python prepare_line_lists.py --run-dir RUN --effective-area-dir AREAS
    python prepare_line_lists.py --run-dir RUN --effective-area-dir AREAS --density

The ChiantiPy backend needs XUVTOP; the fiasco backend its HDF5 database.
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
    task_id_from_environment,
    wavelength_range,
)

from muse.instrument import load_or_create_line_list
from muse.instrument.pipeline import DEFAULT_ABUNDANCES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    abundances = args.abundance or list(DEFAULT_ABUNDANCES)
    task_id = task_id_from_environment()
    if task_id is not None:
        abundances = [abundances[task_id]]
    sg_files, ci_files = effective_area_files(args)
    for abundance in abundances:
        load_or_create_line_list(
            args.run_dir / "cache",
            abundance=abundance,
            wavelength_range=wavelength_range(sg_files + ci_files),
            temperature=DEFAULT_TEMPERATURE,
            minimum_abundance=args.minimum_abundance,
            backend=args.line_list_backend,
            workers=args.workers,
            threads=args.threads,
            **plasma_grid(args),
        )
        print(f"line list ready for {abundance}")


if __name__ == "__main__":
    main()
