"""
Generate the SG continuum responses of the baseline: one product per band and abundance::

    python generate_continuum_baseline.py --run-dir RUN --effective-area-dir AREAS --author ABC --effective-area-note coating
"""

import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_config import DEFAULT_TEMPERATURE, add_common_arguments, effective_area_files, plasma_grid

from muse.instrument import generate_continuum_responses
from muse.instrument.pipeline import DEFAULT_ABUNDANCES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--author", required=True)
    parser.add_argument("--effective-area-note", required=True)
    parser.add_argument("--backend", choices=["chianti", "fiasco"], default="fiasco")
    args = parser.parse_args(argv)
    sg_files, _ = effective_area_files(args)
    for abundance in args.abundance or DEFAULT_ABUNDANCES:
        written = generate_continuum_responses(
            output_directory=args.run_dir / "continuum",
            cache_directory=args.run_dir / "cache",
            author=args.author,
            effective_area_note=args.effective_area_note,
            sg_effective_areas=sg_files,
            abundance=abundance,
            backend=args.backend,
            temperature=DEFAULT_TEMPERATURE,
            minimum_abundance=args.minimum_abundance,
            workers=args.workers,
            **plasma_grid(args),
        )
        print(f"{abundance}: {len(written)} continuum products")


if __name__ == "__main__":
    main()
