"""
Run ONE task of the baseline response generation: the cluster job-array driver.

A task is one effective-area file at one pressure index for one abundance, from the table
``run_config.enumerate_tasks`` builds; the index comes from --task-id or the job array
(SLURM_ARRAY_TASK_ID / PBS_ARRAY_INDEX). For a serial run of the whole baseline use
generate_baseline.py, which loops these same tasks through the same run_task::

    python run_response_pipeline.py --run-dir RUN --effective-area-dir AREAS --author ABC --effective-area-note coating --list-tasks
    python run_response_pipeline.py --run-dir RUN --effective-area-dir AREAS --author ABC --effective-area-note coating --task-id 3

The line-list caches must exist in RUN/cache (prepare_line_lists.py). Products land in
RUN/SG and RUN/CI, one log per task in RUN/logs, provenance in RUN/manifest/task_<id>.json.
"""

import sys
import json
import time
import pathlib
import argparse
import datetime as dt
import platform
import subprocess

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from run_config import DEFAULT_TEMPERATURE, add_common_arguments, plasma_grid, task_id_from_environment, tasks

import muse
from muse.instrument import generate_responses
from muse.instrument.pipeline import DEFAULT_PRESSURE
from muse.log import logger

MAX_ATTEMPTS = 3


def add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--author", required=True)
    parser.add_argument("--effective-area-note", required=True)
    parser.add_argument("--maximum-contaminants", type=int, default=10000)
    parser.add_argument("--no-continuum", action="store_true", help="Leave the continuum out of the CI products.")
    parser.add_argument("--continuum-backend", choices=["chianti", "fiasco"], default="fiasco")
    parser.add_argument(
        "--line-list-file", type=pathlib.Path, default=None, help="A line list to use instead of the cache."
    )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd=pathlib.Path(__file__).parent
    )
    return result.stdout.strip()


def run_task(task: dict, task_id: int, args) -> None:
    run_dir = args.run_dir
    output_directory = run_dir / task["detector"]
    for directory in (output_directory, run_dir / "logs", run_dir / "manifest"):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "manifest" / f"task_{task_id}.json"
    # A preempted task is requeued whole; the attempt count lives in its manifest.
    entry = json.loads(manifest.read_text()) if manifest.exists() else {}
    attempts = entry.get("attempt", 0)
    if attempts >= MAX_ATTEMPTS:
        entry.update(status="abandoned")
        manifest.write_text(json.dumps(entry, indent=2))
        logger.error(f"task {task_id} failed {attempts} times; it needs a person")
        sys.exit(1)
    entry = {
        "task_id": task_id,
        "task": {**task, "effective_area": str(task["effective_area"])},
        "attempt": attempts + 1,
        "muse_version": muse.__version__,
        "git_sha": _git_sha(),
        "host": platform.node(),
        "started": dt.datetime.now(dt.UTC).isoformat(),
        "status": "running",
    }
    manifest.write_text(json.dumps(entry, indent=2))
    start = time.perf_counter()
    is_ci = task["detector"] == "CI"
    plasma = plasma_grid(args)
    if "pressure" in plasma:
        plasma = {"pressure": DEFAULT_PRESSURE.isel(pressure=[task["pressure_index"]])}
    log_file = (
        run_dir / "logs" / f"task_{task_id}_{task['detector']}_{task['effective_area'].stem}_{task['abundance']}.log"
    )
    sink = logger.add(log_file, level="INFO")
    try:
        generate_responses(
            output_directory=output_directory,
            cache_directory=run_dir / "cache",
            author=args.author,
            effective_area_note=args.effective_area_note,
            sg_effective_areas=[] if is_ci else [task["effective_area"]],
            ci_effective_areas=[task["effective_area"]] if is_ci else [],
            abundances=[task["abundance"]],
            temperature=DEFAULT_TEMPERATURE,
            maximum_contaminants=args.maximum_contaminants,
            minimum_abundance=args.minimum_abundance,
            line_list_backend=args.line_list_backend,
            line_list_file=args.line_list_file,
            include_continuum=not args.no_continuum,
            continuum_backend=args.continuum_backend,
            workers=args.workers,
            threads=args.threads,
            **plasma,
        )
    except Exception as exc:
        entry.update(status="failed", error=repr(exc), duration_s=round(time.perf_counter() - start, 1))
        manifest.write_text(json.dumps(entry, indent=2))
        raise
    finally:
        logger.remove(sink)
    entry.update(status="completed", duration_s=round(time.perf_counter() - start, 1))
    manifest.write_text(json.dumps(entry, indent=2))
    print(f"task {task_id} completed in {entry['duration_s']} s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    add_task_arguments(parser)
    parser.add_argument("--task-id", type=int, default=None, help="By default the job-array index.")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args(argv)
    table = tasks(args)
    if args.list_tasks:
        for index, task in enumerate(table):
            print(
                f"task {index}: {task['detector']} {task['effective_area'].name} abundance={task['abundance']} pressure_index={task['pressure_index']}"
            )
        print(f"{len(table)} tasks total")
        return
    task_id = args.task_id if args.task_id is not None else task_id_from_environment()
    if task_id is None:
        msg = "no task id: pass --task-id N or run inside a job array; for a serial run of everything use generate_baseline.py"
        raise SystemExit(msg)
    run_task(table[task_id], task_id, args)


if __name__ == "__main__":
    main()
