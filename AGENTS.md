# AGENTS.md

Guidance for AI agents in repo. Keep changes minimal, typed-by-units, tested. When in doubt, match surrounding code.

## Project

`muse` — Python library to analyze and reduce data from MUSE (Multi-slit Solar Explorer) mission. Pre-alpha. Core domain objects = `xarray` datasets carrying `astropy.units` quantities. Repo: `LM-SAL/muse`.

## Environment & commands

Dev env = **micromamba** env named `muse` (under your `$MAMBA_ROOT_PREFIX/envs/`). Run commands through it — do **not** assume bare `python`:

```bash
micromamba run -n muse python -m pytest muse/tests/test_variables.py -q   # one test file
micromamba run -n muse python -m pytest muse -q                           # package tests
micromamba activate muse                                                  # or activate the shell
tox -e py314                                                              # full env (matrix: py312/313/314)
tox -e build_docs                                                         # build Sphinx docs
```

- `.venv/` in repo is **not** dev env — transient venv created by `tox-uv` when running `tox`. Don't rely on it for ad-hoc commands.
- Python ≥ 3.12 (CI runs 3.12, 3.13, 3.14). `tox` deps resolve via `uv.lock`.
- Lint/format = **ruff** (config in `.ruff.toml`), run via pre-commit:
  ```bash
  pre-commit run --all-files          # docformatter + ruff + ruff-format + isort + typos + checks
  ```
  Ruff may not be on `PATH` outside pre-commit; prefer hook. docformatter owns docstring wrapping (multi-line summary style) — don't hand-wrap docstrings, let the hook reflow them.

## Repo layout

Current layout; new code should follow it.

```
muse/                  package
  variables.py         instrument defaults as InstrumentDefaults instances (DEFAULTS_MUSE, DEFAULTS_AIA)
  variables_schema.py  attrs schema + immutability/units machinery (InstrumentDefaults, FrozenDict)
  log.py               loguru logger; import-time torch-free
  utils/               cross-cutting helpers (documentation.py = format_docstring, ...)
  <subpkg>/tests/      tests live next to the code they cover
  tests/helpers.py     fake data builders (fake_vdem/fake_response style)
  conftest.py          shared pytest fixtures
docs/                  Sphinx (numpydoc); also doctested by pytest
changelog/             towncrier news fragments (required on PRs)
examples/              sphinx-gallery scripts (looser lint rules)
```

## Code conventions

- **`__all__`** in every public implementation module; keep accurate when adding/removing public names.
- **Docstrings: numpy style** (`pydocstyle convention = numpy`) with `Parameters`/`Returns`. Use `@format_docstring("DEFAULTS_MUSE", param="field_name")` decorator (`muse.utils.documentation`) to inject default values into docstrings — maps `{placeholder}` to field name on named defaults object.
- **Errors:** assign message to `msg` variable, then raise (ruff `EM`):
  ```python
  msg = f"Unsupported restype {restype!r}"
  raise ValueError(msg)
  ```
- **Logging:** `from muse.log import logger` (loguru). No `print` in library code. `change_logging_level` is an app-level convenience that replaces all Loguru sinks — for user scripts/notebooks only; never call it from library code or at import.
- **Immutable config:** instrument parameters live on frozen attrs class `InstrumentDefaults`; create variants with `attrs.evolve`, never by mutation.
- **mixedCase allowed for science names** (`logT`, `SG_resp`, `dx_pixel_CI`, `vdop`) — ruff `N8xx` intentionally relaxed. Follow existing names; don't "fix" to snake_case.
- **Line length 120**, double quotes, ruff-format owns formatting (don't hand-format).
- Keep module import cheap: **lazy-import heavy optional deps** (`torch`, `jax`, `ChiantiPy`) inside functions where module imported at package init (see `log.py:log_gpu_status`). Leaf submodule not imported by `muse/__init__` may import them at top level. On `ImportError`, point at the extra (e.g. `pip install muse[chianti]`).

## Units & data model

- Quantities use `astropy.units`; converters normalize to canonical unit on construction (e.g. arcsec, Angstrom, km/s). Don't strip units mid-pipeline.
- Primary containers = `xarray.Dataset`/`DataArray`. Spatial/spectral axes have conventional names: `x`, `y`, `slit`, `step`, `logT`, `vdop`, `channel`, `line`, `SG_xpixel`, `SG_wvl`, `trans_index`, `log_density`.
- Functions returning dataset record provenance via `add_history(ds, locals(), func)`; preserve when editing pipeline functions — recorded call string asserted in some tests. `add_history` alone owns `HISTORY`/`date created`/`date modified`/`version`; multi-input results inherit lineage via `add_history(..., sources=(a, b))`. `update_attrs` copies only non-provenance attrs.
- **Finalizer exception to immutability:** `add_history` and `update_attrs` mutate in place and return `None`; call them only on newly constructed outputs the function owns (name the result first), never on caller-owned inputs.
- **Treat datasets as immutable; never mutate inputs in place.** Build new objects with `assign`/`assign_coords` — these share underlying arrays (cheap, no large copy). Don't `ds.copy(deep=True)` whole dataset just to add/tweak a coord or attr.
- **Deep-copy only the one array you overwrite** (`ds.assign(SG_resp=ds.SG_resp.copy(deep=True))`), never entire dataset.
- **Attrs shared on shallow copies.** Set attrs on freshly computed `DataArray` before `assign_coords`, or use `.assign_attrs(...)`; don't mutate `ds.var.attrs[...]` on shared object — leaks back to caller.
- **Never rely on host xarray options.** `import muse` must not call `xr.set_options`, so code runs under whatever options the host set. Consequences: (1) don't assume attrs survive reductions/arithmetic — set attrs explicitly on outputs (or pass `keep_attrs=True` to that one call); (2) pass combine kwargs (`data_vars`, `coords`, `compat`, `join`) explicitly to every `xr.concat`/`xr.merge` — ambiguous defaults raise `FutureWarning` = CI failure.
- **For large data, stay lazy** (dask-backed chunked zarr): operations build lazy graph, memory stays bounded, originals untouched until `.compute()`/write.
- **Validate units at trust boundaries** with `require_unit` (`muse.utils.utils`); compose domain checks on top (see `synthesis._validate_inputs`).

## Testing

- `pytest` with `pytest.ini`. **Warnings are errors** (`filterwarnings = error`) — new `DeprecationWarning` fails CI; fix it or add justified ignore with comment/issue link.
- Doctests run on docstrings and `.rst` (`doctest_plus`, `--doctest-rst`). Keep doctest examples runnable.
- Tests live in `muse/<subpkg>/tests/test_*.py`. Build inputs from `muse/tests/helpers.py` fakes + `conftest.py` fixtures; avoid real data files.
- `remote_data`/`online` markers gate network tests (`remote_data_strict = true`); default runs offline. `mpl_image_compare` applied via `figure_test` decorator, not by hand.
- In `test_*.py`, `assert` and `N806` allowed (relaxed in `.ruff.toml`).

## Changelog & PRs

- Every PR needs towncrier fragment in `changelog/` named `<PR#>.<type>.rst`, type ∈ `breaking | deprecation | removal | feature | bugfix | doc | trivial`. Gilesbot enforces; see `changelog/README.rst`.
- Commit/PR work on branch, not `main`. Keep diffs small and focused.

## Gotchas

- CHIANTI (`muse/instrument/linelist.py`): needs `[chianti]` extra + `XUVTOP` env var pointing at a local CHIANTI database. Live tests are `remote_data`-gated and run in a dedicated CI job (`chianti` in `ci.yaml`) that downloads/caches the database; validation tests run offline without it.
- `.history/` = editor backup noise — ignore; excluded from pytest.
- Legacy `.flake8` / `.isort.cfg` exist but ruff is source of truth for lint (isort rules `I` delegated to isort pre-commit hook).
- `muse/_version.py` generated by setuptools-scm — never edit by hand.
