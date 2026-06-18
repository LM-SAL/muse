# AGENTS.md

Guidance for AI agents working in this repo. Keep changes minimal, typed-by-units,
and tested. When in doubt, match the surrounding code.

## Project

`muse` — Python library to analyze and reduce data from the MUSE (Multi-slit
Solar Explorer) mission. Pre-alpha. Core domain objects are `xarray` datasets
carrying `astropy.units` quantities. Repo: `LM-SAL/muse`.

## Environment & commands

The dev environment is a **micromamba** env named `muse` (under your
`$MAMBA_ROOT_PREFIX/envs/`). Run commands through it — do **not** assume a bare
`python`:

```bash
micromamba run -n muse python -m pytest muse/tests/test_variables.py -q   # one test file
micromamba run -n muse python -m pytest muse -q                           # package tests
micromamba activate muse                                                  # or activate the shell
tox -e py314                                                              # full env (matrix: py312/313/314)
tox -e build_docs                                                         # build Sphinx docs
```

- `.venv/` in the repo is **not** the dev env — it is a transient venv created by
  `tox-uv` when running `tox`. Don't rely on it for ad-hoc commands.
- Python ≥ 3.12 (CI runs 3.12, 3.13, 3.14). `tox` deps resolve via `uv.lock`.
- Lint/format is **ruff** (config in `.ruff.toml`), run via pre-commit:
  ```bash
  pre-commit run --all-files          # ruff + ruff-format + isort + typos + checks
  ```
  Ruff may not be on `PATH` outside pre-commit; prefer the hook.

## Repo layout

This is the **intended/target** layout. Some pieces (marked *planned*) do not
exist yet — they arrive with upcoming work — but new code should follow it.

```
muse/                  package
  variables.py         instrument defaults as InstrumentDefaults instances (DEFAULTS_MUSE, DEFAULTS_AIA)
  variables_schema.py  attrs schema + immutability/units machinery (InstrumentDefaults, FrozenDict)
  log.py               loguru logger; import-time torch-free
  utils/               cross-cutting helpers (documentation.py = format_docstring, ...)
  <subpkg>/tests/      tests live next to the code they cover (planned for new subpackages)
  tests/helpers.py     fake data builders (fake_vdem/fake_response style)
  conftest.py          shared pytest fixtures (planned)
docs/                  Sphinx (numpydoc); also doctested by pytest
changelog/             towncrier news fragments (required on PRs)
examples/              sphinx-gallery scripts (looser lint rules)
```

## Code conventions

- **`__all__`** in every module; keep it accurate when adding/removing public names.
- **Docstrings: numpy style** (`pydocstyle convention = numpy`) with
  `Parameters`/`Returns`. Use the `@format_docstring("DEFAULTS_MUSE", param="field_name")`
  decorator (`muse.utils.documentation`) to inject default values into docstrings —
  it maps a `{placeholder}` to a field name on a named defaults object.
- **Errors:** assign the message to a `msg` variable, then raise (ruff `EM`):
  ```python
  msg = f"Unsupported restype {restype!r}"
  raise ValueError(msg)
  ```
- **Logging:** `from muse.log import logger` (loguru). No `print` in library code.
- **Immutable config:** instrument parameters live on the frozen attrs class
  `InstrumentDefaults`; create variants with `attrs.evolve`, never by mutation.
- **mixedCase is allowed for science names** (`logT`, `SG_resp`, `dx_pixel_CI`,
  `vdop`) — ruff `N8xx` is intentionally relaxed. Follow existing names; don't
  "fix" them to snake_case.
- **Line length 120**, double quotes, ruff-format owns formatting (don't hand-format).
- Keep module import cheap: **lazy-import `torch`** inside functions where a module
  is imported at package init (see `log.py:log_gpu_status`). A leaf submodule that
  is not imported by `muse/__init__` may import torch at top level.

## Units & data model

- Quantities use `astropy.units`; converters normalize to a canonical unit on
  construction (e.g. arcsec, Angstrom, km/s). Don't strip units mid-pipeline.
- Primary containers are `xarray.Dataset`/`DataArray`. Spatial/spectral axes have
  conventional names: `x`, `y`, `slit`, `step`, `logT`, `vdop`, `channel`, `line`,
  `SG_xpixel`, `SG_wvl`.
- Functions that return a dataset record provenance via `add_history(ds, locals(),
  func)`; preserve this when editing pipeline functions — the recorded call string
  is asserted in some tests.

## Testing

- `pytest` with `pytest.ini`. **Warnings are errors** (`filterwarnings = error`) —
  a new `DeprecationWarning` will fail CI; fix it or add a justified ignore with a
  comment/issue link.
- Doctests run on docstrings and `.rst` (`doctest_plus`, `--doctest-rst`). Keep
  doctest examples runnable.
- Tests live in `muse/<subpkg>/tests/test_*.py`. Build inputs from
  `muse/tests/helpers.py` fakes + `conftest.py` fixtures; avoid real data files.
- `remote_data`/`online` markers gate network tests (`remote_data_strict = true`);
  default runs are offline. `mpl_image_compare` is applied via the `figure_test`
  decorator, not by hand.
- In `test_*.py`, `assert` and `N806` are allowed (relaxed in `.ruff.toml`).

## Changelog & PRs

- Every PR needs a towncrier fragment in `changelog/` named `<PR#>.<type>.rst`,
  type ∈ `breaking | deprecation | removal | feature | bugfix | doc | trivial`.
  Gilesbot enforces this; see `changelog/README.rst`.
- Commit/PR work on a branch, not `main`. Keep diffs small and focused.

## Gotchas

- `.history/` is editor backup noise — ignore it; it is excluded from pytest.
- Legacy `.flake8` / `.isort.cfg` exist but ruff is the source of truth for lint
  (isort rules `I` are delegated to the isort pre-commit hook).
- `muse/_version.py` is generated by setuptools-scm — never edit by hand.
