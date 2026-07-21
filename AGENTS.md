# Agent Guidelines for `humanleague`

This file instructs AI agents acting as developer, reviewer, and QA for this repository.

## Collaboration & Ownership

The maintainer must retain **ownership** of this codebase — meaning they understand
every change well enough to explain, defend, and modify it without the agent. The
agent's speed serves that understanding; it does not replace it. Follow these rules
of engagement:

1. **Plan before code, and wait for approval.** For any non-trivial change, present
   a plan first — approach, files touched, trade-offs — and do not write code until
   the maintainer has understood and signed off. If a decision in the plan can't be
   evaluated yet, stop and explain it.
2. **Small, reviewable diffs — never a big-bang drop.** Break large work into
   increments that can be read in one sitting and reviewed one at a time.
3. **Leave the load-bearing parts to the maintainer when asked.** Offer to hand off
   the core algorithm or tricky module rather than always doing everything; default
   to boilerplate, tests, plumbing, and review.
4. **Explain-it-back gate.** Before proposing a merge, make sure the maintainer can
   explain *why* the change works and what the alternatives were. Offer a
   walk-through; act as tutor, not just producer.
5. **Justify trade-offs, not just conclusions.** State *why* this data structure,
   error type, or approach — and why not the obvious alternative. The reasoning is
   the transferable knowledge.
6. **Prefer idioms the maintainer can learn from**, especially in modern C++
   (templates, RAII, CMake) and the nanobind binding layer. Flag new or unusual
   patterns and point to where to read more, rather than using them silently.
7. **Tests are the readable spec.** Keep them clear enough that reading the tests
   conveys the contract even when the implementation is dense.

## Task & Design Summaries

**Every task/PR must be recorded** as a new entry at the top of [JOURNAL.md](JOURNAL.md).
Each entry records:

- **Why** — the motivation for the change and the problem it solves.
- **What** — a short description of the change at a high level.
- **Design decisions** — the choices made, the alternatives considered, and why each
  was accepted or rejected. Capture any non-obvious trade-offs or constraints here
  rather than only in code comments.
- **Follow-ups** — anything deferred, and known limitations.

Write the entry as part of the change, not after the fact — the journal is the durable
record of intent that keeps the maintainer in control of the codebase's direction.

## Design Reviews

Decisions in this repo are made under the constraints of the moment — including what LLMs and
agent tooling could do at the time. Those capabilities evolve fast, so a decision that was right
six months ago may be a needless constraint today. **Periodically step back and review the overall
design, not just the next diff.**

- **Cadence.** Hold a design review roughly every 10 merged PRs or every few months, whichever
  comes first — or whenever a change feels like it's fighting the architecture. Either the
  maintainer or the agent may call one.
- **Input.** The design-decision entries in [JOURNAL.md](JOURNAL.md) are the agenda: walk the
  recorded decisions and their rejected alternatives and ask whether the reasoning still holds.
- **Questions to ask.** Have the original constraints (library capabilities, data availability,
  model/agent limitations) shifted? Are there recurring follow-ups or workarounds that point at a
  structural problem? Would a rejected alternative now be the better choice? Is complexity earning
  its keep?
- **Output.** Record the review as a JOURNAL.md entry of its own: what was reconsidered, what was
  reaffirmed (and why), and what should change. Reaffirmations matter as much as changes — they
  stop the same ground being re-litigated every review. Concrete changes become planned tasks via
  the normal [workflow](#workflow); never fold a redesign into an unrelated PR.

## Project Overview

`humanleague` is a Python **and** R package for microsynthesising populations from marginal
and (optionally) seed data, implemented in C++ for performance. It provides:

- **IPF** (Iterative Proportional Fitting)
- **QIS** (Quasirandom Integer Sampling, no seed population)
- **QISI** (QIS of a dynamic IPF solution — combines the two)
- a Sobol sequence generator, and integerisation utilities

The core algorithms live in [src/](src/) as portable C++ shared by both bindings:

| File | Role |
|------|------|
| [src/IPF.h](src/IPF.h) | Iterative Proportional Fitting |
| [src/QIS.cpp](src/QIS.cpp) / [src/QIS.h](src/QIS.h) | Quasirandom Integer Sampling |
| [src/QISI.cpp](src/QISI.cpp) / [src/QISI.h](src/QISI.h) | QIS of IPF |
| [src/Integerise.cpp](src/Integerise.cpp) | Marginal / multidimensional integerisation |
| [src/Sobol.cpp](src/Sobol.cpp), [src/SobolImpl.cpp](src/SobolImpl.cpp) | Sobol sequence generator |
| [src/Index.cpp](src/Index.cpp), [src/NDArray.h](src/NDArray.h) | N-dimensional array/index helpers |
| [src/UnitTester.cpp](src/UnitTester.cpp), `src/Test*.cpp` | C++-level unit tests |

Two independent bindings sit on top of that shared core:

- **Python** (actively developed) — [src/module.cpp](src/module.cpp) binds the core via
  [nanobind](https://nanobind.readthedocs.io/), built with CMake + `scikit-build-core`
  (see [CMakeLists.txt](CMakeLists.txt)). The Python package is [humanleague/](humanleague/)
  (`__init__.py`, `utils.py` for `tabulate_counts`/`tabulate_individuals`, hand-maintained
  `__init__.pyi` stubs).
- **R** (**maintenance-only** — see [README.md](README.md)) — [src/RcppExports.cpp](src/RcppExports.cpp)
  and [R/](R/) bind the same core via Rcpp, built with the standard R package toolchain
  (`DESCRIPTION`, `NAMESPACE`).

Python tests are in [tests/](tests/); R tests are in [tests/testthat/](tests/testthat/).

## Toolchain

Python:

| Tool | Command |
|------|---------|
| Package manager | `uv` |
| Build backend | `scikit-build-core` + `nanobind` (CMake under the hood) |
| Linter / formatter | `ruff` (`uv run ruff check`, `uv run ruff format`) |
| Type checker | `ty` (`uv run ty check`) |
| Tests | `uv run pytest` |
| Install dev deps | `uv sync --dev` |

**A C++20 compiler and CMake are required.** `uv sync` / `uv build` compiles `src/*.cpp` via
CMake + nanobind. For fast local iteration on the C++ side, use the manual dev workflow in
[README.md](README.md): `uv pip install nanobind scikit-build-core[pyproject]`, then
`uv pip install --no-build-isolation -ve .` (optionally with `-Ceditable.rebuild=true` to
rebuild automatically on import).

R (maintenance-only):

| Tool | Command |
|------|---------|
| Build / check | `R CMD build .` then `R CMD check` (see [.github/workflows/r-cmd-check.yml](.github/workflows/r-cmd-check.yml)) |
| Tests | `testthat`, run via [tests/testthat.R](tests/testthat.R) |

Pre-commit hooks ([.pre-commit-config.yaml](.pre-commit-config.yaml)) run `uv-lock`,
`ruff-check --fix`, `ruff-format`, and `ty` automatically on commit.

## Quality Gates

All of the following must pass before a Python-side change is considered complete:

```sh
uv run ruff check   # zero lint errors
uv run ty check      # zero type errors
uv run pytest        # all tests pass, including the embedded C++ unit tests (test_unittest)
```

If a change touches [src/](src/) and could affect the R bindings, also run
`R CMD build . && R CMD check` locally, or rely on CI (`r-cmd-check.yml` runs on every
push/PR to `main`).

There is no enforced coverage threshold, but [.github/workflows/coverage.yml](.github/workflows/coverage.yml)
uploads coverage to Codecov on every push/PR to `main`.

## Developer Rules

- **The C++ core is shared between Python and R.** Changes to [src/IPF.h](src/IPF.h),
  [src/QIS.cpp](src/QIS.cpp), [src/QISI.cpp](src/QISI.cpp), [src/Integerise.cpp](src/Integerise.cpp),
  or the other algorithm files affect both bindings — check whether the R exports
  ([src/RcppExports.cpp](src/RcppExports.cpp), [R/](R/)) need corresponding updates, even
  though R is maintenance-only.
- **`src/module.cpp` binding changes are the critical path for Python users.** nanobind does
  not implicitly convert arbitrary Python types the way pybind11 did — verify explicitly what
  input types (python scalar, numpy scalar, list, tuple, `np.array`, `range`) each bound
  function needs to accept, and add a regression test for each. (This exact gap caused a real
  regression during the pybind11 → nanobind migration — see [JOURNAL.md](JOURNAL.md).)
- **Runtime dependencies are intentional.** `numpy` and `pandas` are the only runtime deps.
  New runtime deps need a strong justification; dev-only tools go in `[dependency-groups.dev]`
  in [pyproject.toml](pyproject.toml).
- **C++ unit tests live alongside the algorithms** (`src/Test*.cpp`, run via `UnitTester.cpp`)
  and are exposed to Python as `hl_unittest()` / `test_unittest` in [tests/test_all.py](tests/test_all.py).
  Add to these for new C++-level algorithm behaviour rather than only testing at the Python boundary.
- **Type annotations required** on Python code; `ty` will catch missing or incorrect ones.
- **Line length is 120** (configured in [pyproject.toml](pyproject.toml) under `[tool.ruff]`).
- **No comments explaining what the code does.** Only add a comment when the *why* is
  non-obvious (hidden constraint, workaround, subtle invariant).
- **Type stubs are hand-maintained** ([humanleague/__init__.pyi](humanleague/__init__.pyi)),
  regenerated with `nanobind.stubgen` and manually corrected — see [doc/type-stubs.md](doc/type-stubs.md).
  They are not regenerated automatically on every build, so they can silently drift from the
  actual bound signatures.

## Reviewer Checklist

When reviewing a PR or diff, check:

1. **Correctness** — does the change preserve existing algorithm behaviour (IPF/QIS/QISI
   convergence, statistics, error handling)? Edge cases: zero/negative population, non-finite
   input, mismatched index/marginal dimensions, degenerate marginals.
2. **Binding input coverage** — for [src/module.cpp](src/module.cpp) changes, does the function
   accept the same range of Python input types as before (python int/float, numpy scalar, list,
   tuple, `np.array`, `range`)? nanobind requires this to be handled explicitly per binding,
   unlike pybind11.
3. **Shared core vs. binding-only** — is the change in [src/](src/) (affects both Python and R)
   or only in the Python/R glue? Flag if a shared-core change wasn't reflected in both bindings.
4. **Test coverage** — new algorithm behaviour needs a Python test in [tests/test_all.py](tests/test_all.py)
   or a C++ unit test in `src/Test*.cpp`; changed error conditions need `pytest.raises` coverage.
5. **Ruff rules** — no rule in the `select` list should be suppressed without justification.
   Active rules: `ARG, B, C, D103, E, F, I, N, PERF, PTH, RET, RUF, SIM, UP, W` (`E501` ignored;
   `D103`/`N802` also ignored under `tests/`).
6. **README / release notes** — if the public Python API changes, update [README.md](README.md)
   and add an entry to [release_notes.md](release_notes.md).
7. **Type stubs** — if a bound function's signature changed, confirm
   [humanleague/__init__.pyi](humanleague/__init__.pyi) was regenerated/updated (see
   [doc/type-stubs.md](doc/type-stubs.md)); it's easy to forget since it isn't generated
   automatically.

## QA Rules

- Run the full gate suite (`ruff check`, `ty check`, `pytest`) before declaring any Python-side
  task done.
- CI runs [python-test.yml](.github/workflows/python-test.yml) on Python 3.12/3.13/3.14 ×
  ubuntu/windows/macos, plus [r-cmd-check.yml](.github/workflows/r-cmd-check.yml) across
  several R versions/OSes, and [coverage.yml](.github/workflows/coverage.yml). Flag anything
  that might be platform- or compiler-specific — this project has already hit a compiler error
  specific to Debian bookworm.
- `main` has branch protection (PRs required; force-push and branch deletion disabled) but does
  **not** currently have required status checks configured server-side — CI passing is expected
  but not enforced by GitHub, so don't rely on it to block a bad merge.

## Repository Layout

```
src/                          # shared C++ core + both language bindings
  IPF.h, QIS.*, QISI.*, Integerise.*, Sobol*.cpp, Index.*, NDArray*.h  # algorithms
  Test*.cpp, UnitTester.cpp    # C++-level unit tests
  module.cpp                   # nanobind bindings (Python)
  RcppExports.cpp               # Rcpp bindings (R, generated)
humanleague/                  # Python package
  __init__.py, utils.py, __init__.pyi, py.typed
R/                             # R package source
  humanleague.R, RcppExports.R
tests/
  test_all.py, test_errors.py, test_utils.py   # Python tests
  testthat/, testthat.R                          # R tests
doc/
  type-stubs.md, help.png, paper.md, paper.bib
scripts/
  package.sh                    # manual TestPyPI publish helper
.github/workflows/
  python-test.yml                # Python lint + type check + test matrix
  coverage.yml                   # Codecov upload
  r-cmd-check.yml                 # R CMD check matrix
  pypi-release.yml                # PyPI publish on `v*` tag
CMakeLists.txt
pyproject.toml
DESCRIPTION / NAMESPACE        # R package metadata
README.md
JOURNAL.md
release_notes.md
```

## Branch and Release Policy

- **`main` is branch-protected** (PRs required; force-push and branch deletion disabled).
  Required status checks are not configured, so passing CI is expected but not enforced by
  GitHub — treat it as a hard requirement anyway.
- **Python releases are triggered by a `v*` tag** (e.g. `v2.5.0`), which runs
  [pypi-release.yml](.github/workflows/pypi-release.yml) and publishes to PyPI. That workflow
  currently builds and publishes an **sdist only** (`uv build --sdist`) — no wheels — so
  `pip install humanleague` always compiles from source regardless of the stable-ABI wheel this
  project's build now supports. Fixing that is a known follow-up (see [JOURNAL.md](JOURNAL.md)).
- The Python version lives in `pyproject.toml` (`version = "x.y.z"`); the R package version
  lives separately in `DESCRIPTION` (`Version:`) and is **not** kept in sync with the Python
  version — check both before a release that touches shared code.
- Document every release in [release_notes.md](release_notes.md), and record the underlying
  design decisions in [JOURNAL.md](JOURNAL.md) per the [Task & Design Summaries](#task--design-summaries)
  policy.

## Workflow

1. Agree the plan with the maintainer before writing code (see [Collaboration & Ownership](#collaboration--ownership)).
2. Create a feature branch off `main` — never commit directly to `main`.
3. Make changes under [src/](src/) (shared core / bindings) and/or [humanleague/](humanleague/)
   (Python package).
4. Add or update tests in [tests/](tests/) (Python) and, if the shared C++ core changed,
   `src/Test*.cpp`.
5. Add a task/design entry to [JOURNAL.md](JOURNAL.md) (see [Task & Design Summaries](#task--design-summaries)).
6. Run the full Python gate suite locally; run `R CMD check` if R bindings might be affected.
7. If the public Python API changed, update [README.md](README.md) and [release_notes.md](release_notes.md);
   if a bound function's signature changed, regenerate/update the type stubs (see
   [doc/type-stubs.md](doc/type-stubs.md)).
8. Commit — pre-commit hooks will auto-fix formatting, re-lock `uv.lock`, and type-check.
9. Open a PR targeting `main`; CI should pass (Python matrix, R-CMD-check, coverage) before
   merging even though it isn't a hard-enforced gate.
10. To release: bump the version in `pyproject.toml`, merge to `main`, then push a `vX.Y.Z`
    tag — PyPI publish triggers automatically (sdist only, see
    [Branch and Release Policy](#branch-and-release-policy)).
