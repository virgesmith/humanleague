# Development Journal

A running log of every task/PR: *why* it was done and the *design decisions* made.
Newest entries at the top. This is the durable record of intent that keeps the
maintainer in control of the codebase's direction — see the
[Task & Design Summaries](AGENTS.md#task--design-summaries) policy in `AGENTS.md`.

Entry template:

```markdown
## YYYY-MM-DD — <title> (#PR)

**Why** — the motivation and the problem this solves.

**What** — high-level description of the change.

**Design decisions**
- <decision> — alternatives considered, why this was chosen.

**Follow-ups** — anything deferred, known limitations.
```

---

## 2026-10-11 — Python 3.13–3.15; update build requirements and CI actions (#68)

**Why** — The policy is to support the three most recent CPython releases. With 3.15.0
final released, the window moves from 3.12–3.14 to 3.13–3.15, and 3.12 is dropped.
Separately, the build requirements and CI actions were a major version or more behind.
Isolated builds already fetch the latest scikit-build-core and nanobind, so the old floors
(`>=0.12`, `>=2.12.0`) only described versions nobody tests against. Through
`minimum-version = "build-system.requires"`, they also kept scikit-build-core emulating its
pre-1.0 defaults.

**What**
- Supported Python: `requires-python` `>=3.12` → `>=3.13`; classifiers drop 3.12 and add
  3.15; stable-ABI target `wheel.py-api` `cp312` → `cp313`; cibuildwheel `build` list
  `cp312–cp314` → `cp313–cp315`; CMake `find_package(Python 3.12` → `3.13`. CI test matrix,
  README and AGENTS.md updated to match; the coverage job moves from 3.13 to 3.14. The
  three-version policy, and the checklist of places a version change touches, are now a
  Developer Rule in AGENTS.md.
- Build requirements: `scikit-build-core >=1.1`, `nanobind >=3.1`. Dropped the
  `[pyproject]` extra from the manual dev-install instructions (README, CMakeLists.txt
  comment, AGENTS.md): scikit-build-core 1.0 removed it and installers now warn about it.
- CI actions: `actions/checkout` v6 → v7; `astral-sh/setup-uv` `@v7` (release) and a SHA pin
  of v8.0.0 (test/coverage) → `@v10.2.0` everywhere; `pypa/cibuildwheel` v4.2.0 → v4.3.1.
  `upload-artifact@v7`, `download-artifact@v8` and `r-lib/actions@v2` were already on the
  latest major and are unchanged.
- `uv.lock` is now committed (removed from `.gitignore`).
- Required checks: each of `python-test.yml`, `coverage.yml` and `r-cmd-check.yml` gains an
  `all-passed` summary job (`Python tests passed`, `Coverage passed`, `R CMD check passed`).
  Branch protection on `main` now requires just those three, instead of 15 per-matrix-leg
  job names. AGENTS.md updated: it said no status checks were required, which was out of date.

**Design decisions**
- *Stable-ABI floor follows `requires-python`.* There is no point tagging the wheel `cp312`
  when pip won't install the package on 3.12. One `cp313-abi3` wheel per platform still
  covers every supported version.
- *Build-requirement floors are the versions actually built and tested* (scikit-build-core
  1.1.1, nanobind 3.1.0), not the oldest that might work. Nothing exercises an older
  combination, so a lower floor would be an untested claim. The cost falls only on
  `--no-build-isolation` dev builds, which now need current versions installed.
- *Raising the scikit-build-core floor deliberately opts into its 1.x defaults*, because
  `minimum-version` tracks it. Reviewed: SDists now resolve symlinks (the repo tracks none);
  `project.version` is kept as written (ours is already normalised); renamed config fields
  now error (we use none of them).
- *No source changes for nanobind 3.* Its API breaks were reviewed against
  `src/module.cpp`: no runtime-computed return value policies or `noconvert` annotations,
  no `nb::none()` in a conditional, no `gil_scoped_acquire`, no trampolines, no
  `ndarray_traits`. 3.0.0 was yanked; a 3.1 floor also skips it.
- *nanobind 3's split mode not adopted.* It puts the frontend on the 3.10 stable ABI plus a
  per-version `nanobind-backend` wheel from PyPI. That would add a runtime dependency (see
  the runtime-deps rule in AGENTS.md), and the stable-ABI wheel already covers our range.
- *setup-uv pinned to an exact version tag, replacing the SHA pins.* From v8 setup-uv no
  longer publishes floating major/minor tags, so `@v10` does not resolve. The choice is
  between an exact tag and a SHA. SHA pinning normally wins because tags are mutable (cf.
  the 2025 `tj-actions/changed-files` compromise, where existing tags were repointed to
  malicious code). But setup-uv publishes GitHub *immutable releases*, so its tags cannot
  be moved or deleted, and the readable tag gives the same guarantee. Residual risk:
  immutability is a repo setting upstream controls, not something our workflow enforces.
  Accepted for this project.
- *setup-uv v10.2.0 rather than the latest v10.3.0.* v10.3.0 was two days old, and a
  cooldown before adopting a fresh release lowers exposure to a bad or compromised one.
  Nothing in v10.3.0 is needed. setup-uv's v8–v10 breaking changes don't apply: no custom
  `manifest-file`; the `prune-cache` default flip (v9) only affects cache size; automatic
  cache disabling for `pull_request_target`/`workflow_run` (v10) concerns triggers we
  don't use.
- *cibuildwheel at the one-day-old v4.3.1 despite the same cooldown argument.* Its only
  change moves CPython 3.15 from rc3 to the final 3.15.0, which the new `cp315-*` builds
  need. Its releases are *not* immutable, so this tag pin is weaker than setup-uv's. Other
  actions stay on floating major tags (`@v7` etc.), as before.
- *Commit `uv.lock`.* CI (`uv sync --dev`) and local dev environments now install the same
  pinned versions, so a CI failure can be reproduced exactly, and dependency changes arrive
  as reviewable lockfile diffs instead of silently on the next run. The trade-off: CI no
  longer tests against the latest numpy/pandas, which is what users get, because the lock
  only constrains our environments, not the published package's requirements. A new
  upstream release can therefore break users before CI notices, until the lock is
  refreshed (Dependabot's `uv` ecosystem, planned next). The lock does not cover
  `build-system.requires`: isolated builds still fetch the latest scikit-build-core and
  nanobind.
- *Summary jobs instead of requiring every matrix leg.* Required checks are matched by job
  name, and matrix job names embed the Python/R version (`build (3.12, ubuntu-latest)`).
  So every version bump left branch protection waiting forever for jobs that no longer
  exist, and this PR hit exactly that. A summary job `needs:` the matrix and gives one name
  that stays the same. Details:
  - `if: always()` is essential. Without it, a failed matrix leg makes the summary job
    *skipped*, and GitHub counts a skipped required check as passing.
  - The step checks `needs.<job>.result == success`, so a failure or cancellation in any
    leg fails the summary.
  - One summary job per workflow, because `needs:` cannot cross workflow files. Rejected:
    merging the three workflows into one, which is a bigger restructure and would couple
    R and Python CI triggers.
  - Rejected: a ruleset or third-party "wait for all checks" action. These add a moving
    part, or a dependency, for what three five-line jobs already do.

**Follow-ups**
- Type stubs not regenerated: no bound signatures changed. nanobind 3's stubgen output
  differs in formatting from the hand-corrected stubs, as before.
- No Dependabot config for `github-actions`, so action bumps stay manual. Adding
  `.github/dependabot.yml` would automate them; its `cooldown` option could encode the
  "don't adopt brand-new releases" policy.

---

## 2026-08-28 — Build binary wheels for PyPI releases (#67)

**Why** — The release workflow published only an sdist, so every `pip install humanleague`
compiled the nanobind extension from source and therefore required a C++20 toolchain, CMake
and nanobind on the user's machine. The move to nanobind's stable ABI (#60) made per-platform
binary wheels cheap — one wheel per OS covers 3.12+ — so there is no longer a reason not to
ship them.

**What** — Split the workflow into three jobs: `wheels` (cibuildwheel across
`ubuntu-latest`/`windows-latest`/`macos-latest`, one artifact per platform), `sdist`
(`uv build --sdist`), and `deploy`, which now `needs` both and downloads their artifacts
rather than building anything itself. Added a `workflow_dispatch` trigger so the wheel build
can be exercised on demand, with `deploy` guarded on `refs/tags/v*` so a manual run builds
without publishing.

**Design decisions**
- *cibuildwheel rather than a hand-rolled matrix* — it already knows the manylinux
  containers, the repair/audit step and the ABI tags, none of which are worth reimplementing.
- *Separate `wheels` and `sdist` jobs feeding `deploy` via artifacts* — keeps a single
  publish step (so a partial upload can't happen), lets the platform builds run in parallel,
  and makes `deploy` a pure publish with nothing to rebuild.
- *`workflow_dispatch` plus a tag guard on `deploy` instead of a separate CI workflow* —
  one definition, exercised the same way it will run for real, with the publish step
  unreachable outside a tag.
- *`fail-fast: false` on the wheel matrix* — a failure on one platform should still show
  which of the others would have succeeded.
- *Dropped the `zip examples` / artifact-upload steps* — they packaged an `examples/`
  directory that does not exist in the repo and never has, so `zip -r examples.zip
  examples/` would exit non-zero. They sat in `deploy` after `uv publish`, so a tagged
  release would have published to PyPI and then gone red. Removed rather than fixed:
  attaching example material to a release is a separate concern from publishing the
  package, and there is no material to attach.

**Follow-ups**
- There is no `[tool.cibuildwheel]` section in `pyproject.toml`, so the `build-frontend =
  "build[uv]"` that the added comment assumes is not in effect and cibuildwheel will use its
  default pip frontend — making the `setup-uv` step on that job redundant. Either add the
  config or drop the comment and the step.
- Not yet exercised: a `workflow_dispatch` run on the branch would validate `wheels` and
  `sdist` end to end before merge.

---

## 2026-08-28 — Optimise QIS sampling; compute degeneracy in log space (#66)

**Why** — `QIS::sample` is the hot path: it runs once per person per marginal, and it was
paying for a `std::map` construction and a by-value copy on every call, plus a redundant
`accumulate` over the distribution at every level of the sampling recursion. Separately,
`degeneracy` could silently return `0` for a value that is astronomically large, because
`tgamma` overflows to `inf` for any cell count above ~170 and one `inf` in the denominator
collapses the whole running product.

**What** — Four changes. In `QIS::sample`, the `std::map<int64_t, int64_t> slice_map` becomes
a `std::vector<int64_t>` indexed by the original dimension and is passed by const ref;
the two `dims_to_*` vectors are reserved; and the array copy is skipped when there are no
fixed dimensions to slice. `pick` gains a precomputed-sum parameter threaded down the
recursion. `degeneracy` evaluates the same expression via `lgamma`, exponentiating once.
`chiSq` walks the underlying storage instead of incrementing an `Index`. Measured on
`solve_m`: 16% faster at pop=100k over 8^4, 7% at pop=200k over 16^4, 21% at pop=50k over
4^4, with byte-identical output in every case.

**Design decisions**
- *Vector instead of map for `slice_map`* — the keys are a dense range of dimension indices
  known up front, so the map bought nothing but an allocation per call. A `std::vector`
  sized to `dims.size()` with `-1` marking "not sliced" preserves the sparse-lookup
  semantics the `VERBOSE` printout relied on. Rejected keeping the map and merely passing
  it by const ref: that removes the copy but not the per-call construction.
- *Thread the sum through `pick` for QIS but not QISI* — the sum of a slice is exactly the
  reduced value the parent picked from, so recomputing it is redundant. QIS's marginals are
  `int64_t`, so the sums are exact in `double` and output is unchanged. QISI's are `double`,
  where the reused sum differs from a re-accumulation in the last bits; that flipped picks at
  boundaries (chiSq 400.11 → 394.57 at pop=51200 over 8^4, different population for the same
  seed) and measured no faster, so it was rejected there. It also introduces a failure mode
  that cannot occur today: an externally supplied sum a ULP larger than the true sum lets
  `r * sum` exceed the final running total and throw `pick failed`.
- *`lgamma` rather than a guarded `tgamma`* — clamping or special-casing the overflow still
  loses precision across the product. Working in log space and exponentiating once is both
  shorter and correct over the whole representable range; results that were representable
  before are bit-identical.
- *Raw-pointer loop in `chiSq`* — only correct because sample and reference are always the
  same shape with the same storage order. It is called once per solve, so this is tidying,
  not a speedup; kept because it reads more plainly than the `Index` form.

**Follow-ups**
- `QIS::sample` recomputes the sum of the sliced marginal per call; the population is known
  to the caller, so it could in principle be threaded in from `solve_m` instead.
- `Sobol::reset()` defaults to `nSkip = 0` and so discards the constructor's `skips`. Not
  reachable from either binding (both construct a fresh object per call and never pass
  `reset`), but `QIS`/`QISI` should store `m_skips` and pass it through.
- `QIS`'s constructor still calls `m_sobolSeq.skip(skips)` unguarded, so `qis(..., 0)`
  consumes a Sobol point where `qisi(..., 0)` no longer does (see #64). Deliberate —
  guarding it would change every existing `qis` result — but the asymmetry is undocumented.

---

## 2026-07-21 — Switch Python bindings from pybind11 to nanobind (#60)

**Why** — The pybind11 + hand-rolled `setup.py`/`setuptools` build was slow to iterate on
locally and produced a separate, non-stable-ABI extension per CPython minor version
(3.12/3.13/3.14 built and shipped independently). nanobind + `scikit-build-core` targets
the CPython stable ABI, which can let a single wheel per OS cover all supported Python
versions, and is a smaller, faster-to-compile binding layer.

**What** — Replaced `setup.py` with `CMakeLists.txt` + `scikit-build-core` (declared in
`pyproject.toml`'s `[build-system]`). Ported `src/module.cpp` from pybind11 to nanobind
(`nb::ndarray` instead of `py::array_t`, capsule-owned returned arrays, `nb::class_` for
`SobolSequence`). The compiled extension is now built as a stable-ABI module
(`cp312-abi3`, via `NB_STATIC`) and importable as `humanleague.humanleague_ext` (was the
top-level `_humanleague`); `humanleague/__init__.py` and the tests were updated for the
new import path. Removed the now-unused `pybind11` dev dependency and `setup.py`.

Before merging, fixed two defects introduced by the mechanical pybind11 → nanobind port:

- `collect_indices` (used by `ipf`/`qis`/`qisi` to parse index arguments) had narrowed
  the accepted input types — pybind11's implicit array conversion used to accept numpy
  integer scalars (e.g. `np.int64(0)`) as index elements, but the nanobind rewrite only
  special-cased Python `int` and iterables, silently raising `ValueError` for numpy
  scalars. Fixed with `nb::try_cast<int64_t>`, which also covers numpy integer scalars
  via their `__index__` protocol. Added a regression test.
- Two leftover copy-paste duplicate lines (a dead second `return` in `qis()`, a duplicate
  `stats["degeneracy"]` assignment in `qisi()`) — harmless but removed.

Also updated `doc/type-stubs.md`, which still described the old `pybind11-stubgen`
workflow after `README.md` had already switched to nanobind's own `stubgen`.

**Design decisions**

- nanobind + `scikit-build-core` over keeping pybind11/setuptools — measured, not assumed:
  a clean rebuild dropped from ~17s to ~4.8s (~3.5x faster) and the compiled extension
  shrank from 1.29MB to 534KB (~58% smaller). Per-call binding overhead and import time
  showed no measurable difference (within noise, ~0.3-0.9µs/call either way) — this
  package's binding surface is small (6 functions, 1 class) and each call does real
  numerical work, so call overhead was never the bottleneck. The switch is a build-time
  and packaging win, not a runtime-performance one, and should not be described as the
  latter.
- Manual index parsing in `collect_indices` rather than relying on nanobind's implicit
  conversion — nanobind, unlike pybind11, does not auto-convert arbitrary Python scalars
  to `nb::ndarray`. `nb::try_cast<int64_t>` first, falling back to iterable handling,
  reproduces the pre-existing contract (python int, numpy integer scalar, or any iterable
  including `range()`, per each index element).
- `NB_STATIC` (statically link libnanobind into the extension) over `NB_SHARED` —
  `NB_SHARED` caused path/loading issues in this project's layout, and with only one
  extension module in the package there's no downside to static linking here.

**Follow-ups**

- `.github/workflows/pypi-release.yml` still runs `uv build --sdist` only — no wheels are
  built or published, so the stable-ABI single-wheel-per-OS benefit from this migration
  isn't actually realized yet. End users installing from PyPI still compile from source
  either way. Worth a follow-up to build and publish wheels (e.g. via `cibuildwheel`).
- `humanleague/__init__.pyi` (the hand-maintained type stubs) was not regenerated as part
  of this migration — it's byte-identical to `main`. Worth a one-off regeneration pass
  with `nanobind.stubgen` to confirm it still matches the bindings' actual signatures.
