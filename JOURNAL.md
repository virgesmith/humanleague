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
