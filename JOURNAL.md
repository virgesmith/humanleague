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
