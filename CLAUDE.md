# CLAUDE.md — GSTools Benchmarking

This file is the complete reference for the benchmarking infrastructure added
on the `benchmarking` branch (Jeisson Leal, UFZ). Read it before touching
anything in `benchmarks/`, `.github/workflows/`, or the ASV config files.

---

## Overview

The benchmarking suite uses [Airspeed Velocity (ASV)](https://asv.readthedocs.io)
to measure and track performance of two GSTools computation backends across
three workflow families (variogram, kriging, random field generation):

- **`cython_fallback`** — default Cython-backed implementation from
  [gstools-cython](https://github.com/GeoStat-Framework/GSTools-Cython)
- **`rust_core`** — Rust-backed implementation from
  [gstools_core](https://github.com/GeoStat-Framework/GSTools-Core)

Benchmark history is published at:
`https://geostat-framework.github.io/gstools-benchmarks/`
(served from `jeilealr/gstools-benchmarks-asv` via GitHub Pages)

---

## Repository layout

```
asv.conf.json                        # ASV config, single-thread baseline
asv.openmp.conf.json                 # ASV config, multi-thread / OpenMP
benchmarks/
  benchmark_two_point_statistics.py  # ASV benchmark classes (the actual suite)
  README.md                          # Detailed local-run guide
  tools/
    asv_speedup_summary.py           # Terminal/Markdown/HTML speedup summary
    check_backend_parallel_ready.py  # CI readiness probe (Cython OpenMP + Rust)
    configure_asv_machine.py         # Write a stable ASV machine profile in CI
    install_openmp_cython.py         # Build gstools-cython with OpenMP from source
    install_pyproject_extras.py      # Install pyproject.toml extras inside ASV envs
    plot_case_backend_comparison.py  # Interactive HTML visualization generator
    profile_benchmark_workflows.py   # cProfile helper for local profiling
    render_benchmark_comment.py      # Shared PR comment body renderer (both CI paths)
.github/workflows/
  asv-openmp-pr-comparison.yml       # Full OpenMP benchmark on every PR
  asv-benchmark-publish.yml          # Publish cumulative history after merge to main
  benchmark-readiness.yml            # Smoke check: can benchmarks install and start?
  benchmark-pr-comment.yml          # Post comment on cross-fork PRs via workflow_run
```

---

## Benchmark suite (`benchmark_two_point_statistics.py`)

### Cases

| Family | Case label | Meaning |
|---|---|---|
| variogram | `full_900` | 900 scattered pts, full pairwise |
| variogram | `sampled_5000_to_1500` | 5 000 pts, sampled 1 500 |
| variogram | `sampled_15000_to_4500` | 15 000 pts, sampled 4 500 |
| krige | `small_30x500` | 30 cond. pts → 500 target pts |
| krige | `large_120x2000` | 120 cond. pts → 2 000 target pts |
| krige | `extra_large_360x6000` | 360 cond. pts → 6 000 target pts |
| field | `srf_unstructured_randmeth` | SRF RandMeth, 2 000 pts |
| field | `srf_structured_randmeth` | SRF RandMeth, 64×64 grid |
| field | `srf_structured_fourier` | SRF Fourier, 64×64 grid |
| field | `condsrf_unstructured` | CondSRF, 40 cond. pts → 1 000 target |

### Benchmark method naming

ASV discovers benchmarks by method prefix:

- `time_*` — wall-clock time
- `peakmem_*` — peak memory usage
- `setup_cache()` — runs once per environment; shared across parameter combos
- `setup()` — runs before each parameter combo; can call `raise NotImplementedError`
  to skip unavailable combos (e.g. Rust backend not installed)

Each benchmark family has both a `time_*` and a `peakmem_*` method with
identical bodies — ASV uses the method prefix to select the measurement mode.

### Environment variables that control the suite

| Variable | Default | Purpose |
|---|---|---|
| `GSTOOLS_BENCHMARK_THREADS` | `"1"` | Comma-separated thread counts, e.g. `"1,2,4,8"` |
| `GSTOOLS_BENCHMARK_BACKENDS` | all backends | Comma-separated subset, e.g. `"rust_core"` |

These are read at import time by `_configured_thread_counts()` and
`_configured_backends()`. Changing them after import has no effect.

### `gstools_backend` context manager

```python
with gstools_backend(use_core=True, num_threads=4):
    # gstools.config.USE_GSTOOLS_CORE = True
    # gstools.config.NUM_THREADS = 4
    # ... run benchmark ...
# config restored to original values
```

Always use this context manager instead of mutating `gs.config` directly.

---

## ASV configuration

### `asv.conf.json` (single-thread baseline)

- `environment_type: "conda"` — ASV creates isolated conda environments
- `results_dir: ".asv/results"` — kept out of git
- Timing attributes (`number = 1`, `repeat = 20`) live on the benchmark
  **classes** in `benchmark_two_point_statistics.py`, not in `asv.conf.json`
  (ASV ignores `number`/`repeat` keys there).
- `install_command`: installs `gstools_core` (Rust) via
  `install_pyproject_extras.py rust`, then installs GSTools with `--no-deps`
- Does **not** build gstools-cython from source; relies on conda-forge package

### `asv.openmp.conf.json` (multi-thread / OpenMP)

- `env_dir / results_dir / html_dir` all under `.asv-openmp/`
- `matrix.req` includes `c-compiler`, `cxx-compiler`, `cython`, etc. — needed
  to compile gstools-cython from source inside the ASV environment
- `install_command` runs (in order):
  1. `install_pyproject_extras.py rust` — installs gstools_core
  2. `install_openmp_cython.py {env_dir}` — builds gstools-cython with OpenMP
  3. `pip install --no-deps {build_dir}` — installs the benchmarked GSTools commit
  4. `check_backend_parallel_ready.py --verbose` — verifies readiness; failure
     marks the environment broken and skips all benchmarks for that commit

---

## CI workflows

### 1. `asv-openmp-pr-comparison.yml` — "Benchmarking"

**Trigger:** `pull_request` targeting `main`

**What it does:**
1. Detects whether benchmark-relevant files changed (skips if not)
2. Runs full OpenMP comparison: benchmarks both the base SHA and head SHA
3. Runs `asv compare --factor 1.05 --split` to produce `comparison.txt`
4. Builds the interactive HTML report via `plot_case_backend_comparison.py`
5. Publishes an artifact `openmp-pr-<number>` (30-day retention)
6. Calls `render_benchmark_comment.py` to write `/tmp/benchmark_comment.md`
7. Posts the comment directly to the PR **only for same-repo PRs** (gated on
   `head.repo.full_name == github.repository`)
8. Saves a `benchmark-pr-context` artifact (1-day retention) for the
   cross-fork comment workflow to pick up

**Permissions:** `contents: read`, `pull-requests: write`

**Machine name in CI:** `github-actions-ubuntu` (set via `MACHINE_NAME` env var,
written to the ASV machine profile by `configure_asv_machine.py`)

### 2. `benchmark-pr-comment.yml` — "Post Benchmark PR Comment"

**Trigger:** `workflow_run` on "Benchmarking" completed

**Purpose:** Posts PR comments for **cross-fork PRs** (where the inline step in
workflow 1 is skipped because `GITHUB_TOKEN` is read-only for forks).

**How it works:**
1. Checks out the repo (needed to access `render_benchmark_comment.py`)
2. Downloads the `benchmark-pr-context` artifact from the triggering run
   (carries `pr_number`, the **benchmarked** `base_sha`/`head_sha`, and the
   comparison text)
3. Verifies the PR's current head and the artifact's head both equal the
   triggering run's `head_sha`, then uses the recorded `base_sha` (format-
   validated) — it does **not** re-query `.base.sha`, which would drift if the
   base branch advanced after the benchmark ran
4. Calls `render_benchmark_comment.py` to build the comment body
5. Posts or patches the PR comment using the `<!-- gstools-openmp-benchmark -->`
   HTML marker for idempotency

**Permissions:** `actions: read` (required for cross-workflow artifact download),
`pull-requests: write`

**Important:** `workflow_run` only triggers workflows that exist on the **default
branch** of the target repo. This workflow will not fire on PRs targeting a repo
where this file is not yet on `main`.

**Secrets:** Only `GITHUB_TOKEN` (auto-injected). No custom secrets needed.

### 3. `asv-benchmark-publish.yml` — "Publish OpenMP Benchmark History"

**Trigger:** Push to `main` (benchmark-relevant paths only), or manual dispatch

**What it does:**
1. Checks out GSTools + the `jeilealr/gstools-benchmarks-asv` history repo
2. Restores cumulative ASV results from the history repo
3. Benchmarks the newly accepted commit
4. Builds updated `index.html` + native ASV HTML + raw results
5. Commits and force-pushes to `jeilealr/gstools-benchmarks-asv`

**Gate:** `if: github.repository == 'jeilealr/GSTools'` — prevents running on
forks or the upstream GeoStat-Framework repo.

**Secret required:** `BENCHMARKS_REPO_TOKEN` — a PAT with write access to
`jeilealr/gstools-benchmarks-asv`. Must be set in `jeilealr/GSTools` →
Settings → Secrets → Actions.

### 4. `benchmark-readiness.yml` — "Benchmark Checks"

**Trigger:** Pull request or push to `main` touching benchmark files

**What it does:** Smoke-checks that benchmarks can be installed and started on
all four runner OSes (ubuntu, windows, macos, macos-15-intel). Uses an existing
Python environment (`-E existing`), not a full ASV conda environment. Runs only
one benchmark method quickly with `--quick`.

Also runs `check_backend_parallel_ready.py` in a conda environment to verify
Cython OpenMP + Rust backend readiness across all four OSes.

---

## PR comment mechanism

Two code paths produce the same comment format:

```
<!-- gstools-openmp-benchmark -->
## OpenMP Benchmark Results

**Base:** `<sha8>` → **Head:** `<sha8>` · [Full HTML Report ↗](<url>)

✅/⚠️ <badge> · Mann-Whitney U · 5% threshold

<details>
<summary>Full ASV comparison</summary>
```<asv compare output>```
</details>
```

**Badge logic (in `render_benchmark_comment.py`):**
- Count lines starting with `"+ "` → regressions
- Count lines starting with `"- "` → improvements
- Lines starting with `"~ "` are **not counted** (ratio exceeded threshold but
  Mann-Whitney U test was not significant — measurement noise)

The `<!-- gstools-openmp-benchmark -->` marker enables idempotent PATCH
updates: if a comment with this marker already exists, it is updated via
`gh api PATCH`; otherwise a new comment is created via `gh pr comment`.

### Why `~1.13` shows as "No significant changes"

ASV's `asv compare --factor 1.05 --split` applies two independent criteria:

1. **Ratio threshold:** result ratio must exceed the factor (1.05 = 5% change)
2. **Mann-Whitney U statistical test:** the two measurement distributions must
   be statistically distinguishable

A line is prefixed `+` (regressed) or `-` (improved) only when **both** agree.
A `~` prefix means the ratio exceeded 1.05 but the statistical test failed —
i.e., the measurements overlap too much to conclude there is a real change.
GitHub-hosted runners are shared and noisy, making `~` common even for real
changes. The HTML bar chart shows the raw ratio regardless of statistical
significance.

---

## Interactive HTML visualization (`plot_case_backend_comparison.py`)

Generates a fully self-contained, no-dependency HTML file from ASV result JSON.

### What it shows
- Grouped bar chart (bar mode): backends side by side per commit/tag, with %
  change annotations when exactly 2 commits/tags are selected
- Line chart (line mode): metric over commits/tags per backend
- Filters: Metric, Family, View (bar/line), Machine, Case, Threads, Commits or
  Tags (toggle), Backend. **Machine is single-choice** so results from different
  environments are never merged into one series (defaults to one machine).

### Commit vs Tag toggle
- **Commits mode:** shows all benchmarked commits in a `<select multiple>`
- **Tags mode:** shows only commits that have a git tag attached, by tag name

Tags only appear in the selector if a benchmarked commit has a tag. In a PR
artifact with two commits, at most one or two tags will appear. The full tag
history grows as more commits are merged and published.

### Default selections
The JS `refreshOptions()` function picks smart defaults per family:
- **krige:** defaults to `extra_large_360x6000`
- **variogram:** defaults to `sampled_15000_to_4500`
- **field:** defaults to `srf_unstructured_randmeth`
- fallback: first alphabetical case

### Dark mode
Full `@media (prefers-color-scheme: dark)` support via CSS variables. All
colors use `var(--*)` tokens; bar chart % annotation colors use `var(--danger)`
and `var(--success)` so they adapt correctly to dark mode.

### CLI options
```bash
python benchmarks/tools/plot_case_backend_comparison.py \
  --results-dir .asv-openmp/results \
  --metric time          # or memory, or all (default)
  --benchmark variogram  # or krige, field, srf, condsrf, vario, kriging
  --max-commits 50       # limit displayed history
  --output report.html
```

---

## Tools reference

### `render_benchmark_comment.py`
Shared script used by both CI paths to generate the PR comment body. Keeps
badge wording and comment format in sync.

```bash
python benchmarks/tools/render_benchmark_comment.py \
  --base <sha> --head <sha> \
  --artifact-url <url> \
  --comparison .asv-openmp/artifact/comparison.txt \
  --output /tmp/benchmark_comment.md
```

### `asv_speedup_summary.py`
Local developer tool (not called from CI). Reads ASV result JSON and prints
Rust-vs-Cython speedup ratios.

```bash
python benchmarks/tools/asv_speedup_summary.py                      # terminal table
python benchmarks/tools/asv_speedup_summary.py --format markdown     # Markdown
python benchmarks/tools/asv_speedup_summary.py --format html --output report.html
python benchmarks/tools/asv_speedup_summary.py --results-dir .asv-openmp/results
```

Speedup formula: `cython_fallback_time / rust_core_time`
- `> 1.0` → Rust is faster
- `= 1.0` → equal
- `< 1.0` → Rust is slower

### `check_backend_parallel_ready.py`
Verifies:
1. Each `gstools_cython` module accepts explicit OpenMP thread counts
2. Rust backend runs variogram, SRF field generation, and kriging with
   `NUM_THREADS > 1`

Used both in CI (`asv.openmp.conf.json` install_command) and manually.

```bash
python benchmarks/tools/check_backend_parallel_ready.py --verbose
python benchmarks/tools/check_backend_parallel_ready.py --cython-only
```

### `configure_asv_machine.py`
Writes a named ASV machine profile using detected hardware. Used in CI to
ensure all results are stored under the stable name `github-actions-ubuntu`
regardless of the actual runner hardware.

```bash
python benchmarks/tools/configure_asv_machine.py github-actions-ubuntu
```

---

## Secrets and permissions

| Repo | Secret | Purpose | Where to set |
|---|---|---|---|
| `jeilealr/GSTools` | `BENCHMARKS_REPO_TOKEN` | Push benchmark history to `jeilealr/gstools-benchmarks-asv` | Settings → Secrets → Actions |
| `jeilealr/GSTools` | `GITHUB_TOKEN` | Auto-injected, no setup needed | — |
| `GeoStat-Framework/GSTools` | `GITHUB_TOKEN` | Auto-injected, works for PR comments | — |
| `GeoStat-Framework/GSTools` | `BENCHMARKS_REPO_TOKEN` | Only needed if they adopt the publish workflow | Settings → Secrets → Actions |

Workflow-level permissions (set in YAML, not repo settings):
- `asv-openmp-pr-comparison.yml`: `contents: read`, `pull-requests: write`
- `benchmark-pr-comment.yml`: **`actions: read`** (required for cross-workflow
  artifact download), `pull-requests: write`
- `asv-benchmark-publish.yml`: `contents: read`
- `benchmark-readiness.yml`: `contents: read`

---

## Published benchmark history

**Current (fork development):** cumulative results live in
`jeilealr/gstools-benchmarks-asv`, served at
`https://jeilealr.github.io/gstools-benchmarks-asv/`

**Planned (pending green light from GeoStat-Framework maintainers):** migrate
to `GeoStat-Framework/GSTools-Benchmarks`. When that happens, update
`BENCHMARKS_REPO` in `asv-benchmark-publish.yml` and the `project_url` in
both ASV config files. Until then, everything runs on Jeisson's fork.

Repository layout:
```
gstools-benchmarks-asv/          (or GSTools-Benchmarks/ once migrated)
  index.html        # custom interactive HTML (plot_case_backend_comparison.py)
  asv/              # native ASV HTML report
  results/          # raw ASV result JSON (source of truth)
  .nojekyll         # disables Jekyll processing on GitHub Pages
```

The publish workflow adds one new entry per merge to `jeilealr/GSTools` main.
It restores `results/` before running so history accumulates across commits.

---

## How to add a new benchmark case

1. Add the case label to the relevant tuple in `benchmark_two_point_statistics.py`
   (`VARIOGRAM_CASES`, `KRIGE_CASES`, or `FIELD_CASES`)
2. The corresponding `setup_cache` / `setup` / `time_*` / `peakmem_*` methods
   use the tuple as ASV `params` — no other changes needed in the benchmark file
3. If it is a genuinely new family (not variogram/krige/field), add:
   - New prefix to `BENCHMARK_PREFIXES` in both `plot_case_backend_comparison.py`
     and `asv_speedup_summary.py`
   - New alias(es) in `BENCHMARK_ALIASES` in `plot_case_backend_comparison.py`
   - Update the `defaultCase` heuristic in `refreshOptions()` (JS)

---

## Known behaviours and gotchas

- **`~` in `asv compare` output** means the ratio exceeded the 5% threshold
  but the Mann-Whitney U test was not significant. It is not counted as a
  regression or improvement in PR badge or comment.

- **Tags mode shows few tags on PR artifacts** — expected. A PR artifact only
  contains the two benchmarked commits (base SHA + head SHA). Only tags that
  sit exactly on one of those commits will appear. Full tag history accumulates
  in the published site as more commits are merged.

- **`workflow_run` requires the workflow on the default branch** — if
  `benchmark-pr-comment.yml` is not yet merged to the target repo's default
  branch, cross-fork PR comments will not appear. This applies both to
  `jeilealr/GSTools` (until the benchmarking branch merges to its `main`) and
  to `GeoStat-Framework/GSTools` (until a PR there is accepted).

- **`asv.openmp.conf.json` install_command failure** — if
  `check_backend_parallel_ready.py` exits non-zero, ASV marks the environment
  broken and silently skips all benchmarks for that commit.

- **Duplicate comment posts for first-party PRs** — both the inline step in
  `asv-openmp-pr-comparison.yml` and `benchmark-pr-comment.yml` will attempt
  to post/update for same-repo PRs. This is harmless because the second write
  is an idempotent PATCH of the same comment using the HTML marker. It is
  redundant but not a bug.

- **Python 3.14 in ASV configs** — both configs use `"pythons": ["3.14"]`.
  Update both files together if a Python version change is needed.

- **`BENCHMARKS_REPO_TOKEN` scope** — the PAT only needs `repo` scope scoped
  to `jeilealr/gstools-benchmarks-asv`. It does not need org-level access.

---

## Action SHA pins (as of 2026-06)

All actions in CI workflows are pinned to commit SHAs for supply-chain security:

| Action | SHA | Version |
|---|---|---|
| `actions/checkout` | `08c6903cd8c0fde910a37f88322edcfb5dd907a8` | v5.0.0 |
| `conda-incubator/setup-miniconda` | `fc2d68f6413eb2d87b895e92f8584b5b94a10167` | v3 |
| `actions/setup-python` | `e797f83bcb11b83ae66e0230d6156d7c80228e7c` | v6.0.0 |
| `actions/upload-artifact` | `ea165f8d65b6e75b540449e92b4886f43607fa02` | v4.6.2 |
| `actions/download-artifact` | `d3f86a106a0bac45b974a628896c90dbdf5c8093` | v4.3.0 |

When updating, replace both the SHA and the `# vX.Y.Z` comment together.
