# GSTools Benchmark Guide

This directory contains the Airspeed Velocity ([ASV](https://github.com/airspeed-velocity/asv/)) benchmark suite for GSTools and a complementary profiling helper implemented with cProfile (part of the Python standard library).

This guide benchmarks GSTools, inspects the
results, profiles where runtime is spent, and then decides what to optimize.

Unit tests in `tests/` answer "is the code correct?". The ASV benchmarks in
`benchmarks/` answer "how fast is this workflow, how much memory does it use,
and did that change across commits?". The complementary cProfile helper
answers "inside this workflow, which Python functions are taking most of the
time right now?".

The benchmarks compare two GSTools backends, which gives more context for
deciding where optimization work should go:

- `cython_fallback`: the default Cython-backed fallback implementation from
  [gstools-cython](https://github.com/GeoStat-Framework/GSTools-Cython).
- `rust_core`: the Rust-backed implementation from
  [gstools_core](https://github.com/GeoStat-Framework/GSTools-Core).

## Index

- [Setup](#setup)
- [Benchmarking Scripts](#benchmarking-scripts)
  - [ASV Configuration](#asv-configuration)
  - [Benchmark Naming](#benchmark-naming)
- [Benchmark Coverage](#benchmark-coverage)
  - [Shared Constants](#shared-constants)
  - [Shared Helpers](#shared-helpers)
  - [Benchmark Classes](#benchmark-classes)
    - [VariogramBenchmarks](#variogrambenchmarks)
    - [KrigingBenchmarks](#krigingbenchmarks)
    - [RandomFieldBenchmarks](#randomfieldbenchmarks)
- [Running The Benchmarks](#running-the-benchmarks)
  - [Baseline Benchmark](#baseline-benchmark)
    - [Main Branch Baseline](#main-branch-baseline)
    - [Several Commits Baseline](#several-commits-baseline)
    - [Summary of Results](#summary-of-results)
    - [Visualization of Results](#visualization-of-results)
    - [Profiling With cProfile](#profiling-with-cprofile)
- [Optional Parallelization with OpenMP](#optional-parallelization-with-openmp)
  - [OpenMP ASV Configuration](#openmp-asv-configuration)
  - [Run on macOS and Linux](#run-on-macos-and-linux)
  - [Verify Parallel Backends](#verify-parallel-backends)
  - [Run on Windows](#run-on-windows)
  - [OpenMP Thread Rule](#openmp-thread-rule)
  - [HPC Notes](#hpc-notes)
  - [Profiling With cProfile for Multiple Threads](#profiling-with-cprofile-for-multiple-threads)
- [More ASV Commands](#more-asv-commands)
- [External Reference](#external-reference)

## Setup

The regular installation commands in the main `README.md` install GSTools for
normal use. This benchmark guide uses conda because ASV creates isolated
benchmark environments for the commits it measures.

The default benchmark configuration intentionally compares both backends with
one GSTools thread:

```text
gstools.config.NUM_THREADS = 1
```

That keeps the first comparison simple: Cython fallback vs Rust core without
parallelism as a confounding factor. Parallel/OpenMP scaling is treated as a
separate optional experiment because the correct Cython OpenMP build depends on
the user's operating system, compiler, and runtime environment.

To run the benchmark and the optional cProfile helper, follow these steps (this guide uses Python 3.14):

1. Move to the GSTools repository root:

```bash
cd /path/to/GSTools
```

2. Create and activate a conda environment for local benchmark work:

```bash
conda create -n gstools-benchmark -c conda-forge python=3.14 asv
conda activate gstools-benchmark
```

If you already have a suitable conda environment, activate that instead.

3. If you use an existing environment, make sure ASV is installed:

```bash
conda install -c conda-forge asv
```

4. Create a machine profile once per computer:

```bash
asv machine --yes
```

The machine profile records local hardware information so ASV can label
results correctly. Do not compare absolute times across different machines.

## Benchmarking Scripts

The benchmarking setup currently consists of:

- `asv.conf.json`: tells ASV how to build GSTools, where benchmarks live, where
  to store results, and which Python/environment matrix to use.
- `asv.openmp.conf.json`: optional cross-platform ASV configuration that builds
  `gstools-cython` from source with OpenMP inside ASV's own environment.
- `benchmarks/benchmark_two_point_statistics.py`: contains the ASV benchmark
  classes.
- `benchmarks/README.md`: this practical guide.
- `benchmarks/tools/asv_speedup_summary.py`: reads ASV result JSON files and
  prints Rust-vs-Cython speedup ratios or writes a curated Markdown/HTML
  report.
- `benchmarks/tools/plot_case_backend_comparison.py`: reads ASV result JSON
  files and writes a self-contained HTML report with grouped backend bars and
  commit trends for selected cases, thread counts, and metrics.
- `benchmarks/tools/profile_benchmark_workflows.py`: runs one representative
  workflow from `benchmark_two_point_statistics.py` under Python's built-in
  `cProfile`, so you can see which functions take time in the current
  checkout.
- `benchmarks/tools/check_backend_parallel_ready.py`: CI helper that verifies
  Cython OpenMP detection and Rust backend execution (variogram, field
  generation, and kriging) with more than one GSTools thread.
- `benchmarks/tools/configure_asv_machine.py`: writes a named ASV machine
  profile from detected hardware; used in CI to ensure all results are stored
  under a stable name regardless of the actual runner.
- `benchmarks/tools/install_pyproject_extras.py`: installs selected optional
  dependencies directly from `pyproject.toml` without installing GSTools or
  its regular dependencies.
- `benchmarks/tools/install_openmp_cython.py`: helper used by
  `asv.openmp.conf.json` to compile `gstools-cython` with OpenMP on macOS,
  Linux, and native Windows.
- `benchmarks/tools/render_benchmark_comment.py`: CI helper used by both the
  pull-request comparison workflow and the cross-fork comment workflow to
  render the GitHub PR comment body; keeps badge wording and comment format
  in sync across both code paths.

### ASV Configuration

The repo root `asv.conf.json` is tailored to this GSTools checkout:

```json
{
  "repo": ".",
  "branches": ["main"],
  "benchmark_dir": "benchmarks",
  "env_dir": ".asv/env",
  "results_dir": ".asv/results",
  "html_dir": ".asv/html",
  "environment_type": "conda",
  "pythons": ["3.14"],
  "matrix": {
    "req": {
      "emcee": [""],
      "hankel": [""],
      "meshio": [""],
      "numpy": [""],
      "pyevtk": [""],
      "scipy": [""],
      "gstools-cython": [""]
    }
  },
  "install_command": [
    "in-dir={env_dir} python {conf_dir}/benchmarks/tools/install_pyproject_extras.py --pyproject {conf_dir}/pyproject.toml rust",
    "in-dir={env_dir} python -m pip install --no-deps {build_dir}"
  ],
  "number": 1,
  "repeat": 20
}
```

Important details:

- `environment_type: "conda"` means conda is required for the ASV workflow in
  this guide. ASV creates isolated conda environments for the commits it
  benchmarks.
- `pythons: ["3.14"]` means ASV creates Python 3.14 benchmark environments.
  Both ASV configs use Python 3.14 by default; change `pythons` in
  `asv.conf.json` and `asv.openmp.conf.json` only when intentionally
  validating another Python/GSTools backend stack.
- `number: 1` means ASV runs the benchmark code once per iteration.
- `repeat: 20` means the default baseline records 20 independent measurements
  per benchmark case.
- `matrix.req` asks ASV to install GSTools runtime dependencies before
  installing the checked-out GSTools source. It includes `gstools-cython`
  explicitly because the GSTools commit is installed with `--no-deps`.
- `{build_dir}` is ASV's temporary checkout/build directory for the exact
  GSTools commit being benchmarked.
- `install_command` installs the checked-out GSTools revision with `--no-deps`.
  It also uses `install_pyproject_extras.py` to install the `rust` extra from
  `pyproject.toml`, keeping the `gstools_core` requirement in one place.
- ASV still needs its own `install_command` because it creates isolated
  environments for the commits it benchmarks.
- In CI, `configure_asv_machine.py` replaces the interactive `asv machine
  --yes` step and writes a stable machine name (`github-actions-ubuntu`)
  regardless of the actual runner hardware. Locally, `asv machine --yes` is
  sufficient and `configure_asv_machine.py` is not needed.
- Run the cProfile helper with the Python executable from ASV's isolated
  environment, for example `.asv/env/<env-id>/bin/python`. In that mode, the
  ASV environment provides dependencies while the helper imports the current
  checkout through the repo `src/` path.

ASV creates these generated directories:

```text
.asv/env/      benchmark environments
.asv/results/  local benchmark result JSON files
.asv/html/     generated local benchmark website
```

Those directories are machine-specific generated artifacts. They should
normally stay out of git.

If needed, users can list more than one branch, Python version, benchmark
directory, and so on. For example:

```json
"branches": ["main", "my-feature-branch"]
```

Users can also benchmark any explicit branch, commit, tag, or range without
changing `asv.conf.json`:

```bash
asv run 'my-feature-branch^!' --bench benchmark_two_point_statistics
asv run main..my-feature-branch --bench benchmark_two_point_statistics
```

ASV checks out package code at each git commit being benchmarked. Commit source
changes before benchmarking them with ASV. Otherwise ASV may benchmark the last
committed package code rather than your uncommitted source changes.

### Benchmark Naming

ASV recognizes benchmark methods by name:

- methods starting with `time_` measure runtime
- methods starting with `peakmem_` measure peak memory
- `setup_cache()` creates reusable data once per benchmark environment
- `setup()` can skip or prepare individual parameter combinations

## Benchmark Coverage

This section describes what is measured by the ASV suite and how the benchmark
labels map to real GSTools workflows. The goal is to cover representative
operations that are relevant for geostatistical work, not isolated
micro-functions.

The current suite measures runtime and peak memory for variogram estimation,
global kriging, spatial random field generation, and conditioned random field
generation. Each workflow is run with both backends so the results can show
both absolute performance and Rust-vs-Cython differences.

### Shared Constants

```python
BACKENDS = ("cython_fallback", "rust_core")
THREAD_COUNTS = _configured_thread_counts()
VARIOGRAM_CASES = (
    "full_900",
    "sampled_5000_to_1500",
    "sampled_15000_to_4500",
)
KRIGE_CASES = ("small_30x500", "large_120x2000", "extra_large_360x6000")
FIELD_CASES = (
    "srf_unstructured_randmeth",
    "srf_structured_randmeth",
    "srf_structured_fourier",
    "condsrf_unstructured",
)
```

These constants define backend parameter values and the case/thread labels used
in generated benchmark method names.

`BACKENDS` compares:

- `cython_fallback`
- `rust_core`

`THREAD_COUNTS` defaults to:

- `threads_1`: force `gstools.config.NUM_THREADS = 1`

That is the default because the first benchmark target is a clean
Cython-vs-Rust backend comparison without parallelism.

Both constants are driven by environment variables read at import time:

| Variable | Default | Example override |
|---|---|---|
| `GSTOOLS_BENCHMARK_THREADS` | `"1"` | `"1,2,4,8"` — run with 1, 2, 4, and 8 threads |
| `GSTOOLS_BENCHMARK_BACKENDS` | all backends | `"rust_core"` — run Rust only |

These must be set before ASV imports the benchmark module. When running
locally with ASV, prefix the `asv run` command:

```bash
GSTOOLS_BENCHMARK_BACKENDS=rust_core asv run 'main^!' --bench benchmark_two_point_statistics
```

### Shared Helpers

`gstools_backend(use_core, num_threads)` temporarily forces GSTools to use
either the Cython fallback backend or the Rust `gstools_core` backend, and
sets `gstools.config.NUM_THREADS` for that benchmark run.

`_random_points(seed, count, scale)` creates deterministic 2D point clouds.

`_smooth_field(x, y)` creates deterministic synthetic values:

```python
np.sin(x / 10.0) + np.cos(y / 15.0)
```

`_make_variogram_data(...)` creates positions, field values, and bins for
variogram estimation.

`_make_krige_data(...)` creates conditioning points, conditioning values, and
target points for kriging and conditioned random fields.

The fixed random seeds are intentional. They keep benchmark inputs stable so
changes in results are more likely to come from code changes, not new random
data.

### Benchmark Classes

The ASV benchmarking is organized around workflow classes. Each workflow class
compares `cython_fallback` and `rust_core`, and each class includes both
runtime and peak-memory methods.

The suite currently measures:

- `VariogramBenchmarks`: full pairwise work vs sampled large work
- `KrigingBenchmarks`: small vs larger global kriging systems
- `RandomFieldBenchmarks`: unstructured SRF, structured SRF, Fourier
  SRF, and conditioned SRF

This keeps the ASV suite focused on representative workflows rather than
separate duplicate backend checks.

#### VariogramBenchmarks

This class measures variogram estimation cases:

```text
full_900
sampled_5000_to_1500
sampled_15000_to_4500
```

The labels mean:

- `full_900`: create 900 scattered points and use all 900 points for the
  variogram calculation.
- `sampled_5000_to_1500`: create 5,000 scattered points, then randomly select
  1,500 of those points for the variogram calculation.
- `sampled_15000_to_4500`: create 15,000 scattered points, then randomly select
  4,500 of those points for the variogram calculation.

The sampled cases still represent larger input datasets, but the variogram
calculation is done on the randomly selected subset so the pairwise work stays
practical.

#### KrigingBenchmarks

This class measures global kriging at three scales:

```text
small_30x500
large_120x2000
extra_large_360x6000
```

The labels mean:

- `small_30x500`: 30 conditioning points, 500 target points
- `large_120x2000`: 120 conditioning points, 2,000 target points
- `extra_large_360x6000`: 360 conditioning points, 6,000 target points

#### RandomFieldBenchmarks

This class measures SRF and CondSRF generation workflows:

```text
srf_unstructured_randmeth
srf_structured_randmeth
srf_structured_fourier
condsrf_unstructured
```

The cases are:

- `srf_unstructured_randmeth`: SRF using RandMeth on 2,000 unstructured points
- `srf_structured_randmeth`: SRF using RandMeth on a 64 by 64 structured grid
- `srf_structured_fourier`: SRF using the Fourier generator on a 64 by 64
  structured grid
- `condsrf_unstructured`: conditioned SRF with 40 conditioning points and 1,000
  target points

## Running The Benchmarks

### Baseline Benchmark

The baseline benchmark is the first result set to create before doing any
optimization work. It uses the default ASV configuration, so each workflow is
measured with `threads_1` for both `cython_fallback` and `rust_core`.

#### Main Branch Baseline

- Save a baseline for the latest local `main` commit:

```bash
asv run 'main^!' --bench benchmark_two_point_statistics
```

#### Several Commits Baseline

ASV can also compare several commits. This example runs the last three commits
on local `main`; choose a different range when that better matches your
branch.

- Run the last three commits on the local `main` branch:

```bash
asv run 'main~3..main' --bench benchmark_two_point_statistics
```

#### Summary of Results

After running ASV, inspect the explicit Rust-vs-Cython speedup ratios:

```bash
python benchmarks/tools/asv_speedup_summary.py
```

The helper reads `.asv/results/` and reports ratios per case and thread label:

```text
speedup = cython_fallback_time / rust_core_time
```

Interpret the ratio as:

- `speedup > 1.0` means Rust is faster
- `speedup = 1.0` means similar performance
- `speedup < 1.0` means Rust is slower

The speedup helper prints the backend ratio explicitly in the terminal. By
default, the helper skips removed legacy duplicate rows from older saved
results.

For a report that is easier to paste into a PR or issue:

```bash
python benchmarks/tools/asv_speedup_summary.py --format markdown
```

For a small standalone HTML report:

```bash
python benchmarks/tools/asv_speedup_summary.py --format html --output .asv/backend-report-overview.html
```

#### Visualization of Results

Build a self-contained interactive HTML report from the result files:

```bash
python benchmarks/tools/plot_case_backend_comparison.py
```

This reads `.asv/results/` by default and writes
`case-backend-comparison.html` next to it. Open the file in any browser. Use
`--metric time` or `--metric memory` to show one metric, and `--benchmark` to
filter to one family (e.g. `variogram`, `krige`, `srf`).

You can also inspect results in the native ASV browser report:

```bash
asv publish
asv preview
```

Then open the printed local URL, for example:

```text
http://127.0.0.1:8080/#/
```
(or any other `http://127.0.0.1:<port>/#/` URL shown by the running preview).

The native ASV report shows raw plots and trends. ASV does not draw a line
when there is only one x-axis point, so a single-commit run may show results
without useful trend graphs. The custom HTML report works with one or more
commits.

### Profiling With cProfile

`cProfile` does not update the ASV results shown in the browser report.
Instead, it prints a table in the terminal showing which Python
functions consumed time while one workflow ran.

The helper script is:

```text
benchmarks/tools/profile_benchmark_workflows.py
```

It imports the ASV benchmark classes from `benchmark_two_point_statistics.py`,
selects one case, forces one backend, and runs that case under `cProfile`.

Since ASV has already created an isolated Python environment, select that
environment to execute the profiling helper:

```bash
ASV_ENV="$(ls -td .asv/env/* | head -n 1)"
ASV_PYTHON="$ASV_ENV/bin/python"
```

On Windows PowerShell, use:

```powershell
$asvEnv = Get-ChildItem .asv\env -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
$asvPython = Join-Path $asvEnv.FullName 'python.exe'
```

The helper still profiles the current checkout because
`profile_benchmark_workflows.py` adds the repository `src/` directory to
`sys.path`. The ASV environment provides the installed dependencies, including
`gstools-cython` and `gstools_core`.

List available cases:

```bash
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --list
```

On Windows PowerShell, replace `"$ASV_PYTHON"` with `& $asvPython`.

All available cases:

```text
variogram-full          variogram on 900 points, full pairwise
variogram-sampled       variogram on 5 000 pts, sampled 1 500
variogram-extra-large   variogram on 15 000 pts, sampled 4 500
krige-small             kriging 30 → 500 pts
krige-large             kriging 120 → 2 000 pts
krige-extra-large       kriging 360 → 6 000 pts
srf-unstructured        SRF RandMeth, 2 000 unstructured pts
srf-structured          SRF RandMeth, 64×64 grid
srf-fourier             SRF Fourier, 64×64 grid
condsrf                 conditioned SRF, 40 → 1 000 pts
```

Profile a selected case:

```bash
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case variogram-sampled --backend rust_core --threads threads_1 --limit 10
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case variogram-extra-large --backend rust_core --threads threads_1 --limit 10
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case krige-large --backend rust_core --threads threads_1 --limit 10
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case krige-extra-large --backend rust_core --threads threads_1 --limit 10
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case srf-unstructured --backend rust_core --threads threads_1 --limit 10
"$ASV_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case condsrf --backend rust_core --threads threads_1 --limit 10
```

## Optional Parallelization with OpenMP

This section collects optional workflows for testing Cython and Rust with
several thread counts. OpenMP setup is platform-dependent, so each operating
system must be verified on the machine that produces the results.

The default setup above remains the recommended baseline: one thread, normal
ASV environment, and no extra OpenMP build steps. Use this section only when
you explicitly want to measure backend scaling with multiple thread counts.

### OpenMP ASV Configuration

Use one OpenMP ASV config on all supported desktop platforms:

```text
asv.openmp.conf.json
```

This config keeps the OpenMP experiment separate from the default baseline:

```text
.asv-openmp/env/
.asv-openmp/results/
.asv-openmp/html/
```

It uses the same `number` and `repeat` settings as the baseline config:

```json
"number": 1,
"repeat": 20
```

Each benchmark case records 20 independent measurements, running the benchmark
code once per measurement.

During ASV installation, it runs:

```bash
benchmarks/tools/install_openmp_cython.py
```

That helper compiles `gstools-cython` from source inside ASV's own
environment, not inside your active `gstools-benchmark` driver environment.
The helper sets `GSTOOLS_BUILD_PARALLEL=1` and then uses platform-specific
compiler handling:

- macOS: uses Apple clang through wrapper scripts and conda's `llvm-openmp`.
- Linux: uses the ASV conda compiler toolchain when available.
- Windows: uses native MSVC Build Tools.

### Run on macOS and Linux

Use these commands from a POSIX shell on macOS or Linux. On macOS, install
Xcode command-line tools first. On Linux, make sure the conda compiler packages
from `asv.openmp.conf.json` solve for your platform.

Create the ASV machine profile once:

```bash
asv --config asv.openmp.conf.json machine --yes
```

#### Verify Parallel Backends

Only interpret the Cython rows as OpenMP-enabled, and the Rust rows as
parallel-ready, after this check passes inside the `.asv-openmp/env/...`
environment.

On macOS and Linux:

```bash
ASV_OPENMP_ENV="$(ls -td .asv-openmp/env/* | head -n 1)"
"$ASV_OPENMP_ENV/bin/python" benchmarks/tools/check_backend_parallel_ready.py --verbose
```

On Windows PowerShell:

```powershell
$asvOpenmpEnv = Get-ChildItem .asv-openmp\env -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
$asvOpenmpPython = Join-Path $asvOpenmpEnv.FullName 'python.exe'
& $asvOpenmpPython benchmarks\tools\check_backend_parallel_ready.py --verbose
```

Expected passing output contains:

```text
Cython OpenMP readiness: PASS
Rust backend readiness: PASS with NUM_THREADS=2
```

If the check passes, run the last five commits:

```bash
GSTOOLS_BENCHMARK_THREADS=1,2,4,8 \
asv --config asv.openmp.conf.json run 'main~5..main' --bench benchmark_two_point_statistics --show-stderr
```

Print Rust-vs-Cython ratios from the OpenMP result folder:

```bash
python benchmarks/tools/asv_speedup_summary.py --results-dir .asv-openmp/results
```

Build a curated OpenMP HTML report:

```bash
python benchmarks/tools/asv_speedup_summary.py \
  --results-dir .asv-openmp/results \
  --format html \
  --output .asv-openmp/backend-report-openmp-overview.html
```

Build the case/backend comparison report:

```bash
python benchmarks/tools/plot_case_backend_comparison.py \
  --results-dir .asv-openmp/results \
  --max-commits 50 \
  --output .asv-openmp/case-backend-comparison.html
```

By default this report includes both time and peak-memory benchmarks. Use
`--metric time` or `--metric memory` when you want only one metric.
`--max-commits` limits only the generated custom report; it does not delete
raw ASV results. Use `--benchmark` to filter to one family:

```bash
python benchmarks/tools/plot_case_backend_comparison.py \
  --results-dir .asv-openmp/results \
  --benchmark variogram   # or krige, kriging, field, srf, condsrf, vario
  --output .asv-openmp/variogram-comparison.html
```

Build and preview the OpenMP browser report:

```bash
asv --config asv.openmp.conf.json publish
asv --config asv.openmp.conf.json preview
```

### Run on Windows

Use native Windows, not WSL, when you want Windows benchmark results. Install
Microsoft C++ Build Tools first, including the C++ build tools workload and a
Windows SDK. Run the commands from PowerShell or Anaconda Prompt after
activating the `gstools-benchmark` conda environment.

Create the ASV machine profile once:

```powershell
asv --config asv.openmp.conf.json machine --yes
```

Run a quick OpenMP smoke run for the latest local `main` commit:

```powershell
$env:GSTOOLS_BENCHMARK_THREADS = '1,2,4,8'
asv --config asv.openmp.conf.json run 'main^!' --quick --bench benchmark_two_point_statistics --show-stderr
```

Verify the parallel backends with the PowerShell commands in
[Verify Parallel Backends](#verify-parallel-backends). If the check passes,
run the last five commits:

```powershell
$env:GSTOOLS_BENCHMARK_THREADS = '1,2,4,8'
asv --config asv.openmp.conf.json run 'main~5..main' --bench benchmark_two_point_statistics --show-stderr
```

Print Rust-vs-Cython ratios from the OpenMP result folder:

```powershell
python benchmarks\tools\asv_speedup_summary.py --results-dir .asv-openmp\results
```

Build a curated OpenMP HTML report:

```powershell
python benchmarks\tools\asv_speedup_summary.py `
  --results-dir .asv-openmp\results `
  --format html `
  --output .asv-openmp\backend-report-openmp-overview.html
```

Build the case/backend comparison report:

```powershell
python benchmarks\tools\plot_case_backend_comparison.py `
  --results-dir .asv-openmp\results `
  --output .asv-openmp\case-backend-comparison.html
```

By default this report includes both time and peak-memory benchmarks. Use
`--metric time` or `--metric memory` when you want only one metric, and
`--benchmark` to filter to one family (e.g. `variogram`, `krige`, `field`,
`srf`, `condsrf`).

Build and preview the OpenMP browser report:

```powershell
asv --config asv.openmp.conf.json publish
asv --config asv.openmp.conf.json preview
```

When finished, clear the thread-count override if the PowerShell session will
be reused:

```powershell
Remove-Item Env:\GSTOOLS_BENCHMARK_THREADS
```

### OpenMP Thread Rule

The `GSTOOLS_BENCHMARK_THREADS=1,2,4,8` setting only tells the benchmark code to
run the same workflows with different `gstools.config.NUM_THREADS` values. It
does not, by itself, make the Cython backend parallel.

For Cython OpenMP scaling, `gstools-cython` must be compiled with OpenMP inside
the same `.asv-openmp/env/...` environment that runs the benchmark. The
`asv.openmp.conf.json` install command does that through
`benchmarks/tools/install_openmp_cython.py`; the commands in
[Verify Parallel Backends](#verify-parallel-backends) are the proof that the
environment is ready. If that check fails, the benchmark may still run, but the
Cython rows should not be interpreted as OpenMP-enabled Cython results.

### HPC Notes

The `asv.openmp.conf.json` workflow is intended for local macOS, Linux, and
native Windows machines. Managed HPC systems often use custom compiler modules,
MPI/OpenMP runtimes, scheduler pinning, and CPU affinity rules. Use the same
validation rule there: only interpret Cython as OpenMP-enabled if
`check_backend_parallel_ready.py --verbose` passes inside the exact ASV
environment used for the benchmark run.

### Profiling With cProfile for Multiple Threads

To profile how a workflow changes across configured thread counts, run the
same cProfile case several times with the OpenMP ASV environment:

```bash
ASV_OPENMP_ENV="$(ls -td .asv-openmp/env/* | head -n 1)"
ASV_OPENMP_PYTHON="$ASV_OPENMP_ENV/bin/python"

for threads in threads_1 threads_2 threads_4 threads_8; do
  "$ASV_OPENMP_PYTHON" benchmarks/tools/profile_benchmark_workflows.py --case krige-extra-large --backend rust_core --threads "$threads" --limit 10
done
```

On Windows PowerShell:

```powershell
$asvOpenmpEnv = Get-ChildItem .asv-openmp\env -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
$asvOpenmpPython = Join-Path $asvOpenmpEnv.FullName 'python.exe'

foreach ($threads in 'threads_1', 'threads_2', 'threads_4', 'threads_8') {
  & $asvOpenmpPython benchmarks\tools\profile_benchmark_workflows.py --case krige-extra-large --backend rust_core --threads $threads --limit 10
}
```

Useful options:

- `--case`: choose one workflow, or use `all`
- `--backend`: choose `cython_fallback` or `rust_core`
- `--threads`: choose `threads_1` for a baseline profile, or `threads_2`,
  `threads_4`, or `threads_8` for parallel profiles
- `--limit`: number of function rows to print from the cProfile table
- `--sort cumtime`: sort by cumulative time, usually the best first view
- `--sort tottime`: sort by time spent directly in each function
- `--repeat`: repeat a workflow inside the profiler

For example, `--limit 10` means "print the top 10 function rows after sorting".

## More ASV Commands

Save results for only the latest local `main` commit:

```bash
asv run 'main^!' --bench benchmark_two_point_statistics
```

Compare the latest local `main` commit with the previous local `main` commit:

```bash
asv run 'main~1^!' --bench benchmark_two_point_statistics
asv run 'main^!' --bench benchmark_two_point_statistics
asv compare main~1 main
```

Compare local `main` with the latest remote `main`:

```bash
asv run 'main^!' --bench benchmark_two_point_statistics
git fetch origin main
asv run 'origin/main^!' --bench benchmark_two_point_statistics
asv compare origin/main main
```

Compare the previous local `main` commit with the latest remote `main`:

```bash
git fetch origin main
asv run 'main~1^!' --bench benchmark_two_point_statistics
asv run 'origin/main^!' --bench benchmark_two_point_statistics
asv compare main~1 origin/main
```

Run the last three commits on local `main`:

```bash
asv run 'main~3..main' --bench benchmark_two_point_statistics --show-stderr
```

Run the last three commits on the latest remote `main`:

```bash
git fetch origin main
asv run 'origin/main~3..origin/main' --bench benchmark_two_point_statistics --show-stderr
```

On a linear branch, `main~5..main` benchmarks:

```text
main~4
main~3
main~2
main~1
main
```

Run a selected list of commits:

```bash
git rev-parse main main~3 origin/main e20c88f7 > /tmp/gstools-asv-commits.txt
asv run HASHFILE:/tmp/gstools-asv-commits.txt --bench benchmark_two_point_statistics
```

Use full commit hashes when sharing results. Short hashes and branch names are
fine locally but can become ambiguous later.

If running ASV from outside the repo root, pass the config explicitly:

```bash
asv --config /path/to/MPS-Tools/GSTools/asv.conf.json run --quick --bench benchmark_two_point_statistics
```

## External Reference

For complete ASV command syntax, see:

```text
https://asv.readthedocs.io/en/stable/commands.html
```
