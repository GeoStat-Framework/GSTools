# Direct Sampling maintenance tools

These source-only entry points are included in the source archive, not installed
in the runtime wheel. Run them with a Python environment containing GSTools.

`validate_ds_pairing.py --core ../GSTools-Core --out /absolute/path/new-ds-pair`
builds release wheels with Cargo/Rust, Maturin, and `python -m build`, then
installs the two explicit artifacts and test dependencies into a new venv.
The output must be fresh and have no symlink path components. The default output is a new system temporary directory.
Each step has a bounded timeout (`--timeout`, seconds), logs, and a running,
failed, or passed JSON manifest. Import origins and native/artifact hashes are
recorded after resolution; `pip check`, native smoke, child-process boundary
checks, and frontend MPS tests must all pass for exit zero.

Use `--maturin PATH` to select the build executable, `--constraints FILE` for an
approved dependency profile, and `--wheelhouse DIR` for offline resolution
(`--no-index`). Build dependencies must also be available to the build frontend.
`--no-build` reuses exactly one wheel per project under an owned output's
`wheels/`, only when no venv exists. Copy wheels into a newly owned directory
or retain a failed pre-install run; arbitrary environments are never deleted.
Source association is unknown unless a matching hash is recorded in
`build-provenance.json`. `--system-site` is a non-isolated convenience profile:
inherited dependencies do not establish an isolated dependency-floor gate.
The validator is manually invoked; its existence does not establish hosted CI,
platform support, or publication approval.

`validate_mps_ensembles.py --count 1 --out /tmp/new-ds-study` uses the current
interpreter and requires optional Matplotlib and SciPy tool dependencies.
It writes design/provenance, raw fields, per-realization metrics, summaries, and
plots to a new directory, refusing to overwrite an existing study. Included
synthetic cases are illustrations, not certification for every TI/application.
Exit zero means the diagnostic completed, not scientific equivalence. Diagnostic
tolerances are not calibrated scientific acceptance margins. Realizations,
rather than spatially dependent pixels, are the units for Monte Carlo summaries.
`--count 1` is only a smoke profile; sample-variance intervals are unavailable.
The default count is 100; larger studies are explicitly requested work.
