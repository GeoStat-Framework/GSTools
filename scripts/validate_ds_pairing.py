"""Build and validate an exact GSTools/GSTools-Core wheel pair.

Install selected artifacts in a disposable environment, verify import origins,
and run native smoke, boundary, and frontend integration checks. Record artifact
hashes, source provenance, and step outcomes. This manually invoked tool does
not publish artifacts or certify scientific equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

OWNER = "gstools-ds-pairing-v1"


def isolated_env():
    """Apply the same isolation policy to builds, tests, and child processes."""
    env = dict(os.environ)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
    ):
        env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        MPLBACKEND="Agg",
    )
    return env


def run(cmd, log, env=None, cwd=None, timeout=1800):
    """Preserve command output, including timeouts, and propagate failure."""
    cmd = [str(c) for c in cmd]
    with Path(log).open("ab") as handle:
        handle.write(("+ " + " ".join(cmd) + "\n").encode())
        handle.flush()
        try:
            subprocess.run(
                cmd,
                env=env or isolated_env(),
                cwd=cwd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired:
            handle.write(f"\nTIMEOUT after {timeout}s\n".encode())
            raise


def git(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            env=isolated_env(),
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one_wheel(directory, pattern):
    hits = sorted(Path(directory).glob(pattern))
    if len(hits) != 1 or not hits[0].is_file() or hits[0].is_symlink():
        raise ValueError(f"expected exactly one regular {pattern}: {hits}")
    return hits[0]


def safe_output(path, reuse=False):
    """Never delete an environment; only reuse tool-owned wheel directories."""
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlink output path is unsafe: {path}")
    for name in (
        "venv",
        "wheels",
        "logs",
        "work",
        ".pairing-owner",
        "manifest.json",
        "build-provenance.json",
        "origins.json",
    ):
        if (path / name).is_symlink():
            raise ValueError(f"symlink output child is unsafe: {path / name}")
    if path.exists():
        marker = path / ".pairing-owner"
        if not reuse or not marker.is_file() or marker.read_text() != OWNER:
            raise ValueError(f"choose a fresh output directory: {path}")
        if (path / "venv").exists():
            raise ValueError(
                "existing venv is never removed; use a fresh output"
            )
    else:
        path.mkdir(parents=True)
        (path / ".pairing-owner").write_text(OWNER)
    return path


def install_command(python, core, frontend, wheelhouse=None, constraints=None):
    """Resolve dependencies and both exact project wheels in one transaction."""
    cmd = [
        python,
        "-m",
        "pip",
        "--isolated",
        "install",
        core,
        frontend,
        "pytest",
        "matplotlib>=3.7",
    ]
    if wheelhouse:
        cmd += ["--no-index", "--find-links", wheelhouse]
    if constraints:
        cmd += ["--constraint", constraints]
    return cmd


def reuse_provenance(wheel, recorded):
    """Associate reused artifacts with source only when their hash is recorded."""
    if recorded and recorded.get("sha256") == sha256(wheel):
        return recorded.get("source", "unknown")
    return "unknown"


def write_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True, type=Path)
    parser.add_argument(
        "--frontend", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="fresh directory; default: new system temp dir",
    )
    parser.add_argument("--maturin", default=shutil.which("maturin"))
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="reuse wheels in an owned output without a venv",
    )
    parser.add_argument(
        "--system-site",
        action="store_true",
        help="non-isolated convenience profile: inherit deps",
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="offline dependency wheels; disables index access",
    )
    parser.add_argument("--constraints", type=Path)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="seconds per build/install/test step",
    )
    args = parser.parse_args(argv)
    core, frontend = args.core.resolve(), args.frontend.resolve()
    for repo, files in (
        (
            core,
            (
                "Cargo.toml",
                "scripts/check_mps_wheel.py",
                "scripts/check_mps_boundaries.py",
            ),
        ),
        (frontend, ("pyproject.toml", "tests")),
    ):
        for name in files:
            if not (repo / name).exists():
                parser.error(f"missing checkout prerequisite: {repo / name}")
    for name in ("wheelhouse", "constraints"):
        value = getattr(args, name)
        if value and not (
            value.is_dir() if name == "wheelhouse" else value.is_file()
        ):
            parser.error(f"invalid --{name}: {value}")
    args.wheelhouse = args.wheelhouse.resolve() if args.wheelhouse else None
    args.constraints = args.constraints.resolve() if args.constraints else None
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.no_build:
        for tool in ("cargo", "rustc"):
            if not shutil.which(tool):
                parser.error(f"missing build tool: {tool}")
        if not args.maturin or not shutil.which(args.maturin):
            parser.error("missing build tool: maturin; pass --maturin PATH")
        args.maturin = str(Path(shutil.which(args.maturin)).resolve())
        if importlib.util.find_spec("build") is None:
            parser.error("missing Python build tool: python -m build")
    # A fresh default directory has no owner data to replace.
    default_output = args.out is None
    if default_output:
        args.out = Path(tempfile.mkdtemp(prefix="gstools-ds-pair-")).resolve()
        (args.out / ".pairing-owner").write_text(OWNER)
    for repo in (frontend, core):
        if args.out.resolve() == repo or repo in args.out.resolve().parents:
            parser.error("output must be outside source checkouts")
    out = safe_output(args.out, reuse=args.no_build or default_output)
    wheels, logs = out / "wheels", out / "logs"
    wheels.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    manifest = {
        "status": "running",
        "steps": {},
        "dependency_profile": "inherited" if args.system_site else "isolated",
        "dependency_floor_gate": False,
        "checkouts": {
            name: {
                "path": str(repo),
                "head": git(repo, "rev-parse", "HEAD"),
                "status": git(repo, "status", "--porcelain"),
            }
            for name, repo in (("core", core), ("frontend", frontend))
        },
    }
    manifest_path = out / "manifest.json"
    write_manifest(manifest_path, manifest)

    def step(name, cmd):
        manifest["current_step"] = name
        manifest["steps"][name] = "running"
        write_manifest(manifest_path, manifest)
        run(cmd, logs / (name + ".log"), cwd=work, timeout=args.timeout)
        manifest["steps"][name] = "passed"
        write_manifest(manifest_path, manifest)

    try:
        build_record_path = out / "build-provenance.json"
        if not args.no_build:
            step(
                "core-build",
                [
                    args.maturin,
                    "build",
                    "--release",
                    "--locked",
                    "--interpreter",
                    sys.executable,
                    "-m",
                    core / "Cargo.toml",
                    "-o",
                    wheels,
                ],
            )
            step(
                "frontend-build",
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "-o",
                    wheels,
                    frontend,
                ],
            )
        cw = one_wheel(wheels, "gstools_core-*.whl")
        fw = one_wheel(wheels, "gstools-*.whl")
        recorded = (
            json.loads(build_record_path.read_text())
            if (args.no_build and build_record_path.is_file())
            else {}
        )
        manifest["artifacts"] = {}
        for name, wheel in (("core", cw), ("frontend", fw)):
            source = (
                reuse_provenance(wheel, recorded.get(name))
                if args.no_build
                else manifest["checkouts"][name]
            )
            manifest["artifacts"][name] = dict(
                path=str(wheel), sha256=sha256(wheel), source=source
            )
        if not args.no_build:
            write_manifest(build_record_path, manifest["artifacts"])
        env_dir = out / "venv"
        manifest["current_step"] = "venv"
        write_manifest(manifest_path, manifest)
        venv.EnvBuilder(
            with_pip=True, system_site_packages=args.system_site
        ).create(env_dir)
        vpy = env_dir / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        step(
            "install",
            install_command(vpy, cw, fw, args.wheelhouse, args.constraints),
        )
        step("pip-check", [vpy, "-m", "pip", "--isolated", "check"])
        # Native modules may be exported through a package __init__.
        origins_code = """
import hashlib, importlib, json, pathlib, sys
import gstools, gstools_core
root = pathlib.Path(sys.prefix).resolve()
modules = {}
for name, module in [('frontend', gstools), ('core', gstools_core)]:
    path = pathlib.Path(module.__file__).resolve()
    assert root in path.parents, path
    modules[name] = dict(path=str(path), version=module.__version__)
native = importlib.import_module(gstools_core.mps_simulate.__module__)
path = pathlib.Path(native.__file__).resolve()
assert root in path.parents, path
modules['native'] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
assert gstools_core.mps_api_version() == 8
pathlib.Path(sys.argv[1]).write_text(json.dumps(modules, indent=2))
"""
        step("origins", [vpy, "-c", origins_code, out / "origins.json"])
        manifest["imports"] = json.loads((out / "origins.json").read_text())
        step("smoke", [vpy, core / "scripts/check_mps_wheel.py"])
        step(
            "boundaries",
            [
                vpy,
                core / "scripts/check_mps_boundaries.py",
                "--output",
                out / "boundaries.json",
            ],
        )
        step(
            "mps-tests",
            [
                vpy,
                "-m",
                "pytest",
                frontend / "tests",
                "-k",
                "mps",
                "-q",
                "--junitxml",
                out / "mps-junit.xml",
            ],
        )
    except BaseException as exc:
        manifest["status"] = "failed"
        failed = manifest.get("current_step", "artifact-selection")
        manifest["failed_step"] = failed
        manifest["steps"][failed] = "failed"
        manifest["error"] = str(exc)
        write_manifest(manifest_path, manifest)
        raise
    manifest["status"] = "passed"
    manifest.pop("current_step", None)
    write_manifest(manifest_path, manifest)
    print(f"PAIRED DS VALIDATION: PASS — {manifest_path}")


if __name__ == "__main__":
    main()
