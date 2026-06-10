#!/usr/bin/env python
"""Install GSTools-Cython with OpenMP inside an ASV environment.

This helper is called by ``asv.openmp.conf.json`` after ASV has created a
conda environment with the common build requirements. It is kept for local or
manual OpenMP ASV runs. GitHub Actions uses a faster readiness check with the
conda-forge ``gstools-cython`` package instead of building ``gstools-cython``
from source. This helper keeps the ASV config portable and handles
platform-specific compiler/OpenMP details here.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def run(command, env=None, check=True):
    """Run a subprocess command and echo it for ASV logs."""
    print("+ " + " ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, check=check, env=env)


def prepend_path(build_env, *paths):
    """Prepend existing paths to a build environment's ``PATH``."""
    current = build_env.get("PATH", "")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        build_env["PATH"] = os.pathsep.join(existing + [current])


def first_match(directory, patterns):
    """Return the first file matching one of the glob patterns."""
    for pattern in patterns:
        matches = sorted(
            path for path in directory.glob(pattern) if path.is_file()
        )
        if matches:
            return matches[0]
    return None


def require_file(path, message):
    """Return true when ``path`` exists, otherwise print ``message``."""
    if not path.exists():
        print(message, file=sys.stderr)
        return False
    return True


def conda_executable():
    """Find the conda executable used to amend ASV environments."""
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda is None:
        print("No conda executable was found.", file=sys.stderr)
    return conda


def conda_install(env_dir, *packages):
    """Install conda-forge packages into an ASV environment directory."""
    conda = conda_executable()
    if conda is None:
        return False
    run(
        [
            conda,
            "install",
            "-y",
            "-p",
            str(env_dir),
            "-c",
            "conda-forge",
            *packages,
        ]
    )
    return True


def ensure_macos_llvm_openmp(env_dir):
    """Ensure that macOS ASV environments contain ``llvm-openmp``."""
    include_dir = env_dir / "include"
    lib_dir = env_dir / "lib"
    omp_header = include_dir / "omp.h"
    omp_lib = lib_dir / "libomp.dylib"

    if omp_header.exists() and omp_lib.exists():
        return True

    if not conda_install(env_dir, "llvm-openmp"):
        print(
            "llvm-openmp is required on macOS but was not found in the ASV "
            "environment, and it could not be installed with conda.",
            file=sys.stderr,
        )
        return False

    return require_file(
        omp_header,
        f"llvm-openmp install did not create expected header: {omp_header}",
    ) and require_file(
        omp_lib,
        f"llvm-openmp install did not create expected library: {omp_lib}",
    )


def ensure_linux_libgomp(env_dir):
    """Ensure that Linux ASV environments contain an OpenMP runtime."""
    lib_dir = env_dir / "lib"
    if first_match(lib_dir, ["libgomp.so*", "libomp.so*"]) is not None:
        return True

    if not conda_install(env_dir, "libgomp"):
        print(
            "libgomp is required on Linux but could not be installed with conda.",
            file=sys.stderr,
        )
        return False

    if first_match(lib_dir, ["libgomp.so*", "libomp.so*"]) is not None:
        return True

    print(
        f"libgomp install did not create an OpenMP runtime under {lib_dir}.",
        file=sys.stderr,
    )
    return False


def write_macos_wrapper(path, force_cxx=False):
    """Write a clang wrapper that maps ``-fopenmp`` to Apple clang flags."""
    text = """#!/bin/bash
set -e
prefix="${GSTOOLS_OPENMP_PREFIX:-${CONDA_PREFIX:-}}"
name="$(basename "$0")"
if [[ "${GSTOOLS_FORCE_CXX:-0}" == "1" || "$name" == *++* ]]; then
  real="${GSTOOLS_REAL_CXX:-/usr/bin/clang++}"
else
  real="${GSTOOLS_REAL_CC:-/usr/bin/clang}"
fi
is_compile=0
for arg in "$@"; do
  [[ "$arg" == "-c" ]] && is_compile=1
done
args=()
for arg in "$@"; do
  if [[ "$arg" == "-fopenmp" ]]; then
    if [[ "$is_compile" == "1" ]]; then
      args+=("-Xpreprocessor" "-fopenmp" "-I${prefix}/include")
    else
      args+=("-L${prefix}/lib" "-lomp" "-Wl,-rpath,${prefix}/lib")
    fi
  else
    args+=("$arg")
  fi
done
exec "$real" "${args[@]}"
"""
    if force_cxx:
        text = """#!/bin/bash
GSTOOLS_FORCE_CXX=1 exec "$(dirname "$0")/gstools-asv-clang-openmp" "$@"
"""
    path.write_text(text, encoding="utf8")
    path.chmod(
        path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )


def macos_build_env(env_dir):
    """Build environment variables for macOS OpenMP compilation."""
    if not ensure_macos_llvm_openmp(env_dir):
        return None

    bin_dir = env_dir / "bin"
    include_dir = env_dir / "include"
    lib_dir = env_dir / "lib"
    cc_wrapper = bin_dir / "gstools-asv-clang-openmp"
    cxx_wrapper = bin_dir / "gstools-asv-clang-openmp++"
    write_macos_wrapper(cc_wrapper)
    write_macos_wrapper(cxx_wrapper, force_cxx=True)

    build_env = os.environ.copy()
    prepend_path(build_env, bin_dir)
    build_env.update(
        {
            "GSTOOLS_BUILD_PARALLEL": "1",
            "GSTOOLS_OPENMP_PREFIX": str(env_dir),
            "CC": str(cc_wrapper),
            "CXX": str(cxx_wrapper),
            "CFLAGS": f"-I{include_dir}",
            "LDFLAGS": f"-L{lib_dir}",
        }
    )
    return build_env


def linux_build_env(env_dir):
    """Build environment variables for Linux OpenMP compilation."""
    if not ensure_linux_libgomp(env_dir):
        return None

    bin_dir = env_dir / "bin"
    cc = first_match(
        bin_dir,
        [
            "*-conda-linux-gnu-cc",
            "*-conda-linux-gnu-gcc",
            "gcc",
            "cc",
        ],
    )
    cxx = first_match(
        bin_dir,
        [
            "*-conda-linux-gnu-c++",
            "*-conda-linux-gnu-g++",
            "g++",
            "c++",
        ],
    )

    build_env = os.environ.copy()
    prepend_path(build_env, bin_dir)
    build_env["GSTOOLS_BUILD_PARALLEL"] = "1"
    if cc is not None:
        build_env["CC"] = str(cc)
    if cxx is not None:
        build_env["CXX"] = str(cxx)
    return build_env


def windows_build_env(env_dir):
    """Build environment variables for native Windows OpenMP compilation."""
    build_env = os.environ.copy()
    prepend_path(
        build_env,
        env_dir,
        env_dir / "Scripts",
        env_dir / "Library" / "bin",
    )
    build_env["GSTOOLS_BUILD_PARALLEL"] = "1"
    return build_env


def install_gstools_cython(build_env):
    """Reinstall GSTools-Cython from source with OpenMP enabled."""
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "gstools-cython",
            "gstools_cython",
        ],
        env=build_env,
        check=False,
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-cache-dir",
            "--force-reinstall",
            "--no-binary=gstools-cython",
            "--no-deps",
            "gstools-cython",
        ],
        env=build_env,
    )


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "env_dir",
        nargs="?",
        default=sys.prefix,
        help="ASV or active conda environment directory. Defaults to sys.prefix.",
    )
    return parser.parse_args()


def main():
    """Prepare the platform build environment and install GSTools-Cython."""
    args = parse_args()
    env_dir = Path(args.env_dir).resolve()
    system = platform.system()
    print(f"Preparing GSTools-Cython OpenMP build for {system}")

    if system == "Darwin":
        build_env = macos_build_env(env_dir)
    elif system == "Linux":
        build_env = linux_build_env(env_dir)
    elif system == "Windows":
        build_env = windows_build_env(env_dir)
    else:
        print(
            f"Unsupported platform for OpenMP ASV build: {system}",
            file=sys.stderr,
        )
        return 2

    if build_env is None:
        return 2

    install_gstools_cython(build_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
