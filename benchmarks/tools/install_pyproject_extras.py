#!/usr/bin/env python
"""Install selected optional dependencies declared in ``pyproject.toml``.

This installs only the requirements listed under the requested
``[project.optional-dependencies]`` extras. It does not install the project or
its regular dependencies, so ASV can keep control of its environment matrix.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "extras",
        nargs="+",
        help="Optional dependency extras to install.",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml. Default: ./pyproject.toml",
    )
    return parser.parse_args()


def extra_requirements(pyproject, extras):
    """Return deduplicated requirements for named pyproject extras."""
    with pyproject.open("rb") as handle:
        config = tomllib.load(handle)

    optional = config.get("project", {}).get("optional-dependencies", {})
    missing = sorted(set(extras) - set(optional))
    if missing:
        available = ", ".join(sorted(optional)) or "none"
        names = ", ".join(missing)
        raise ValueError(
            f"Unknown optional dependency extras: {names}. "
            f"Available extras: {available}"
        )

    requirements = []
    for extra in extras:
        for requirement in optional[extra]:
            if requirement not in requirements:
                requirements.append(requirement)
    return requirements


def install_requirements(requirements):
    """Install requirements with pip in the current Python environment."""
    if not requirements:
        print("No requirements to install.", flush=True)
        return
    command = [sys.executable, "-m", "pip", "install", *requirements]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    """Install the requested optional dependency extras."""
    args = parse_args()
    try:
        requirements = extra_requirements(args.pyproject, args.extras)
    except (FileNotFoundError, ValueError) as err:
        print(err, file=sys.stderr)
        return 2
    try:
        install_requirements(requirements)
    except subprocess.CalledProcessError as err:
        print(f"pip install failed with exit code {err.returncode}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
