"""MPS is Rust-only: it requires gstools_core>=1.4.0 with the MPS exports.

Replaces the old ``test_mps_optional_core.py`` (which asserted the now-removed
pure-Python fallback). Here we assert the inverse contract: importing the MPS
core validator raises a clear error when the Rust core is missing or too old.
"""

import importlib
import sys
import types

import pytest

_MSG = r"gstools-core>=1\.4\.0 with MPS support"


def _expect_core_raises(fake_core):
    """Reload gstools.mps._core against *fake_core* and require a clear error."""
    saved = {
        name: sys.modules.get(name)
        for name in ("gstools_core", "gstools.mps._core")
    }
    try:
        sys.modules.pop("gstools.mps._core", None)
        sys.modules["gstools_core"] = fake_core
        with pytest.raises(ImportError, match=_MSG):
            importlib.import_module("gstools.mps._core")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_mps_core_required_when_missing():
    """No gstools_core installed -> clear ImportError."""
    # A ``None`` entry in sys.modules makes ``import gstools_core`` raise.
    _expect_core_raises(None)


def test_mps_core_required_with_mps_exports():
    """An older gstools_core (released 1.3.0, no MPS exports) -> clear error.

    This is the important case: pip could resolve the published 1.3.0 wheel,
    which predates every MPS export.
    """
    old = types.ModuleType("gstools_core")
    old.__version__ = "1.3.0"
    _expect_core_raises(old)
