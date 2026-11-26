"""Port-contract guards for the Rust-only MPS backend.

These lock in the invariants that the port of Niklas's features
must hold across the Python and native integration:

1. the deleted Python scheduler modules stay gone,
2. ``DirectSampling.__call__`` exposes no ``progress`` hook (there is no
   per-node Python callback path anymore), and
3. a single ``DirectSampling`` call runs exactly one complete Rust-engine
   pass (no hidden Python scheduler loop, no extra passes).

They are deliberately cheap and import-light so they act as a fast tripwire
if a future feature accidentally reintroduces a Python numerical path.
"""

import importlib
import inspect

import numpy as np

import gstools as gs
from gstools.mps import simulate


def test_python_scheduler_modules_are_gone():
    """scan.py / runner.py / data_event.py must not be importable."""
    for module in ("scan", "runner", "data_event"):
        try:
            importlib.import_module(f"gstools.mps.{module}")
        except ModuleNotFoundError:
            continue
        raise AssertionError(
            f"gstools.mps.{module} is importable; the Python scheduler must "
            "stay removed on the Rust-only backend."
        )


def test_direct_sampling_has_no_progress_hook():
    """The per-node ``progress`` callback path was removed with the scheduler."""
    params = inspect.signature(gs.DirectSampling.__call__).parameters
    assert "progress" not in params, (
        "DirectSampling.__call__ must not expose a 'progress' hook; there is "
        "no Python per-node loop to drive it."
    )


def test_single_rust_engine_pass_per_call():
    """One DirectSampling call == exactly one complete Rust-engine pass."""
    captured = []
    simulate._MPS_RUST_ENGINE_STATS_HOOK = captured.append
    try:
        data = (np.indices((20, 20)).sum(axis=0) % 2).astype(float)
        ti = gs.TrainingImage(data, n_neighbors=8)
        model = gs.MPSModel(ti, scan_fraction=0.3, threshold=0.0)
        gs.DirectSampling(model, seed=1)(
            [np.arange(16.0), np.arange(16.0)], store=False
        )
    finally:
        simulate._MPS_RUST_ENGINE_STATS_HOOK = None
    assert len(captured) == 1, (
        f"expected exactly one complete Rust-engine pass, saw {len(captured)}"
    )
