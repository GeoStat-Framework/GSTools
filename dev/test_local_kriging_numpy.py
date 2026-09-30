"""
Validation script for the pure NumPy local-kriging draft.

Checks the core assumption of local_kriging_numpy.local_krige: with a
``local_radius`` big enough to cover the whole domain (so every
conditioning point is a "neighbor" of every target point), it must
reproduce global kriging exactly (up to floating point noise), for
Simple, Ordinary and Universal kriging, with and without
anisotropy/rotation.

Run with:
    uv run pytest dev/test_local_kriging_numpy.py -v
"""

import numpy as np
import pytest

import gstools as gs
from local_kriging_numpy import local_krige

# radius comfortably bigger than any pairwise distance the test domains
# below can produce -> every conditioning point is always a neighbor
FULL_RADIUS = 1e6


def make_cond(dim, seed=1):
    rng = np.random.RandomState(seed)
    n = 20
    cond_pos = [rng.uniform(0, 10, n) for _ in range(dim)]
    cond_val = rng.uniform(-1, 1, n)
    return cond_pos, cond_val


def make_targets(dim, seed=2):
    rng = np.random.RandomState(seed)
    n = 30
    return [rng.uniform(-2, 12, n) for _ in range(dim)]


MODELS = [gs.Gaussian, gs.Exponential, gs.Matern]
DIMS = [1, 2, 3]


@pytest.mark.parametrize("Model", MODELS)
@pytest.mark.parametrize("dim", DIMS)
def test_simple_matches_global(Model, dim):
    cond_pos, cond_val = make_cond(dim)
    pos = make_targets(dim)
    model = Model(dim=dim, var=1.5, len_scale=3, nugget=0.1)
    krige = gs.krige.Simple(model, cond_pos, cond_val, mean=0.3)

    field_g, var_g = krige(pos, return_var=True, post_process=False, store=False)
    field_l, var_l = local_krige(krige, pos, local_radius=FULL_RADIUS)

    np.testing.assert_allclose(field_l, field_g, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(var_l, var_g, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("Model", MODELS)
@pytest.mark.parametrize("dim", DIMS)
def test_ordinary_matches_global(Model, dim):
    cond_pos, cond_val = make_cond(dim)
    pos = make_targets(dim)
    model = Model(dim=dim, var=1.5, len_scale=3, nugget=0.1)
    krige = gs.krige.Ordinary(model, cond_pos, cond_val)

    field_g, var_g = krige(pos, return_var=True, post_process=False, store=False)
    field_l, var_l = local_krige(krige, pos, local_radius=FULL_RADIUS)

    np.testing.assert_allclose(field_l, field_g, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(var_l, var_g, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("Model", MODELS)
@pytest.mark.parametrize("dim", DIMS)
def test_universal_matches_global(Model, dim):
    cond_pos, cond_val = make_cond(dim)
    pos = make_targets(dim)
    model = Model(dim=dim, var=1.5, len_scale=3, nugget=0.1)
    krige = gs.krige.Universal(model, cond_pos, cond_val, drift_functions="linear")

    field_g, var_g = krige(pos, return_var=True, post_process=False, store=False)
    field_l, var_l = local_krige(krige, pos, local_radius=FULL_RADIUS)

    np.testing.assert_allclose(field_l, field_g, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(var_l, var_g, atol=1e-6, rtol=1e-6)


def test_anisotropic_rotated_matches_global():
    dim = 2
    cond_pos, cond_val = make_cond(dim)
    pos = make_targets(dim)
    model = gs.Gaussian(
        dim=dim, var=2.0, len_scale=3, anis=0.5, angles=0.7, nugget=0.05
    )
    krige = gs.krige.Ordinary(model, cond_pos, cond_val)

    field_g, var_g = krige(pos, return_var=True, post_process=False, store=False)
    field_l, var_l = local_krige(krige, pos, local_radius=FULL_RADIUS)

    np.testing.assert_allclose(field_l, field_g, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(var_l, var_g, atol=1e-8, rtol=1e-6)


def test_exact_matches_global():
    dim = 2
    cond_pos, cond_val = make_cond(dim)
    pos = make_targets(dim)
    model = gs.Gaussian(dim=dim, var=2.0, len_scale=3, nugget=0.2)
    krige = gs.krige.Ordinary(model, cond_pos, cond_val, exact=True)

    field_g, var_g = krige(pos, return_var=True, post_process=False, store=False)
    field_l, var_l = local_krige(krige, pos, local_radius=FULL_RADIUS)

    np.testing.assert_allclose(field_l, field_g, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(var_l, var_g, atol=1e-8, rtol=1e-6)


def test_reproduces_cond_values_at_cond_pos():
    """Sanity check independent of global kriging: with exact=True the
    local field must reproduce the conditioning values at the
    conditioning points themselves (like normal kriging does)."""
    dim = 2
    cond_pos, cond_val = make_cond(dim)
    model = gs.Gaussian(dim=dim, var=2.0, len_scale=3, nugget=0.0)
    krige = gs.krige.Ordinary(model, cond_pos, cond_val, exact=True)

    field_l, _ = local_krige(krige, cond_pos, local_radius=4.0)
    np.testing.assert_allclose(field_l, cond_val, atol=1e-6)


def test_error_decreases_as_radius_increases():
    """Not a strict correctness check, but a sanity check: as
    local_radius grows towards the full domain, the local field should
    converge monotonically (in RMSE) towards the global field."""
    dim = 2
    cond_pos, cond_val = make_cond(dim)
    pos = make_targets(dim)
    model = gs.Gaussian(dim=dim, var=1.5, len_scale=2, nugget=0.1)
    krige = gs.krige.Ordinary(model, cond_pos, cond_val)
    field_g, _ = krige(pos, return_var=True, post_process=False, store=False)

    rmses = []
    for r in (3.5, 5.0, 7.0, 9.0, FULL_RADIUS):
        field_l, _ = local_krige(krige, pos, local_radius=r)
        rmses.append(np.sqrt(np.mean((field_l - field_g) ** 2)))

    assert rmses[-1] < 1e-6
    assert all(a >= b - 1e-12 for a, b in zip(rmses, rmses[1:]))


def test_local_radius_restricts_neighborhood():
    """A small radius must actually cut down which neighbors are used:
    field should differ from the full-radius case and the error
    variance should be larger (less information nearby)."""
    dim = 2
    cond_pos, cond_val = make_cond(dim)
    pos = make_targets(dim)
    model = gs.Gaussian(dim=dim, var=1.5, len_scale=3, nugget=0.1)
    krige = gs.krige.Ordinary(model, cond_pos, cond_val)

    field_unrestricted, var_unrestricted = local_krige(
        krige, pos, local_radius=FULL_RADIUS
    )
    field_restricted, var_restricted = local_krige(
        krige, pos, local_radius=3.5
    )

    assert not np.allclose(field_unrestricted, field_restricted)
    assert np.all(var_restricted >= var_unrestricted - 1e-12)


def test_local_radius_raises_when_no_neighbor_in_radius():
    dim = 2
    cond_pos, cond_val = make_cond(dim)
    model = gs.Gaussian(dim=dim, var=1.5, len_scale=3, nugget=0.1)
    krige = gs.krige.Ordinary(model, cond_pos, cond_val)

    far_away = [np.array([1e6]), np.array([1e6])]
    with pytest.raises(ValueError, match="no conditioning points within"):
        local_krige(krige, far_away, local_radius=1.0)


def test_ext_drift_matches_global():
    dim = 2
    cond_pos, cond_val = make_cond(dim)
    pos = make_targets(dim)
    model = gs.Gaussian(dim=dim, var=1.5, len_scale=3, nugget=0.1)

    rng = np.random.RandomState(3)
    cond_ext = rng.uniform(-1, 1, len(cond_val))
    targ_ext = rng.uniform(-1, 1, len(pos[0]))

    krige = gs.krige.ExtDrift(model, cond_pos, cond_val, cond_ext)

    field_g, var_g = krige(
        pos, ext_drift=targ_ext, return_var=True, post_process=False, store=False
    )
    field_l, var_l = local_krige(
        krige, pos, local_radius=FULL_RADIUS, ext_drift=targ_ext
    )

    np.testing.assert_allclose(field_l, field_g, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(var_l, var_g, atol=1e-6, rtol=1e-6)


def test_ext_drift_and_functional_drift_combined():
    """Sanity check for the unified drift array: functional (Universal)
    drift together with external drift in the same system, compared
    directly against a global Krige base instance configured the same
    way (drift_functions + ext_drift)."""
    dim = 2
    cond_pos, cond_val = make_cond(dim)
    pos = make_targets(dim)
    model = gs.Gaussian(dim=dim, var=1.5, len_scale=3, nugget=0.1)

    rng = np.random.RandomState(4)
    cond_ext = rng.uniform(-1, 1, len(cond_val))
    targ_ext = rng.uniform(-1, 1, len(pos[0]))

    krige = gs.krige.Krige(
        model,
        cond_pos,
        cond_val,
        drift_functions="linear",
        ext_drift=cond_ext,
    )

    field_g, var_g = krige(
        pos, ext_drift=targ_ext, return_var=True, post_process=False, store=False
    )
    field_l, var_l = local_krige(
        krige, pos, local_radius=FULL_RADIUS, ext_drift=targ_ext
    )

    np.testing.assert_allclose(field_l, field_g, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(var_l, var_g, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
