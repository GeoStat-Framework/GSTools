"""Call-time resolution of :class:`MPSModel` non-stationarity specs.

``MPSModel`` stores ``rotation``/``scale`` specs grid-independently; once
:meth:`DirectSampling.__call__` knows the grid, :func:`resolve_spec` turns each
spec into its canonical per-node array of shape ``grid_shape + (n_comp,)`` plus
a ``stationary`` flag:

- ``n_comp = no_of_angles(dim)`` for rotation, ``dim`` for scale.
- Stationary (node-independent) specs return a **zero-copy read-only**
  ``np.broadcast_to`` view, so the engine can hoist the transform-matrix build
  out of the node loop without paying per-node memory.

Resolution is by **exact shape match**, never ndim heuristics, so a flattened
per-node map can never be silently mistaken for a stationary vector. A genuine
collision — a 1-D grid whose length equals ``n_comp`` — raises, naming both
interpretations.

Callables follow the GSTools mean/trend convention: invoked ``f(*pos)`` with one
flat coordinate array per dimension, returning ``(n_nodes,)`` (broadcast across
components) or ``(n_nodes, n_comp)`` (the trailing component axis is what
per-node indexing consumes).

It also resolves zonation (:func:`resolve_where` / :func:`build_zone_selector`,
mapping each node to its ``Zone``), and is **stricter** than the upstream source
on rotation/scale: all resolved values must be finite, and ``scale`` must
additionally be strictly positive (the upstream ``> 0`` check alone admits
``+inf``).
"""

import numpy as np

__all__ = ["resolve_spec", "resolve_where", "build_zone_selector"]


def _check_finite(arr, name):
    """Reject NaN/inf; rotation angles and scales must be finite."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"MPSModel: all {name} values must be finite.")


def _check_positive(arr, name):
    """Reject non-positive values (scale only)."""
    if not np.all(np.asarray(arr) > 0):
        raise ValueError(
            f"MPSModel: all {name} values must be > 0 "
            f"(minimum found: {float(np.min(arr))!r})."
        )


def _stationary_view(values, grid_shape, n_comp):
    """Zero-copy read-only broadcast of an ``(n_comp,)`` vector to the grid."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    return np.broadcast_to(
        values.reshape((1,) * len(grid_shape) + (n_comp,)),
        grid_shape + (n_comp,),
    )


def _materialize_callable(func, grid_shape, n_comp, pos, name):
    """Evaluate a coordinate callable once and reshape to canonical form."""
    n_nodes = int(np.prod(grid_shape))
    vals = np.asarray(func(*pos), dtype=np.float64)
    if vals.shape == (n_nodes,):
        return np.broadcast_to(
            np.ascontiguousarray(vals).reshape(grid_shape + (1,)),
            grid_shape + (n_comp,),
        )
    if vals.shape == (n_nodes, n_comp):
        return np.ascontiguousarray(vals).reshape(grid_shape + (n_comp,))
    raise ValueError(
        f"MPSModel: the {name} callable must return shape ({n_nodes},) "
        f"(one value per node, broadcast across components) or "
        f"({n_nodes}, {n_comp}) (per-node components); got {vals.shape!r}."
    )


def resolve_spec(
    spec, grid_shape, n_comp, pos, name, positive=False, angles=False
):
    """Resolve a rotation/scale spec to its canonical per-node array.

    Parameters
    ----------
    spec : None, scalar, array-like, or callable
        The grid-independent spec stored on :class:`MPSModel`.
    grid_shape : tuple of int
        Simulation grid shape.
    n_comp : int
        Component count (``no_of_angles(dim)`` for rotation, ``dim`` for scale).
    pos : tuple of numpy.ndarray
        One flat coordinate array per dimension; callables are invoked
        ``f(*pos)``.
    name : str
        Parameter name for error messages (``"rotation"``/``"scale"``).
    positive : bool, optional
        Require all resolved values strictly positive (scale). Default: False.
    angles : bool, optional
        Rotation convention for a **scalar** spec: when True, a scalar sets
        only the first rotation plane (remaining angles zero), matching
        :func:`gstools.tools.geometric.set_angles`; when False a scalar fills
        all components (a scalar scale dilates every axis). Default: False.
        Only affects the scalar case; vectors/maps/callables are unchanged.

    Returns
    -------
    arr : numpy.ndarray of shape ``grid_shape + (n_comp,)``, or None
        Canonical per-node array (``None`` when ``spec`` is ``None``).
        Stationary specs return a zero-copy read-only broadcast view.
    stationary : bool
        True when the value is node-independent (the engine may hoist the
        transform-matrix construction out of the node loop).
    """
    if spec is None:
        return None, True
    grid_shape = tuple(int(s) for s in grid_shape)
    if callable(spec):
        arr = _materialize_callable(spec, grid_shape, n_comp, pos, name)
        _check_finite(arr, name)
        if positive:
            _check_positive(arr, name)
        return arr, False
    arr = np.asarray(spec, dtype=np.float64)
    _check_finite(arr, name)
    if positive:
        _check_positive(arr, name)
    if arr.ndim == 0:
        if angles:
            # set_angles convention: scalar sets the first plane, rest zero.
            vec = np.zeros(n_comp, dtype=np.float64)
            if n_comp:
                vec[0] = float(arr)
        else:
            vec = np.full(n_comp, float(arr), dtype=np.float64)
        return _stationary_view(vec, grid_shape, n_comp), True
    is_stationary_form = arr.shape == (n_comp,)
    is_grid_scalar = arr.shape == grid_shape
    is_grid_full = arr.shape == grid_shape + (n_comp,)
    if is_stationary_form and (is_grid_scalar or is_grid_full):
        raise ValueError(
            f"MPSModel: {name} of shape {arr.shape!r} is ambiguous on a grid "
            f"of shape {grid_shape!r}: it matches both a stationary "
            f"{n_comp}-component value and a per-node map. Disambiguate by "
            f"adding the trailing component axis (shape "
            f"{grid_shape + (n_comp,)!r}) or by using a callable."
        )
    if is_stationary_form:
        return _stationary_view(arr, grid_shape, n_comp), True
    if is_grid_scalar:
        return (
            np.broadcast_to(arr[..., None], grid_shape + (n_comp,)),
            False,
        )
    if is_grid_full:
        return arr, False
    raise ValueError(
        f"MPSModel: {name} of shape {arr.shape!r} matches no accepted form "
        f"for a grid of shape {grid_shape!r}: scalar, ({n_comp},) stationary "
        f"vector, {grid_shape!r} per-node map, or "
        f"{grid_shape + (n_comp,)!r} per-node components. Per-node maps must "
        "be shaped like the grid, not flattened."
    )


def resolve_where(where, grid_shape, pos):
    """Resolve a ``Zone.where`` spec to a boolean mask of the grid shape.

    Parameters
    ----------
    where : numpy.ndarray of bool or callable
        A boolean mask of ``grid_shape``, or a callable ``f(*pos) -> bool``
        evaluated on flat coordinate arrays.
    grid_shape : tuple of int
        Simulation grid shape.
    pos : tuple of numpy.ndarray
        One flat coordinate array per dimension; callables are invoked as
        ``f(*pos)``.

    Returns
    -------
    numpy.ndarray of bool, shape grid_shape
        Boolean mask indicating the region.
    """
    grid_shape = tuple(int(s) for s in grid_shape)
    n_nodes = int(np.prod(grid_shape))
    if callable(where):
        mask = np.asarray(where(*pos))
        if mask.dtype != np.bool_:
            raise ValueError(
                f"Zone: the where callable must return a boolean mask, got "
                f"dtype {mask.dtype!r} — values are never silently coerced."
            )
        if mask.shape == (n_nodes,):
            return mask.reshape(grid_shape)
        if mask.shape == grid_shape:
            return mask
        raise ValueError(
            f"Zone: the where callable must return shape ({n_nodes},) or "
            f"{grid_shape!r}, got {mask.shape!r}."
        )
    mask = np.asarray(where)
    if mask.dtype != np.bool_:
        raise ValueError(
            f"Zone: where must be a boolean array, got dtype {mask.dtype!r}."
        )
    if mask.shape != grid_shape:
        raise ValueError(
            f"Zone: where shape {mask.shape!r} does not match the simulation "
            f"grid shape {grid_shape!r}."
        )
    return mask


def build_zone_selector(zones, grid_shape, pos):
    """Fold resolved zone masks into one integer selector over the grid.

    Parameters
    ----------
    zones : list of Zone
        Zones to fold into a selector.
    grid_shape : tuple of int
        Simulation grid shape.
    pos : tuple of numpy.ndarray
        One flat coordinate array per dimension; passed to
        :func:`resolve_where` for each zone.

    Returns
    -------
    numpy.ndarray of numpy.intp, shape grid_shape
        ``0`` = primary TI, ``i + 1`` = ``zones[i]``.

    Raises
    ------
    ValueError
        When two zones' resolved regions overlap (deterministic — no silent
        precedence order).
    """
    grid_shape = tuple(int(s) for s in grid_shape)
    selector = np.zeros(grid_shape, dtype=np.intp)
    covered = np.zeros(grid_shape, dtype=bool)
    for i, zone in enumerate(zones):
        mask = resolve_where(zone.where, grid_shape, pos)
        overlap = covered & mask
        if overlap.any():
            raise ValueError(
                f"MPSModel: zone {i}'s region overlaps an earlier zone at "
                f"{int(overlap.sum())} node(s) (first at index "
                f"{tuple(int(c) for c in np.argwhere(overlap)[0])}). Zone "
                "regions must be disjoint."
            )
        covered |= mask
        selector[mask] = i + 1
    return selector
