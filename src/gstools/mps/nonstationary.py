"""Call-time resolution of non-stationarity and zonation specs.

`MPSModel` stores rotation/scale/zone specs grid-independently; once
`DirectSampling.__call__` knows the grid, this module resolves each spec into
its canonical per-node form:

- rotation/scale -> array of shape ``grid_shape + (n_comp,)`` plus a
  ``stationary`` flag. Stationary specs resolve to zero-copy read-only
  ``np.broadcast_to`` views, so the engine can hoist the transform-matrix
  construction out of the node loop without paying per-node memory.
- ``Zone.where`` masks -> one integer selector array over the grid
  (0 = primary TI, i+1 = ``zones[i]``); overlaps raise.

Resolution is by **exact shape match**, never ndim heuristics — this is what
structurally eliminates bug B5 (flattened per-node maps silently treated as
stationary). Genuine collisions (a 1-D grid of length ``n_comp``) raise a
ValueError naming both interpretations.

Callables follow the established GSTools mean/trend convention: invoked as
``f(*pos)`` with one flat coordinate array per dimension, returning
``(n_nodes,)`` (broadcast across components) or ``(n_nodes, n_comp)``. The
trailing component axis is a deliberate deviation from GSTools vector-field
callables (which lead with a ``(dim,)`` axis): trailing is what per-node
indexing ``arr[tuple(x_i)]`` consumes.
"""

import numpy as np

__all__ = ["resolve_spec", "resolve_where", "build_zone_selector"]


def _check_positive(arr, name):
    if not np.all(np.asarray(arr) > 0):
        raise ValueError(
            f"MPSModel: all {name} values must be > 0 "
            f"(minimum found: {float(np.min(arr))!r})."
        )


def _stationary_view(values, grid_shape, n_comp):
    """Zero-copy read-only broadcast of an (n_comp,) vector to the grid."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    return np.broadcast_to(
        values.reshape((1,) * len(grid_shape) + (n_comp,)),
        grid_shape + (n_comp,),
    )


def _materialize_callable(func, grid_shape, n_comp, pos, name):
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


def resolve_spec(spec, grid_shape, n_comp, pos, name, positive=False):
    """Resolve a rotation/scale spec to its canonical per-node array.

    Parameters
    ----------
    spec : None, scalar, array-like, or callable
        The grid-independent spec stored on :class:`MPSModel`.
    grid_shape : tuple of int
        Simulation grid shape.
    n_comp : int
        Component count: ``no_of_angles(dim)`` for rotation, ``dim`` for
        scale.
    pos : tuple of numpy.ndarray
        One flat coordinate array per dimension
        (:func:`gstools.tools.geometric.generate_grid` output); callables are
        invoked as ``f(*pos)``.
    name : str
        Parameter name for error messages (``"rotation"``/``"scale"``).
    positive : bool, optional
        Require all resolved values > 0 (scale). Default: False.

    Returns
    -------
    arr : numpy.ndarray of shape grid_shape + (n_comp,), or None
        Canonical per-node array (``None`` when ``spec`` is ``None``).
        Stationary specs return a zero-copy read-only broadcast view.
    stationary : bool
        True when the value is node-independent (the engine may hoist the
        transform-matrix construction out of the node loop).

    Examples
    --------
    >>> import numpy as np
    >>> from gstools.mps.nonstationary import resolve_spec
    >>> grid_shape = (5, 5)
    >>> x = np.arange(5)
    >>> y = np.zeros(5)
    >>> pos = (x, y)
    >>> spec = 45.0
    >>> n_comp = 1
    >>> arr, stationary = resolve_spec(spec, grid_shape, n_comp, pos, "rotation")
    >>> arr.shape
    (5, 5, 1)
    >>> stationary
    True
    >>> arr[0, 0]
    array([45.])
    """
    if spec is None:
        return None, True
    grid_shape = tuple(int(s) for s in grid_shape)
    if callable(spec):
        arr = _materialize_callable(spec, grid_shape, n_comp, pos, name)
        if positive:
            _check_positive(arr, name)
        return arr, False
    arr = np.asarray(spec, dtype=np.float64)
    if positive:
        _check_positive(arr, name)
    if arr.ndim == 0:
        return (
            _stationary_view(np.full(n_comp, float(arr)), grid_shape, n_comp),
            True,
        )
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
        One flat coordinate array per dimension
        (:func:`gstools.tools.geometric.generate_grid` output); callables are
        invoked as ``f(*pos)``.

    Returns
    -------
    numpy.ndarray of bool, shape grid_shape
        Boolean mask indicating the region.

    Examples
    --------
    >>> import numpy as np
    >>> from gstools.mps.nonstationary import resolve_where
    >>> grid_shape = (3, 3)
    >>> x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> pos = (x, y)
    >>> where = lambda x, y: x >= 1
    >>> mask = resolve_where(where, grid_shape, pos)
    >>> mask.shape
    (3, 3)
    >>> mask
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
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
        One flat coordinate array per dimension
        (:func:`gstools.tools.geometric.generate_grid` output); passed to
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

    Examples
    --------
    >>> import numpy as np
    >>> from gstools.mps.nonstationary import build_zone_selector
    >>> from gstools.mps.zone import Zone
    >>> from gstools.mps.training_image import TrainingImage
    >>> ti_a = TrainingImage(np.ones((3, 3)))
    >>> ti_b = TrainingImage(np.ones((3, 3)) * 2)
    >>> zone_a = Zone(ti_a, where=lambda x, y: x < 1)
    >>> zone_b = Zone(ti_b, where=lambda x, y: x >= 2)
    >>> zones = [zone_a, zone_b]
    >>> grid_shape = (3, 3)
    >>> x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> pos = (x, y)
    >>> selector = build_zone_selector(zones, grid_shape, pos)
    >>> selector
    array([[1, 1, 1],
           [0, 0, 0],
           [2, 2, 2]])
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
