"""
GStools subpackage providing the PostProcessing class for MPS simulations.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   PostProcessing
"""

import numpy as np

from gstools.mps._validation import exact_count
from gstools.mps.training_image import SEARCH_FIELDS

__all__ = ["PostProcessing"]

_VALID_PATH = ("random", "sequential", "same")


def _validate_factor(value):
    """Return ``value`` as a float that is finite, > 0, with a finite reciprocal."""
    v = float(value)
    if not (np.isfinite(v) and v > 0.0 and np.isfinite(1.0 / v)):
        raise ValueError(
            f"PostProcessing: factor must be finite and > 0, got {value!r}"
        )
    return v


def _read_only_copy(arr):
    """Return a private, non-writeable copy of ``arr``."""
    arr = np.array(arr)
    arr.flags.writeable = False
    return arr


def _validate_path(value):
    """None, one of ``_VALID_PATH``, or an explicit ``(N, dim)`` array."""
    if value is None or (isinstance(value, str) and value in _VALID_PATH):
        return value
    if isinstance(value, str):
        raise ValueError(
            f"PostProcessing: path must be one of {_VALID_PATH!r}, an "
            f"(N, dim) integer array, or None (inherit the main path mode), "
            f"got {value!r}"
        )
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(
            "PostProcessing: an explicit path must be a 2-D (N, dim) array, "
            f"got shape {arr.shape!r}"
        )
    return _read_only_copy(arr)


def _validate_overrides(value):
    """Return a private copy of ``{variable name: {search field: value}}``."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            "PostProcessing: overrides must be a dict "
            f"{{variable name: {{field: value}}}}, got {type(value)!r}."
        )
    out = {}
    for name, fields in value.items():
        if not isinstance(name, str) or not isinstance(fields, dict):
            raise TypeError(
                "PostProcessing: overrides must map a variable name to a "
                f"dict of fields, got {name!r}: {fields!r}."
            )
        unknown = set(fields) - set(SEARCH_FIELDS)
        if unknown:
            raise ValueError(
                f"PostProcessing: overrides[{name!r}] has unknown field(s) "
                f"{sorted(unknown)!r}; valid fields are "
                f"{sorted(SEARCH_FIELDS)!r}."
            )
        out[name] = dict(fields)
    return out


def _validate_mask(value):
    """None or a boolean array (its shape is checked against the grid at call time)."""
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype != bool:
        raise ValueError(
            f"PostProcessing: mask must be a boolean array or None, got dtype {arr.dtype!r}."
        )
    return _read_only_copy(arr)


class _ArrayRepr:
    """Prints as ``array(shape=...)`` inside a repr."""

    def __init__(self, arr):
        self._text = f"array(shape={arr.shape})"

    def __repr__(self):
        return self._text


def _overrides_repr(overrides):
    """Repr of the overrides dict with array values shown by shape."""
    return repr(
        {
            name: {
                key: _ArrayRepr(val) if isinstance(val, np.ndarray) else val
                for key, val in fields.items()
            }
            for name, fields in overrides.items()
        }
    )


class PostProcessing:
    """Post-processing passes of Direct Sampling (Meerschman2013 §4).

    After the main pass, each post-processing pass re-simulates every
    non-conditioning node against a fully informed neighbourhood, with the
    search effort divided by ``factor``, to remove simulation noise. Pass
    an instance as ``MPSModel(post_processing=...)``.

    Parameters
    ----------
    passes : :class:`int`
        Number of passes, at least 1 (an integral value; a bool counts as an
        int). Each pass re-visits essentially the whole grid and runs over the
        same dependency graph as the main pass, so it honours ``num_threads``
        and gives the same result for any thread count.
    factor : :class:`float`, optional
        Factor ``p_f > 0`` dividing ``scan_fraction`` and every
        ``n_neighbors`` during the passes; ``threshold`` is unaffected.
        Values above 1 reduce the search effort, values below 1 increase it.
        Default: 1.0.
    path : :class:`str`, array-like or None, optional
        Visit order of the passes (an implementation extension, the paper
        does not specify it): ``None`` follows the main path (raster order if
        the main ``path`` is ``"sequential"``, otherwise a fresh random
        permutation per pass, also for an explicit main path array),
        ``"random"`` draws a fresh permutation per pass, ``"sequential"``
        uses raster order, ``"same"`` reuses the main visit order, and an
        ``(N, dim)`` integer array gives an explicit order over the nodes
        with at least one non-conditioned variable (conditioned nodes in it
        are dropped). Default: ``None``.
    overrides : :class:`dict` or None, optional
        Per-variable search settings used only during the passes, as
        ``{variable name: {field: value}}``; the name is ``"field"`` for a
        training image built from an array. Allowed fields are
        ``distance``, ``weight``, ``n_neighbors``, ``max_radius``,
        ``penalty_matrix``, ``distance_power`` and ``cond_weight``; unset
        fields inherit the variable's own value. Weights must end up set on
        every variable or on none, and a ``"variation"`` variable must keep
        at least 2 neighbours after dividing by ``factor``. Names and values
        are checked against the training image by :class:`MPSModel`.
        Default: ``None``.
    mask : numpy.ndarray of bool or None, optional
        Simulation-grid-shaped mask of the nodes the passes may re-simulate
        (``True`` = eligible). Conditioning nodes are never re-simulated;
        masked-out nodes still inform their neighbours. Joint across
        variables. The shape is checked at simulation time. Default:
        ``None`` (every non-conditioning node).

    Raises
    ------
    ValueError
        If ``passes`` is not an integer >= 1, ``factor`` is not finite and
        > 0, ``path`` is an unknown string or not a 2-D array, ``overrides``
        names an unknown field, or ``mask`` is not boolean.
    TypeError
        If ``overrides`` is not a ``{variable name: {field: value}}`` dict.

    Examples
    --------
    >>> import numpy as np
    >>> import gstools as gs
    >>> ti = gs.mps.TrainingImage(np.indices((20, 20)).sum(axis=0) % 3.0)
    >>> post = gs.mps.PostProcessing(
    ...     2, factor=2, overrides={"field": {"n_neighbors": 8}}
    ... )
    >>> model = gs.mps.MPSModel(ti, post_processing=post)
    >>> model.post_processing.passes
    2
    """

    def __init__(
        self, passes, *, factor=1.0, path=None, overrides=None, mask=None
    ):
        self._passes = exact_count(
            passes, minimum=1, context="PostProcessing: passes"
        )
        self._factor = _validate_factor(factor)
        self._path = _validate_path(path)
        self._overrides = _validate_overrides(overrides)
        self._mask = _validate_mask(mask)

    @property
    def passes(self):
        """:class:`int`: Number of post-processing passes."""
        return self._passes

    @property
    def factor(self):
        """:class:`float`: Divisor of ``scan_fraction`` and ``n_neighbors`` in the passes."""
        return self._factor

    @property
    def path(self):
        """:class:`str`, array-like or None: Visit order of the passes (``None`` inherits the main-path mode)."""
        return self._path

    @property
    def overrides(self):
        """:class:`dict`: Copy of the ``{variable name: {field: value}}`` overrides."""
        return {name: dict(fields) for name, fields in self._overrides.items()}

    @property
    def mask(self):
        """:class:`numpy.ndarray` of bool or None: Nodes eligible for re-simulation."""
        return self._mask

    def __repr__(self):
        args = [repr(self._passes)]
        if self._factor != 1.0:
            args.append(f"factor={self._factor!r}")
        if self._path is not None:
            path = self._path
            args.append(
                f"path={path!r}"
                if isinstance(path, str)
                else f"path=array(shape={path.shape})"
            )
        if self._overrides:
            args.append(f"overrides={_overrides_repr(self._overrides)}")
        if self._mask is not None:
            args.append(f"mask=array(shape={self._mask.shape})")
        return f"PostProcessing({', '.join(args)})"
