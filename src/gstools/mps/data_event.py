"""Per-variable data-event value object for Direct Sampling.

Bundles the five parallel arrays that describe one variable's data event so
they cannot silently desync.  All operations are functional (return a new
``DataEvent``); arrays are never mutated in place.  This preserves the
aliasing invariant in :mod:`gstools.mps.simulate`: ``lags_sg`` and ``lags_ti``
stay independent across truncation, so trimming the TI-frame lags never
mutates the SG-frame lags.
"""


class DataEvent:
    """One variable's data event: parallel SG arrays plus the TI-frame lags.

    Parameters
    ----------
    lags_sg : numpy.ndarray, shape (k, dim)
        Lag vectors in the simulation-grid frame (float64).
    values : numpy.ndarray, shape (k,)
        Data-event values gathered from the simulation grid.
    cond_mask : numpy.ndarray of bool, shape (k,)
        True where the entry is conditioning data.
    lag_norms : numpy.ndarray, shape (k,)
        Euclidean norm of each SG lag vector.
    lags_ti : numpy.ndarray, shape (k, dim) or None, optional
        Lag vectors mapped into the training-image frame.  ``None`` until
        seam 2 (:meth:`_transform_and_reduce_lags`) computes them.
    """

    __slots__ = ("lags_sg", "values", "cond_mask", "lag_norms", "lags_ti")

    def __init__(self, lags_sg, values, cond_mask, lag_norms, lags_ti=None):
        self.lags_sg = lags_sg
        self.values = values
        self.cond_mask = cond_mask
        self.lag_norms = lag_norms
        self.lags_ti = lags_ti

    def __len__(self):
        """Number of lags ``k`` in the data event."""
        return len(self.values)

    def with_lags_ti(self, lags_ti):
        """Return a new :class:`DataEvent` with ``lags_ti`` set, SG arrays shared."""
        return DataEvent(
            self.lags_sg, self.values, self.cond_mask, self.lag_norms, lags_ti
        )

    def truncate(self, k):
        """Return a new :class:`DataEvent` with every array sliced to ``[:k]``.

        Each field is a new array reference (a slice) on the returned object;
        because ``truncate`` is functional — it never writes in place — the
        SG/TI aliasing invariant holds regardless of whether ``lags_ti`` shares
        ``lags_sg``'s buffer. Callers that need a buffer-independent ``lags_ti``
        must pass a copy (see :meth:`with_lags_ti`).
        """
        return DataEvent(
            self.lags_sg[:k],
            self.values[:k],
            self.cond_mask[:k],
            self.lag_norms[:k],
            None if self.lags_ti is None else self.lags_ti[:k],
        )
