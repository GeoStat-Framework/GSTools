"""
GStools subpackage providing the Zone class for zonated MPS simulation.

.. currentmodule:: gstools.mps

The following classes are provided

.. autosummary::
   Zone
"""

import numpy as np

from gstools.mps.training_image import TrainingImage

__all__ = ["Zone"]


class Zone:
    """A training image bound to a region of the simulation grid.

    Zonated Direct Sampling (Mariethoz et al. 2010, para [40]): nodes inside
    the zone's region scan this zone's TI instead of the primary one. The
    data event is still built from simulation-grid neighbours regardless of
    their zones, which keeps zone boundaries coherent.

    Parameters
    ----------
    ti : :any:`TrainingImage`
        Zone training image. May be a sub-region view created with
        :meth:`TrainingImage.window`. Must share the primary TI's variable
        set (validated by :class:`MPSModel`).
    where : :class:`numpy.ndarray` of :class:`bool` or callable
        Region of the simulation grid this zone covers: a boolean array of
        the grid shape, or a callable ``f(x, [y, z, ...]) -> bool mask``
        evaluated on flat coordinate arrays (resolved at call time).
        Non-boolean arrays raise — no silent ``!= 0`` coercion. A user
        holding an integer zonation raster writes
        ``Zone(ti_k, where=(labels == k))``.

    Examples
    --------
    >>> import numpy as np
    >>> import gstools as gs
    >>> ti_b = gs.TrainingImage(np.ones((50, 50)))
    >>> zone = gs.Zone(ti_b, where=lambda x, y: x >= 20)
    >>> zone.ti.shape
    (50, 50)
    """

    def __init__(self, ti, where):
        if not isinstance(ti, TrainingImage):
            raise TypeError(
                f"Zone: ti must be a TrainingImage, got {type(ti)!r}"
            )
        self._ti = ti
        if callable(where):
            self._where = where
        else:
            arr = np.asarray(where)
            if arr.dtype != np.bool_:
                raise ValueError(
                    f"Zone: where must be a boolean array or a callable, got "
                    f"dtype {arr.dtype!r}. For an integer zonation raster "
                    "use e.g. where=(labels == k) — values are never "
                    "silently coerced."
                )
            self._where = arr
        self._where_is_callable = callable(where)

    @property
    def ti(self):
        """:any:`TrainingImage`: This zone's training image."""
        return self._ti

    @property
    def where(self):
        """:class:`numpy.ndarray` of bool or callable: The zone region spec."""
        return self._where

    def __repr__(self):
        if self._where_is_callable:
            wr = getattr(self._where, "__name__", "callable")
        else:
            wr = f"array(shape={self._where.shape})"
        return f"Zone({self._ti!r}, where={wr})"
