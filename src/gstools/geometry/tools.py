"""
GStools subpackage providing helper tools for stochastic geometry.

.. currentmodule:: gstools.geometry.tools

The following functions are provided

.. autosummary::
   interfaces_to_pyvista
   topology
"""

import numpy as np

from gstools.tools.geometric import generate_grid

__all__ = ["interfaces_to_pyvista", "topology"]


def interfaces_to_pyvista(stack, lat_pos):
    """Export the interfaces of a stack as pyvista surfaces.

    Height functions need no meshing algorithm: ``(x, y, f_i(x, y))`` on a
    structured lateral grid already is a structured surface.

    Parameters
    ----------
    stack : :class:`gstools.geometry.LayerStack` or :class:`gstools.geometry.SurfaceStack`
        The stack whose interfaces are exported. Must have ``dim == 3``.
    lat_pos : :class:`tuple` of :class:`numpy.ndarray`
        Structured lateral axes ``(x, y)``.

    Returns
    -------
    :class:`list` of :class:`pyvista.StructuredGrid`
        One surface per interface, bottom to top.

    Raises
    ------
    ImportError
        If pyvista is not installed.
    """
    import pyvista as pv  # optional dependency, imported on demand

    if stack.dim != 3:
        raise ValueError("interfaces_to_pyvista: needs a stack with dim=3.")
    x_ax, y_ax = (np.asarray(ax, dtype=np.double) for ax in lat_pos)
    ifaces = stack.interfaces(generate_grid((x_ax, y_ax)))
    x_grid, y_grid = np.meshgrid(x_ax, y_ax, indexing="ij")
    return [
        pv.StructuredGrid(x_grid, y_grid, iface.reshape(x_grid.shape))
        for iface in ifaces
    ]


def topology(labels, background=None):
    """Count adjacent label pairs on a structured label grid.

    Useful for ensemble QA, e.g. rejecting realizations that violate a
    known unit adjacency.

    Parameters
    ----------
    labels : :class:`numpy.ndarray`
        Structured label grid (as returned by
        :class:`gstools.geometry.FaciesField` with ``mesh_type="structured"``).
    background : :class:`float`, optional
        Label to ignore. Default: None

    Returns
    -------
    :class:`dict`
        ``{(label_a, label_b): count}`` with ``label_a < label_b``, counting
        face-adjacent cell pairs along every axis.
    """
    labels = np.asarray(labels)
    res = {}
    for axis in range(labels.ndim):
        n_ax = labels.shape[axis]
        lab_a = np.take(labels, range(n_ax - 1), axis=axis).ravel()
        lab_b = np.take(labels, range(1, n_ax), axis=axis).ravel()
        sel = lab_a != lab_b
        if background is not None:
            sel &= (lab_a != background) & (lab_b != background)
        pairs = np.sort(np.stack((lab_a[sel], lab_b[sel]), axis=1), axis=1)
        if pairs.size == 0:
            continue
        uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
        for (l_a, l_b), num in zip(uniq.tolist(), cnt.tolist()):
            res[(l_a, l_b)] = res.get((l_a, l_b), 0) + num
    return res
