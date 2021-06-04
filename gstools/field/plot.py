# -*- coding: utf-8 -*-
"""
GStools subpackage providing plotting routines for spatial fields.

.. currentmodule:: gstools.field.plot

The following classes and functions are provided

.. autosummary::
   plot_field
   plot_vec_field
"""
# pylint: disable=C0103, W0613, E1101
import numpy as np
from scipy import interpolate as inter
from scipy.spatial import ConvexHull
from gstools.tools.misc import get_fig_ax
from gstools.tools.geometric import rotation_planes

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons
except ImportError as exc:
    raise ImportError("Plotting: Matplotlib not installed.") from exc


__all__ = ["plot_field", "plot_vec_field"]


# plotting routines #######################################################


def plot_field(
    fld, field="field", fig=None, ax=None, **kwargs
):  # pragma: no cover
    """
    Plot a spatial field.

    Parameters
    ----------
    fld : :class:`Field`
        The given Field class instance.
    field : :class:`str`, optional
        Field that should be plotted. Default: "field"
    fig : :class:`Figure` or :any:`None`, optional
        Figure to plot the axes on. If `None`, a new one will be created.
        Default: `None`
    ax : :class:`Axes` or :any:`None`, optional
        Axes to plot on. If `None`, a new one will be added to the figure.
        Default: `None`
    **kwargs
        Forwarded to the plotting routine.
    """
    plt_fld = getattr(fld, field)
    assert not (fld.pos is None or plt_fld is None)
    if fld.dim == 1:
        return plot_1d(fld.pos, plt_fld, fig, ax, **kwargs)
    return plot_nd(
        fld.pos, plt_fld, fld.mesh_type, fig, ax, fld.latlon, **kwargs
    )


def plot_1d(pos, field, fig=None, ax=None, ax_names=None):  # pragma: no cover
    """
    Plot a 1D field.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing either the point coordinates (x, y, ...)
        or the axes descriptions (for mesh_type='structured')
    field : :class:`numpy.ndarray`
        Field values.
    fig : :class:`Figure` or :any:`None`, optional
        Figure to plot the axes on. If `None`, a new one will be created.
        Default: `None`
    ax : :class:`Axes` or :any:`None`, optional
        Axes to plot on. If `None`, a new one will be added to the figure.
        Default: `None`
    ax_names : :class:`list` of :class:`str`, optional
        Axes names. The default is ["$x$", "field"].

    Returns
    -------
    ax : :class:`Axes`
        Axis containing the plot.
    """
    fig, ax = get_fig_ax(fig, ax)
    title = f"Field 1D: {field.shape}"
    x = pos[0]
    x = x.flatten()
    arg = np.argsort(x)
    ax_names = _ax_names(1, ax_names=ax_names)
    ax.plot(x[arg], field.ravel()[arg])
    ax.set_xlabel(ax_names[0])
    ax.set_ylabel(ax_names[1])
    ax.set_title(title)
    fig.show()
    return ax


def plot_nd(
    pos,
    field,
    mesh_type,
    fig=None,
    ax=None,
    latlon=False,
    resolution=128,
    ax_names=None,
    aspect="quad",
    show_colorbar=True,
    convex_hull=False,
    contour_plot=True,
    **kwargs,
):  # pragma: no cover
    """
    Plot field in arbitrary dimensions.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing either the point coordinates (x, y, ...)
        or the axes descriptions (for mesh_type='structured')
    field : :class:`numpy.ndarray`
        Field values.
    fig : :class:`Figure` or :any:`None`, optional
        Figure to plot the axes on. If `None`, a new one will be created.
        Default: `None`
    ax : :class:`Axes` or :any:`None`, optional
        Axes to plot on. If `None`, a new one will be added to the figure.
        Default: `None`
    latlon : :class:`bool`, optional
        Whether the data is representing 2D fields on earths surface described
        by latitude and longitude. When using this, the estimator will
        use great-circle distance for variogram estimation.
        Note, that only an isotropic variogram can be estimated and a
        ValueError will be raised, if a direction was specified.
        Bin edges need to be given in radians in this case.
        Default: False
    resolution : :class:`int`, optional
        Resolution of the imshow plot. The default is 128.
    ax_names : :class:`list` of :class:`str`, optional
        Axes names. The default is ["$x$", "field"].
    aspect : :class:`str` or :any:`None` or :class:`float`, optional
        Aspect of the plot. Can be "auto", "equal", "quad", None or a number
        describing the aspect ratio.
        The default is "quad".
    show_colorbar : :class:`bool`, optional
        Whether to show the colorbar. The default is True.
    convex_hull : :class:`bool`, optional
        Whether to show the convex hull in 2D with unstructured data.
        The default is False.
    contour_plot : :class:`bool`, optional
        Whether to use a contour-plot in 2D. The default is True.

    Returns
    -------
    ax : :class:`Axes`
        Axis containing the plot.
    """
    dim = len(pos)
    assert dim > 1
    assert not latlon or dim == 2
    if dim == 2 and contour_plot:
        return _plot_2d(
            pos, field, mesh_type, fig, ax, latlon, ax_names, **kwargs
        )
    pos = pos[::-1] if latlon else pos
    field = field.T if (latlon and mesh_type != "unstructured") else field
    ax_names = _ax_names(dim, latlon, ax_names)
    # init planes
    planes = rotation_planes(dim)
    plane_names = [f" {ax_names[p[0]]} - {ax_names[p[1]]}" for p in planes]
    ax_ends = [[p.min(), p.max()] for p in pos]
    ax_rngs = [end[1] - end[0] for end in ax_ends]
    ax_steps = [rng / resolution for rng in ax_rngs]
    ax_extents = [ax_ends[p[0]] + ax_ends[p[1]] for p in planes]
    # create figure
    reformat = fig is None and ax is None
    fig, ax = get_fig_ax(fig, ax)
    ax.set_title(f"Field {dim}D {mesh_type} {field.shape}")
    if reformat:  # only format fig if it was created here
        fig.set_size_inches(8, 5.5 + 0.5 * (dim - 2))
    # init additional axis, radio-buttons and sliders
    s_frac = 0.5 * (dim - 2) / (6 + 0.5 * (dim - 2))
    s_size = s_frac / max(dim - 2, 1)
    left, bottom = (0.25, s_frac + 0.13) if dim > 2 else (None, None)
    fig.subplots_adjust(left=left, bottom=bottom)
    slider = []
    for i in range(dim - 2, 0, -1):
        slider_ax = fig.add_axes([0.3, i * s_size, 0.435, s_size * 0.6])
        slider.append(Slider(slider_ax, "", 0, 1, facecolor="grey"))
        slider[-1].vline.set_color("k")
    # create radio buttons
    if dim > 2:
        rax = fig.add_axes(
            [0.05, 0.85 - 2 * s_frac, 0.15, 2 * s_frac], frame_on=0, alpha=0
        )
        rax.set_title("  Plane", loc="left")
        radio = RadioButtons(rax, plane_names, activecolor="grey")
        # make radio buttons circular
        rpos = rax.get_position().get_points()
        fh, fw = fig.get_figheight(), fig.get_figwidth()
        rscale = (rpos[:, 1].ptp() / rpos[:, 0].ptp()) * (fh / fw)
        for circ in radio.circles:
            circ.set_radius(0.06)
            circ.height /= rscale
    elif mesh_type == "unstructured" and convex_hull:
        # show convex hull in 2D
        hull = ConvexHull(pos.T)
        for simplex in hull.simplices:
            ax.plot(pos[0, simplex], pos[1, simplex], "k")
    # init imshow and colorbar axis
    grid = np.mgrid[0 : 1 : resolution * 1j, 0 : 1 : resolution * 1j]
    f_ini, vmin, vmax = np.full_like(grid[0], np.nan), field.min(), field.max()
    im = ax.imshow(
        f_ini.T, interpolation="bicubic", origin="lower", vmin=vmin, vmax=vmax
    )

    # actions
    def inter_plane(cuts, axes):
        """Interpolate plane."""
        plane_ax = []
        for i, (rng, end, cut) in enumerate(zip(ax_rngs, ax_ends, cuts)):
            if i in axes:
                plane_ax.append(grid[axes.index(i)] * rng + end[0])
            else:
                plane_ax.append(np.full_like(grid[0], cut, dtype=float))
        # needs to be a tuple
        plane_ax = tuple(plane_ax)
        if mesh_type != "unstructured":
            return inter.interpn(pos, field, plane_ax, bounds_error=False)
        return inter.griddata(pos.T, field, plane_ax, method="nearest")

    def update_field(*args):
        """Sliders update."""
        p = plane_names.index(radio.value_selected) if dim > 2 else 0
        # dummy cut values for selected plane-axes (setting to 0)
        cuts = [s.val for s in slider]
        cuts.insert(planes[p][0], 0)
        cuts.insert(planes[p][1], 0)
        im.set_array(inter_plane(cuts, planes[p]).T)
        fig.canvas.draw_idle()

    def update_plane(label):
        """Radio button update."""
        p = plane_names.index(label)
        cut_select = [i for i in range(dim) if i not in planes[p]]
        # reset sliders
        for i, s in zip(cut_select, slider):
            s.label.set_text(ax_names[i])
            s.valmin, s.valmax = ax_ends[i]
            s.valinit = ax_ends[i][0] + ax_rngs[i] / 2.0
            s.valstep = ax_steps[i]
            s.ax.set_xlim(*ax_ends[i])
            # update representation
            s.poly.xy[:2] = (s.valmin, 0), (s.valmin, 1)
            s.vline.set_data(2 * [s.valinit], [-0.1, 1.1])
            s.reset()
        im.set_extent(ax_extents[p])
        if aspect == "quad":
            asp = ax_rngs[planes[p][0]] / ax_rngs[planes[p][1]]
        if aspect is not None:
            ax.set_aspect(asp if aspect == "quad" else aspect)
        ax.set_xlabel(ax_names[planes[p][0]])
        ax.set_ylabel(ax_names[planes[p][1]])
        update_field()

    # initial plot on xy plane
    update_plane(plane_names[0])
    # bind actions
    if dim > 2:
        radio.on_clicked(update_plane)
    for s in slider:
        s.on_changed(update_field)
    if show_colorbar:
        fig.colorbar(im, ax=ax)
    fig.show()
    return ax


def plot_vec_field(fld, field="field", fig=None, ax=None):  # pragma: no cover
    """
    Plot a spatial random vector field.

    Parameters
    ----------
    fld : :class:`Field`
        The given field class instance.
    field : :class:`str`, optional
        Field that should be plotted. Default: "field"
    fig : :class:`Figure` or :any:`None`, optional
        Figure to plot the axes on. If `None`, a new one will be created.
        Default: `None`
    ax : :class:`Axes` or :any:`None`, optional
        Axes to plot on. If `None`, a new one will be added to the figure.
        Default: `None`
    """
    if fld.mesh_type == "unstructured":
        raise RuntimeError(
            "Only structured vector fields are supported "
            "for plotting. Please create one on a structured grid."
        )
    plt_fld = getattr(fld, field)
    assert not (fld.pos is None or plt_fld is None)

    norm = np.sqrt(plt_fld[0, :].T ** 2 + plt_fld[1, :].T ** 2)

    fig, ax = get_fig_ax(fig, ax)
    title = f"Field 2D {fld.mesh_type}: {plt_fld.shape}"
    x = fld.pos[0]
    y = fld.pos[1]

    sp = plt.streamplot(
        x,
        y,
        plt_fld[0, :].T,
        plt_fld[1, :].T,
        color=norm,
        linewidth=norm / 2,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    fig.colorbar(sp.lines)
    fig.show()
    return ax


def _ax_names(dim, latlon=False, ax_names=None):
    if ax_names is not None:
        assert len(ax_names) >= dim
        return ax_names[:dim]
    if dim == 2 and latlon:
        return ["lon", "lat"]
    if dim <= 3:
        return ["$x$", "$y$", "$z$"][:dim] + (dim == 1) * ["field"]
    return [f"$x_{{{i}}}$" for i in range(dim)]


def _plot_2d(
    pos,
    field,
    mesh_type,
    fig=None,
    ax=None,
    latlon=False,
    ax_names=None,
    levels=64,
    antialias=True,
):  # pragma: no cover
    """Plot a 2d field with a contour plot."""
    fig, ax = get_fig_ax(fig, ax)
    title = f"Field 2D {mesh_type}: {field.shape}"
    ax_names = _ax_names(2, latlon, ax_names=ax_names)
    x, y = pos[::-1] if latlon else pos
    if mesh_type == "unstructured":
        cont = ax.tricontourf(x, y, field.ravel(), levels=levels)
        if antialias:
            ax.tricontour(x, y, field.ravel(), levels=levels, zorder=-10)
    else:
        plt_fld = field if latlon else field.T
        cont = ax.contourf(x, y, plt_fld, levels=levels)
        if antialias:
            ax.contour(x, y, plt_fld, levels=levels, zorder=-10)
    ax.set_xlabel(ax_names[0])
    ax.set_ylabel(ax_names[1])
    ax.set_title(title)
    fig.colorbar(cont)
    fig.show()
    return ax
