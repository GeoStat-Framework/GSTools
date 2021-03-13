# -*- coding: utf-8 -*-
"""
GStools subpackage providing plotting routines for spatial fields.

.. currentmodule:: gstools.field.plot

The following classes and functions are provided

.. autosummary::
   plot_field
   plot_vec_field
"""
# pylint: disable=C0103
import numpy as np
from scipy import interpolate as inter
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from gstools.covmodel.plot import _get_fig_ax
from gstools.tools.geometric import rotation_planes


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
    plot_field = getattr(fld, field)
    assert not (fld.pos is None or plot_field is None)
    if fld.dim == 1:
        return _plot_1d(fld.pos, plot_field, fig, ax, **kwargs)
    if fld.dim == 2:
        return _plot_2d(
            fld.pos,
            plot_field,
            fld.mesh_type,
            fig,
            ax,
            fld.model.latlon,
            **kwargs
        )
    return _plot_nd(
        fld.pos, plot_field, fld.mesh_type, fig, ax, fld.model.latlon, **kwargs
    )


def _ax_names(dim, latlon=False, ax_names=None):
    if ax_names is not None:
        assert len(ax_names) >= dim
        return ax_names[:dim]
    if dim == 2 and latlon:
        return ["lon", "lat"]
    if dim <= 3:
        return ["$x$", "$y$", "$z$"][:dim]
    return ["$x_{" + str(i) + "}$" for i in range(dim)]


def _plot_1d(pos, field, fig=None, ax=None, ax_names=None):  # pragma: no cover
    """Plot a 1d field."""
    fig, ax = _get_fig_ax(fig, ax)
    title = "Field 1D: " + str(field.shape)
    x = pos[0]
    x = x.flatten()
    arg = np.argsort(x)
    ax_name = _ax_names(1, ax_names=ax_names)[0]
    ax.plot(x[arg], field.ravel()[arg])
    ax.set_xlabel(ax_name)
    ax.set_ylabel("field")
    ax.set_title(title)
    fig.show()
    return ax


def _plot_2d(
    pos, field, mesh_type, fig=None, ax=None, latlon=False, ax_names=None
):  # pragma: no cover
    """Plot a 2d field."""
    fig, ax = _get_fig_ax(fig, ax)
    title = "Field 2D " + mesh_type + ": " + str(field.shape)
    ax_names = _ax_names(2, latlon, ax_names=ax_names)
    y = pos[0] if latlon else pos[1]
    x = pos[1] if latlon else pos[0]
    if mesh_type == "unstructured":
        cont = ax.tricontourf(x, y, field.ravel(), levels=256)
    else:
        plot_field = field if latlon else field.T
        try:
            cont = ax.contourf(x, y, plot_field, levels=256)
        except TypeError:
            cont = ax.contourf(x, y, plot_field, 256)
    ax.set_xlabel(ax_names[0])
    ax.set_ylabel(ax_names[1])
    ax.set_title(title)
    fig.colorbar(cont)
    fig.show()
    return ax


def _plot_nd(
    pos,
    field,
    mesh_type,
    fig=None,
    ax=None,
    latlon=False,
    resolution=100,
    ax_names=None,
    aspect="quad",
):  # pragma: no cover
    """Plot N-D field."""
    dim = len(pos)
    assert dim > 1
    assert not latlon or dim == 2
    pos = pos[::-1] if latlon else pos
    ax_names = _ax_names(dim, latlon, ax_names)
    # init planes
    planes = rotation_planes(dim)
    plane_names = [
        " {} - {}".format(ax_names[p[0]], ax_names[p[1]]) for p in planes
    ]
    ax_ends = [[p.min(), p.max()] for p in pos]
    ax_rngs = [end[1] - end[0] for end in ax_ends]
    ax_steps = [rng / resolution for rng in ax_rngs]
    ax_ticks = [np.linspace(end[0], end[1], 5) for end in ax_ends]
    ax_extents = [ax_ends[p[0]] + ax_ends[p[1]] for p in planes]
    # create figure
    reformat = fig is None and ax is None
    fig, ax = _get_fig_ax(fig, ax)
    ax.set_title("Field {}D {} {}".format(dim, mesh_type, field.shape))
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
    elif mesh_type == "unstructured":
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
            asp = ax_rngs[planes[p][1]] / ax_rngs[planes[p][0]]
        if aspect is not None:
            ax.set_aspect(asp if aspect == "quad" else aspect)
        ax.set_xticks(ax_ticks[planes[p][0]])
        ax.set_yticks(ax_ticks[planes[p][1]])
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
    plot_field = getattr(fld, field)
    assert not (fld.pos is None or plot_field is None)

    norm = np.sqrt(plot_field[0, :].T ** 2 + plot_field[1, :].T ** 2)

    fig, ax = _get_fig_ax(fig, ax)
    title = "Field 2D " + fld.mesh_type + ": " + str(plot_field.shape)
    x = fld.pos[0]
    y = fld.pos[1]

    sp = plt.streamplot(
        x,
        y,
        plot_field[0, :].T,
        plot_field[1, :].T,
        color=norm,
        linewidth=norm / 2,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    fig.colorbar(sp.lines)
    fig.show()
    return ax
