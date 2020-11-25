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
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from gstools.covmodel.plot import _get_fig_ax

__all__ = ["plot_field", "plot_vec_field"]


# plotting routines #######################################################


def plot_field(fld, field="field", fig=None, ax=None):  # pragma: no cover
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
    """
    plot_field = getattr(fld, field)
    assert not (fld.pos is None or plot_field is None)
    if fld.model.field_dim == 1:
        ax = _plot_1d(fld.pos, plot_field, fig, ax)
    elif fld.model.field_dim == 2:
        ax = _plot_2d(
            fld.pos, plot_field, fld.mesh_type, fig, ax, fld.model.latlon
        )
    elif fld.model.field_dim == 3:
        ax = _plot_3d(fld.pos, plot_field, fld.mesh_type, fig, ax)
    else:
        raise ValueError("Field.plot: only possible for dim=1,2,3!")
    return ax


def _plot_1d(pos, field, fig=None, ax=None):  # pragma: no cover
    """Plot a 1d field."""
    fig, ax = _get_fig_ax(fig, ax)
    title = "Field 1D: " + str(field.shape)
    x = pos[0]
    x = x.flatten()
    arg = np.argsort(x)
    ax.plot(x[arg], field.ravel()[arg])
    ax.set_xlabel("X")
    ax.set_ylabel("field")
    ax.set_title(title)
    fig.show()
    return ax


def _plot_2d(
    pos, field, mesh_type, fig=None, ax=None, latlon=False
):  # pragma: no cover
    """Plot a 2d field."""
    fig, ax = _get_fig_ax(fig, ax)
    title = "Field 2D " + mesh_type + ": " + str(field.shape)
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
    if latlon:
        ax.set_ylabel("Lat in deg")
        ax.set_xlabel("Lon in deg")
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    ax.set_title(title)
    fig.colorbar(cont)
    fig.show()
    return ax


def _plot_3d(pos, field, mesh_type, fig=None, ax=None):  # pragma: no cover
    """Plot 3D field."""
    dir1, dir2 = np.mgrid[0:1:51j, 0:1:51j]
    levels = np.linspace(field.min(), field.max(), 100, endpoint=True)

    x_min = pos[0].min()
    x_max = pos[0].max()
    y_min = pos[1].min()
    y_max = pos[1].max()
    z_min = pos[2].min()
    z_max = pos[2].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    x_step = x_range / 50.0
    y_step = y_range / 50.0
    z_step = z_range / 50.0
    ax_info = {
        "x": [x_min, x_max, x_range, x_step],
        "y": [y_min, y_max, y_range, y_step],
        "z": [z_min, z_max, z_range, z_step],
    }
    fig, ax = _get_fig_ax(fig, ax)
    title = "Field 3D " + mesh_type + ": " + str(field.shape)
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.25)
    sax = plt.axes([0.15, 0.1, 0.65, 0.03])
    z_height = Slider(
        sax,
        "z value",
        z_min,
        z_max,
        valinit=z_min + z_range / 2.0,
        valstep=z_step,
    )
    rax = plt.axes([0.05, 0.7, 0.1, 0.15])
    radio = RadioButtons(rax, ("x slice", "y slice", "z slice"), active=2)
    z_dir_tmp = "z"
    # create container
    container_class = type(
        "info", (object,), {"z_height": z_height, "z_dir_tmp": z_dir_tmp}
    )
    container = container_class()

    def get_plane(z_val_in, z_dir):
        """Get the plane."""
        if z_dir == "z":
            x_io = dir1 * x_range + x_min
            y_io = dir2 * y_range + y_min
            z_io = np.full_like(x_io, z_val_in)
        elif z_dir == "y":
            x_io = dir1 * x_range + x_min
            z_io = dir2 * z_range + z_min
            y_io = np.full_like(x_io, z_val_in)
        else:
            y_io = dir1 * y_range + y_min
            z_io = dir2 * z_range + z_min
            x_io = np.full_like(y_io, z_val_in)

        if mesh_type != "unstructured":
            # contourf plots image like for griddata, therefore transpose
            plane = inter.interpn(
                pos, field, np.array((x_io, y_io, z_io)).T, bounds_error=False
            ).T
        else:
            plane = inter.griddata(
                pos, field, (x_io, y_io, z_io), method="linear"
            )
        if z_dir == "x":
            return y_io, z_io, plane
        elif z_dir == "y":
            return x_io, z_io, plane
        return x_io, y_io, plane

    def update(__):
        """Widget update."""
        z_dir_in = radio.value_selected[0]
        if z_dir_in != container.z_dir_tmp:
            sax.clear()
            container.z_height = Slider(
                sax,
                z_dir_in + " value",
                ax_info[z_dir_in][0],
                ax_info[z_dir_in][1],
                valinit=ax_info[z_dir_in][0] + ax_info[z_dir_in][2] / 2.0,
                valstep=ax_info[z_dir_in][3],
            )
            container.z_height.on_changed(update)
            container.z_dir_tmp = z_dir_in
        z_val = container.z_height.val
        ax.clear()
        xx, yy, zz = get_plane(z_val, z_dir_in)
        cont = ax.contourf(
            xx,
            yy,
            zz,
            vmin=field.min(),
            vmax=field.max(),
            levels=levels,
        )
        # cont.cmap.set_under("k", alpha=0.0)
        # cont.cmap.set_bad("k", alpha=0.0)
        if z_dir_in == "x":
            ax.set_xlabel("Y")
            ax.set_ylabel("Z")
        elif z_dir_in == "y":
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
        else:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_title(title)
        fig.canvas.draw_idle()
        return cont

    container.z_height.on_changed(update)
    radio.on_clicked(update)
    cont = update(0)
    cax = plt.axes([0.85, 0.2, 0.03, 0.6])
    fig.colorbar(cont, cax=cax, ax=ax)
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
            "Only structured vector fields are supported"
            + " for plotting. Please create one on a structured grid."
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
