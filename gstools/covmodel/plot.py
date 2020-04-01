# -*- coding: utf-8 -*-
"""
GStools subpackage providing plotting routines for the covariance models.

.. currentmodule:: gstools.covmodel.plot

The following classes and functions are provided

.. autosummary::
   plot_variogram
   plot_covariance
   plot_correlation
   plot_vario_spatial
   plot_cov_spatial
   plot_cor_spatial
   plot_spectrum
   plot_spectral_density
   plot_spectral_rad_pdf
"""
# pylint: disable=C0103
import numpy as np

import gstools
from gstools.field.tools import reshape_axis_from_struct_to_unstruct

__all__ = [
    "plot_variogram",
    "plot_covariance",
    "plot_correlation",
    "plot_vario_spatial",
    "plot_cov_spatial",
    "plot_cor_spatial",
    "plot_spectrum",
    "plot_spectral_density",
    "plot_spectral_rad_pdf",
]


# plotting routines #######################################################


def _get_fig_ax(fig, ax, ax_name="rectilinear"):  # pragma: no cover
    from matplotlib import pyplot as plt

    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ax_name)
    elif ax is None:
        ax = fig.add_subplot(111, projection=ax_name)
    elif fig is None:
        fig = ax.get_figure()
        assert ax.name == ax_name
    else:
        assert ax.name == ax_name
        assert ax.get_figure() == fig
    return fig, ax


def plot_vario_spatial(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot spatial variogram of a given CovModel."""
    field = gstools.field.base.Field(model)
    field._value_type = "scalar"
    if x_max is None:
        x_max = 3 * model.integral_scale
    field.mesh_type = "structured"
    x_s = np.linspace(-x_max, x_max) + x_min
    pos = [x_s] * model.dim
    x, y, z, shape = reshape_axis_from_struct_to_unstruct(model.dim, *pos)
    vario = model.vario_spatial([x, y, z][: model.dim]).reshape(shape)
    field.pos = pos
    field.field = vario
    return field.plot(fig=fig, ax=ax)


def plot_cov_spatial(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot spatial covariance of a given CovModel."""
    field = gstools.field.base.Field(model)
    field._value_type = "scalar"
    if x_max is None:
        x_max = 3 * model.integral_scale
    field.mesh_type = "structured"
    x_s = np.linspace(-x_max, x_max) + x_min
    pos = [x_s] * model.dim
    x, y, z, shape = reshape_axis_from_struct_to_unstruct(model.dim, *pos)
    vario = model.cov_spatial([x, y, z][: model.dim]).reshape(shape)
    field.pos = pos
    field.field = vario
    return field.plot(fig=fig, ax=ax)


def plot_cor_spatial(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot spatial correlation of a given CovModel."""
    field = gstools.field.base.Field(model)
    field._value_type = "scalar"
    if x_max is None:
        x_max = 3 * model.integral_scale
    field.mesh_type = "structured"
    x_s = np.linspace(-x_max, x_max) + x_min
    pos = [x_s] * model.dim
    x, y, z, shape = reshape_axis_from_struct_to_unstruct(model.dim, *pos)
    vario = model.cor_spatial([x, y, z][: model.dim]).reshape(shape)
    field.pos = pos
    field.field = vario
    return field.plot(fig=fig, ax=ax)


def plot_variogram(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot variogram of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    ax.plot(x_s, model.variogram(x_s), label=model.name + " variogram")
    ax.legend()
    fig.show()
    return ax


def plot_covariance(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot covariance of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    ax.plot(x_s, model.covariance(x_s), label=model.name + " covariance")
    ax.legend()
    fig.show()
    return ax


def plot_correlation(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot correlation function of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    ax.plot(x_s, model.correlation(x_s), label=model.name + " correlation")
    ax.legend()
    fig.show()
    return ax


def plot_spectrum(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot specturm of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 / model.integral_scale
    x_s = np.linspace(x_min, x_max)
    ax.plot(
        x_s,
        model.spectrum(x_s),
        label=model.name + " " + str(model.dim) + "D spectrum",
    )
    ax.legend()
    fig.show()
    return ax


def plot_spectral_density(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot spectral density of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 / model.integral_scale
    x_s = np.linspace(x_min, x_max)
    ax.plot(
        x_s,
        model.spectral_density(x_s),
        label=model.name + " " + str(model.dim) + "D spectral-density",
    )
    ax.legend()
    fig.show()
    return ax


def plot_spectral_rad_pdf(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot radial spectral pdf of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 / model.integral_scale
    x_s = np.linspace(x_min, x_max)
    ax.plot(
        x_s,
        model.spectral_rad_pdf(x_s),
        label=model.name + " " + str(model.dim) + "D spectral-rad-pdf",
    )
    ax.legend()
    fig.show()
    return ax
