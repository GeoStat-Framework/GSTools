# -*- coding: utf-8 -*-
"""
GStools subpackage providing plotting routines for the covariance models.

.. currentmodule:: gstools.covmodel.plot

The following classes and functions are provided

.. autosummary::
   plot_variogram
   plot_covariance
   plot_correlation
   plot_vario_yadrenko
   plot_cov_yadrenko
   plot_cor_yadrenko
   plot_vario_axis
   plot_cov_axis
   plot_cor_axis
   plot_vario_spatial
   plot_cov_spatial
   plot_cor_spatial
   plot_spectrum
   plot_spectral_density
   plot_spectral_rad_pdf
"""
# pylint: disable=C0103, C0415
import numpy as np

import gstools


__all__ = [
    "plot_variogram",
    "plot_covariance",
    "plot_correlation",
    "plot_vario_yadrenko",
    "plot_cov_yadrenko",
    "plot_cor_yadrenko",
    "plot_vario_axis",
    "plot_cov_axis",
    "plot_cor_axis",
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
        x_max = 3 * model.len_scale
    x_s = np.linspace(-x_max, x_max) + x_min
    iso_pos, shape = field.pre_pos([x_s] * model.dim, "structured")
    field.field = model.vario_spatial(model.anisometrize(iso_pos)).reshape(
        shape
    )
    return field.plot(fig=fig, ax=ax)


def plot_cov_spatial(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot spatial covariance of a given CovModel."""
    field = gstools.field.base.Field(model)
    field._value_type = "scalar"
    if x_max is None:
        x_max = 3 * model.len_scale
    x_s = np.linspace(-x_max, x_max) + x_min
    iso_pos, shape = field.pre_pos([x_s] * model.dim, "structured")
    field.field = model.cov_spatial(model.anisometrize(iso_pos)).reshape(shape)
    return field.plot(fig=fig, ax=ax)


def plot_cor_spatial(
    model, x_min=0.0, x_max=None, fig=None, ax=None
):  # pragma: no cover
    """Plot spatial correlation of a given CovModel."""
    field = gstools.field.base.Field(model)
    field._value_type = "scalar"
    if x_max is None:
        x_max = 3 * model.len_scale
    x_s = np.linspace(-x_max, x_max) + x_min
    iso_pos, shape = field.pre_pos([x_s] * model.dim, "structured")
    field.field = model.cor_spatial(model.anisometrize(iso_pos)).reshape(shape)
    return field.plot(fig=fig, ax=ax)


def plot_variogram(
    model, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot variogram of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 * model.len_scale
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault("label", model.name + " variogram")
    ax.plot(x_s, model.variogram(x_s), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_covariance(
    model, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot covariance of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 * model.len_scale
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault("label", model.name + " covariance")
    ax.plot(x_s, model.covariance(x_s), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_correlation(
    model, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot correlation function of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 * model.len_scale
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault("label", model.name + " correlation")
    ax.plot(x_s, model.correlation(x_s), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_vario_yadrenko(
    model, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot Yadrenko variogram of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = min(3 * model.len_rescaled, np.pi)
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault("label", model.name + " Yadrenko variogram")
    ax.plot(x_s, model.vario_yadrenko(x_s), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_cov_yadrenko(
    model, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot Yadrenko covariance of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = min(3 * model.len_rescaled, np.pi)
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault("label", model.name + " Yadrenko covariance")
    ax.plot(x_s, model.cov_yadrenko(x_s), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_cor_yadrenko(
    model, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot Yadrenko correlation function of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = min(3 * model.len_rescaled, np.pi)
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault("label", model.name + " Yadrenko correlation")
    ax.plot(x_s, model.cor_yadrenko(x_s), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_vario_axis(
    model, axis=0, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot variogram of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 * model.len_scale
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault(
        "label", model.name + " variogram on axis {}".format(axis)
    )
    ax.plot(x_s, model.vario_axis(x_s, axis), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_cov_axis(
    model, axis=0, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot variogram of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 * model.len_scale
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault(
        "label", model.name + " covariance on axis {}".format(axis)
    )
    ax.plot(x_s, model.cov_axis(x_s, axis), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_cor_axis(
    model, axis=0, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot variogram of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 * model.len_scale
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault(
        "label", model.name + " correlation on axis {}".format(axis)
    )
    ax.plot(x_s, model.cor_axis(x_s, axis), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_spectrum(
    model, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot specturm of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 / model.len_scale
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault(
        "label", model.name + " " + str(model.dim) + "D spectrum"
    )
    ax.plot(x_s, model.spectrum(x_s), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_spectral_density(
    model, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot spectral density of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 / model.len_scale
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault(
        "label", model.name + " " + str(model.dim) + "D spectral-density"
    )
    ax.plot(x_s, model.spectral_density(x_s), **kwargs)
    ax.legend()
    fig.show()
    return ax


def plot_spectral_rad_pdf(
    model, x_min=0.0, x_max=None, fig=None, ax=None, **kwargs
):  # pragma: no cover
    """Plot radial spectral pdf of a given CovModel."""
    fig, ax = _get_fig_ax(fig, ax)
    if x_max is None:
        x_max = 3 / model.len_scale
    x_s = np.linspace(x_min, x_max)
    kwargs.setdefault(
        "label", model.name + " " + str(model.dim) + "D spectral-rad-pdf"
    )
    ax.plot(x_s, model.spectral_rad_pdf(x_s), **kwargs)
    ax.legend()
    fig.show()
    return ax
