# -*- coding: utf-8 -*-
"""
GStools subpackage providing plotting routines for the covariance models.

.. currentmodule:: gstools.covmodel.plot

The following classes and functions are provided

.. autosummary::
   plot_variogram
   plot_variogram_normed
   plot_covariance
   plot_correlation
   plot_spectrum
   plot_spectral_density
   plot_spectral_rad_pdf
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from matplotlib import pyplot as plt

__all__ = [
    "plot_variogram",
    "plot_variogram_normed",
    "plot_covariance",
    "plot_correlation",
    "plot_spectrum",
    "plot_spectral_density",
    "plot_spectral_rad_pdf",
]


# plotting routines #######################################################


def plot_variogram(model, x_min=0.0, x_max=None):
    """plot variogram of a given CovModel"""
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(x_s, model.variogram(x_s), label=model.name + " variogram")
    plt.legend()
    plt.show()


def plot_covariance(model, x_min=0.0, x_max=None):
    """plot covariance of a given CovModel"""
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(x_s, model.covariance(x_s), label=model.name + " cov")
    plt.legend()
    plt.show()


def plot_correlation(model, x_min=0.0, x_max=None):
    """plot correlation function of a given CovModel"""
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(x_s, model.correlation(x_s), label=model.name + " cov normed")
    plt.legend()
    plt.show()


def plot_variogram_normed(model, x_min=0.0, x_max=None):
    """plot normalized variogram of a given CovModel"""
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s, model.variogram_normed(x_s), label=model.name + " vario normed"
    )
    plt.legend()
    plt.show()


def plot_spectrum(model, x_min=0.0, x_max=None):
    """plot specturm of a given CovModel"""
    if x_max is None:
        x_max = 3 / model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s,
        model.spectrum(x_s),
        label=model.name + " " + str(model.dim) + "D spec",
    )
    plt.legend()
    plt.show()


def plot_spectral_density(model, x_min=0.0, x_max=None):
    """plot spectral density of a given CovModel"""
    if x_max is None:
        x_max = 3 / model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s,
        model.spectral_density(x_s),
        label=model.name + " " + str(model.dim) + "D spec-dens",
    )
    plt.legend()
    plt.show()


def plot_spectral_rad_pdf(model, x_min=0.0, x_max=None):
    """plot radial spectral propability density function of a given CovModel"""
    if x_max is None:
        x_max = 3 / model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s,
        model.spectral_rad_pdf(x_s),
        label=model.name + " " + str(model.dim) + "D spec-rad-pdf",
    )
    plt.legend()
    plt.show()
