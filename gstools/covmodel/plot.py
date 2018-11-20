# -*- coding: utf-8 -*-
"""
GStools subpackage providing plotting routines for the covariance models.

.. currentmodule:: gstools.covmodel.plot

The following classes and functions are provided

.. autosummary::
   plot_variogram
   plot_variogram_normed
   plot_covariance
   plot_covariance_normed
   plot_spectrum
   plot_spectral_density
   plot_spectral_rad_pdf
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from matplotlib import pyplot as plt


# plotting routines #######################################################


def plot_variogram(model, x_min=0.0, x_max=None):
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s, model.variogram(x_s), label=model.__class__.__name__ + " vario"
    )
    plt.legend()
    plt.show()


def plot_covariance(model, x_min=0.0, x_max=None):
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s, model.covariance(x_s), label=model.__class__.__name__ + " cov"
    )
    plt.legend()
    plt.show()


def plot_covariance_normed(model, x_min=0.0, x_max=None):
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s,
        model.covariance_normed(x_s),
        label=model.__class__.__name__ + " cov normed",
    )
    plt.legend()
    plt.show()


def plot_variogram_normed(model, x_min=0.0, x_max=None):
    if x_max is None:
        x_max = 3 * model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s,
        model.variogram_normed(x_s),
        label=model.__class__.__name__ + " vario normed",
    )
    plt.legend()
    plt.show()


def plot_spectrum(model, x_min=0.0, x_max=None):
    if x_max is None:
        x_max = 3 / model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s,
        model.spectrum(x_s),
        label=model.__class__.__name__ + " " + str(model.dim) + "D spec",
    )
    plt.legend()
    plt.show()


def plot_spectral_density(model, x_min=0.0, x_max=None):
    if x_max is None:
        x_max = 3 / model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s,
        model.spectral_density(x_s),
        label=model.__class__.__name__ + " " + str(model.dim) + "D spec-dens",
    )
    plt.legend()
    plt.show()


def plot_spectral_rad_pdf(model, x_min=0.0, x_max=None):
    if x_max is None:
        x_max = 3 / model.integral_scale
    x_s = np.linspace(x_min, x_max)
    plt.plot(
        x_s,
        model.spectral_rad_pdf(x_s),
        label=model.__class__.__name__
        + " "
        + str(model.dim)
        + "D spec-rad-pdf",
    )
    plt.legend()
    plt.show()
