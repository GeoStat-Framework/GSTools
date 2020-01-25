.. _tutorial_02_cov:

Tutorial 2: The Covariance Model
================================

One of the core-features of GSTools is the powerful :any:`CovModel`
class, which allows you to easily define arbitrary covariance models by
yourself. The resulting models provide a bunch of nice features to explore the
covariance models.


A covariance model is used to characterize the
`semi-variogram <https://en.wikipedia.org/wiki/Variogram#Semivariogram>`_,
denoted by :math:`\gamma`, of a spatial random field.
In GSTools, we use the following form for an isotropic and stationary field:

.. math::
   \gamma\left(r\right)=
   \sigma^2\cdot\left(1-\rho\left(r\right)\right)+n

Where:

  - :math:`\rho(r)` is the so called
    `correlation <https://en.wikipedia.org/wiki/Autocovariance#Normalization>`_
    function depending on the distance :math:`r`
  - :math:`\sigma^2` is the variance
  - :math:`n` is the nugget (subscale variance)

.. note::

   We are not limited to isotropic models. GSTools supports anisotropy ratios
   for length scales in orthogonal transversal directions like:

   - :math:`x` (main direction)
   - :math:`y` (1. transversal direction)
   - :math:`z` (2. transversal direction)

   These main directions can also be rotated.
   Just have a look at the corresponding examples.

Provided Covariance Models
--------------------------

.. currentmodule:: gstools.covmodel

The following standard covariance models are provided by GSTools

.. autosummary::
    Gaussian
    Exponential
    Matern
    Stable
    Rational
    Linear
    Circular
    Spherical
    Intersection

As a special feature, we also provide truncated power law (TPL) covariance models

.. autosummary::
    TPLGaussian
    TPLExponential
    TPLStable

Gallery
-------
