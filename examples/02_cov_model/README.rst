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
   \sigma^2\cdot\left(1-\mathrm{cor}\left(s\cdot\frac{r}{\ell}\right)\right)+n

Where:

  - :math:`r` is the lag distance
  - :math:`\ell` is the main correlation length
  - :math:`s` is a scaling factor for unit conversion or normalization
  - :math:`\sigma^2` is the variance
  - :math:`n` is the nugget (subscale variance)
  - :math:`\mathrm{cor}(h)` is the normalized correlation function depending on
    the non-dimensional distance :math:`h=s\cdot\frac{r}{\ell}`

Depending on the normalized correlation function, all covariance models in
GSTools are providing the following functions:

  - :math:`\rho(r)=\mathrm{cor}\left(s\cdot\frac{r}{\ell}\right)`
    is the so called
    `correlation <https://en.wikipedia.org/wiki/Autocovariance#Normalization>`_
    function
  - :math:`C(r)=\sigma^2\cdot\rho(r)` is the so called
    `covariance <https://en.wikipedia.org/wiki/Covariance_function>`_
    function, which gives the name for our GSTools class

.. note::

   We are not limited to isotropic models. GSTools supports anisotropy ratios
   for length scales in orthogonal transversal directions like:

   - :math:`x_0` (main direction)
   - :math:`x_1` (1. transversal direction)
   - :math:`x_2` (2. transversal direction)
   - ...

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
    HyperSpherical
    SuperSpherical
    JBessel

As a special feature, we also provide truncated power law (TPL) covariance models

.. autosummary::
    TPLGaussian
    TPLExponential
    TPLStable
    TPLSimple

Gallery
-------
