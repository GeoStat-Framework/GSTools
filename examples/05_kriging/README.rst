.. _tutorial_05_kriging:

Kriging
=======

The subpackage :py:mod:`gstools.krige` provides routines for Gaussian process regression,
also known as kriging.
Kriging is a method of data interpolation based on predefined covariance models.

The aim of kriging is to derive the value of a field at some point :math:`x_0`,
when there are fixed observed values :math:`z(x_1)\ldots z(x_n)` at given points :math:`x_i`.

The resluting value :math:`z_0` at :math:`x_0` is calculated as a weighted mean:

.. math::

   z_0 = \sum_{i=1}^n w_i \cdot z_i

The weights :math:`W = (w_1,\ldots,w_n)` depent on the given covariance model and the location of the target point.

The different kriging approaches provide different ways of calculating :math:`W`.

The :any:`Krige` class provides everything in one place and you can switch on/off
the features you want:

* `unbiased`: the weights have to sum up to `1`. If true, this results in
  :any:`Ordinary` kriging, where the mean is estimated, otherwise it will result in
  :any:`Simple` kriging, where the mean has to be given.
* `drift_functions`: you can give a polynomial order or a list of self defined
  functions representing the internal drift of the given values. This drift will
  be fitted internally during the kriging interpolation. This results in :any:`Universal` kriging.
* `ext_drift`: You can also give an external drift per point to the routine.
  In contrast to the internal drift, that is evaluated at the desired points with
  the given functions, the external drift has to given for each point form an "external"
  source. This results in :any:`ExtDrift` kriging.
* `trend_function`: If you already have fitted a trend model, that is provided as a
  callable, you can give it to the kriging routine. This trend is subtracted from the
  conditional values before the kriging is done, meaning, that only the residuals are
  used for kriging. This can be used with separate regression of your data.
  This results in :any:`Detrended` kriging.
* `exact` and `cond_err`: To incorporate the nugget effect and/or measurement errors,
  one can set `exact` to `False` and provide either individual measurement errors
  for each point or set the nugget as a constant measurement error everywhere.
* `pseudo_inv`: Sometimes the inversion of the kriging matrix can be numerically unstable.
  This occurs for examples in cases of redundant input values. In this case we provide a switch to
  use the pseudo-inverse of the matrix. Then redundant conditional values will automatically
  be averaged.

.. note::

   All mentioned features can be combined within the :any:`Krige` class.
   All other kriging classes are just shortcuts to this class with a limited list
   of input parameters.

The routines for kriging are almost identical to the routines for spatial random fields,
with regard to their handling.
First you define a covariance model, as described in :ref:`tutorial_02_cov`,
then you initialize the kriging class with this model:

.. code-block:: python

    import gstools as gs
    # condtions
    cond_pos = [...]
    cond_val = [...]
    model = gs.Gaussian(dim=1, var=0.5, len_scale=2)
    krig = gs.krige.Simple(model, cond_pos=cond_pos, cond_val=cond_val, mean=1)

The resulting field instance ``krig`` has the same methods as the
:any:`SRF` class.
You can call it to evaluate the kriged field at different points,
you can plot the latest field or you can export the field and so on.

Provided Kriging Methods
------------------------

.. currentmodule:: gstools.krige

The following kriging methods are provided within the
submodule :any:`gstools.krige`.

.. autosummary::
    Krige
    Simple
    Ordinary
    Universal
    ExtDrift
    Detrended

.. only:: html

   Gallery
   -------

   Below is a gallery of examples
