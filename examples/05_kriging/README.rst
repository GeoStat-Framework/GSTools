.. _tutorial_05_kriging:

Tutorial 5: Kriging
===================

The subpackage :py:mod:`gstools.krige` provides routines for Gaussian process regression, also known as kriging.
Kriging is a method of data interpolation based on predefined covariance models.

We provide two kinds of kriging routines:

* Simple: The data is interpolated with a given mean value for the kriging field.
* Ordinary: The mean of the resulting field is unkown and estimated during interpolation.


The aim of kriging is to derive the value of a field at some point :math:`x_0`,
when there are fixed observed values :math:`z(x_1)\ldots z(x_n)` at given points :math:`x_i`.

The resluting value :math:`z_0` at :math:`x_0` is calculated as a weighted mean:

.. math::

   z_0 = \sum_{i=1}^n w_i \cdot z_i

The weights :math:`W = (w_1,\ldots,w_n)` depent on the given covariance model and the location of the target point.

The different kriging approaches provide different ways of calculating :math:`W`.



The routines for kriging are almost identical to the routines for spatial random fields.
First you define a covariance model, as described in :ref:`tutorial_02_cov`,
then you initialize the kriging class with this model:

.. code-block:: python

    from gstools import Gaussian, krige
    # condtions
    cond_pos = ...
    cond_val = ...
    model = Gaussian(dim=1, var=0.5, len_scale=2)
    krig = krige.Simple(model, mean=1, cond_pos=cond_pos, cond_val=cond_val)

The resulting field instance ``krig`` has the same methods as the :any:`SRF` class.
You can call it to evaluate the kriged field at different points,
you can plot the latest field or you can export the field and so on.
Have a look at the documentation of :any:`Simple` and :any:`Ordinary`.
