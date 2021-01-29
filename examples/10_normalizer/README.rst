Normalizing Data
================

When dealing with real-world data, one can't assume it to be normal distributed.
In fact, many properties are modeled by applying different transformations,
for example conductivity is often assumed to be log-normal or precipitation
is transformed using the famous box-cox power transformation.

These "normalizers" are often represented as parameteric power transforms and
one is interested in finding the best parameter to gain normality in the input
data.

This is of special interest when kriging should be applied, since the target
variable of the kriging interpolation is assumed to be normal distributed.

GSTools provides a set of Normalizers and routines to automatically fit these
to input data by minimizing the likelihood function.

Provided Normalizers
--------------------

The following normalizers can be passed to all Field-classes
(:any:`SRF` or :any:`Krige`) or can be used as standalone tools to analyse data.

.. currentmodule:: gstools.normalizer

.. autosummary::
   LogNormal
   BoxCox
   BoxCoxShift
   YeoJohnson
   Modulus
   Manly

Examples
--------
