Tutorial 7: Field transformations
=================================

The generated fields of gstools are ordinary Gaussian random fields.
In application there are several transformations to describe real world
problems in an appropriate manner.

GStools provides a submodule :py:mod:`gstools.transform` with a range of
common transformations:

.. currentmodule:: gstools.transform

.. autosummary::
   binary
   discrete
   boxcox
   zinnharvey
   normal_force_moments
   normal_to_lognormal
   normal_to_uniform
   normal_to_arcsin
   normal_to_uquad


All the transformations take a field class, that holds a generated field,
as input and will manipulate this field inplace.

Simply import the transform submodule and apply a transformation to the srf class:

.. code-block:: python

    from gstools import transform as tf
    ...
    tf.normal_to_lognormal(srf)

Gallery
-------
