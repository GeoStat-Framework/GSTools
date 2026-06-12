Field transformations
=====================

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
   apply_function


All the transformations take a field class, that holds a generated field,
as input and will manipulate this field inplace or store it with a given name.

Simply apply a transformation to a field class:

.. code-block:: python

    import gstools as gs
    ...
    srf = gs.SRF(model)
    srf(...)
    gs.transform.normal_to_lognormal(srf)

Or use the provided wrapper:

.. code-block:: python

    import gstools as gs
    ...
    srf = gs.SRF(model)
    srf(...)
    srf.transform("lognormal")

Examples
--------
