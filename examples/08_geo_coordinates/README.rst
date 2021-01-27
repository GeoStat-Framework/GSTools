Geographic Coordinates
======================

GSTools provides support for
`geographic coordinates <https://en.wikipedia.org/wiki/Geographic_coordinate_system>`_
given by:

- latitude ``lat``: specifies the north–south position of a point on the Earth's surface
- longitude ``lon``: specifies the east–west position of a point on the Earth's surface

If you want to use this feature for field generation or Kriging, you
have to set up a geographical covariance Model by setting ``latlon=True``
in your desired model (see :any:`CovModel`):

.. code-block:: python

    import numpy as np
    import gstools as gs

    model = gs.Gaussian(latlon=True, var=2, len_scale=np.pi / 16)

By doing so, the model will use the associated `Yadrenko` model on a sphere
(see `here <https://onlinelibrary.wiley.com/doi/abs/10.1002/sta4.84>`_).
The `len_scale` is given in radians to scale the arc-length.
In order to have a more meaningful length scale, one can use the ``rescale``
argument:

.. code-block:: python

    import gstools as gs

    model = gs.Gaussian(latlon=True, var=2, len_scale=500, rescale=gs.EARTH_RADIUS)

Then ``len_scale`` can be interpreted as given in km.

A `Yadrenko` model :math:`C` is derived from a valid
isotropic covariance model in 3D :math:`C_{3D}` by the following relation:

.. math::
   C(\zeta)=C_{3D}\left(2 \cdot \sin\left(\frac{\zeta}{2}\right)\right)

Where :math:`\zeta` is the
`great-circle distance <https://en.wikipedia.org/wiki/Great-circle_distance>`_.

.. note::

   ``lat`` and ``lon`` are given in degree, whereas the great-circle distance
   :math:`zeta` is given in radians.

Note, that :math:`2 \cdot \sin(\frac{\zeta}{2})` is the
`chordal distance <https://en.wikipedia.org/wiki/Chord_(geometry)>`_
of two points on a sphere, which means we simply think of the earth surface
as a sphere, that is cut out of the surrounding three dimensional space,
when using the `Yadrenko` model.

.. note::

   Anisotropy is not available with the geographical models, since their
   geometry is not euclidean. When passing values for :any:`CovModel.anis`
   or :any:`CovModel.angles`, they will be ignored.

   Since the Yadrenko model comes from a 3D model, the model dimension will
   be 3 (see :any:`CovModel.dim`) but the `field_dim` will be 2 in this case
   (see :any:`CovModel.field_dim`).

Examples
--------
