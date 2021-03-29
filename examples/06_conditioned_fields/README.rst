Conditioned Fields
==================

Kriged fields tend to approach the field mean outside the area of observations.
To generate random fields, that coincide with given observations, but are still
random according to a given covariance model away from the observations proximity,
we provide the generation of conditioned random fields.

The idea behind conditioned random fields builds up on kriging.
First we generate a field with a kriging method, then we generate a random field,
with 0 as mean and 1 as variance that will be multiplied with the kriging
standard deviation.

To do so, you can instantiate a :any:`CondSRF` class with a configured
:any:`Krige` class.

The setup of the a conditioned random field should be as follows:

.. code-block:: python

    krige = gs.Krige(model, cond_pos, cond_val)
    cond_srf = CondSRF(krige)
    field = cond_srf(grid)

Examples
--------
