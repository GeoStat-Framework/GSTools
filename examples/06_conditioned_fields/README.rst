Tutorial 6: Conditioned Fields
==============================

Kriged fields tend to approach the field mean outside the area of observations.
To generate random fields, that coincide with given observations, but are still
random according to a given covariance model away from the observations proximity,
we provide the generation of conditioned random fields.


The idea behind conditioned random fields builds up on kriging.
First we generate a field with a kriging method, then we generate a random field,
and finally we generate another kriged field to eliminate the error between
the random field and the kriged field of the given observations.

To do so, you can choose between ordinary and simple kriging.
In case of ordinary kriging, the mean of the SRF will be overwritten by the
estimated mean.

The setup of the spatial random field is the same as described in
:ref:`tutorial_02_cov`.
You just need to add the conditions as described in :ref:`tutorial_05_kriging`:

.. code-block:: python

    srf.set_condition(cond_pos, cond_val, "simple")

or:

.. code-block:: python

    srf.set_condition(cond_pos, cond_val, "ordinary")

Gallery
-------
