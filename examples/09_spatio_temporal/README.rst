Spatio-Temporal Modeling
========================

Spatio-Temporal modelling can provide insights into time dependent processes
like rainfall, air temperature or crop yield.

GSTools provides the metric spatio-temporal model for all covariance models
by setting ``temporal=True``, which enhances the spatial model dimension with
a time dimension to result in the spatio-temporal dimension and setting a
spatio-temporal anisotropy ratio like this:

.. code-block:: python

    import gstools as gs
    dim = 3  # spatial dimension
    st_anis = 0.4
    st_model = gs.Exponential(dim=dim, temporal=True, anis=st_anis)

Since it is given in the name "spatio-temporal", time is always treated as last dimension.
There are three different dimension attributes giving information about (i) the
model dimension (``dim``), (ii) the field dimension (``field_dim``, including time) and
(iii) the spatial dimension (``spatial_dim`` always 1 less than ``field_dim`` for temporal models).
Model and field dimension can differ in case of geographic coordinates where the model dimension is 3,
but the field or parametric dimension is 2.
If the model is spatio-temporal one with geographic coordinates, the model dimension is 4,
the field dimension is 3 and the spatial dimension is 2.

In the case above we get:

.. code-block:: python

    st_model.dim == 4
    st_model.field_dim == 4
    st_model.spatial_dim == 3

This formulation enables us to have spatial anisotropy and rotation defined as in
non-temporal models, without altering the behavior in the time dimension:

.. code-block:: python

    anis = [0.4, 0.2]  # spatial anisotropy in 3D
    angles = [0.5, 0.4, 0.3]  # spatial rotation in 3D
    st_model = gs.Exponential(dim=dim, temporal=True, anis=anis+[st_anis], angles=angles)

In order to generate spatio-temporal position tuples, GSTools provides a
convenient function :any:`generate_st_grid`. The output can be used for
spatio-temporal random field generation (or kriging resp. conditioned fields):

.. code-block:: python

    pos = dim * [1, 2, 3]  # 3 points in space (1,1,1), (2,2,2) and (3,3,3)
    time = range(10)  # 10 time steps
    st_grid = gs.generate_st_grid(pos, time)
    st_rf = gs.SRF(st_model)
    st_field = st_rf(st_grid).reshape(-1, len(time))

Then we can access the different time-steps by the last array index.

Examples
--------
