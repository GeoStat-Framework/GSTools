Spatio-Temporal Modeling
========================

Spatio-Temporal modelling can provide insights on time depending processes
like rainfall, air temperature or crop yield.

GSTools provides the metric spatio-temporal model for all covariance models
by enhancing the spatial model dimension with a time dimesion to result in
the spatio-temporal dimension ``st_dim`` and setting a
spatio-temporal anisotropy ratio ``st_anis``:

.. code-block:: python

    import gstools as gs
    dim = 3  # spatial dimension
    st_dim = dim + 1
    st_anis = 0.4
    model = gs.Exponential(dim=st_dim, anis=st_anis)

Since it is given in the name "spatio-temporal",
we will always treat the time as last dimension.

This enables to have spatial anisotropy and rotation defined as in
non-temporal models, without altering the behavior in the time dimension:

.. code-block:: python

    anis = [0.4, 0.2]  # spatial anisotropy in 3D
    angles = [0.5, 0.4, 0.3]  # spatial rotation in 3D
    model = gs.Exponential(dim=st_dim, anis=anis + [st_anis], angles=angles)

In order to generate spatio-temporal position tuples, GSTools provides a
convenient function :any:`generate_st_grid`:

.. code-block:: python

    pos = ...
    time = range(10)
    st_pos = gs.generate_st_grid(pos, time)
    st_rf = SRF(st_model)
    st_field = st_rf(st_pos).reshape(-1, len(time))

Then you we access the different time-steps with the last index.

Examples
--------
