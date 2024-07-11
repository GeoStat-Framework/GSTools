"""
Generating a Simple Periodic Random Field
-----------------------------------------

In this simple example we are going to learn how to generate periodic spatial
random fields. The Fourier method comes naturally with the property of
periodicity, so we'll use it to create the random field.
"""

import numpy as np

import gstools as gs

# We start off by defining the spatial grid. For the sake of simplicity, we
# use a square domain. We set the optional argument `endpoint` to `False`, to
# not make the domain in each dimension one grid cell larger than the
# periodicity.
L = 500.0
x = np.linspace(0, L, 256, endpoint=False)
y = np.linspace(0, L, 128, endpoint=False)

Now, we create a Gaussian covariance model with a correlation length which is
# roughly half the size of the grid.
model = gs.Gaussian(dim=2, var=1, len_scale=200)

# Next, we hand the cov. model to the spatial random field class `SRF`
# and set the generator to `"Fourier"`. The argument `period` is set to the
# domain size. If only a single number is given, the same periodicity is
# applied in each dimension, as shown in this example. The `mode_no` argument
# sets the number of Fourier modes. If only an integer is given, that number
# of modes is used for all dimensions.
srf = gs.SRF(
    model,
    generator="Fourier",
    period=L,
    mode_no=32,
    seed=1681903,
)

# Now, we can calculate the field with the given parameters.
srf((x, y), mesh_type="structured")

# GSTools has a few simple visualization methods built in.
srf.plot()
