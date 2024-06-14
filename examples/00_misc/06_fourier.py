"""
Generating a Simple Periodic Random Field
-----------------------------------------

In this simple example we are going to learn how to generate periodic spatial
random fields. The Fourier method comes naturally with the property of
periodicity, so we'll use it to create the random field.
"""

import numpy as np

import gstools as gs

# We start off by defining the spatial grid.
L = np.array((500, 500))
x = np.linspace(0, L[0], 256)
y = np.linspace(0, L[1], 128)

# And by setting up a Gaussian covariance model with a correlation length
# scale which is roughly half the size of the grid.
model = gs.Gaussian(dim=2, var=1, len_scale=200)

# Next, we hand the cov. model to the spatial random field class
# and set the generator to `Fourier`. The `mode_no` argument sets the number of
# Fourier modes per dimension. The argument `period` is set to the domain size.
srf = gs.SRF(
    model,
    generator="Fourier",
    mode_no=[32, 32],
    period=L,
    seed=1681903,
)

# Now, we can calculate the field with the given parameters.
srf((x, y), mesh_type="structured")

# GSTools has a few simple visualization methods built in.
srf.plot()
