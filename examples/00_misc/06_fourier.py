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
x = np.linspace(0, 500, 256)
y = np.linspace(0, 500, 128)

# And by setting up a Gaussian covariance model with a correlation length
# scale which is roughly half the size of the grid.
model = gs.Gaussian(dim=2, var=1, len_scale=200)

# Next, we hand the cov. model to the spatial random field class
# and set the generator to `Fourier`. The higher the modes_no, the better
# the quality of the generated field, but also the computing time increases.
# The modes_truncation are the cut-off values of the Fourier modes and finally,
# the seed ensures that we generate the same random field each time.
srf = gs.SRF(
    model,
    generator="Fourier",
    modes_no=[16, 8],
    modes_truncation=[16, 8],
    seed=1681903,
)

# Now, we can finally calculate the field with the given parameters.
srf((x, y), mesh_type="structured")

# GSTools has a few simple visualization methods built in.
srf.plot()
