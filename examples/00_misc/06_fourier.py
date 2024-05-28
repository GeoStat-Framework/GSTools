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
# and set the generator to `Fourier`. We will let the class figure out the
# modes internally, by handing over `period` and `mode_rel_cutoff` which is the cutoff
# value of the spectral density, relative to the maximum spectral density at
# the origin. Simply put, we will use `mode_rel_cutoff`% of the spectral
# density for the calculations. The argument `period` is set to the domain
# size.
srf = gs.SRF(
    model,
    generator="Fourier",
    mode_rel_cutoff=0.99,
    period=L,
    seed=1681903,
)

# Now, we can calculate the field with the given parameters.
srf((x, y), mesh_type='structured')

# GSTools has a few simple visualization methods built in.
srf.plot()

# Alternatively, we could calculate the modes ourselves and hand them over to
# GSTools. Therefore, we set the cutoff values to absolut values in Fourier
# space. But always check, if you cover enough of the spectral density to not
# run into numerical problems.
modes_cutoff = [1., 1.]

# Next, we have to compute the numerical step size in Fourier space. This choice
# influences the periodicity, which we want to set to the domain size by
modes_delta = 2 * np.pi / L

# Now, we calculate the modes with
modes = [np.arange(0, modes_cutoff[d], modes_delta[d]) for d in 2]

# And we can create a new instance of the SRF class with our own modes.
srf_modes = gs.SRF(
    model,
    generator="Fourier",
    modes=modes,
    seed=494754,
)
