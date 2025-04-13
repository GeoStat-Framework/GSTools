"""
Creating PGS with periodic boundaries
-------------------------------------

Plurigaussian fields with periodic boundaries (P-PGS) are used in various
applications, including the simulation of interactions between the landsurface
and the atmosphere, as well as the application of homogenisation theory to
porous media, e.g. [Ricketts 2024](https://doi.org/10.1007/s11242-024-02074-z).

In this example we will use GSTools's Fourier generator to create periodic
random fields, which can in turn be used to generate P-PGS.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

dim = 2
# define the spatial grid, see the periodic random field [examples](https://geostat-framework.readthedocs.io/projects/gstools/en/latest/examples/01_random_field/08_fourier.html)
# for details.

# domain size and periodicity
lithotypes = 200
# no. of cells in both dimensions
N = [170, 153]

x = np.linspace(0, lithotypes, N[0], endpoint=False)
y = np.linspace(0, lithotypes, N[1], endpoint=False)

###############################################################################
# The parameters of the covariance model are very similar to previous examples.
# The interesting part is the SRF class. We set the `generator` to `"Fourier"`,
# which inherently generates periodic SRFs. The Fourier generator needs an
# extra parameter `period` which defines the periodicity.

model = gs.Gaussian(dim=dim, var=0.8, len_scale=40)
srf = gs.SRF(model, generator="Fourier", period=lithotypes)
field1 = srf.structured([x, y], seed=19770319)
field2 = srf.structured([x, y], seed=19860912)

###############################################################################
# Very similar to previous examples, we create a simple lithotypes field.

M = [200, 160]

# size of the rectangle
rect = [40, 32]

lithotypes = np.zeros(M)
lithotypes[
    M[0] // 2 - rect[0] // 2 : M[0] // 2 + rect[0] // 2,
    M[1] // 2 - rect[1] // 2 : M[1] // 2 + rect[1] // 2,
] = 1

###############################################################################
# With the two SRFs and the lithotypes ready, we can create our first P-PGS.

pgs = gs.PGS(dim, [field1, field2])
P = pgs(lithotypes)

###############################################################################
# Finally, we can plot the PGS, but we will also show the lithotypes and the
# two original periodic Gaussian fields. Especially with `field1` you can
# nicely see the periodic structures in the black structure in the upper right
# corner. This transfers to the P-PGS, where you can see that the structures
# seemlessly match the opposite boundaries.

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(field1, cmap="copper", origin="lower")
axs[0, 1].imshow(field2, cmap="copper", origin="lower")
axs[1, 0].imshow(lithotypes, cmap="copper", origin="lower")
axs[1, 1].imshow(P, cmap="copper", origin="lower")

plt.show()
