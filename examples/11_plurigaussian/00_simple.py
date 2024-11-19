"""
A First and Simple Example
--------------------------

As a first example, we will create a two dimensional plurigaussian field
(PGS). Thus, we need two spatial random fields(SRF) and on top of that, we
need a field describing the categorical data and its spatial relation.
We will start off by creating the two SRFs with a Gaussian variogram, which
makes the fields nice and smooth. But before that, we will import all
necessary libraries and define a few variables, like the number of grid
cells in each dimension.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

dim = 2
# no. of cells in both dimensions
N = [180, 140]

x = np.arange(N[0])
y = np.arange(N[1])

###############################################################################
# In this first example we will use the same geostatistical parameters for
# both fields for simplicity. Thus, we can use the same SRF instance for the
# two fields.

model = gs.Gaussian(dim=dim, var=1, len_scale=10)
srf = gs.SRF(model)
field1 = srf.structured([x, y], seed=20170519)
field2 = srf.structured([x, y], seed=19970221)

###############################################################################
# Now, we will create the field describing the categorical data. For now, we
# will only have two categories and we will address them by the integers 0 and 1.
# We start off by creating a matrix of 0s from which we will select a rectangle
# and fill that with 1s.
# This field does not have to match the shape of the SRFs.

M = [200, 160]

# size of the rectangle
R = [40, 32]

L = np.zeros(M)
L[
    M[0] // 2 - R[0] // 2 : M[0] // 2 + R[0] // 2,
    M[1] // 2 - R[1] // 2 : M[1] // 2 + R[1] // 2,
] = 1

###############################################################################
# With the two SRFs and `L` ready, we can create our first PGS.

pgs = gs.PGS(dim, [field1, field2], L)

###############################################################################
# Finally, we can plot the PGS, but we will also show the field `L`.

fig, axs = plt.subplots(1, 2)

axs[0].imshow(L, cmap="copper")
axs[1].imshow(pgs.P, cmap="copper")
