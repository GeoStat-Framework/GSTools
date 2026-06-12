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
# Now, we will create the lithotypes field describing the categorical data. For
# now, we will only have two categories and we will address them by the
# integers 0 and 1. We start off by creating a matrix of 0s from which we will
# select a rectangle and fill that with 1s. This field does not have to match
# the shape of the SRFs.

centroid = [200, 160]

# size of the rectangle
rect = [40, 32]

lithotypes = np.zeros(centroid)
lithotypes[
    centroid[0] // 2 - rect[0] // 2 : centroid[0] // 2 + rect[0] // 2,
    centroid[1] // 2 - rect[1] // 2 : centroid[1] // 2 + rect[1] // 2,
] = 1

###############################################################################
# With the two SRFs and the L-field ready, we can create our first PGS. First, we
# set up an instance of the PGS class and then we are ready to calculate the
# field by calling the instance and handing over the L-field.

pgs = gs.PGS(dim, [field1, field2])
P = pgs(lithotypes)

###############################################################################
# Finally, we can plot the PGS, but we will also show the L-field and the two
# original Gaussian fields.

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(field1, cmap="copper", origin="lower")
axs[0, 1].imshow(field2, cmap="copper", origin="lower")
axs[1, 0].imshow(lithotypes, cmap="copper", origin="lower")
axs[1, 1].imshow(P, cmap="copper", origin="lower")

# For more information on Plurigaussian fields and how they naturally extend
# truncated Gaussian fields, we can recommend the book
# [Plurigaussian Simulations in Geosciences](https://doi.org/10.1007/978-3-642-19607-2)
