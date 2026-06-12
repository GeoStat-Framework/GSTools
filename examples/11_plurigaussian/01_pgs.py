"""
Understanding PGS
-----------------

In this example we want to try to understand how exactly PGS are generated
and how to influence them with the categorical field.
First of all, we will set everything up very similar to the first example.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

dim = 2
# no. of cells in both dimensions
N = [100, 80]

x = np.arange(N[0])
y = np.arange(N[1])

###############################################################################
# In this example we will use different geostatistical parameters for the
# SRFs. We will create fields with a strong anisotropy, and on top of that they
# will both be rotated.

model1 = gs.Gaussian(dim=dim, var=1, len_scale=[20, 1], angles=np.pi / 8)
srf1 = gs.SRF(model1, seed=20170519)
field1 = srf1.structured([x, y])
model2 = gs.Gaussian(dim=dim, var=1, len_scale=[1, 20], angles=np.pi / 4)
srf2 = gs.SRF(model2, seed=19970221)
field2 = srf2.structured([x, y])
field1 += 5.0

###############################################################################
# Internally, each field's values are mapped along an axis, which can be nicely
# visualized with a scatter plot. We can easily do that by flattening the 2d
# field values and simply use matplotlib's scatter plotting functionality.
# The x-axis shows field1's values and the y-axis shows field2's values.

plt.scatter(field1.flatten(), field2.flatten(), s=0.1)

###############################################################################
# This mapping always has a multivariate Gaussian distribution and this is also
# the field on which we define our categorical data `lithotypes` and their
# relations to each other. Before providing further explanations, we will
# create the lithotypes field, which again will have only two categories, but
# this time we will not prescribe a rectangle, but a circle.

# no. of grid cells of L-field
M = [51, 41]
# we need the indices of `lithotypes` later
x_lith = np.arange(M[0])
y_lith = np.arange(M[1])

# radius of circle
radius = 7

lithotypes = np.zeros(M)
mask = (x_lith[:, np.newaxis] - M[0] // 2) ** 2 + (
    y_lith[np.newaxis, :] - M[1] // 2
) ** 2 < radius**2
lithotypes[mask] = 1

###############################################################################
# We can compute the actual PGS now. As a second step, we use a helper function
# to recalculate the axes on which the lithotypes are defined. Normally, this
# is handled internally. But in order to show the scatter plot together with
# the lithotypes, we need the axes here.

pgs = gs.PGS(dim, [field1, field2])
P = pgs(lithotypes)

x_lith, y_lith = pgs.calc_lithotype_axes(lithotypes.shape)

###############################################################################
# And now to some plotting. Unfortunately, matplotlib likes to mess around with
# the aspect ratios of the plots, so the left panel is a bit stretched.

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(field1, cmap="copper", origin="lower")
axs[0, 1].imshow(field2, cmap="copper", origin="lower")
axs[1, 0].scatter(field1.flatten(), field2.flatten(), s=0.1, color="C0")
axs[1, 0].pcolormesh(x_lith, y_lith, lithotypes.T, alpha=0.3, cmap="copper")

axs[1, 1].imshow(P, cmap="copper", origin="lower")

###############################################################################
# The black areas show the category 0 and the orange areas show category 1. We
# see that the majority of all points in the scatter plot are within the
# yellowish circle, thus the orange areas are larger than the black ones. The
# strong anisotropy and the rotation of the fields create these interesting
# patterns which remind us of fractures in the subsurface.
