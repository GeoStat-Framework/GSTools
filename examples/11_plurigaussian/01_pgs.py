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

###############################################################################
# Internally, each field's values are mapped along an axis, which can be nicely
# visualized with a scatter plot. We can easily do that by flattening the 2d
# field values and simply use matplotlib's scatter plotting functionality.
# The x-axis shows field1's values and the y-axis shows field2's values.

plt.scatter(field1.flatten(), field2.flatten(), s=0.1)

###############################################################################
# This mapping always has a multivariate Gaussian distribution and this is also
# the field on which we define our categorical data `L` and their relations to each
# other. Before providing further explanations, we will create `L`, which again
# will have only two categories, but this time we will not prescribe a rectangle,
# but a circle.

# no. of grid cells of field L
M = [51, 41]
# we need the indices of L later
x_L = np.arange(M[0])
y_L = np.arange(M[1])

# radius of circle
R = 7

L = np.zeros(M)
mask = (x_L[:, np.newaxis] - M[0] // 2) ** 2 + (
    y_L[np.newaxis, :] - M[1] // 2
) ** 2 < R**2
L[mask] = 1

###############################################################################
# Now, we look at every point in the scatter plot (which shows the field values)
# shown above and map the categorical values of `L` to the positions (variables
# `x` and `y` defined above) of these field values. This is probably much easier
# to understand with a plot.
# First, we calculate the indices of the L field, which we need for manually
# plotting L together with the scatter plot. Normally they are computed internally.

x_l = np.linspace(
    np.floor(field1[0].min()) - 1,
    np.ceil(field1[0].max()) + 1,
    L.shape[0],
)
y_l = np.linspace(
    np.floor(field1[1].min()) - 1,
    np.ceil(field1[1].max()) + 1,
    L.shape[1],
)

###############################################################################
# We also compute the actual PGS now, to also plot that.

pgs = gs.PGS(dim, [field1, field2], L)

###############################################################################
# And now to some plotting. Unfortunately, matplotlib likes to mess around with
# the aspect ratios of the plots, so the left panel is a bit stretched.

fig, axs = plt.subplots(1, 2)
axs[0].scatter(field1.flatten(), field2.flatten(), s=0.1, color="C0")
axs[0].pcolormesh(x_l, y_l, L.T, alpha=0.3)
axs[1].imshow(pgs.P, cmap="copper")

###############################################################################
# The black areas show the category 0 and the orange areas show category 1. We
# see that the majority of all points in the scatter plot are within the
# yellowish circle, thus the orange areas are larger than the black ones. The
# strong anisotropy and the rotation of the fields create these interesting
# patterns which remind us of fractures in the subsurface.
