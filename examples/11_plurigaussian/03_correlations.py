"""
Understanding the Influence of Variograms
-----------------------------------------

Up until now, we have only used very smooth Gaussian variograms for the
underlying spatial random fields. Now, we will combine a smooth Gaussian
field with a much rougher exponential field. This example should feel
familiar, if you had a look at the previous examples.
"""

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs

dim = 2
# no. of cells in both dimensions
N = [200, 200]

x = np.arange(N[0])
y = np.arange(N[1])

###############################################################################
# Now, we generate fields with a Gaussian and an Exponential variogram.

model1 = gs.Gaussian(dim=dim, var=1, len_scale=[50, 25])
srf1 = gs.SRF(model1)
field1 = srf1.structured([x, y], seed=20170519)
model2 = gs.Exponential(dim=dim, var=1, len_scale=[40, 40])
srf2 = gs.SRF(model2)
field2 = srf2.structured([x, y], seed=19970221)

###############################################################################
# The `L` field will consist of a circle which contains one category and the
# surrounding is the second category.

# no. of grid cells of field L
M = [200, 200]

# radius of circle
R = 25

x_L = np.arange(M[0])
y_L = np.arange(M[1])
L = np.zeros(M)
mask = (x_L[:, np.newaxis] - M[0] // 2) ** 2 + (
    y_L[np.newaxis, :] - M[1] // 2
) ** 2 < R**2
L[mask] = 1

###############################################################################
# With the two SRFs and `L` ready, we can create the PGS.
pgs = gs.PGS(dim, [field1, field2], L)

###############################################################################
# And now the plotting of `L` and the PGS.

fig, axs = plt.subplots(1, 2)

axs[0].imshow(L, cmap="copper")
axs[1].imshow(pgs.P, cmap="copper")
plt.show()

###############################################################################
# In this PGS, we can see two different spatial structures combined. We see large
# and rather smooth structures and shapes, which are surrounded by very rough and
# unconnected patches.
