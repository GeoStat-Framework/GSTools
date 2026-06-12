"""
Controlling Spatial Relations
-----------------------------

In this example we will try to understand how we can influence the spatial
relationships of the different categories with the lithotypes field. For
simplicity, we will start very similarly to the very first example.
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
# Again, we will use the same parameters for both fields.

model = gs.Gaussian(dim=dim, var=1, len_scale=10)
srf = gs.SRF(model)
field1 = srf.structured([x, y], seed=20170519)
field2 = srf.structured([x, y], seed=19970221)

###############################################################################
# Now, we will prepare the lithotypes field, which will be a bit more
# complicated this # time. First, we will create a triangle. Next, we will
# create two rectangles touching each other along one of their edges and both
# being directly above the triangle, but without touching it directly.
# Finally, we will create a few very narrow rectangles, which will not touch
# any other category shapes. The implementation details are not very
# interesting, and can be skipped.

# no. of grid cells of lithotypes field
M = [60, 50]

# size of the rectangles
rect = [10, 8]

# positions of some of the shapes for concise indexing
S1 = [1, -9]
S2 = [-5, 3]
S3 = [-5, -5]

lithotypes = np.zeros(M)
# a small upper triangular helper matrix to create the triangle
triu = np.triu(np.ones((rect[0], rect[0])))
# the triangle
lithotypes[
    M[0] // 2 + S1[0] : M[0] // 2 + S1[0] + rect[0],
    M[1] // 2 + S1[1] : M[1] // 2 + S1[1] + rect[0],
] = triu
# the first rectangle
lithotypes[
    M[0] // 2 + S2[0] - rect[0] // 2 : M[0] // 2 + S2[0] + rect[0] // 2,
    M[1] // 2 + S2[1] - rect[1] // 2 : M[1] // 2 + S2[1] + rect[1] // 2,
] = 2
# the second rectangle
lithotypes[
    M[0] // 2 + S3[0] - rect[0] // 2 : M[0] // 2 + S3[0] + rect[0] // 2,
    M[1] // 2 + S3[1] - rect[1] // 2 : M[1] // 2 + S3[1] + rect[1] // 2,
] = 3
# some very narrow rectangles
for i in range(4):
    lithotypes[
        M[0] // 2 + S1[0] : M[0] // 2 + S1[0] + rect[0],
        M[1] // 2 + S1[1] + rect[1] + 3 + 2 * i : M[1] // 2
        + S1[1]
        + rect[1]
        + 4
        + 2 * i,
    ] = 4 + i

###############################################################################
# With the two SRFs and the L-field ready, we can create the PGS.
pgs = gs.PGS(dim, [field1, field2])
P = pgs(lithotypes)

###############################################################################
# And now the plotting of the two Gaussian fields, the L-field, and the PGS.

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(field1, cmap="copper", origin="lower")
axs[0, 1].imshow(field2, cmap="copper", origin="lower")
axs[1, 0].imshow(lithotypes, cmap="copper", origin="lower")
axs[1, 1].imshow(P, cmap="copper", origin="lower")
plt.show()

###############################################################################
# We can see that the two lower light and medium brown rectangles both fill up
# large and rather smooth areas of the PGS. And they share very long common
# borders due to the fact that these categories touch each other along one of
# their edges. The next large area is the dark brown of the upper triangle.
# This category is always very close to the light brown areas, but only
# sometimes close to the medium brown areas, as they only share small parts in
# close proximity to each other. Finally, we have the four stripes. They create
# distorted stripes in the PGS. The lighter they get, the less area they fill.
# This is due to the fact that their area is not only relatively small, but
# also because they are increasingly further away from the center of the
# lithotypes.
