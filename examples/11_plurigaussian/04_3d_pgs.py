"""
Creating a Three Dimensional PGS
--------------------------------

Let's create a 3d PGS! This will mostly feel very familiar, but the plotting
will be a bit more involved.
"""

# sphinx_gallery_thumbnail_path = 'pics/3d_pgs.png'
import numpy as np

import gstools as gs

dim = 3
# no. of cells in all dimensions
N = [40] * dim

x = np.arange(N[0])
y = np.arange(N[1])
z = np.arange(N[2])

###############################################################################
# Because we want to create a 3d PGS, we have to generate 3 SRFs. If we are
# interested in even higher dimensions, we could solve this code repetition
# by using a loop...

model1 = gs.Gaussian(dim=dim, var=1, len_scale=[20, 10, 15])
srf1 = gs.SRF(model1)
field1 = srf1.structured([x, y, z], seed=20170519)
model2 = gs.Exponential(dim=dim, var=1, len_scale=[5, 5, 5])
srf2 = gs.SRF(model2)
field2 = srf2.structured([x, y, z], seed=19970221)
model3 = gs.Gaussian(dim=dim, var=1, len_scale=[7, 12, 18])
srf3 = gs.SRF(model3)
field3 = srf3.structured([x, y, z], seed=20011012)

###############################################################################
# The 3d lithotypes field will consist of a cube which contains one category
# and the surrounding is the second category.

# size of cube
cube = [18] * dim

lithotypes = np.zeros(N)
lithotypes[
    N[0] // 2 - cube[0] // 2 : N[0] // 2 + cube[0] // 2,
    N[1] // 2 - cube[1] // 2 : N[1] // 2 + cube[1] // 2,
    N[2] // 2 - cube[2] // 2 : N[2] // 2 + cube[2] // 2,
] = 1

###############################################################################
# With the three SRFs and the lithotypes ready, we can create the 3d PGS.
pgs = gs.PGS(dim, [field1, field2, field3])
P = pgs(lithotypes)

# ###############################################################################
# For ploting the 3d PGS, we will use [PyVista](https://pyvista.org/) which works
# nicely together with GSTools.

import pyvista as pv

grid = pv.ImageData(dimensions=N)

# uncomment, if you want to see lithotypes field, which is just a cube...
# grid.point_data['lithotypes'] = np.meshgrid(lithotypes, indexing="ij")[0]
# grid.plot(show_edges=True)

grid.point_data["PGS"] = P.reshape(-1)

###############################################################################
# .. note::
#    PyVista does not work on readthedocs, but you can try it out yourself by
#    running the example yourself. You will get an interactive version of this
#    screenshot.

# grid.contour(isosurfaces=8).plot()

###############################################################################
#
# .. image:: ../../pics/3d_pgs.png
#    :width: 400px
#    :align: center
