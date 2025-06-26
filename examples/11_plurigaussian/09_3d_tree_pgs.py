"""
Three dimensional PGS through decision trees
--------------------------------------------

In 
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

dim = 3
# no. of cells in all dimensions
N = [60] * dim

x = np.arange(N[0])
y = np.arange(N[1])
z = np.arange(N[2])

model1 = gs.Gaussian(dim=dim, var=1, len_scale=[20, 10, 15])
srf1 = gs.SRF(model1)
field1 = srf1.structured([x, y, z], seed=20170519)
model2 = gs.Exponential(dim=dim, var=1, len_scale=[5, 5, 5])
srf2 = gs.SRF(model2)
field2 = srf2.structured([x, y, z], seed=19970221)
model3 = gs.Gaussian(dim=dim, var=1, len_scale=[7, 12, 18])
srf3 = gs.SRF(model3)
field3 = srf3.structured([x, y, z], seed=20011012)

pgs = gs.PGS(dim, [field1, field2, field3])

from gstools.field.pgs import ellipsoid

config = {
    'root': {
        'type': 'decision',
        'func': ellipsoid,
        'args': {
            'key1': 'Z1',
            'key2': 'Z2',
            'key3': 'Z4',
            'c1': 0,
            'c2': 0,
            'c3': 0,
            's1': 3,
            's2': 1,
            's3': 0.4
        },
        'yes_branch': 'phase1',
        'no_branch': 'phase0'
    },
    'phase0': {
        'type': 'leaf',
        'action': 0
    },
    'phase1': {
        'type': 'leaf',
        'action': 1
    },
}

L, P = pgs(tree=config)

import pyvista as pv

grid = pv.ImageData(dimensions=N)
grid.point_data["PGS"] = P.reshape(-1)
grid.contour(isosurfaces=8).plot()