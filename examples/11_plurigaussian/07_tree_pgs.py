"""
PGS through decision trees
--------------------------

In 
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

dim = 2

N = [150,150]

x = np.arange(N[0])
y = np.arange(N[1])

model = gs.Gaussian(dim=dim, var=1, len_scale=10)
srf = gs.SRF(model)
field1 = srf.structured([x, y], seed=201519)
field2 = srf.structured([x, y], seed=199221)

from gstools.field.pgs import ellipse

config = {
    'root': {
        'type': 'decision',
        'func': ellipse,
        'args': {
            'key1': 'Z1',
            'key2': 'Z2',
            'c1': 0,
            'c2': 0,
            's1': 2.5,
            's2': 0.8,
            'angle': -45
        },
        'yes_branch': 'phase1',
        'no_branch': 'phase0'
    },
    'phase1': {
        'type': 'leaf',
        'action': 1
    },
    'phase0': {
        'type': 'leaf',
        'action': 0
    },
}


pgs = gs.PGS(dim, [field1, field2])

L = pgs.compute_lithotype(tree=config)
P = pgs(tree=config)

# can compute L after also with L = pgs.compute_lithotype()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

im0 = axs[0].imshow(L, cmap="copper", origin="lower")
axs[0].set_title("L")
im1 = axs[1].imshow(P, cmap="copper", origin="lower")
axs[1].set_title("P")

plt.tight_layout()

plt.show()