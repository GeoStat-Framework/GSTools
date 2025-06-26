"""
From bigaussian to plurigaussian simulation
--------------------------------------------------

In 
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

dim = 2

N = [200,200]

x = np.arange(N[0])
y = np.arange(N[1])

model = gs.Gaussian(dim=dim, var=1, len_scale=15)
srf = gs.SRF(model)
field1 = srf.structured([x, y], seed=201519)
field2 = srf.structured([x, y], seed=199221)
model = gs.Gaussian(dim=dim, var=1, len_scale=3)
srf = gs.SRF(model)
field3 = srf.structured([x, y], seed=1345134)
field4 = srf.structured([x, y], seed=1351455)

from gstools.field.pgs import ellipse

config = {
    'root': {
        'type': 'decision',
        'func': ellipse,
        'args': {
            'key1': 'Z1',
            'key2': 'Z2',
            'c1': 0.7,
            'c2': 0.7,
            's1': 1.3,
            's2': 1.3,
        },
        'yes_branch': 'phase1',
        'no_branch': 'node1'
    },
    'node1': {
        'type': 'decision',
        'func': ellipse,
        'args': {
            'key1': 'Z3',
            'key2': 'Z4',
            'c1': -0.7,
            'c2': -0.7,
            's1': 1.3,
            's2': 1.3,
            'angle': 30
        },
        'yes_branch': 'phase2',
        'no_branch': 'phase0'
    },
    'phase2': {
        'type': 'leaf',
        'action': 2
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

pgs = gs.PGS(dim, [field1, field2, field3, field4])

P = pgs(tree=config)

plt.imshow(P, cmap="copper", origin="lower")
plt.tight_layout()
plt.show()