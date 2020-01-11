"""
SRF Merge
=========
"""
import numpy as np
import matplotlib.pyplot as pt
from gstools import SRF, Exponential
from gstools.random import MasterRNG

# creating our own unstructured grid
seed = MasterRNG(19970221)
rng = np.random.RandomState(seed())
x = rng.randint(0, 100, size=10000)
y = rng.randint(0, 100, size=10000)

model = Exponential(dim=2, var=1, len_scale=[12.0, 3.0], angles=np.pi / 8.0)

srf = SRF(model, seed=20170519)

field = srf((x, y))

# new grid
seed = MasterRNG(20011012)
rng = np.random.RandomState(seed())
x2 = rng.randint(99, 150, size=10000)
y2 = rng.randint(20, 80, size=10000)

field2 = srf((x2, y2))

pt.tricontourf(x, y, field.T)
pt.tricontourf(x2, y2, field2.T)
pt.axes().set_aspect("equal")
pt.show()
