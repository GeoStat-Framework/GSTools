"""
SRF Ensemble
============
"""
import numpy as np
import matplotlib.pyplot as pt
from gstools import SRF, Gaussian
from gstools.random import MasterRNG

x = y = np.arange(100)

model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model)

ens_no = 4
field = []
seed = MasterRNG(20170519)
for i in range(ens_no):
    field.append(srf((x, y), seed=seed(), mesh_type="structured"))

fig, ax = pt.subplots(2, 2, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(ens_no):
    ax[i].imshow(field[i].T, origin="lower")

pt.show()
