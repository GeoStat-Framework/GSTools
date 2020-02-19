"""
discrete fields
-------------

Here we transform a field to a discrete field with five values.
If we do not give thresholds, the pairwise means of the given
values are taken as thresholds.
"""
import numpy as np
import gstools as gs

# structured field with a size of 100x100 and a grid-size of 1x1
x = y = range(100)
model = gs.Gaussian(dim=2, var=1, len_scale=10)
srf = gs.SRF(model, seed=20170519)
srf.structured([x, y])
# create 5 equidistanly spaced values
discrete_values = np.linspace(np.min(srf.field), np.max(srf.field), 5)
gs.transform.discrete(srf, discrete_values)
srf.plot()
