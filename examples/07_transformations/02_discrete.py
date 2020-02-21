"""
discrete fields
-------------

Here we transform a field to a discrete field with values.
If we do not give thresholds, the pairwise means of the given
values are taken as thresholds.
If thresholds are given, arbitrary values can be applied to the field.
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

# use 3 thresholds for separation
thresholds = [-1, 0, 1]
# but apply different values to the separated classes
discrete_values2 = [0, 10, 100, 1000]

srf.structured([x, y])
gs.transform.discrete(srf, discrete_values2, thresholds)
srf.plot()
