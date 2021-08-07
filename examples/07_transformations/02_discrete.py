"""
Discrete fields
---------------

Here we transform a field to a discrete field with values.
If we do not give thresholds, the pairwise means of the given
values are taken as thresholds.
If thresholds are given, arbitrary values can be applied to the field.

See :any:`transform.discrete`
"""
import numpy as np
import gstools as gs

# Structured field with a size of 100x100 and a grid-size of 0.5x0.5
x = y = np.arange(200) * 0.5
model = gs.Gaussian(dim=2, var=1, len_scale=5)
srf = gs.SRF(model, seed=20170519)
srf.structured([x, y])

###############################################################################
# Create 5 equidistanly spaced values, thresholds are the arithmetic means

values1 = np.linspace(np.min(srf.field), np.max(srf.field), 5)
srf.transform("discrete", store="f1", values=values1)
srf.plot("f1")

###############################################################################
# Calculate thresholds for equal shares
# but apply different values to the separated classes

values2 = [0, -1, 2, -3, 4]
srf.transform("discrete", store="f2", values=values2, thresholds="equal")
srf.plot("f2")

###############################################################################
# Create user defined thresholds
# and apply different values to the separated classes

values3 = [0, 1, 10]
thresholds = [-1, 1]
srf.transform("discrete", store="f3", values=values3, thresholds=thresholds)
srf.plot("f3")
