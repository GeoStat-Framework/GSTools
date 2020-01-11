"""
Gaussian
========
"""

from gstools import SRF, Gaussian

# structured field with a size of 100x100 and a grid-size of 1x1
x = y = range(100)
model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model)
srf.structured([x, y])
srf.plot()
