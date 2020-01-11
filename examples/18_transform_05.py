"""
Transform 5
===========
"""
from gstools import SRF, Gaussian
from gstools import transform as tf

# structured field with a size of 100x100 and a grid-size of 1x1
x = y = range(100)
model = Gaussian(dim=2, var=1, len_scale=10)
srf = SRF(model, mean=-9, seed=20170519)
srf.structured([x, y])
tf.normal_force_moments(srf)
tf.zinnharvey(srf, conn="low")
tf.binary(srf)
tf.normal_to_lognormal(srf)
srf.plot()
