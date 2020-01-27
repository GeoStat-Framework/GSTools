"""
log-normal fields
-----------------

Here we transform a field to a log-normal distribution:
"""
import gstools as gs

# structured field with a size of 100x100 and a grid-size of 1x1
x = y = range(100)
model = gs.Gaussian(dim=2, var=1, len_scale=10)
srf = gs.SRF(model, seed=20170519)
srf.structured([x, y])
gs.transform.normal_to_lognormal(srf)
srf.plot()
