import numpy as np
import matplotlib.pyplot as plt
from gstools import SRF, Gaussian, Exponential


# the grid
x = np.arange(100)
y = np.arange(100)

# a smooth Gaussian covariance model
model = Gaussian(dim=2, var=1, len_scale=10)

srf = SRF(model, generator='VectorField')
srf((x, y), mesh_type='structured', seed=19841203)
srf.plot()

# a rougher exponential covariance model
model2 = Exponential(dim=2, var=1, len_scale=10)

srf.model = model2
srf((x, y), mesh_type='structured', seed=19841203)
srf.plot()
