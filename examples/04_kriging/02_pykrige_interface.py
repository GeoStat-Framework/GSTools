"""
Interface to PyKrige
--------------------

To use fancier methods like `universal kriging <https://en.wikipedia.org/wiki/Kriging#Universal>`__,
we provide an interface to `PyKrige <https://github.com/bsmurphy/PyKrige>`__.

You can pass a GSTools Covariance Model to the PyKrige routines as ``variogram_model``.

To demonstrate the general workflow, we compare the ordinary kriging of PyKrige
with GSTools in 2D:
"""
import numpy as np
from gstools import Gaussian, krige
from pykrige.ok import OrdinaryKriging
from matplotlib import pyplot as plt

# conditioning data
data = np.array(
    [
        [0.3, 1.2, 0.47],
        [1.9, 0.6, 0.56],
        [1.1, 3.2, 0.74],
        [3.3, 4.4, 1.47],
        [4.7, 3.8, 1.74],
    ]
)

# grid definition for output field
gridx = np.arange(0.0, 5.5, 0.1)
gridy = np.arange(0.0, 6.5, 0.1)

###############################################################################

# a GSTools based covariance model
cov_model = Gaussian(
    dim=2, len_scale=1, anis=0.2, angles=-0.5, var=0.5, nugget=0.1
)

###############################################################################
# ordinary kriging with pykrige
OK1 = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], cov_model)
z1, ss1 = OK1.execute("grid", gridx, gridy)
plt.imshow(z1, origin="lower")
plt.show()

###############################################################################

# ordinary kriging with gstools for comparison
OK2 = krige.Ordinary(cov_model, [data[:, 0], data[:, 1]], data[:, 2])
OK2.structured([gridx, gridy])
OK2.plot()
