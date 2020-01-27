"""
Interface to PyKrige
--------------------

To use fancier methods like
`regression kriging <https://en.wikipedia.org/wiki/Regression-kriging>`__,
we provide an interface to
`PyKrige <https://github.com/bsmurphy/PyKrige>`__.

In the future you can pass a GSTools Covariance Model
to the PyKrige routines as ``variogram_model``.

At the moment we only provide prepared
keyword arguments for the pykrige routines.

To demonstrate the general workflow, we compare the ordinary kriging of PyKrige
with GSTools in 2D:
"""
import numpy as np
import gstools as gs
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
# A GSTools based covariance model.

cov_model = gs.Gaussian(
    dim=2, len_scale=1, anis=0.2, angles=-0.5, var=0.5, nugget=0.1
)

###############################################################################
# Ordinary kriging with pykrige.
# A dictionary containing keyword arguments for the pykrige routines is
# provided by the gstools covariance models.

pk_kwargs = cov_model.pykrige_kwargs
OK1 = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], **pk_kwargs)
z1, ss1 = OK1.execute("grid", gridx, gridy)
plt.imshow(z1, origin="lower")
plt.show()

###############################################################################
# Ordinary kriging with gstools for comparison.

OK2 = gs.krige.Ordinary(cov_model, [data[:, 0], data[:, 1]], data[:, 2])
OK2.structured([gridx, gridy])
ax = OK2.plot()
ax.set_aspect("equal")
