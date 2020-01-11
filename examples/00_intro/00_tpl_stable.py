"""
TPL Stable
----------
"""

import numpy as np
from gstools import SRF, TPLStable

x = y = np.linspace(0, 100, 100)
model = TPLStable(
    dim=2,  # spatial dimension
    var=1,  # variance (C is calculated internally, so that the variance is actually 1)
    len_low=0,  # lower truncation of the power law
    len_scale=10,  # length scale (a.k.a. range), len_up = len_low + len_scale
    nugget=0.1,  # nugget
    anis=0.5,  # anisotropy between main direction and transversal ones
    angles=np.pi / 4,  # rotation angles
    alpha=1.5,  # shape parameter from the stable model
    hurst=0.7,  # hurst coefficient from the power law
)
srf = SRF(model, mean=1, mode_no=1000, seed=19970221, verbose=True)
srf.structured([x, y])
srf.plot()
