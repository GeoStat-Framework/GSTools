"""
Universal Kriging
-----------------

You can give a polynomial order or a list of self defined
functions representing the internal drift of the given values.
This drift will be fitted internally during the kriging interpolation.

In the following we are creating artificial data, where a linear drift
was added. The resulting samples are then used as input for Universal kriging.

The "linear" drift is then estimated during the interpolation.
To access only the estimated mean/drift, we provide a switch `only_mean`
in the call routine.
"""
import numpy as np
from gstools import SRF, Gaussian, krige

# synthetic condtions with a drift
drift_model = Gaussian(dim=1, var=0.1, len_scale=2)
drift = SRF(drift_model, seed=101)
cond_pos = np.linspace(0.1, 8, 10)
cond_val = drift(cond_pos) + cond_pos * 0.1 + 1
# resulting grid
gridx = np.linspace(0.0, 15.0, 151)
drift_field = drift(gridx) + gridx * 0.1 + 1
# kriging
model = Gaussian(dim=1, var=0.1, len_scale=2)
krig = krige.Universal(model, cond_pos, cond_val, "linear")
krig(gridx)
ax = krig.plot()
ax.scatter(cond_pos, cond_val, color="k", zorder=10, label="Conditions")
ax.plot(gridx, gridx * 0.1 + 1, ":", label="linear drift")
ax.plot(gridx, drift_field, "--", label="original field")

mean, mean_err = krig(gridx, only_mean=True)
ax.plot(gridx, mean, label="estimated drift")

ax.legend()
