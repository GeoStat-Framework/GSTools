"""
Compare Kriging
---------------
"""
import numpy as np
from gstools import Gaussian, krige
import matplotlib.pyplot as plt

# condtions
cond_pos = [0.3, 1.9, 1.1, 3.3, 4.7]
cond_val = [0.47, 0.56, 0.74, 1.47, 1.74]
# resulting grid
gridx = np.linspace(0.0, 15.0, 151)

###############################################################################
# A gaussian variogram model.

model = Gaussian(dim=1, var=0.5, len_scale=2)

###############################################################################
# Two kriged fields. One with simple and one with ordinary kriging.

kr1 = krige.Simple(model=model, mean=1, cond_pos=cond_pos, cond_val=cond_val)
kr2 = krige.Ordinary(model=model, cond_pos=cond_pos, cond_val=cond_val)
kr1(gridx)
kr2(gridx)

###############################################################################

plt.plot(gridx, kr1.field, label="simple kriged field")
plt.plot(gridx, kr2.field, label="ordinary kriged field")
plt.scatter(cond_pos, cond_val, color="k", zorder=10, label="Conditions")
plt.legend()
plt.show()
