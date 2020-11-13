"""
Directional variogram estimation in 2D
--------------------------------------

In this example, we demonstrate how to estimate a directional variogram by
setting the direction angles in 2D.
"""
import numpy as np
import gstools as gs
from matplotlib import pyplot as plt

###############################################################################
# Generating synthetic field with anisotropy and a rotation of 22.5 degree.

angle = np.pi / 8
model = gs.Exponential(dim=2, len_scale=[10, 5], angles=angle)
x = y = range(100)
srf = gs.SRF(model, seed=123456)
field = srf((x, y), mesh_type="structured")

###############################################################################
# Now we are going to estimate a directional variogram with an angular
# tolerance of 11.25 degree and a bandwith of 1.
# We provide the rotation angle of the covariance model and the orthogonal
# direction by adding 90 degree.

bins = range(0, 40, 3)
bin_c, vario, cnt = gs.vario_estimate(
    *((x, y), field, bins),
    direction=gs.rotated_main_axes(dim=2, angles=angle),
    angles_tol=np.pi / 16,
    bandwidth=1.0,
    mesh_type="structured",
    return_counts=True,
)

###############################################################################
# Plotting.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 5])

ax1.plot(bin_c, vario[0], label="emp. vario: pi/8")
ax1.plot(bin_c, vario[1], label="emp. vario: pi*5/8")
ax1.legend(loc="lower right")

srf.plot(ax=ax2)
ax2.set_aspect("equal")

plt.show()

###############################################################################
# Without fitting a model, we see that the correlation length in the main
# direction is greater than the transversal one.
