"""
Directional variogram estimation in 3D
--------------------------------------

In this example, we demonstrate how to estimate a directional variogram by
setting the estimation directions in 3D.
"""
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# Generating synthetic field with anisotropy and rotation by Tait-Bryan angles.

dim = 3
# rotation around z, y, x
angles = [np.pi / 2, np.pi / 4, np.pi / 8]
model = gs.Gaussian(dim=3, len_scale=[16, 8, 4], angles=angles)
x = y = z = range(50)
srf = gs.SRF(model, seed=1001)
field = srf.structured((x, y, z))

###############################################################################
# Here we generate the axes of the rotated coordinate system
# to get an impression what the rotation angles do.

# All 3 axes of the rotated coordinate-system
main_axes = gs.rotated_main_axes(dim, angles)
axis1, axis2, axis3 = main_axes

###############################################################################
# Now we estimate the variogram along the main axes. When the main axes are
# unknown, one would need to sample multiple directions and look for the one
# with the longest correlation length (flattest gradient).
# Then check the transversal directions and so on.

bins = range(0, 40, 3)
bin_c, vario = gs.vario_estimate(
    *([x, y, z], field, bins),
    direction=main_axes,
    bandwidth=10,
    sampling_size=2000,
    sampling_seed=1001,
    mesh_type="structured"
)

###############################################################################
# Plotting.

fig = plt.figure(figsize=[15, 5])
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132, projection=Axes3D.name)
ax3 = fig.add_subplot(133)

srf.plot(ax=ax1)
ax1.set_aspect("equal")

ax2.plot([0, axis1[0]], [0, axis1[1]], [0, axis1[2]], label="1.")
ax2.plot([0, axis2[0]], [0, axis2[1]], [0, axis2[2]], label="2.")
ax2.plot([0, axis3[0]], [0, axis3[1]], [0, axis3[2]], label="3.")
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.set_title("Tait-Bryan main axis")
ax2.legend(loc="lower left")

ax3.plot(bin_c, vario[0], label="1. axis")
ax3.plot(bin_c, vario[1], label="2. axis")
ax3.plot(bin_c, vario[2], label="3. axis")
ax3.legend()
plt.show()
