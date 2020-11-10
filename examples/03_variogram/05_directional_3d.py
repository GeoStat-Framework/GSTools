"""
Directional variogram estimation in 2D
--------------------------------------

In this example, we demonstrate how to estimate a directional variogram by
setting the estimation directions in 3D.
"""
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=[15, 5])
ax1 = fig.add_subplot(131, projection=Axes3D.name)
ax2 = fig.add_subplot(132, projection=Axes3D.name)
ax3 = fig.add_subplot(133)

###############################################################################
# Generating synthetic field with anisotropy and rotation by Tait-Bryan angles.

dim = 3
# rotation around z, y, x
angles = [np.pi / 2, np.pi / 4, np.pi / 8]
model = gs.Gaussian(dim=3, len_scale=[16, 8, 4], angles=angles)
x = y = z = range(50)
srf = gs.SRF(model, seed=1001)
field = srf.structured((x, y, z))
srf.plot(ax=ax1)
ax1.set_aspect("equal")

###############################################################################
# Here we plot the rotated coordinate system to get an impression, what
# the rotation angles do.

x1, x2, x3 = (1, 0, 0), (0, 1, 0), (0, 0, 1)
ret = np.array(gs.field.tools.rotate_mesh(dim, angles, x1, x2, x3))
dir0 = ret[:, 0]  # main direction
dir1 = ret[:, 1]  # first lateral direction
dir2 = ret[:, 2]  # second lateral direction
ax2.plot([0, dir0[0]], [0, dir0[1]], [0, dir0[2]], label="1.")
ax2.plot([0, dir1[0]], [0, dir1[1]], [0, dir1[2]], label="2.")
ax2.plot([0, dir2[0]], [0, dir2[1]], [0, dir2[2]], label="3.")
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.set_aspect("equal")
ax2.set_title("Tait-Bryan main axis")
ax2.legend(loc="lower left")

###############################################################################
# Now we estimate the variogram along the main axis. When the main axis are
# unknown, one would need to sample multiple directions and look out for
# the one with the longest correlation length (flattest gradient).
# Then check the transversal directions and so on.

bins = range(0, 40, 3)
bin_c, vario = gs.vario_estimate(
    *([x, y, z], field, bins),
    direction=(dir0, dir1, dir2),
    bandwidth=10,
    sampling_size=2000,
    sampling_seed=1001,
    mesh_type="structured"
)
ax3.plot(bin_c, vario[0], label="1. axis")
ax3.plot(bin_c, vario[1], label="2. axis")
ax3.plot(bin_c, vario[2], label="3. axis")
ax3.legend()
fig.show()
