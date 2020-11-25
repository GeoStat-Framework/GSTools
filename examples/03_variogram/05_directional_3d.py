"""
Directional variogram estimation and fitting in 3D
--------------------------------------------------

In this example, we demonstrate how to estimate a directional variogram by
setting the estimation directions in 3D.

Afterwards we will fit a model to this estimated variogram and show the result.
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
bin_center, dir_vario, counts = gs.vario_estimate(
    *([x, y, z], field, bins),
    direction=main_axes,
    bandwidth=10,
    sampling_size=2000,
    sampling_seed=1001,
    mesh_type="structured",
    return_counts=True,
)

###############################################################################
# Afterwards we can use the estimated variogram to fit a model to it.
# Note, that the rotation angles need to be set beforehand.
#
# We can use the `counts` of data pairs per bin as weights for the fitting
# routines to give more attention to areas where more data was available.
# In order to not introduce to much offset at the origin, we disable
# fitting the nugget.

print("Original:")
print(model)
model.fit_variogram(bin_center, dir_vario, weights=counts, nugget=False)
print("Fitted:")
print(model)

###############################################################################
# Plotting.

fig = plt.figure(figsize=[15, 5])
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132, projection=Axes3D.name)
ax3 = fig.add_subplot(133)

srf.plot(ax=ax1)
ax1.set_aspect("equal")

ax2.plot([0, axis1[0]], [0, axis1[1]], [0, axis1[2]], label="0.")
ax2.plot([0, axis2[0]], [0, axis2[1]], [0, axis2[2]], label="1.")
ax2.plot([0, axis3[0]], [0, axis3[1]], [0, axis3[2]], label="2.")
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.set_title("Tait-Bryan main axis")
ax2.legend(loc="lower left")

ax3.scatter(bin_center, dir_vario[0], label="0. axis")
ax3.scatter(bin_center, dir_vario[1], label="1. axis")
ax3.scatter(bin_center, dir_vario[2], label="2. axis")
model.plot("vario_axis", axis=0, ax=ax3, label="fit on axis 0")
model.plot("vario_axis", axis=1, ax=ax3, label="fit on axis 1")
model.plot("vario_axis", axis=2, ax=ax3, label="fit on axis 2")
ax3.set_title("Fitting an anisotropic model")
ax3.legend()

plt.show()
