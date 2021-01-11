"""
Creating Synthetic Precipitation Fields
---------------------------------------

In this example we want to create a time series of a synthetic precipitation
field.

We'll start off by creating a Gaussian random field with an exponential
variogram, which seems to reproduce the spatial correlations of precipitation
fields quite well. We'll create a daily timeseries over a one dimensional cross
section of 50km. This workflow is suited for sub daily precipitation time
series.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import gstools as gs

# fix the seed for reproducibility
seed = 20170521
# half daily timesteps over three months
t = np.arange(0.0, 90.0, 0.5)
# spatial axis of 50km with a resolution of 1km
x = np.arange(0, 50, 1.0)

# an exponential variogram with a corr. lengths of 2d and 5km
model = gs.Exponential(dim=2, var=1.0, len_scale=2.0, anis=2.5)
# create a spatial random field instance
srf = gs.SRF(model)

# a Gaussian random field which is also saved internally for the transformations
srf.structured((t, x), seed=seed)
P_gau = copy.deepcopy(srf.field)

###############################################################################
# Now we should take care of the dry periods. Therefore we simply introduce a
# lower threshold value.

threshold = 0.4
srf.field[srf.field <= threshold] = 0.0
P_cut = srf.field

###############################################################################
# With the above lines of code we have created a cut off Gaussian spatial
# random field with an exponential variogram. But precipitation fields are not
# distributed Gaussian. Thus, we will now transform the field with a box-cox
# transformation, which is often used to account for the skewness of
# precipitation fields. Different values have been suggested for the
# transformation parameter lambda, but we will stick to 1/2. We call the
# resulting field Gaussian anamorphosis.

gs.transform.boxcox(srf, lmbda=0.5, shift=-1.0)

###############################################################################
# As a last step, the amount of precipitation is set. This should of course be
# calibrated towards observations (the same goes for the threshold, the
# variance, correlation length, and so on).

amount = 3.0
srf.field *= amount
P_ana = srf.field

###############################################################################
# Finally we can have a look at the fields resulting from each step. For a
# closer look, we will examine a cross section at an arbitrary location. And
# afterwards we will create a contour plot for visual candy.

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

axs[0, 0].set_title("Gaussian")
axs[0, 0].plot(t, P_gau[:, 20])
axs[0, 0].set_ylabel(r"$P$ / mm")

axs[0, 1].set_title("Cut Gaussian")
axs[0, 1].plot(t, P_cut[:, 20])

axs[1, 0].set_title("Cut Gaussian Anamorphosis")
axs[1, 0].plot(t, P_ana[:, 20])
axs[1, 0].set_xlabel(r"$t$ / d")
axs[1, 0].set_ylabel(r"$P$ / mm")

axs[1, 1].set_title("Different Cross Section")
axs[1, 1].plot(t, P_ana[:, 10])
axs[1, 1].set_xlabel(r"$t$ / d")

plt.tight_layout()

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

axs[0, 0].set_title("Gaussian")
cont = axs[0, 0].contourf(t, x, P_gau.T, cmap="PuBu")
cbar = fig.colorbar(cont, ax=axs[0, 0])
cbar.ax.set_ylabel(r"$P$ / mm")
axs[0, 0].set_ylabel(r"$x$ / km")

axs[0, 1].set_title("Cut Gaussian")
cont = axs[0, 1].contourf(t, x, P_cut.T, cmap="PuBu")
cbar = fig.colorbar(cont, ax=axs[0, 1])
cbar.ax.set_ylabel(r"$P$ / mm")
axs[0, 1].set_xlabel(r"$t$ / d")

axs[1, 0].set_title("Cut Gaussian Anamorphosis")
cont = axs[1, 0].contourf(t, x, P_ana.T, cmap="PuBu")
cbar = fig.colorbar(cont, ax=axs[1, 0])
cbar.ax.set_ylabel(r"$P$ / mm")
axs[1, 0].set_xlabel(r"$t$ / d")
axs[1, 0].set_ylabel(r"$x$ / km")

fig.delaxes(axs[1, 1])
plt.tight_layout()

###############################################################################
# In this example we have created precipitation fields which have one spatial
# dimension, but it is very easy do the same steps with two spatial dimension.
# For the 2d case, we will not save the field after every step, making the
# workflow a little bit easier.

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs

# fix the seed for reproducibility
seed = 20170521
# half daily timesteps over three months
t = np.arange(0.0, 90.0, 0.5)
# 1st spatial axis of 50km with a resolution of 1km
x = np.arange(0, 50, 1.0)
# 2nd spatial axis of 40km with a resolution of 1km
y = np.arange(0, 40, 1.0)

# an exponential variogram with a corr. lengths of 2d, 5km, and 5km
model = gs.Exponential(dim=3, var=1.0, len_scale=2.0, anis=(2.5, 2.5))
# create a spatial random field instance
srf = gs.SRF(model)

# the Gaussian random field
srf.structured((t, x, y), seed=seed)

# the dry periods
threshold = 0.4
srf.field[srf.field <= threshold] = 0.0

# account for the skewness
gs.transform.boxcox(srf, lmbda=0.5, shift=-1.0)

# adjust the amount of precipitation
amount = 3.0
srf.field *= amount

###############################################################################
# plot the 2d precipitation field together with the time axis as a 3d plot.
# We will cut out the volumes with low precipitation in order to have a look
# inside the cuboid.

mesh = srf.to_pyvista()
mesh.threshold_percent(0.25).plot()
