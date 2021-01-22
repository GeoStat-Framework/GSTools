"""
Creating a 2D Synthetic Precipitation Field
-------------------------------------------

In this example we'll create a time series of a 2D synthetic precipitation
field.

Very similar to the previous tutorial, we'll start off by creating a Gaussian
random field with an exponential variogram, which seems to reproduce the
spatial correlations of precipitation fields quite well. We'll create a daily
timeseries over a two dimensional domain of 50km x 40km. This workflow is
suited for sub daily precipitation time series.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gstools as gs

# fix the seed for reproducibility
seed = 20170521
# 1st spatial axis of 50km with a resolution of 1km
x = np.arange(0, 50, 1.0)
# 2nd spatial axis of 40km with a resolution of 1km
y = np.arange(0, 40, 1.0)
# half daily timesteps over three months
t = np.arange(0.0, 90.0, 0.5)

# total spatio-temporal dimension
st_dim = 2 + 1
# space-time anisotropy ratio given in units d / km
st_anis = 0.4

# an exponential variogram with a corr. lengths of 5km, 5km, and 2d
model = gs.Exponential(dim=st_dim, var=1.0, len_scale=5.0, anis=st_anis)
# create a spatial random field instance
srf = gs.SRF(model, seed=seed)

pos, time = [x, y], [t]

# the Gaussian random field
srf.structured(pos + time)

# the dry periods
threshold = 0.4
srf.field[srf.field <= threshold] = 0.0

# account for the skewness
gs.transform.boxcox(srf, lmbda=0.5, shift=-1.0)

# adjust the amount of precipitation
amount = 4.0
srf.field *= amount

###############################################################################
# plot the 2d precipitation field over time as an animation.

def _update_ani(time_step):
    im.set_array(srf.field[:, :, time_step].T)
    return im,

fig, ax = plt.subplots()
im = ax.imshow(
    srf.field[:,:,0].T,
    cmap="Blues",
    interpolation="bicubic",
    origin="lower",
)
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(r"Precipitation $P$ / mm")
ax.set_xlabel(r"$x$ / km")
ax.set_ylabel(r"$y$ / km")

ani = animation.FuncAnimation(fig, _update_ani, len(t), interval=100, blit=True)
