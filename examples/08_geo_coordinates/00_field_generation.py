"""
Working with lat-lon random fields
----------------------------------

In this example, we demonstrate how to generate a random field on
geographical coordinates.

First we setup a model, with ``latlon=True``, to get the associated
Yadrenko model.

In addition we will use the earth radius provided by :any:`EARTH_RADIUS`,
to have a meaningful length scale in km.

To generate the field, we simply pass ``(lat, lon)`` as position tuple
to the :any:`SRF` class.
"""
# sphinx_gallery_thumbnail_number = 3
import gstools as gs

model = gs.Gaussian(latlon=True, var=1, len_scale=777, rescale=gs.EARTH_RADIUS)

lat = lon = range(-80, 81)
srf = gs.SRF(model, seed=12345)
field = srf.structured((lat, lon))
srf.plot()

###############################################################################
# This was easy as always! Now we can use this field to estimate the empirical
# variogram in order to prove, that the generated field has the correct
# geo-statistical properties.
# The :any:`vario_estimate` routine also provides a ``latlon`` switch to
# indicate, that the given field is defined on geographical variables.
#
# As we will see, everthing went well... Phew!

bin_edges = [0.01 * i for i in range(30)]
bin_center, emp_vario = gs.vario_estimate(
    *((lat, lon), field, bin_edges),
    latlon=True,
    mesh_type="structured",
    sampling_size=2000,
    sampling_seed=12345,
)

ax = model.plot("vario_yadrenko", x_max=0.3)
model.fit_variogram(bin_center, emp_vario, init_guess="current", nugget=False)
model.plot("vario_yadrenko", ax=ax, label="fitted", x_max=0.3)
ax.scatter(bin_center, emp_vario, color="k")
print(model)

###############################################################################
# The resulting field can also be easily visualized with the aid of
# `cartopy <https://scitools.org.uk/cartopy/docs/latest/index.html>`_.

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Orthographic(-45, 45)})
cont = ax.contourf(lon, lat, field, transform=ccrs.PlateCarree())
ax.set_title("lat-lon random field generated with GSTools")
ax.coastlines()
ax.set_global()
fig.colorbar(cont)
