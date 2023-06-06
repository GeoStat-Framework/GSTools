"""
Working with spatio-temporal lat-lon fields
-------------------------------------------

In this example, we demonstrate how to generate a spatio-temporal
random field on geographical coordinates.

First we setup a model, with ``latlon=True`` and ``time=True``,
to get the associated spatio-temporal Yadrenko model.

In addition, we will use the earth radius provided by :any:`EARTH_RADIUS`,
to have a meaningful length scale in km.

To generate the field, we simply pass ``(lat, lon, time)`` as the position tuple
to the :any:`SRF` class.

The anisotropy factor of `0.1` will result in a time length-scale of `77.7` days.
"""
import gstools as gs

model = gs.Gaussian(
    latlon=True,
    time=True,
    var=1,
    len_scale=777,
    anis=0.1,
    rescale=gs.EARTH_RADIUS,
)

lat = lon = range(-80, 81)
time = range(0, 110, 10)
srf = gs.SRF(model, seed=1234)
field = srf.structured((lat, lon, time))
srf.plot()
