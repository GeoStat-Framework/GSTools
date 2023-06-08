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

The anisotropy factor of `0.1` (days/km) will result in a time length-scale of `77.7` days.
"""
import numpy as np

import gstools as gs

model = gs.Gaussian(
    latlon=True,
    time=True,
    var=1,
    len_scale=777,
    anis=0.1,
    radius=gs.EARTH_RADIUS,
)

lat = lon = np.linspace(-80, 81, 50)
time = np.linspace(0, 777, 50)
srf = gs.SRF(model, seed=1234)
field = srf.structured((lat, lon, time))
srf.plot()
