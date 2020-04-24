"""
Example: Using a mesh for GSTools
---------------------------------

This example shows how external data can be analysed with GSTools.


Pretending we have some Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will pretend that we have some external data, by creating some random data
with GSTools, but we will delete the objects and only use the data, without the
backend GSTools provides.
"""
from datetime import date
import numpy as np
import gstools as gs

# create a circular mesh
point_no = 10000
rng = np.random.RandomState(20170521)
r = 50.0 * np.sqrt(rng.uniform(size=point_no))
theta = 2.0 * np.pi * rng.uniform(size=point_no)
x = r * np.cos(theta)
y = r * np.sin(theta)

tmp_model = gs.Exponential(dim=2, var=1.5, len_scale=10.0)
tmp_srf = gs.SRF(tmp_model)
field = tmp_srf((x, y))
tmp_srf.plot()

# Now that we have our data, let's delete everything GSTools related and pretend
# that this has never happend
del(tmp_model)
del(tmp_srf)


# Creating the Mesh
# ^^^^^^^^^^^^^^^^^
#
# Starting out fresh, we want to feed the mesh with our data
mesh = gs.Mesh(pos=(x, y), values=field)

# We can add meta data too
mesh.set_field_data("location", "Süderbrarup")
mesh.set_field_data("date", date(year=2020, month=2, day=28))

# This can be conviniently accessed
print(mesh.location)
print(mesh.date)

# But the meta data is also collected as a dictionary in case you want to export
# it
print(mesh.field_data)


# Estimating the Variogram
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Now, with our mesh, which was loaded from completely external sources, we can
# estimate the variogram of the data.
# To speed things up, we will only use a fraction of the available data

bins = np.linspace(0, 50, 50)
bin_centre, gamma = gs.vario_estimate_unstructured(
    (x, y), field, bins, sampling_size=2000, sampling_seed=19900408)

# As we are experts, we'll do an expert guess and say, that we will most likely
# have data that has an exponential variogram. Non-experts can have a look at
# the "Finding the best fitting variogram model" tutorial in
# :ref:`tutorial_03_variogram`.
fit_model = gs.Exponential(dim=2)
fit_model.fit_variogram(bin_centre, gamma, nugget=False)

ax = fit_model.plot(x_max=max(bin_centre))
ax.plot(bin_centre, gamma)
