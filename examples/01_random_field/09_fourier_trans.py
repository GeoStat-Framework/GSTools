"""
Generating a Transformed Periodic Random Field
----------------------------------------------

Building on the precious example, we are now going to generate periodic
spatial random fields with a transformation applied, resulting in a level set.
"""

import numpy as np

import gstools as gs

# We start off by defining the spatial grid. As in the previous example, we do
# not want to include the endpoints.
L = np.array((500, 400))
x = np.linspace(0, L[0], 300, endpoint=False)
y = np.linspace(0, L[1], 200, endpoint=False)

# Instead of using a Gaussian covariance model, we will use the much rougher
# exponential model and we will introduce an anisotropy by using two different
# length scales in the x- and y-directions
model = gs.Exponential(dim=2, var=2, len_scale=[80, 20])

# Same as before, we set up the spatial random field. But this time, we will
#  use a periodicity which is equal to the domain size in x-direction, but
# half the domain size in y-direction. And we will use different `mode_no` for
# the different dimensions.
srf = gs.SRF(
    model,
    generator="Fourier",
    period=[L[0], L[1]/2],
    mode_no=[30, 20],
    seed=1681903,
)
# and compute it on our spatial domain
srf((x, y), mesh_type="structured")

# With the field generated, we can now apply transformations starting with a
# discretization of the field into 4 different values
thresholds = np.linspace(np.min(srf.field), np.max(srf.field), 4)
srf.transform("discrete", store="transform_discrete", values=thresholds)
srf.plot("transform_discrete")

# This is already a nice result, but we want to pronounce the peaks of the
# field. We can do this by applying a log-normal transformation on top
srf.transform(
    "lognormal", field="transform_discrete", store="transform_lognormal"
)
srf.plot("transform_lognormal")
