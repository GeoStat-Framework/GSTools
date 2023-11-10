"""
Generating a Transformed Periodic Random Field
----------------------------------------------

Building on the precious example, we are now going to generate periodic
spatial random fields with a transformation applied, resulting in a level set.
"""

import numpy as np
import gstools as gs

# We start off by defining the spatial grid.
x = np.linspace(0, 500, 300)
y = np.linspace(0, 500, 200)

# Instead of using a Gaussian covariance model, we will use the much rougher
# exponential model and we will introduce an anisotropy by using two different
# length scales in the x- and y-axes
model = gs.Exponential(dim=2, var=2, len_scale=[30, 20])

# Very similar as before, setting up the spatial random field
srf = gs.SRF(
    model,
    generator="Fourier",
    modes_no=[30, 20],
    modes_truncation=[30, 20],
    seed=1681903,
)
# and computing it
srf((x, y), mesh_type="structured")

# With the field generated, we can now apply transformations
# starting with a discretization of the field into 4 different values
thresholds = np.linspace(np.min(srf.field), np.max(srf.field), 4)
srf.transform("discrete", store="transform_discrete", values=thresholds)
srf.plot("transform_discrete")

# This is already a nice result, but we want to pronounce the peaks of the
# field. We can do this by applying a log-normal transformation on top
srf.transform(
    "lognormal", field="transform_discrete", store="transform_lognormal"
)
srf.plot("transform_lognormal")
