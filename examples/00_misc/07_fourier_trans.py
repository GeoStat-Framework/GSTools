"""
Generating a Transformed Periodic Random Field
----------------------------------------------

Building on the precious example, we are now going to generate periodic
spatial random fields with a transformation applied, resulting in a level set.
"""

import numpy as np
import gstools as gs

# We start off by defining the spatial grid.
L = np.array((500, 500))
x = np.linspace(0, L[0], 300)
y = np.linspace(0, L[1], 200)

# Instead of using a Gaussian covariance model, we will use the much rougher
# exponential model and we will introduce an anisotropy by using two different
# length scales in the x- and y-axes
model = gs.Exponential(dim=2, var=2, len_scale=[30, 20])

# Same as before, we set up the spatial random field
srf = gs.SRF(
    model,
    generator="Fourier",
    mode_rel_cutoff=0.999,
    period=L,
    seed=1681903,
)
# and compute it on our spatial domain
srf((x, y), mesh_type='structured')

# With the field generated, we can now apply transformations
# starting with a discretization of the field into 4 different values
thresholds = np.linspace(np.min(srf.field), np.max(srf.field), 4)
srf.transform("discrete", store="transform_discrete", values=thresholds)
srf.plot("transform_discrete")

# This is already a nice result, but we want to pronounce the peaks of the
# field. We can do this by applying a log-normal transformation on top
srf.transform("lognormal", field="transform_discrete", store="transform_lognormal")
srf.plot("transform_lognormal")
