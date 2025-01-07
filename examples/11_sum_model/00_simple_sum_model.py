r"""
Creating a Sum Model
--------------------

This tutorial demonstrates how to create and use sum models in GSTools.
We'll combine a Spherical and a Gaussian covariance model to construct
a sum model, visualize its variogram, and generate spatial random fields.

Let's start with importing GSTools setting up the domain size.
"""

import gstools as gs

x = y = range(100)

###############################################################################
# First, we create two individual covariance models: a :any:`Spherical` model and a
# :any:`Gaussian` model. The Spherical model will emphasize small-scale variability,
# while the Gaussian model captures larger-scale patterns.

m0 = gs.Spherical(dim=2, var=2.0, len_scale=5.0)
m1 = gs.Gaussian(dim=2, var=1.0, len_scale=10.0)

###############################################################################
# Next, we create a sum model by adding these two models together.
# Let's visualize the resulting variogram alongside the individual models.

model = m0 + m1
ax = model.plot(x_max=20)
m0.plot(x_max=20, ax=ax)
m1.plot(x_max=20, ax=ax)

###############################################################################
# As shown, the Spherical model controls the behavior at shorter distances,
# while the Gaussian model dominates at longer distances.
#
# Using the sum model, we can generate a spatial random field. Let's visualize
# the field created by the sum model.

srf = gs.SRF(model, seed=20250107)
srf.structured((x, y))
srf.plot()

###############################################################################
# For comparison, we generate random fields using the individual models
# to observe their contributions more clearly.

srf0 = gs.SRF(m0, seed=20250107)
srf0.structured((x, y))
srf0.plot()

srf1 = gs.SRF(m1, seed=20250107)
srf1.structured((x, y))
srf1.plot()

###############################################################################
# As seen, the Gaussian model introduces large-scale structures, while the
# Spherical model influences the field's roughness. The sum model combines
# these effects, resulting in a field that reflects multi-scale variability.
