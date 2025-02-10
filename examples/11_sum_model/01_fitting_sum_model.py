r"""
Fitting a Sum Model
--------------------

In this tutorial, we demonstrate how to fit a sum model consisting of two
covariance models to an empirical variogram.

We will generate synthetic data, compute an empirical variogram, and fit a
sum model combining a Spherical and Gaussian model to it.
"""

import gstools as gs

x = y = range(100)

###############################################################################
# First, we create a synthetic random field based on a known sum model.
# This will serve as the ground truth for fitting.

# Define the true sum model
m0 = gs.Spherical(dim=2, var=2.0, len_scale=5.0)
m1 = gs.Gaussian(dim=2, var=1.0, len_scale=10.0)
true_model = m0 + m1

# Generate synthetic field
srf = gs.SRF(true_model, seed=20250107)
field = srf.structured((x, y))

###############################################################################
# Next, we calculate the empirical variogram from the synthetic data.

# Compute empirical variogram
bin_center, gamma = gs.vario_estimate((x, y), field)

###############################################################################
# Now we define a sum model to fit to the empirical variogram.
# Initially, the parameters of the models are arbitrary.
#
# A sum model can also be created by a list of model classes together with
# the common arguments (like dim in this case).

fit_model = gs.SumModel(gs.Spherical, gs.Gaussian, dim=2)

###############################################################################
# We fit the sum model to the empirical variogram using GSTools' built-in
# fitting capabilities. As seen in the representation, the variances and length
# scales of the individual models can be accessed by the attributes
# :any:`SumModel.vars` and :any:`SumModel.len_scales`.

fit_model.fit_variogram(bin_center, gamma)
print(f"{true_model=}")
print(f" {fit_model=}")

###############################################################################
# The variance of a sum model is the sum of the sub variances
# from the contained models. The length scale is a weighted sum of the sub
# length scales where the weights are the ratios of the sub variances
# to the total variance of the sum model.

print(f"{true_model.var=:.2}, {true_model.len_scale=:.2}")
print(f" {fit_model.var=:.2},  {fit_model.len_scale=:.2}")

###############################################################################
# After fitting, we can visualize the empirical variogram alongside the
# fitted sum model and its components. A Sum Model is subscriptable to access
# the individual models its contains.

ax = fit_model.plot(x_max=max(bin_center))
ax.scatter(bin_center, gamma)
# Extract individual components
fit_model[0].plot(x_max=max(bin_center), ax=ax)
fit_model[1].plot(x_max=max(bin_center), ax=ax)

###############################################################################
# As we can see, the fitted sum model closely matches the empirical variogram,
# demonstrating its ability to capture multi-scale variability effectively.
