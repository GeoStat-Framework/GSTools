"""
Fit Variogram
-------------
"""
import numpy as np
import gstools as gs

###############################################################################
# Generate a synthetic field with an exponential model.

x = np.random.RandomState(19970221).rand(1000) * 100.0
y = np.random.RandomState(20011012).rand(1000) * 100.0
model = gs.Exponential(dim=2, var=2, len_scale=8)
srf = gs.SRF(model, mean=0, seed=19970221)
field = srf((x, y))

###############################################################################
# Estimate the variogram of the field with 40 bins.

bins = np.arange(40)
bin_center, gamma = gs.vario_estimate((x, y), field, bins)

###############################################################################
# Fit the variogram with a stable model (no nugget fitted).

fit_model = gs.Stable(dim=2)
fit_model.fit_variogram(bin_center, gamma, nugget=False)

###############################################################################
# Plot the fitting result.

ax = fit_model.plot(x_max=40)
ax.scatter(bin_center, gamma)
print(fit_model)
