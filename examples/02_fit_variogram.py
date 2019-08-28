import numpy as np
from gstools import SRF, Exponential, Stable, vario_estimate_unstructured
# generate a synthetic field with an exponential model
x = np.random.RandomState(19970221).rand(1000) * 100.
y = np.random.RandomState(20011012).rand(1000) * 100.
model = Exponential(dim=2, var=2, len_scale=8)
srf = SRF(model, mean=0, seed=19970221)
field = srf((x, y))
# estimate the variogram of the field with 40 bins
bins = np.arange(40)
bin_center, gamma = vario_estimate_unstructured((x, y), field, bins)
# fit the variogram with a stable model. (no nugget fitted)
fit_model = Stable(dim=2)
fit_model.fit_variogram(bin_center, gamma, nugget=False)
# output
ax = fit_model.plot(x_max=40)
ax.plot(bin_center, gamma)
print(fit_model)
