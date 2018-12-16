import numpy as np
from gstools import SRF, Exponential, Stable, estimate_unstructured
from gstools.covmodel.plot import plot_variogram
import matplotlib.pyplot as plt
# generate a synthetic field with an exponential model
x = np.random.RandomState(19970221).rand(1000) * 100.
y = np.random.RandomState(20011012).rand(1000) * 100.
model = Exponential(dim=2, var=2, len_scale=8)
srf = SRF(model, mean=0, seed=19970221)
field = srf((x, y))
# estimate the variogram of the field with 40 bins
bins = np.arange(40)
bin_center, gamma = estimate_unstructured((x, y), field, bins)
plt.plot(bin_center, gamma)
# fit the variogram with a stable model. (no nugget fitted)
fit_model = Stable(dim=2)
fit_model.fit_variogram(bin_center, gamma, nugget=False)
plot_variogram(fit_model, x_max=40)
# output
print(fit_model)
plt.show()
