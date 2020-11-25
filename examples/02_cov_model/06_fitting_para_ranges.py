"""
Fitting variogram data
======================

The model class comes with a routine to fit the model-parameters to given
variogram data. In the following we will use the self defined stable model
from a previous example.
"""
import numpy as np
import gstools as gs


class Stab(gs.CovModel):
    def default_opt_arg(self):
        return {"alpha": 1.5}

    def cor(self, h):
        return np.exp(-(h ** self.alpha))


# Exemplary variogram data (e.g. estimated from field observations)
bins = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
est_vario = [0.2, 0.5, 0.6, 0.8, 0.8, 0.9]
# fitting model
model = Stab(dim=2)
# we have to provide boundaries for the parameters
model.set_arg_bounds(alpha=[0, 3])
results, pcov = model.fit_variogram(bins, est_vario, nugget=False)
print("Results:", results)

###############################################################################

ax = model.plot()
ax.scatter(bins, est_vario, color="k", label="sample variogram")
ax.legend()


###############################################################################
# As you can see, we have to provide boundaries for the parameters.
# As a default, the following bounds are set:
#
# - additional parameters: ``[-np.inf, np.inf]``
# - variance: ``[0.0, np.inf]``
# - len_scale: ``[0.0, np.inf]``
# - nugget: ``[0.0, np.inf]``
#
# Also, you can deselect parameters from fitting, so their predefined values
# will be kept. In our case, we fixed a ``nugget`` of ``0.0``, which was set
# by default. You can deselect any standard or
# optional argument of the covariance model.
# The second return value ``pcov`` is the estimated covariance of ``popt`` from
# the used scipy routine :any:`scipy.optimize.curve_fit`.
#
# You can use the following methods to manipulate the used bounds:
#
# .. currentmodule:: gstools.covmodel
#
# .. autosummary::
#    CovModel.default_opt_arg_bounds
#    CovModel.default_arg_bounds
#    CovModel.set_arg_bounds
#    CovModel.check_arg_bounds
#
# You can override the :any:`CovModel.default_opt_arg_bounds`
# to provide standard bounds for your additional parameters.
#
# To access the bounds you can use:
#
# .. autosummary::
#    CovModel.var_bounds
#    CovModel.len_scale_bounds
#    CovModel.nugget_bounds
#    CovModel.opt_arg_bounds
#    CovModel.arg_bounds
