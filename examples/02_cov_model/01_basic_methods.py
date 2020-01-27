r"""
Basic Methods
=============

The covariance model class :any:`CovModel` of GSTools provides a set of handy
methods.

One of the following functions defines the main characterization of the
variogram:

- ``CovModel.variogram`` : The variogram of the model given by

  .. math::
      \gamma\left(r\right)=
      \sigma^2\cdot\left(1-\rho\left(r\right)\right)+n

- ``CovModel.covariance`` : The (auto-)covariance of the model given by

  .. math::
      C\left(r\right)= \sigma^2\cdot\rho\left(r\right)

- ``CovModel.correlation`` : The (auto-)correlation
  (or normalized covariance) of the model given by

  .. math::
      \rho\left(r\right)

- ``CovModel.cor`` : The normalized correlation taking a
  normalized range given by:

  .. math::
      \mathrm{cor}\left(\frac{r}{\ell}\right) = \rho\left(r\right)


As you can see, it is the easiest way to define a covariance model by giving a
correlation function as demonstrated in the introductory example.
If one of the above functions is given, the others will be determined:
"""
import gstools as gs

model = gs.Exponential(dim=3, var=2.0, len_scale=10, nugget=0.5)
ax = model.plot("variogram")
model.plot("covariance", ax=ax)
model.plot("correlation", ax=ax)
