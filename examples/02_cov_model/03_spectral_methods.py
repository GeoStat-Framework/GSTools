r"""
Spectral methods
================

The spectrum of a covariance model is given by:

.. math:: S(\mathbf{k}) = \left(\frac{1}{2\pi}\right)^n
    \int C(\Vert\mathbf{r}\Vert) e^{i b\mathbf{k}\cdot\mathbf{r}} d^n\mathbf{r}

Since the covariance function :math:`C(r)` is radially symmetric, we can
calculate this by the
`hankel-transformation <https://en.wikipedia.org/wiki/Hankel_transform>`_:

.. math:: S(k) = \left(\frac{1}{2\pi}\right)^n \cdot
    \frac{(2\pi)^{n/2}}{(bk)^{n/2-1}}
    \int_0^\infty r^{n/2-1} C(r) J_{n/2-1}(bkr) r dr

Where :math:`k=\left\Vert\mathbf{k}\right\Vert`.

Depending on the spectrum, the spectral-density is defined by:

.. math:: \tilde{S}(k) = \frac{S(k)}{\sigma^2}

You can access these methods by:
"""
import gstools as gs

model = gs.Gaussian(dim=3, var=2.0, len_scale=10)
ax = model.plot("spectrum")
model.plot("spectral_density", ax=ax)

###############################################################################
# .. note::
#    The spectral-density is given by the radius of the input phase. But it is
#    **not** a probability density function for the radius of the phase.
#    To obtain the pdf for the phase-radius, you can use the methods
#    :any:`CovModel.spectral_rad_pdf`
#    or :any:`CovModel.ln_spectral_rad_pdf` for the logarithm.
#
#    The user can also provide a cdf (cumulative distribution function) by
#    defining a method called ``spectral_rad_cdf``
#    and/or a ppf (percent-point function)
#    by ``spectral_rad_ppf``.
#
#    The attributes :any:`CovModel.has_cdf`
#    and :any:`CovModel.has_ppf` will check for that.
