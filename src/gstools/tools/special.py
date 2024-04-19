"""
GStools subpackage providing special functions.

.. currentmodule:: gstools.tools.special

The following functions are provided

.. autosummary::
   inc_gamma
   inc_gamma_low
   exp_int
   inc_beta
   tplstable_cor
   tpl_exp_spec_dens
   tpl_gau_spec_dens
"""

# pylint: disable=C0103, E1101
import numpy as np
from scipy import special as sps

__all__ = [
    "confidence_scaling",
    "inc_gamma",
    "inc_gamma_low",
    "exp_int",
    "inc_beta",
    "tplstable_cor",
    "tpl_exp_spec_dens",
    "tpl_gau_spec_dens",
]


# special functions ###########################################################


def confidence_scaling(per=0.95):
    """
    Scaling of standard deviation to get the desired confidence interval.

    Parameters
    ----------
    per : :class:`float`, optional
        Confidence level. The default is 0.95.

    Returns
    -------
    :class:`float`
        Scale to multiply the standard deviation with.
    """
    return np.sqrt(2) * sps.erfinv(per)


def inc_gamma(s, x):
    r"""Calculate the (upper) incomplete gamma function.

    Given by: :math:`\Gamma(s,x) = \int_x^{\infty} t^{s-1}\,e^{-t}\,{\rm d}t`

    Parameters
    ----------
    s : :class:`float`
        exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    if np.isclose(s, 0):
        return sps.exp1(x)
    if np.isclose(s, np.around(s)) and s < -0.5:
        return x**s * sps.expn(int(1 - np.around(s)), x)
    if s < 0:
        return (inc_gamma(s + 1, x) - x**s * np.exp(-x)) / s
    return sps.gamma(s) * sps.gammaincc(s, x)


def inc_gamma_low(s, x):
    r"""Calculate the lower incomplete gamma function.

    Given by: :math:`\gamma(s,x) = \int_0^x t^{s-1}\,e^{-t}\,{\rm d}t`

    Parameters
    ----------
    s : :class:`float`
        exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    if np.isclose(s, np.around(s)) and s < 0.5:
        return np.full_like(x, np.inf, dtype=np.double)
    if s < 0:
        return (inc_gamma_low(s + 1, x) + x**s * np.exp(-x)) / s
    return sps.gamma(s) * sps.gammainc(s, x)


def exp_int(s, x):
    r"""Calculate the exponential integral :math:`E_s(x)`.

    Given by: :math:`E_s(x) = \int_1^\infty \frac{e^{-xt}}{t^s}\,\mathrm dt`

    Parameters
    ----------
    s : :class:`float`
        exponent in the integral (should be > -100)
    x : :class:`numpy.ndarray`
        input values
    """
    if np.isclose(s, 1):
        return sps.exp1(x)
    if np.isclose(s, np.around(s)) and s > -0.5:
        return sps.expn(int(np.around(s)), x)
    x = np.asarray(x, dtype=np.double)
    x_neg = x < 0
    x = np.abs(x)
    x_compare = x ** min((10, max(((1 - s), 1))))
    res = np.empty_like(x)
    # use asymptotic behavior for zeros
    x_zero = np.isclose(x_compare, 0, atol=1e-20)
    x_inf = x > max(30, -s / 2)  # function is like exp(-x)*(1/x + s/x^2)
    x_fin = np.logical_not(np.logical_or(x_zero, x_inf))
    x_fin_pos = np.logical_and(x_fin, np.logical_not(x_neg))
    if s > 1.0:  # limit at x=+0
        res[x_zero] = 1.0 / (s - 1.0)
    else:
        res[x_zero] = np.inf
    res[x_inf] = np.exp(-x[x_inf]) * (x[x_inf] ** -1 - s * x[x_inf] ** -2)
    res[x_fin_pos] = inc_gamma(1 - s, x[x_fin_pos]) * x[x_fin_pos] ** (s - 1)
    res[x_neg] = np.nan  # nan for x < 0
    return res


def inc_beta(a, b, x):
    r"""Calculate the incomplete Beta function.

    Given by: :math:`B(a,b;\,x) = \int_0^x t^{a-1}\,(1-t)^{b-1}\,dt`

    Parameters
    ----------
    a : :class:`float`
        first exponent in the integral
    b : :class:`float`
        second exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    return sps.betainc(a, b, x) * sps.beta(a, b)


def tplstable_cor(r, len_scale, hurst, alpha):
    r"""Calculate the correlation function of the TPLStable model.

    Given by the following correlation function:

    .. math::
       \rho(r) =
       \frac{2H}{\alpha} \cdot
       E_{1+\frac{2H}{\alpha}}
       \left(\left(\frac{r}{\ell}\right)^{\alpha} \right)


    Parameters
    ----------
    r : :class:`numpy.ndarray`
        input values
    len_scale : :class:`float`
        length-scale of the model.
    hurst : :class:`float`
        Hurst coefficient of the power law.
    alpha : :class:`float`, optional
        Shape parameter of the stable model.
    """
    r = np.asarray(np.abs(r / len_scale), dtype=np.double)
    r[np.isclose(r, 0)] = 0  # hack to prevent numerical errors
    res = np.ones_like(r)
    res[r > 0] = (2 * hurst / alpha) * exp_int(
        1 + 2 * hurst / alpha, (r[r > 0]) ** alpha
    )
    return res


def tpl_exp_spec_dens(k, dim, len_scale, hurst, len_low=0.0):
    r"""
    Spectral density of the TPLExponential covariance model.

    Parameters
    ----------
    k : :class:`float`
        Radius of the phase: :math:`k=\left\Vert\mathbf{k}\right\Vert`
    dim : :class:`int`
        Dimension of the model.
    len_scale : :class:`float`
        Length scale of the model.
    hurst : :class:`float`
        Hurst coefficient of the power law.
    len_low : :class:`float`, optional
        The lower length scale truncation of the model.
        Default: 0.0

    Returns
    -------
    :class:`float`
        spectral density of the TPLExponential model
    """
    if np.isclose(len_low, 0.0):
        k = np.asarray(k, dtype=np.double)
        z = (k * len_scale) ** 2
        a = hurst + dim / 2.0
        b = hurst + 0.5
        c = hurst + dim / 2.0 + 1.0
        d = dim / 2.0 + 0.5
        fac = len_scale**dim * hurst * sps.gamma(d) / (np.pi**d * a)
        return fac / (1.0 + z) ** a * sps.hyp2f1(a, b, c, z / (1.0 + z))
    fac_up = (len_scale + len_low) ** (2 * hurst)
    spec_up = tpl_exp_spec_dens(k, dim, len_scale + len_low, hurst)
    fac_low = len_low ** (2 * hurst)
    spec_low = tpl_exp_spec_dens(k, dim, len_low, hurst)
    return (fac_up * spec_up - fac_low * spec_low) / (fac_up - fac_low)


def tpl_gau_spec_dens(k, dim, len_scale, hurst, len_low=0.0):
    r"""
    Spectral density of the TPLGaussian covariance model.

    Parameters
    ----------
    k : :class:`float`
        Radius of the phase: :math:`k=\left\Vert\mathbf{k}\right\Vert`
    dim : :class:`int`
        Dimension of the model.
    len_scale : :class:`float`
        Length scale of the model.
    hurst : :class:`float`
        Hurst coefficient of the power law.
    len_low : :class:`float`, optional
        The lower length scale truncation of the model.
        Default: 0.0

    Returns
    -------
    :class:`float`
        spectral density of the TPLExponential model
    """
    if np.isclose(len_low, 0.0):
        k = np.asarray(k, dtype=np.double)
        z = np.array((k * len_scale / 2.0) ** 2)
        res = np.empty_like(z)
        z_gz = z > 0.1  # greater zero
        z_nz = np.logical_not(z_gz)  # near zero
        a = hurst + dim / 2.0
        fac = (len_scale / 2.0) ** dim * hurst / np.pi ** (dim / 2.0)
        res[z_gz] = fac * inc_gamma_low(a, z[z_gz]) / z[z_gz] ** a
        # first order approximation for z near zero
        res[z_nz] = fac * (1.0 / a - z[z_nz] / (a + 1.0))
        return res
    fac_up = (len_scale + len_low) ** (2 * hurst)
    spec_up = tpl_gau_spec_dens(k, dim, len_scale + len_low, hurst)
    fac_low = len_low ** (2 * hurst)
    spec_low = tpl_gau_spec_dens(k, dim, len_low, hurst)
    return (fac_up * spec_up - fac_low * spec_low) / (fac_up - fac_low)
