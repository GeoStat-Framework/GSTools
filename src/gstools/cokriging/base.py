"""
GStools subpackage providing collocated cokriging.

.. currentmodule:: gstools.cokriging.base

The following classes are provided

.. autosummary::
   CollocatedCokriging
"""

import numpy as np

from gstools.cokriging.correlogram import Correlogram
from gstools.krige.base import Krige

__all__ = ["CollocatedCokriging"]


class CollocatedCokriging(Krige):
    """
    Collocated cokriging base class using Correlogram models.

    Collocated cokriging uses secondary data at the estimation location
    to improve the primary variable estimate. The cross-covariance structure
    is defined by a :any:`Correlogram` object (e.g., :any:`MarkovModel1`).

    Two algorithms are supported: Simple Collocated ("simple") uses only
    collocated secondary at the estimation point, while Intrinsic Collocated
    ("intrinsic") additionally uses secondary data at all primary locations
    for more accurate variance estimation.

    Parameters
    ----------
    correlogram : :any:`Correlogram`
        Correlogram object defining the cross-covariance structure between
        primary and secondary variables (e.g., :any:`MarkovModel1`).
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the primary variable conditions (nan values will be ignored)
    algorithm : :class:`str`
        Cokriging algorithm to use. Either "simple" (SCCK) or "intrinsic" (ICCK).
    secondary_cond_pos : :class:`list`, optional
        tuple, containing secondary variable condition positions (only for ICCK)
    secondary_cond_val : :class:`numpy.ndarray`, optional
        values of secondary variable at primary locations (only for ICCK)
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the input data to gain normality.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        A callable trend function. Should have the signature: f(x, [y, z, ...])
        This is used for detrended kriging, where the trend is subtracted
        from the conditions before kriging is applied.
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class:`float` or :class:`list`, optional
        The measurement error at the conditioning points.
        Either "nugget" to apply the model-nugget, a single value applied to
        all points or an array with individual values for each point.
        The measurement error has to be <= nugget.
        The "exact=True" variant only works with "cond_err='nugget'".
        Default: "nugget"
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: True
    pseudo_inv_type : :class:`str` or :any:`callable`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:

            * `"pinv"`: use `pinv` from `scipy` which uses `SVD`
            * `"pinvh"`: use `pinvh` from `scipy` which uses eigen-values

        If you want to use another routine to invert the kriging matrix,
        you can pass a callable which takes a matrix and returns the inverse.
        Default: `"pinv"`
    fit_normalizer : :class:`bool`, optional
        Whether to fit the data-normalizer to the given conditioning data.
        Default: False
    fit_variogram : :class:`bool`, optional
        Whether to fit the given variogram model to the data.
        Directional variogram fitting is triggered by setting
        any anisotropy factor of the model to anything unequal 1
        but the main axes of correlation are taken from the model
        rotation angles. If the model is a spatio-temporal latlon
        model, this will raise an error.
        This assumes the sill to be the data variance and with
        standard bins provided by the :any:`standard_bins` routine.
        Default: False

    References
    ----------
    .. [Samson2020] Samson, M., & Deutsch, C. V. (2020). Collocated Cokriging.
       In J. L. Deutsch (Ed.), Geostatistics Lessons. Retrieved from
       http://geostatisticslessons.com/lessons/collocatedcokriging
    .. [Wackernagel2003] Wackernagel, H. Multivariate Geostatistics,
       Springer, Berlin, 2003.
    """

    def __init__(
        self,
        correlogram,
        cond_pos,
        cond_val,
        algorithm,
        secondary_cond_pos=None,
        secondary_cond_val=None,
        normalizer=None,
        trend=None,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type="pinv",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        # Validate correlogram
        if not isinstance(correlogram, Correlogram):
            raise TypeError(
                f"correlogram must be a Correlogram instance, got {type(correlogram)}"
            )
        self.correlogram = correlogram

        # validate algorithm parameter
        if algorithm not in ["simple", "intrinsic"]:
            raise ValueError("algorithm must be 'simple' or 'intrinsic'")
        self.algorithm = algorithm

        # handle secondary conditioning data (required for intrinsic)
        if algorithm == "intrinsic":
            if secondary_cond_pos is None or secondary_cond_val is None:
                raise ValueError(
                    "secondary_cond_pos and secondary_cond_val required for ICCK"
                )
            # ICCK requires secondary data collocated with the primary data
            prim = np.asarray(cond_pos, dtype=np.double).reshape(
                correlogram.primary_model.dim, -1
            )
            sec_pos = np.asarray(secondary_cond_pos, dtype=np.double).reshape(
                correlogram.primary_model.dim, -1
            )
            if sec_pos.shape != prim.shape or not np.allclose(sec_pos, prim):
                raise ValueError(
                    "ICCK requires secondary_cond_pos to be collocated with "
                    "cond_pos (secondary data given at the primary locations)"
                )
            # length must match the primary values *before* NaN filtering
            raw_cond_val = np.asarray(cond_val, dtype=np.double).reshape(-1)
            secondary_cond_val = np.asarray(
                secondary_cond_val, dtype=np.double
            ).reshape(-1)
            if len(secondary_cond_val) != len(raw_cond_val):
                raise ValueError(
                    "secondary_cond_val must have same length as primary cond_val"
                )
            # drop secondary values whose primary is non-finite, matching the
            # mask that Krige.set_condition applies to cond_pos/cond_val
            finite_mask = np.isfinite(raw_cond_val)
            self.secondary_cond_pos = secondary_cond_pos
            self.secondary_cond_val = secondary_cond_val[finite_mask]
        else:
            self.secondary_cond_pos = None
            self.secondary_cond_val = None

        # initialize as simple kriging (unbiased=False)
        super().__init__(
            model=correlogram.primary_model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            mean=correlogram.primary_mean,
            unbiased=False,  # Simple kriging base
            normalizer=normalizer,
            trend=trend,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )

    def __call__(self, pos=None, secondary_data=None, **kwargs):
        """
        Generate the collocated cokriging field.

        The cokriging field is saved as `self.field` and is also returned.
        The cokriging error variance is saved as `self.krige_var` and is
        also returned (if ``return_var`` is True).

        Parameters
        ----------
        pos : :class:`list`
            the position tuple, containing main direction and transversal
            directions (x, [y, z])
        secondary_data : :class:`numpy.ndarray`
            Secondary variable values at the given evaluation positions.
            Must have one value per evaluation point.
        **kwargs
            Keyword arguments passed to :any:`Krige.__call__`
            (e.g. ``mesh_type``, ``chunk_size``, ``return_var``, ``store``,
            ``post_process``).

        Returns
        -------
        field : :class:`numpy.ndarray`
            the collocated cokriging field
        krige_var : :class:`numpy.ndarray`, optional
            the collocated cokriging error variance (if ``return_var`` is True)
        """
        if secondary_data is None:
            raise ValueError(
                "secondary_data required for collocated cokriging. "
                "Note: collocated cokriging objects cannot be wrapped in "
                "CondSRF, which provides no secondary_data channel."
            )
        if kwargs.get("only_mean", False):
            raise NotImplementedError(
                "only_mean is not supported for collocated cokriging"
            )

        return_var = kwargs.pop("return_var", True)
        store = kwargs.pop("store", True)
        post_process = kwargs.pop("post_process", True)

        # SCCK's collocated weight depends on SK variance, so it always
        # needs the variance; ICCK's field path does not.
        need_var = return_var or self.algorithm == "simple"

        # solve simple kriging in residual/normal space (no store, no post)
        sk_result = super().__call__(
            pos=pos,
            return_var=need_var,
            store=False,
            post_process=False,
            **kwargs,
        )
        if need_var:
            sk_field, sk_var = sk_result
        else:
            sk_field, sk_var = sk_result, None

        secondary_data = self._prepare_secondary(
            secondary_data, sk_field.shape
        )

        if self.algorithm == "simple":
            ck_field, ck_var = self._apply_simple_collocated(
                sk_field, sk_var, secondary_data, return_var
            )
        else:  # "intrinsic" (validated in __init__)
            ck_field, ck_var = self._apply_intrinsic_collocated(
                sk_field, sk_var, secondary_data, return_var
            )

        # post-process (mean/normalizer/trend) and store exactly once
        ck_field = self.post_field(ck_field, "field", post_process, store)
        if return_var:
            ck_var = self.post_field(ck_var, "krige_var", False, store)
            return ck_field, ck_var
        return ck_field

    def _apply_simple_collocated(
        self, sk_field, sk_var, secondary_data, return_var
    ):
        """Apply simple collocated cokriging in residual/normal space."""
        C_Z0, C_Y0, C_YZ0 = self._compute_covariances()
        k = C_YZ0 / C_Z0

        # collocated secondary weight (depends on SK variance)
        numerator = k * sk_var
        denominator = C_Y0 - (k**2) * (C_Z0 - sk_var)
        collocated_weights = np.where(
            np.abs(denominator) < 1e-15, 0.0, numerator / denominator
        )

        # residual-space estimator: (1 - k*lam)*sk_resid + lam*(sec - m_Y)
        scck_field = sk_field * (1 - k * collocated_weights) + (
            collocated_weights
            * (secondary_data - self.correlogram.secondary_mean)
        )

        if return_var:
            scck_variance = np.maximum(
                0.0, sk_var * (1 - collocated_weights * k)
            )
        else:
            scck_variance = None
        return scck_field, scck_variance

    def _apply_intrinsic_collocated(
        self, sk_field, sk_var, secondary_data, return_var
    ):
        """
        Apply intrinsic collocated cokriging in residual/normal space.

        The secondary-at-primary contribution is already added to the
        residual-space field during the kriging solve in :any:`_summate`.
        Here we add only the collocated secondary contribution at the
        evaluation points, so both secondary terms live in the same
        (normal) space before a single post-processing step.
        """
        C_Z0, C_Y0, C_YZ0 = self._compute_covariances()
        if C_Y0 < 1e-15:
            lambda_Y0 = 0.0
        else:
            lambda_Y0 = C_YZ0 / C_Y0
        icck_field = sk_field + lambda_Y0 * (
            secondary_data - self.correlogram.secondary_mean
        )

        if return_var:
            if C_Y0 * C_Z0 < 1e-15:
                rho_squared = 0.0
            else:
                rho_squared = (C_YZ0**2) / (C_Y0 * C_Z0)
            icck_var = np.maximum(0.0, (1.0 - rho_squared) * sk_var)
        else:
            icck_var = None
        return icck_field, icck_var

    def _summate(self, field, krige_var, c_slice, k_vec, return_var):
        """Fill the residual-space field (and variance) for one chunk.

        Computes kriging weights once in NumPy and derives field and variance
        directly, avoiding the redundant Cython solve from super()._summate.
        For the intrinsic algorithm, the secondary-at-primary contribution is
        also added here so both secondary terms are in residual space before
        the single post_field call in __call__.
        """
        sk_weights = self._krige_mat @ k_vec
        field[c_slice] = self._krige_cond @ sk_weights
        if return_var:
            krige_var[c_slice] = np.sum(k_vec * sk_weights, axis=0)

        if self.algorithm == "simple":
            return

        # intrinsic: add secondary-at-primary contribution
        C_Z0, C_Y0, C_YZ0 = self._compute_covariances()
        if abs(C_YZ0) < 1e-15:
            return

        lambda_weights = sk_weights[: self.cond_no]
        mu_weights = -(C_YZ0 / C_Y0) * lambda_weights
        secondary_residuals = (
            self.secondary_cond_val - self.correlogram.secondary_mean
        )
        field[c_slice] += np.sum(
            mu_weights * secondary_residuals[:, None], axis=0
        )

    def _prepare_secondary(self, secondary_data, field_shape):
        """
        Validate and reshape secondary data to the evaluation field shape.

        Parameters
        ----------
        secondary_data : array_like
            Secondary variable values, one per evaluation point.
        field_shape : tuple
            Shape of the (residual-space) simple-kriging field.

        Returns
        -------
        numpy.ndarray
            secondary_data reshaped to ``field_shape``.
        """
        secondary_data = np.asarray(secondary_data, dtype=np.double)
        n_expected = int(np.prod(field_shape))
        if secondary_data.size != n_expected:
            raise ValueError(
                "secondary_data must have one value per evaluation point: "
                f"expected {n_expected}, got {secondary_data.size}"
            )
        return secondary_data.reshape(field_shape)

    def _compute_covariances(self):
        """
        Compute covariances at zero lag.

        Delegates to the correlogram object.
        """
        return self.correlogram.compute_covariances()
