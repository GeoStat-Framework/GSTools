====================
The covariance model
====================

One of the core-features of GSTools is the powerfull :any:`CovModel`
class, which allows to easy define covariance models by yourself.
The resulting models provide a bunch of nice features to explore the
covariance model.

Let us start with a short example on a self defined model.


Examples
========

Here we reimplement the Gaussian covariance model by defining just the
`correlation <https://en.wikipedia.org/wiki/Autocovariance#Normalization>`_ function:

.. code-block:: python

    from gstools import CovModel
    import numpy as np

    class Gau(CovModel):
        def correlation(self, r):
            return np.exp(-(r/self.len_scale)**2)

