#######################
Welcome to GeoStatTools
#######################

.. image:: gstools.png
   :width: 251px
   :align: center


Purpose
=======

GeoStatTools is a library providing geostatistical tools.


Installation
============

    ``pip install gstools``


Spatial Random Field Generation
===============================

The core of this library is the generation of spatial random fields. These fields are generated using the randomisation method, described by `Heße et al. 2014 <https://doi.org/10.1016/j.envsoft.2014.01.013>`_.


Examples
--------

Gaussian Covariance Model
^^^^^^^^^^^^^^^^^^^^^^^^^

This is an example of how to generate a 2 dimensional spatial random field with a gaussian covariance model.

.. code-block:: python

    from gstools import SRF, Gaussian
    import matplotlib.pyplot as pt
    # structured field with a size 100x100 and a grid-size of 1x1
    x = y = range(100)
    model = Gaussian(dim=2, var=1, len_scale=10)
    srf = SRF(model)
    field = srf((x, y), mesh_type='structured')
    pt.imshow(field)
    pt.show()

.. image:: gau_field.png
   :width: 600px
   :align: center


Truncated Power Law Model
^^^^^^^^^^^^^^^^^^^^^^^^^

GSTools also implements truncated power law variograms, which can be represented as a
superposition of scale dependant modes in form of standard variograms, which are truncated by
an upper lengthscale :math:`l_u`.

This example shows the truncated power law based on the `stable model <https://en.wikipedia.org/wiki/Stable_distribution>`_ and is given by

.. image:: http://mathurl.com/yasd47ef.png
   :alt: 'Truncated Power Low - Stable'
   :align: center

which gives Gaussian modes for ``alpha=2`` or exponential modes for ``alpha=1``

This results in:

.. image:: http://mathurl.com//yc669evd.png
   :alt: 'Truncated Power Low - Stable'
   :align: center

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as pt
    from gstools import SRF, TPLStable
    x = y = np.linspace(0, 100, 100)
    model = TPLStable(
        dim=2,           # spatial dimension
        var=1,           # variance (C is calculated internally, so that the variance is actually 1)
        len_scale=10,    # length scale (a.k.a. range)
        nugget=0.1,      # nugget
        anis=0.5,        # anisotropy between main direction and transversal ones
        angles=np.pi/4,  # rotation angles
        alpha=1.5,       # shape parameter from the stable model
        hurst=0.7,       # hurst coefficient from the power law
    )
    srf = SRF(model, mean=1, mode_no=1000, seed=19970221, verbose=True)
    field = srf((x, y), mesh_type='structured', force_moments=True)
    # show the field in xy coordinates
    pt.imshow(field.T, origin="lower")
    pt.show()

.. image:: tplstable_field.png
   :width: 600px
   :align: center


Estimating and fitting variograms
=================================

The spatial structure of a field can be analyzed with the variogram, which contains the same information as the covariance function.

All covariance models can be used to fit given variogram data by a simple interface.


Examples
--------

This is an example of how to estimate the variogram of a 2 dimensional unstructured field and estimate the parameters of the covariance
model again.

.. code-block:: python

    import numpy as np
    from gstools import SRF, Exponential, Stable, estimate_unstructured
    from gstools.covmodel.plot import plot_variogram
    import matplotlib.pyplot as pt
    # generate a synthetic field with an exponential model
    x = np.random.RandomState(19970221).rand(1000) * 100.
    y = np.random.RandomState(20011012).rand(1000) * 100.
    model = Exponential(dim=2, var=2, len_scale=8)
    srf = SRF(model, mean=0, seed=19970221)
    field = srf((x, y))
    # estimate the variogram of the field with 40 bins
    bins = np.arange(40)
    bin_center, gamma = estimate_unstructured((x, y), field, bins)
    pt.plot(bin_center, gamma)
    # fit the variogram with a stable model. (no nugget fitted)
    fit_model = Stable(dim=2)
    fit_model.fit_variogram(bin_center, gamma, nugget=False)
    plot_variogram(fit_model, x_max=40)
    print(fit_model)

Which gives:

``Stable(dim=2, var=1.9235043464004502, len_scale=8.151129163855275, nugget=0.0, anis=[1.], angles=[0.], alpha=1.0518003172227908)``

.. image:: exp_vario_fit.png
   :width: 600px
   :align: center


Requirements
============
- `Numpy >= 1.8.2 <http://www.numpy.org>`_
- `SciPy >= 0.19.1 <http://www.scipy.org>`_
- `hankel >= 0.3.6 <https://github.com/steven-murray/hankel>`_
- `emcee <https://github.com/dfm/emcee>`_
- `pyevtk <https://bitbucket.org/pauloh/pyevtk>`_
- `six <https://github.com/benjaminp/six>`_


License
=======

`GPL <https://github.com/LSchueler/GSTools/blob/master/LICENSE>`_ © 2018


Modules
=======

.. toctree::
   :maxdepth: 2

   main
